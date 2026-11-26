"""Unit, MockTransport, and loopback-HTTP tests for open-source LLM providers and grounding."""

from __future__ import annotations

import ast
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from sre_triage_agent.config import LLMConfig
from sre_triage_agent.ingestion import load_bundled_scenario
from sre_triage_agent.llm import (
    DeterministicFallbackProvider,
    LLMConfigError,
    LLMConnectionError,
    LLMGroundingError,
    LLMProvider,
    LLMProviderError,
    LLMResponseParseError,
    LLMTriageResponse,
    MockLLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    ResilientLLMProvider,
    build_llm_provider,
    build_triage_messages,
    extract_json_object,
    validate_and_ground_response,
)
from sre_triage_agent.models import (
    AlertEvent,
    EvidenceRef,
    IncidentBundle,
    LogEntry,
    RemediationStep,
    RunbookStepTemplate,
    RunbookTemplate,
)
from sre_triage_agent.runbooks import RunbookCatalog


class _LoopbackLLMServer:
    """Ephemeral localhost HTTP server supporting GET and POST for LLM adapter tests."""

    def __init__(
        self,
        handler_fn: Callable[[str, str, dict[str, str], bytes], tuple[int, Any, float]],
    ) -> None:
        self.requests: list[tuple[str, str, dict[str, str], str]] = []
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def _dispatch(self, method: str) -> None:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw_body = self.rfile.read(length) if length > 0 else b""
                headers_lower = {k.lower(): v for k, v in self.headers.items()}
                parent.requests.append(
                    (method, self.path, headers_lower, raw_body.decode("utf-8", errors="replace"))
                )
                status, body, delay_s = handler_fn(method, self.path, headers_lower, raw_body)
                if delay_s > 0:
                    time.sleep(delay_s)
                if isinstance(body, (dict, list)):
                    payload_bytes = json.dumps(body).encode("utf-8")
                    content_type = "application/json"
                else:
                    payload_bytes = str(body).encode("utf-8")
                    content_type = "text/plain"
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(payload_bytes)))
                    self.end_headers()
                    self.wfile.write(payload_bytes)
                except OSError:
                    pass

            def do_GET(self) -> None:  # noqa: N802
                self._dispatch("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._dispatch("POST")

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


@pytest.fixture
def start_llm_server():
    servers: list[_LoopbackLLMServer] = []

    def _start(
        handler_fn: Callable[[str, str, dict[str, str], bytes], tuple[int, Any, float]],
    ) -> _LoopbackLLMServer:
        srv = _LoopbackLLMServer(handler_fn)
        servers.append(srv)
        return srv

    yield _start

    for srv in servers:
        srv.close()


def _valid_llm_json_for_bundle(bundle: IncidentBundle, *, fenced: bool = False) -> str:
    alert_id = bundle.alerts[0].alert_id
    log_id = bundle.logs[0].log_id
    payload = {
        "severity": "critical",
        "root_cause_hypothesis": f"Memory exhaustion observed on {bundle.service}",
        "summary": f"Triaged {bundle.service} using verified alerts and logs.",
        "remediation_steps": [
            {
                "step_id": "step-1",
                "runbook_id": "k8s_oom_killed",
                "title": "Inspect pod memory limits",
                "description": "Describe pod to verify OOMKilled exit code 137.",
                "risk_level": "low",
                "command_preview": f"kubectl describe pod -l app={bundle.service} -n production",
                "target_resource": f"deployment/{bundle.service}",
                "expected_effect": "Confirms OOMKilled status without mutating workload.",
                "diff_template": "",
                "evidence_refs": [
                    {
                        "kind": "alert",
                        "ref_id": alert_id,
                        "excerpt": "OOMKilled alert",
                        "reason": "Direct alert trigger",
                    },
                    {
                        "kind": "log",
                        "ref_id": log_id,
                        "excerpt": "Log evidence",
                        "reason": "Direct log line",
                    },
                ],
            }
        ],
    }
    text = json.dumps(payload)
    if fenced:
        return f"Here is the analysis:\n```json\n{text}\n```\nEnd of response."
    return text


def test_build_triage_messages_includes_bundle_ids_and_runbooks() -> None:
    bundle = load_bundled_scenario("pod_oom_killed")
    catalog = RunbookCatalog.load_default()
    candidates = catalog.match_candidates(bundle)

    messages = build_triage_messages(bundle, candidates, max_logs=5)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_payload = json.loads(messages[1]["content"])
    assert user_payload["valid_alert_ids"] == [a.alert_id for a in bundle.alerts]
    assert len(user_payload["logs"]) == 5
    assert len(user_payload["valid_log_ids"]) == 5
    # Prioritized selection keeps CRITICAL/ERROR/WARN logs over INFO logs
    log_levels = {entry["level"] for entry in user_payload["logs"]}
    assert "CRITICAL" in log_levels
    assert user_payload["candidate_runbooks"][0]["runbook_id"] == "k8s_oom_killed"


def test_extract_json_object_repair_and_errors() -> None:
    # Clean JSON with BOM
    assert extract_json_object('\ufeff  {"a": 1}  ') == {"a": 1}

    # Markdown json fence with prose
    fenced = 'Analysis follows:\n```json\n{"severity": "high", "note": "ok {braces}"}\n```\nDone.'
    assert extract_json_object(fenced) == {"severity": "high", "note": "ok {braces}"}

    # Plain fence
    plain_fence = '```\n{"k": "v"}\n```'
    assert extract_json_object(plain_fence) == {"k": "v"}

    # Unclosed leading fence
    unclosed = '```json\n{"partial_fence": true}'
    assert extract_json_object(unclosed) == {"partial_fence": True}

    # Prose with stray non-JSON brace before actual object containing braces in strings
    tricky = 'Note: {not json} prefix {"msg": "value {inside}", "nested": {"ok": 1}} trailing'
    assert extract_json_object(tricky) == {"msg": "value {inside}", "nested": {"ok": 1}}

    # Error cases
    for bad in ("", "   \n\t ", "plain prose without json", "[1, 2, 3]"):
        with pytest.raises(LLMResponseParseError):
            extract_json_object(bad, provider="ollama", model="llama3.1:8b")


def test_extract_json_object_prefers_triage_payload_over_auxiliary_json_blocks() -> None:
    bundle = load_bundled_scenario("pod_oom_killed")
    triage_json = _valid_llm_json_for_bundle(bundle, fenced=False)

    # Fenced auxiliary block before fenced triage block
    two_fences = (
        "Alert context:\n"
        "```json\n"
        '{"alert_id": "alert-real-1", "summary": "context only"}\n'
        "```\n"
        "Remediation payload:\n"
        f"```json\n{triage_json}\n```"
    )
    extracted_two_fences = extract_json_object(two_fences)
    assert "remediation_steps" in extracted_two_fences
    assert "root_cause_hypothesis" in extracted_two_fences

    # Inline JSON object in prose before fenced triage block
    inline_then_fenced = (
        'Considering alert {"alert_id": "alert-real-1"} from Prometheus:\n'
        f"```json\n{triage_json}\n```"
    )
    extracted_inline_fenced = extract_json_object(inline_then_fenced)
    assert "remediation_steps" in extracted_inline_fenced

    # Fenced auxiliary block before inline (unfenced) triage JSON object
    fenced_then_inline = (
        '```json\n{"alert_id": "alert-real-1"}\n```\n'
        f"Final response: {triage_json}"
    )
    extracted_fenced_inline = extract_json_object(fenced_then_inline)
    assert "remediation_steps" in extracted_fenced_inline

    # Single/non-triage fallback still returns first valid JSON object and rejects null/arrays
    assert extract_json_object('prefix {"first": 1} then {"second": 2}') == {"first": 1}
    for non_dict in ("null", "true", "42", '["a", "b"]'):
        with pytest.raises(LLMResponseParseError):
            extract_json_object(non_dict, provider="ollama", model="llama3.1:8b")


def test_validate_and_ground_response_drops_hallucinated_refs_and_ungrounded_steps() -> None:
    bundle = load_bundled_scenario("pod_oom_killed")
    real_alert_id = bundle.alerts[0].alert_id
    real_log_id = bundle.logs[0].log_id

    # Step 1 has 1 fake ref + 1 real ref -> fake pruned, step kept
    # Step 2 has only fabricated/wrong-kind refs -> step dropped
    payload = {
        "severity": "warn",
        "root_cause_hypothesis": "Container memory ceiling too low",
        "summary": "Raise memory limits",
        "remediation_steps": [
            {
                "step_id": "keep-me",
                "runbook_id": "k8s_oom_killed",
                "title": "Inspect pod",
                "description": "Inspect OOM event",
                "risk_level": "low",
                "command_preview": "kubectl describe pod checkout-api",
                "target_resource": "pod/checkout-api",
                "expected_effect": "Shows OOMKilled",
                "evidence_refs": [
                    {"kind": "alert", "ref_id": "hallucinated-alert", "excerpt": "x", "reason": "y"},
                    {"kind": "alert", "ref_id": real_alert_id, "excerpt": "", "reason": ""},
                ],
            },
            {
                "step_id": "drop-me",
                "runbook_id": "k8s_oom_killed",
                "title": "Hallucinated step",
                "risk_level": "medium",
                "command_preview": "kubectl delete pod fake",
                "evidence_refs": [
                    {"kind": "alert", "ref_id": real_log_id, "excerpt": "wrong kind", "reason": "bad"},
                    {"kind": "log", "ref_id": "fake-log", "excerpt": "fake", "reason": "bad"},
                ],
            },
        ],
    }
    resp = validate_and_ground_response(payload, bundle, provider="ollama", model="llama3.1:8b")
    assert resp.severity == "warning"
    assert len(resp.remediation_steps) == 1
    assert resp.remediation_steps[0].step_id == "keep-me"
    assert [ref.ref_id for ref in resp.remediation_steps[0].evidence_refs] == [real_alert_id]
    assert resp.remediation_steps[0].evidence_refs[0].excerpt
    assert resp.remediation_steps[0].evidence_refs[0].reason

    # All steps failing grounding raises LLMGroundingError
    all_hallucinated = {
        "severity": "critical",
        "root_cause_hypothesis": "OOM",
        "summary": "OOM",
        "remediation_steps": [
            {
                "step_id": "bad-step",
                "title": "Bad",
                "risk_level": "low",
                "command_preview": "echo noop",
                "evidence_refs": [{"kind": "log", "ref_id": "nonexistent", "excerpt": "x", "reason": "y"}],
            }
        ],
    }
    with pytest.raises(LLMGroundingError):
        validate_and_ground_response(all_hallucinated, bundle, provider="ollama", model="llama3.1:8b")

    # Empty remediation_steps on non-empty bundle raises LLMResponseParseError
    empty_steps = {
        "severity": "critical",
        "root_cause_hypothesis": "OOM",
        "summary": "OOM",
        "remediation_steps": [],
    }
    with pytest.raises(LLMResponseParseError):
        validate_and_ground_response(empty_steps, bundle, provider="ollama", model="llama3.1:8b")


def test_ollama_provider_mock_transport_triage_and_healthcheck() -> None:
    bundle = load_bundled_scenario("pod_oom_killed")
    catalog = RunbookCatalog.load_default()
    candidates = catalog.match_candidates(bundle)
    captured_requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else None
        captured_requests.append((request.method, request.url.path, body))
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": _valid_llm_json_for_bundle(bundle, fenced=True)}},
            )
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "llama3.1:latest"}, {"name": "qwen2.5:7b"}]},
            )
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    provider = OllamaProvider("http://localhost:11434", "llama3.1", transport=transport)
    assert isinstance(provider, LLMProvider)

    health = provider.healthcheck()
    assert health.healthy is True
    assert health.model_available is True

    resp = provider.triage(bundle, candidates)
    assert resp.provider_used == "ollama"
    assert resp.model_used == "llama3.1"
    assert resp.fallback_used is False
    assert len(resp.remediation_steps) == 1

    chat_req = next(req for req in captured_requests if req[1] == "/api/chat")
    assert chat_req[2] is not None
    assert chat_req[2]["stream"] is False
    assert chat_req[2]["format"] == "json"

    # Missing model in /api/tags -> healthy=False
    missing_provider = OllamaProvider("http://localhost:11434", "mistral:7b", transport=transport)
    missing_health = missing_provider.healthcheck()
    assert missing_health.healthy is False
    assert missing_health.model_available is False


def test_ollama_provider_loopback_http_and_error_mapping(start_llm_server) -> None:
    bundle = load_bundled_scenario("pod_oom_killed")
    catalog = RunbookCatalog.load_default()
    candidates = catalog.match_candidates(bundle)

    # Healthy loopback server
    ok_srv = start_llm_server(
        lambda method, path, _hdrs, _body: (
            (200, {"models": [{"name": "llama3.1:8b"}]}, 0.0)
            if method == "GET" and path == "/api/tags"
            else (200, {"message": {"content": _valid_llm_json_for_bundle(bundle)}}, 0.0)
            if method == "POST" and path == "/api/chat"
            else (404, {"error": "unknown"}, 0.0)
        )
    )
    ok_provider = OllamaProvider(ok_srv.base_url, "llama3.1:8b", timeout_s=5.0)
    assert ok_provider.healthcheck().healthy is True
    assert len(ok_provider.triage(bundle, candidates).remediation_steps) == 1

    # HTTP 500 -> LLMConnectionError
    err_srv = start_llm_server(lambda _m, _p, _h, _b: (500, "ollama crashed", 0.0))
    with pytest.raises(LLMConnectionError) as exc_500:
        OllamaProvider(err_srv.base_url, "llama3.1:8b").triage(bundle, candidates)
    assert exc_500.value.status_code == 500

    # Timeout -> LLMConnectionError
    slow_srv = start_llm_server(
        lambda _m, _p, _h, _b: (200, {"message": {"content": "{}"}}, 0.35)
    )
    with pytest.raises(LLMConnectionError, match="timed out"):
        OllamaProvider(slow_srv.base_url, "llama3.1:8b", timeout_s=0.05).triage(bundle, candidates)

    # Malformed non-JSON content -> LLMResponseParseError
    bad_json_srv = start_llm_server(
        lambda _m, _p, _h, _b: (200, {"message": {"content": "not json at all"}}, 0.0)
    )
    with pytest.raises(LLMResponseParseError):
        OllamaProvider(bad_json_srv.base_url, "llama3.1:8b").triage(bundle, candidates)

    # Invalid configuration -> LLMConfigError
    with pytest.raises(LLMConfigError):
        OllamaProvider("ftp://localhost:11434", "llama3.1:8b")
    with pytest.raises(LLMConfigError):
        OllamaProvider("http://localhost:11434", "   ")
    with pytest.raises(LLMConfigError):
        OllamaProvider("http://localhost:11434", "llama3.1:8b", timeout_s=0)


def test_openai_compat_mock_transport_and_v1_url_normalization() -> None:
    bundle = load_bundled_scenario("api_5xx_db_pool")
    catalog = RunbookCatalog.load_default()
    candidates = catalog.match_candidates(bundle)
    seen_requests: list[tuple[str, dict[str, str], dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else None
        seen_requests.append((request.url.path, dict(request.headers), body))
        if request.url.path == "/v1/chat/completions":
            if body and "response_format" in body:
                return httpx.Response(400, text="unsupported parameter: response_format")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": _valid_llm_json_for_bundle(bundle)}}]},
            )
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    # /v1 suffix must not produce /v1/v1/chat/completions
    provider = OpenAICompatibleProvider(
        "http://localhost:8000/v1/",
        "qwen2.5:7b",
        api_key="local-secret",
        transport=transport,
    )

    health = provider.healthcheck()
    assert health.healthy is True
    assert health.model_available is None

    resp = provider.triage(bundle, candidates)
    assert resp.provider_used == "openai_compatible"
    assert len(resp.remediation_steps) == 1

    # First POST used response_format, second retried without response_format; zero /v1/v1 paths
    post_reqs = [req for req in seen_requests if req[2] is not None]
    assert len(post_reqs) == 2
    assert all(req[0] == "/v1/chat/completions" for req in post_reqs)
    assert post_reqs[0][1].get("authorization") == "Bearer local-secret"
    assert "response_format" in post_reqs[0][2]
    assert "response_format" not in post_reqs[1][2]


def test_openai_compat_loopback_http_auth_and_healthcheck(start_llm_server) -> None:
    bundle = load_bundled_scenario("tls_cert_expiring")
    catalog = RunbookCatalog.load_default()
    candidates = catalog.match_candidates(bundle)

    def handler(method: str, path: str, headers: dict[str, str], _raw: bytes) -> tuple[int, Any, float]:
        if headers.get("authorization") != "Bearer valid-key":
            return (401, {"error": "unauthorized"}, 0.0)
        if method == "GET" and path == "/v1/models":
            return (200, {"data": [{"id": "qwen2.5:7b"}]}, 0.0)
        if method == "POST" and path == "/v1/chat/completions":
            return (
                200,
                {"choices": [{"message": {"content": _valid_llm_json_for_bundle(bundle)}}]},
                0.0,
            )
        return (404, {"error": "not found"}, 0.0)

    srv = start_llm_server(handler)

    unauth = OpenAICompatibleProvider(srv.base_url, "qwen2.5:7b", api_key="wrong-key")
    assert unauth.healthcheck().healthy is False
    with pytest.raises(LLMConnectionError) as exc_401:
        unauth.triage(bundle, candidates)
    assert exc_401.value.status_code == 401

    authed = OpenAICompatibleProvider(srv.base_url, "qwen2.5:7b", api_key="valid-key")
    health = authed.healthcheck()
    assert health.healthy is True
    assert health.model_available is True
    resp = authed.triage(bundle, candidates)
    assert len(resp.remediation_steps) == 1


@pytest.mark.parametrize(
    "scenario_name",
    ["pod_oom_killed", "api_5xx_db_pool", "disk_pressure", "tls_cert_expiring"],
)
def test_deterministic_fallback_all_four_scenarios_and_edge_cases(scenario_name: str) -> None:
    bundle = load_bundled_scenario(scenario_name)
    catalog = RunbookCatalog.load_default()
    candidates = catalog.match_candidates(bundle)

    provider = DeterministicFallbackProvider()
    assert provider.healthcheck().healthy is True

    resp = provider.triage(bundle, candidates)
    assert resp.provider_used == "deterministic"
    assert resp.model_used == "rule-based"
    assert resp.fallback_used is False
    assert len(resp.remediation_steps) >= 3

    valid_alerts = {a.alert_id for a in bundle.alerts}
    valid_logs = {entry.log_id for entry in bundle.logs}
    for step in resp.remediation_steps:
        assert "{" not in step.command_preview
        for ref in step.evidence_refs:
            if ref.kind == "alert":
                assert ref.ref_id in valid_alerts
            else:
                assert ref.ref_id in valid_logs


def test_deterministic_fallback_unmatched_alert_only_log_only_and_empty() -> None:
    provider = DeterministicFallbackProvider()
    now = datetime(2026, 11, 25, 14, 0, 0, tzinfo=timezone.utc)

    # Alert-only unmatched
    alert_only = IncidentBundle(
        incident_id="inc-a",
        title="Custom alert",
        service="billing-api",
        severity="high",
        source_mode="live",
        alerts=[
            AlertEvent(
                alert_id="a-1",
                alertname="CustomQueueLag",
                status="firing",
                severity="high",
                service="billing-api",
                starts_at=now,
            )
        ],
        logs=[],
        created_at=now,
    )
    resp_a = provider.triage(alert_only, [])
    assert len(resp_a.remediation_steps) == 2
    assert all(s.runbook_id == "generic_triage" for s in resp_a.remediation_steps)
    assert resp_a.remediation_steps[0].evidence_refs[0].ref_id == "a-1"

    # Log-only unmatched
    log_only = IncidentBundle(
        incident_id="inc-l",
        title="Custom log",
        service="billing-api",
        severity="warning",
        source_mode="live",
        alerts=[],
        logs=[
            LogEntry(
                log_id="l-1",
                timestamp=now,
                level="ERROR",
                service="billing-api",
                message="Socket reset",
                source="app.log",
                line_number=4,
            )
        ],
        created_at=now,
    )
    resp_l = provider.triage(log_only, [])
    assert len(resp_l.remediation_steps) == 2
    assert resp_l.remediation_steps[0].evidence_refs[0].ref_id == "l-1"

    # Empty bundle
    empty_bundle = IncidentBundle(
        incident_id="inc-empty",
        title="Empty",
        service="unknown",
        severity="info",
        source_mode="live",
        alerts=[],
        logs=[],
        created_at=now,
    )
    resp_empty = provider.triage(empty_bundle, [])
    assert resp_empty.remediation_steps == []


def test_deterministic_fallback_empty_step_runbooks_word_boundary_and_severity_aliases() -> None:
    provider = DeterministicFallbackProvider()
    now = datetime(2026, 11, 25, 14, 0, 0, tzinfo=timezone.utc)

    bundle = IncidentBundle(
        incident_id="inc-empty-rb",
        title="Service outage",
        service="checkout-api",
        severity="fatal",
        source_mode="live",
        alerts=[
            AlertEvent(
                alert_id="a-oom",
                alertname="KubePodOOMKilled",
                status="firing",
                severity="critical",
                service="checkout-api",
                starts_at=now,
            )
        ],
        logs=[
            LogEntry(
                log_id="l-info-collision",
                timestamp=now,
                level="INFO",
                service="checkout-api",
                message="Listener bound on port 15032",
                source="app.log",
                line_number=1,
            ),
            LogEntry(
                log_id="l-err-real",
                timestamp=now,
                level="ERROR",
                service="checkout-api",
                message="Upstream database failure without status code",
                source="app.log",
                line_number=2,
            ),
        ],
        created_at=now,
    )

    empty_step_rb = RunbookTemplate(
        runbook_id="rb_empty",
        title="Empty Steps Runbook",
        description="Has no steps",
        match_alertnames=["KubePodOOMKilled"],
        match_keywords=["503"],
        steps=[],
    )

    # All matched runbooks have empty steps -> falls back cleanly to generic_triage (not [])
    resp_generic = provider.triage(bundle, [empty_step_rb])
    assert resp_generic.severity == "critical"  # "fatal" normalized via _SEVERITY_ALIASES
    assert len(resp_generic.remediation_steps) == 2
    assert all(step.runbook_id == "generic_triage" for step in resp_generic.remediation_steps)

    # Matched empty-step runbook followed by non-empty runbook -> skips empty and uses non-empty
    non_empty_rb = RunbookTemplate(
        runbook_id="rb_real",
        title="Real Steps Runbook",
        description="Has steps",
        match_alertnames=["KubePodOOMKilled"],
        match_keywords=["503"],
        steps=[
            RunbookStepTemplate(
                step_id="step-real-1",
                title="Inspect {service}",
                description="Inspect workload",
                risk_level="low",
                command_template="kubectl get pods -l app={service}",
            )
        ],
    )
    resp_second = provider.triage(bundle, [empty_step_rb, non_empty_rb])
    assert len(resp_second.remediation_steps) == 1
    assert resp_second.remediation_steps[0].runbook_id == "rb_real"
    # Word boundary: keyword "503" must NOT match "15032" in INFO log, so ERROR log is chosen first
    log_ref_ids = [
        ref.ref_id for ref in resp_second.remediation_steps[0].evidence_refs if ref.kind == "log"
    ]
    assert log_ref_ids[0] == "l-err-real"

    # Empty bundle with empty-step runbook still returns []
    empty_bundle = IncidentBundle(
        incident_id="inc-none",
        title="No evidence",
        service="checkout-api",
        severity="warn",
        source_mode="live",
        alerts=[],
        logs=[],
        created_at=now,
    )
    resp_empty = provider.triage(empty_bundle, [empty_step_rb])
    assert resp_empty.severity == "warning"
    assert resp_empty.remediation_steps == []


@pytest.mark.parametrize(
    "ungrounded_mode",
    ["fabricated_ref", "cross_kind", "ws_excerpt", "empty_steps"],
)
def test_resilient_provider_enforces_grounding_standalone(ungrounded_mode: str) -> None:
    bundle = load_bundled_scenario("pod_oom_killed")
    candidates = RunbookCatalog.load_default().match_candidates(bundle)
    real_log_id = bundle.logs[0].log_id
    real_alert_id = bundle.alerts[0].alert_id

    if ungrounded_mode == "fabricated_ref":
        bad_resp = LLMTriageResponse(
            severity="critical",
            root_cause_hypothesis="h",
            summary="s",
            remediation_steps=[
                RemediationStep(
                    step_id="s1",
                    runbook_id="k8s_oom_killed",
                    title="t",
                    description="d",
                    risk_level="low",
                    command_preview="echo 1",
                    target_resource="svc",
                    expected_effect="e",
                    evidence_refs=[
                        EvidenceRef(kind="alert", ref_id="fake-id", excerpt="x", reason="y")
                    ],
                )
            ],
            provider_used="mock",
            model_used="mock-model",
        )
    elif ungrounded_mode == "cross_kind":
        bad_resp = LLMTriageResponse(
            severity="critical",
            root_cause_hypothesis="h",
            summary="s",
            remediation_steps=[
                RemediationStep(
                    step_id="s1",
                    runbook_id="k8s_oom_killed",
                    title="t",
                    description="d",
                    risk_level="low",
                    command_preview="echo 1",
                    target_resource="svc",
                    expected_effect="e",
                    evidence_refs=[
                        EvidenceRef(kind="alert", ref_id=real_log_id, excerpt="x", reason="y")
                    ],
                )
            ],
            provider_used="mock",
            model_used="mock-model",
        )
    elif ungrounded_mode == "ws_excerpt":
        bad_resp = LLMTriageResponse(
            severity="critical",
            root_cause_hypothesis="h",
            summary="s",
            remediation_steps=[
                RemediationStep(
                    step_id="s1",
                    runbook_id="k8s_oom_killed",
                    title="t",
                    description="d",
                    risk_level="low",
                    command_preview="echo 1",
                    target_resource="svc",
                    expected_effect="e",
                    evidence_refs=[
                        EvidenceRef(kind="alert", ref_id=real_alert_id, excerpt="   ", reason="y")
                    ],
                )
            ],
            provider_used="mock",
            model_used="mock-model",
        )
    else:
        bad_resp = LLMTriageResponse(
            severity="critical",
            root_cause_hypothesis="h",
            summary="s",
            remediation_steps=[],
            provider_used="mock",
            model_used="mock-model",
        )

    bad_primary = MockLLMProvider(response=bad_resp, provider_name="buggy", model_name="m")

    # Strict mode raises LLMGroundingError directly from ResilientLLMProvider
    strict = ResilientLLMProvider(bad_primary, allow_fallback=False)
    with pytest.raises(LLMGroundingError):
        strict.triage(bundle, candidates)

    # Fallback mode triggers deterministic fallback and sets fallback_used=True
    resilient = ResilientLLMProvider(bad_primary, allow_fallback=True)
    recovered = resilient.triage(bundle, candidates)
    assert recovered.fallback_used is True
    assert recovered.provider_used == "deterministic"
    assert recovered.fallback_reason is not None and "LLMGroundingError" in recovered.fallback_reason
    assert len(recovered.remediation_steps) >= 3

    # Even a buggy fallback provider cannot bypass grounding in ResilientLLMProvider
    bad_fallback_resilient = ResilientLLMProvider(
        bad_primary,
        fallback=bad_primary,
        allow_fallback=True,
    )
    with pytest.raises(LLMGroundingError):
        bad_fallback_resilient.triage(bundle, candidates)


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: LLMConnectionError("Ollama offline", provider="ollama", model="llama3.1:8b"),
        lambda: LLMResponseParseError("Malformed JSON", provider="ollama", model="llama3.1:8b"),
        lambda: LLMGroundingError("All refs hallucinated", provider="ollama", model="llama3.1:8b"),
    ],
)
def test_resilient_wrapper_fallback_vs_strict_mode(error_factory: Callable[[], LLMProviderError]) -> None:
    bundle = load_bundled_scenario("pod_oom_killed")
    candidates = RunbookCatalog.load_default().match_candidates(bundle)
    failing_primary = MockLLMProvider(error=error_factory(), provider_name="ollama", model_name="llama3.1:8b")

    resilient = ResilientLLMProvider(failing_primary, allow_fallback=True)
    resp = resilient.triage(bundle, candidates)
    assert resp.fallback_used is True
    assert resp.provider_used == "deterministic"
    assert resp.model_used == "rule-based"
    assert resp.fallback_reason is not None and len(resp.fallback_reason) > 0
    assert len(resp.remediation_steps) >= 1

    strict = ResilientLLMProvider(failing_primary, allow_fallback=False)
    with pytest.raises(LLMProviderError):
        strict.triage(bundle, candidates)


def test_mock_provider_and_build_llm_provider_factory() -> None:
    bundle = load_bundled_scenario("pod_oom_killed")
    candidates = RunbookCatalog.load_default().match_candidates(bundle)

    mock = MockLLMProvider(response=_valid_llm_json_for_bundle(bundle, fenced=True))
    resp = mock.triage(bundle, candidates)
    assert resp.provider_used == "mock"
    assert len(mock.calls) == 1
    assert mock.healthcheck().healthy is True

    det = build_llm_provider(LLMConfig(provider="deterministic"))
    assert isinstance(det, DeterministicFallbackProvider)

    ollama_resilient = build_llm_provider(LLMConfig(provider="ollama", allow_fallback=True))
    assert isinstance(ollama_resilient, ResilientLLMProvider)
    assert isinstance(ollama_resilient.primary, OllamaProvider)

    oai_raw = build_llm_provider(
        LLMConfig(provider="openai_compatible", base_url="http://localhost:8000", model="qwen2.5:7b"),
        wrap_resilient=False,
    )
    assert isinstance(oai_raw, OpenAICompatibleProvider)


def test_no_proprietary_llm_sdk_imports() -> None:
    llm_dir = Path(__file__).resolve().parents[2] / "src" / "sre_triage_agent" / "llm"
    forbidden = {"openai", "anthropic", "google", "ollama"}

    for py_file in sorted(llm_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_pkg = alias.name.split(".", 1)[0]
                    assert top_pkg not in forbidden, f"Forbidden import {alias.name} in {py_file}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_pkg = node.module.split(".", 1)[0]
                assert top_pkg not in forbidden, f"Forbidden import {node.module} in {py_file}"
