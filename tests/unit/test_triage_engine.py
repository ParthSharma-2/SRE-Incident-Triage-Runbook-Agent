"""Unit and loopback-HTTP integration tests for TriageEngine and evidence grounding."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from sre_triage_agent.config import AgentConfig, ConfigError, IngestionConfig, LLMConfig
from sre_triage_agent.ingestion import IngestionError
from sre_triage_agent.llm import (
    LLMProviderError,
    LLMTriageResponse,
    MockLLMProvider,
)
from sre_triage_agent.models import (
    AlertEvent,
    EvidenceRef,
    IncidentBundle,
    RemediationStep,
    RunbookTemplate,
    TriageResult,
)
from sre_triage_agent.runbooks import RunbookCatalog
from sre_triage_agent.triage import TriageEngine


class _LoopbackPromServer:
    """Ephemeral localhost HTTP server for live Prometheus/Alertmanager triage tests."""

    def __init__(self, handler_fn: Callable[[str, dict[str, str]], tuple[int, Any, float]]) -> None:
        self.requests: list[tuple[str, dict[str, str]]] = []
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                headers_lower = {k.lower(): v for k, v in self.headers.items()}
                parent.requests.append((self.path, headers_lower))
                status, body, delay_s = handler_fn(self.path, headers_lower)
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
def start_prom_server():
    servers: list[_LoopbackPromServer] = []

    def _start(
        handler_fn: Callable[[str, dict[str, str]], tuple[int, Any, float]],
    ) -> _LoopbackPromServer:
        srv = _LoopbackPromServer(handler_fn)
        servers.append(srv)
        return srv

    yield _start

    for srv in servers:
        srv.close()


def _assert_all_refs_grounded(result: TriageResult) -> None:
    valid_alerts = {a.alert_id for a in result.alerts}
    valid_logs = {entry.log_id for entry in result.logs}
    assert len(result.remediation_steps) >= 1
    for step in result.remediation_steps:
        assert len(step.evidence_refs) >= 1
        for ref in step.evidence_refs:
            assert ref.excerpt.strip()
            assert ref.reason.strip()
            if ref.kind == "alert":
                assert ref.ref_id in valid_alerts
            else:
                assert ref.ref_id in valid_logs


@pytest.mark.parametrize(
    ("scenario_name", "expected_runbook_id"),
    [
        ("pod_oom_killed", "k8s_oom_killed"),
        ("api_5xx_db_pool", "api_db_pool_exhaustion"),
        ("disk_pressure", "host_disk_pressure"),
        ("tls_cert_expiring", "tls_cert_renewal"),
    ],
)
def test_triage_bundled_scenarios_deterministic(
    scenario_name: str,
    expected_runbook_id: str,
) -> None:
    cfg = AgentConfig(llm=LLMConfig(provider="deterministic"))
    engine = TriageEngine(cfg)

    result = engine.triage_scenario(scenario_name)
    assert result.scenario_name == scenario_name
    assert result.source_mode == "bundled"
    assert result.provider_used == "deterministic"
    assert result.model_used == "rule-based"
    assert result.fallback_used is False
    assert result.fallback_reason is None
    assert result.created_at.tzinfo == timezone.utc
    assert len(result.remediation_steps) >= 3
    assert any(step.runbook_id == expected_runbook_id for step in result.remediation_steps)
    for step in result.remediation_steps:
        assert "{" not in step.command_preview
        assert "{" not in step.target_resource
    _assert_all_refs_grounded(result)


def test_triage_all_scenarios_shares_run_id() -> None:
    engine = TriageEngine(AgentConfig(llm=LLMConfig(provider="deterministic")))
    results = engine.triage_all_scenarios()

    assert len(results) == 4
    run_ids = {r.run_id for r in results}
    incident_ids = {r.incident_id for r in results}
    scenarios = {r.scenario_name for r in results}
    assert len(run_ids) == 1
    assert len(incident_ids) == 4
    assert scenarios == {
        "pod_oom_killed",
        "api_5xx_db_pool",
        "disk_pressure",
        "tls_cert_expiring",
    }
    for result in results:
        _assert_all_refs_grounded(result)


def test_triage_with_mock_llm_provider_and_fence_stripping() -> None:
    def fenced_handler(bundle, _runbooks) -> str:
        payload = {
            "severity": "critical",
            "root_cause_hypothesis": "Pod exceeded 512Mi memory limit",
            "summary": "Increase memory limit and verify rollout.",
            "remediation_steps": [
                {
                    "step_id": "mock-step-1",
                    "runbook_id": "k8s_oom_killed",
                    "title": "Inspect OOM pod",
                    "description": "Describe pod state",
                    "risk_level": "low",
                    "command_preview": f"kubectl describe pod -l app={bundle.service}",
                    "target_resource": f"deployment/{bundle.service}",
                    "expected_effect": "Confirms OOMKilled",
                    "evidence_refs": [
                        {
                            "kind": "alert",
                            "ref_id": bundle.alerts[0].alert_id,
                            "excerpt": "OOMKilled alert",
                            "reason": "Matched firing alert",
                        }
                    ],
                }
            ],
        }
        return f"Sure! Here is the JSON triage output:\n```json\n{json.dumps(payload)}\n```"

    mock_provider = MockLLMProvider(handler=fenced_handler)
    engine = TriageEngine(AgentConfig(llm=LLMConfig(allow_fallback=False)), provider=mock_provider)
    result = engine.triage_scenario("pod_oom_killed")

    assert result.provider_used == "mock"
    assert result.fallback_used is False
    assert result.remediation_steps[0].step_id == "mock-step-1"
    _assert_all_refs_grounded(result)


@pytest.mark.parametrize(
    "bad_mode",
    ["fabricated_id", "wrong_kind", "empty_refs", "bypassed_fake_response"],
)
def test_triage_grounding_rejects_fabricated_or_mismatched_evidence_refs(bad_mode: str) -> None:
    def make_bad_provider() -> MockLLMProvider:
        if bad_mode == "fabricated_id":
            return MockLLMProvider(
                response={
                    "severity": "critical",
                    "root_cause_hypothesis": "OOM",
                    "summary": "OOM",
                    "remediation_steps": [
                        {
                            "step_id": "s1",
                            "title": "Step",
                            "risk_level": "low",
                            "command_preview": "echo 1",
                            "evidence_refs": [
                                {"kind": "alert", "ref_id": "does-not-exist", "excerpt": "x", "reason": "y"}
                            ],
                        }
                    ],
                }
            )
        if bad_mode == "wrong_kind":
            return MockLLMProvider(
                handler=lambda bundle, _rbs: {
                    "severity": "critical",
                    "root_cause_hypothesis": "OOM",
                    "summary": "OOM",
                    "remediation_steps": [
                        {
                            "step_id": "s1",
                            "title": "Step",
                            "risk_level": "low",
                            "command_preview": "echo 1",
                            "evidence_refs": [
                                {
                                    "kind": "alert",
                                    "ref_id": bundle.logs[0].log_id,
                                    "excerpt": "wrong kind",
                                    "reason": "mismatch",
                                }
                            ],
                        }
                    ],
                }
            )
        if bad_mode == "empty_refs":
            return MockLLMProvider(
                response={
                    "severity": "critical",
                    "root_cause_hypothesis": "OOM",
                    "summary": "OOM",
                    "remediation_steps": [
                        {
                            "step_id": "s1",
                            "title": "Step",
                            "risk_level": "low",
                            "command_preview": "echo 1",
                            "evidence_refs": [],
                        }
                    ],
                }
            )
        # Direct LLMTriageResponse bypassing validate_and_ground_response (tests TriageEngine guard)
        return MockLLMProvider(
            response=LLMTriageResponse(
                severity="critical",
                root_cause_hypothesis="OOM",
                summary="OOM",
                remediation_steps=[
                    RemediationStep(
                        step_id="s1",
                        runbook_id="k8s_oom_killed",
                        title="Step",
                        description="desc",
                        risk_level="low",
                        command_preview="echo 1",
                        target_resource="pod/x",
                        expected_effect="none",
                        evidence_refs=[
                            EvidenceRef(
                                kind="log",
                                ref_id="fabricated-direct-log-id",
                                excerpt="fake",
                                reason="fake",
                            )
                        ],
                    )
                ],
                provider_used="mock",
                model_used="mock-model",
                fallback_used=False,
            )
        )

    strict_engine = TriageEngine(
        AgentConfig(llm=LLMConfig(allow_fallback=False)),
        provider=make_bad_provider(),
    )
    with pytest.raises(LLMProviderError):
        strict_engine.triage_scenario("pod_oom_killed")

    fallback_engine = TriageEngine(
        AgentConfig(llm=LLMConfig(allow_fallback=True)),
        provider=make_bad_provider(),
    )
    res = fallback_engine.triage_scenario("pod_oom_killed")
    assert res.fallback_used is True
    assert res.provider_used == "deterministic"
    assert res.fallback_reason is not None and len(res.fallback_reason) > 0
    _assert_all_refs_grounded(res)


def test_triage_strict_vs_fallback_on_unreachable_ollama(start_prom_server) -> None:
    closed_srv = start_prom_server(lambda _p, _h: (200, {}, 0.0))
    closed_url = closed_srv.base_url
    closed_srv.close()

    fallback_cfg = AgentConfig(
        llm=LLMConfig(provider="ollama", base_url=closed_url, model="llama3.1:8b", allow_fallback=True)
    )
    fallback_res = TriageEngine(fallback_cfg).triage_scenario("pod_oom_killed")
    assert fallback_res.fallback_used is True
    assert fallback_res.provider_used == "deterministic"
    assert fallback_res.fallback_reason is not None and "LLMConnectionError" in fallback_res.fallback_reason
    _assert_all_refs_grounded(fallback_res)

    strict_cfg = AgentConfig(
        llm=LLMConfig(provider="ollama", base_url=closed_url, model="llama3.1:8b", allow_fallback=False)
    )
    with pytest.raises(LLMProviderError):
        TriageEngine(strict_cfg).triage_scenario("pod_oom_killed")


def test_triage_live_end_to_end_with_loopback_http_and_log_files(
    tmp_path: Path,
    start_prom_server,
) -> None:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prom_payload = {
        "status": "success",
        "data": {
            "alerts": [
                {
                    "labels": {
                        "alertname": "DatabaseConnectionPoolExhausted",
                        "severity": "critical",
                        "service": "orders-service",
                        "namespace": "production",
                        "pod": "orders-service-abc",
                    },
                    "annotations": {
                        "summary": "PostgreSQL connection pool exhausted (50/50)",
                    },
                    "state": "firing",
                    "activeAt": now_iso,
                }
            ]
        },
    }
    srv = start_prom_server(
        lambda path, _hdrs: (200, prom_payload, 0.0)
        if path == "/api/v1/alerts"
        else (404, {"error": "not found"}, 0.0)
    )

    log1 = tmp_path / "orders_a.log"
    log1.write_text(
        f"{now_iso} ERROR service=orders-service pod=orders-service-abc namespace=production "
        'db_host=orders-pg.prod.svc msg="TimeoutError: connection pool exhausted HTTP 503"\n',
        encoding="utf-8",
    )
    log2 = tmp_path / "orders_b.log"
    log2.write_text(
        f"{now_iso} WARN service=orders-service pod=orders-service-abc "
        'msg="Health check degraded: db_pool_waiters=184"\n',
        encoding="utf-8",
    )

    engine = TriageEngine(AgentConfig(llm=LLMConfig(provider="deterministic")))
    results = engine.triage_live(
        log_files=[log1, log2],
        alert_url=srv.base_url,
        alert_api_type="prometheus",
    )
    assert len(results) == 1
    res = results[0]
    assert res.source_mode == "live"
    assert res.scenario_name is None
    assert len(res.alerts) == 1
    assert len(res.logs) == 2
    assert res.remediation_steps[0].runbook_id == "api_db_pool_exhaustion"
    _assert_all_refs_grounded(res)


def test_triage_live_log_only_alert_only_and_empty_sources(
    tmp_path: Path,
    start_prom_server,
) -> None:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    engine = TriageEngine(AgentConfig(llm=LLMConfig(provider="deterministic")))

    # Log-only live triage
    log_path = tmp_path / "disk.log"
    log_path.write_text(
        f"Nov 25 12:18:30 worker-node-03 containerd[1420]: ERROR host=worker-node-03 "
        "No space left on device under /var/lib/containerd\n",
        encoding="utf-8",
    )
    log_results = engine.triage_live(log_files=[log_path])
    assert len(log_results) == 1
    assert log_results[0].alerts == []
    assert log_results[0].remediation_steps[0].runbook_id == "host_disk_pressure"
    _assert_all_refs_grounded(log_results[0])

    # Alert-only live triage
    prom_srv = start_prom_server(
        lambda path, _h: (
            200,
            [
                {
                    "labels": {
                        "alertname": "TlsCertificateExpiringSoon",
                        "severity": "warning",
                        "service": "ingress-gateway",
                        "namespace": "ingress-nginx",
                        "domain": "api.prod.example.com",
                    },
                    "annotations": {"summary": "TLS cert expiring soon"},
                    "startsAt": now_iso,
                    "status": {"state": "active"},
                }
            ],
            0.0,
        )
        if path == "/api/v2/alerts"
        else (404, "not found", 0.0)
    )
    alert_results = engine.triage_live(alert_url=prom_srv.base_url, alert_api_type="alertmanager")
    assert len(alert_results) == 1
    assert alert_results[0].logs == []
    assert alert_results[0].remediation_steps[0].runbook_id == "tls_cert_renewal"
    _assert_all_refs_grounded(alert_results[0])

    # Empty alerts + empty log file -> []
    empty_log = tmp_path / "empty.log"
    empty_log.write_text("", encoding="utf-8")
    empty_srv = start_prom_server(
        lambda _p, _h: (200, {"status": "success", "data": {"alerts": []}}, 0.0)
    )
    assert engine.triage_live(log_files=[empty_log], alert_url=empty_srv.base_url) == []


def test_triage_error_propagation(tmp_path: Path, start_prom_server) -> None:
    engine = TriageEngine(
        AgentConfig(
            llm=LLMConfig(provider="deterministic", allow_fallback=True),
            ingestion=IngestionConfig(),
        )
    )

    # Missing live inputs -> ConfigError
    with pytest.raises(ConfigError):
        engine.triage_live()

    # Unknown scenario -> IngestionError
    with pytest.raises(IngestionError):
        engine.triage_scenario("nonexistent_scenario")

    # Missing live log file -> IngestionError
    with pytest.raises(IngestionError):
        engine.triage_live(log_files=[tmp_path / "missing.log"])

    # HTTP 500 from Prometheus -> IngestionError (must NOT fall back to LLM fallback)
    err_srv = start_prom_server(lambda _p, _h: (500, "internal error", 0.0))
    with pytest.raises(IngestionError):
        engine.triage_live(alert_url=err_srv.base_url, alert_api_type="prometheus")

    # Whitespace-only IngestionConfig.log_file -> ConfigError (not IngestionError)
    ws_engine = TriageEngine(
        AgentConfig(
            llm=LLMConfig(provider="deterministic"),
            ingestion=IngestionConfig(log_file="   "),
        )
    )
    with pytest.raises(ConfigError):
        ws_engine.triage_live()


def test_triage_engine_matched_runbook_empty_steps_falls_back_to_generic_triage() -> None:
    now = datetime(2026, 11, 25, 14, 0, 0, tzinfo=timezone.utc)
    bundle = IncidentBundle(
        incident_id="inc-empty-steps-rb",
        title="OOM alert",
        service="checkout-api",
        severity="critical",
        source_mode="live",
        alerts=[
            AlertEvent(
                alert_id="a1",
                alertname="KubePodOOMKilled",
                status="firing",
                severity="critical",
                service="checkout-api",
                starts_at=now,
            )
        ],
        logs=[],
        created_at=now,
    )
    catalog = RunbookCatalog(
        [
            RunbookTemplate(
                runbook_id="empty_steps_rb",
                title="Empty steps",
                description="Matched runbook with no steps",
                match_alertnames=["KubePodOOMKilled"],
                steps=[],
            )
        ]
    )
    engine = TriageEngine(AgentConfig(llm=LLMConfig(provider="deterministic")), catalog=catalog)
    result = engine.triage_bundle(bundle)
    assert len(result.remediation_steps) == 2
    assert all(step.runbook_id == "generic_triage" for step in result.remediation_steps)
    _assert_all_refs_grounded(result)
