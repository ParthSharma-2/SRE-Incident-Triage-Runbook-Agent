"""Opaque-box E2E test harness for the DevOps SRE Operational Incident Triage Agent.

Strictly interacts with ``sre_triage_agent`` via CLI subprocesses, local loopback
HTTP servers (``127.0.0.1:0``), files, stdin/stdout/stderr, environment variables,
and JSONL audit logs. Zero ``sre_triage_agent`` Python imports exist anywhere here.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

CANONICAL_SCENARIOS: tuple[str, ...] = (
    "pod_oom_killed",
    "api_5xx_db_pool",
    "disk_pressure",
    "tls_cert_expiring",
)

VALID_SEVERITIES: set[str] = {"critical", "high", "warning", "medium", "low", "info"}


@dataclass(frozen=True)
class CLIResult:
    """Captured result of a CLI or Python subprocess invocation."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def parse_json(self) -> Any:
        """Parse stdout as JSON, raising AssertionError with full diagnostics on failure."""
        text = self.stdout.strip()
        if not text:
            raise AssertionError(
                f"Expected JSON on stdout, got empty output.\n"
                f"args={self.args!r}\nreturncode={self.returncode}\n"
                f"timed_out={self.timed_out}\nstderr={self.stderr!r}"
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"Failed to parse stdout as JSON: {exc}\n"
                f"args={self.args!r}\nreturncode={self.returncode}\n"
                f"stdout={self.stdout!r}\nstderr={self.stderr!r}"
            ) from exc

    @property
    def combined_output(self) -> str:
        """Return stdout and stderr joined for diagnostic or message assertions."""
        return f"{self.stdout}\n{self.stderr}".strip()


def build_clean_env(extra_env: dict[str, str | None] | None = None) -> dict[str, str]:
    """Build an isolated, Windows UTF-8 safe environment for subprocess calls."""
    proc_env = {
        key: val
        for key, val in os.environ.items()
        if not key.upper().startswith("SRE_AGENT_")
    }
    proc_env["PYTHONUTF8"] = "1"
    proc_env["PYTHONIOENCODING"] = "utf-8"
    proc_env["NO_COLOR"] = "1"

    src_str = str(SRC_ROOT)
    existing_pp = proc_env.get("PYTHONPATH", "")
    if existing_pp:
        parts = [p for p in existing_pp.split(os.pathsep) if p]
        if src_str not in parts:
            proc_env["PYTHONPATH"] = os.pathsep.join([src_str, *parts])
    else:
        proc_env["PYTHONPATH"] = src_str

    if extra_env:
        for key, val in extra_env.items():
            if val is None:
                for existing_key in list(proc_env.keys()):
                    if existing_key.upper() == key.upper():
                        proc_env.pop(existing_key, None)
            else:
                proc_env[key] = str(val)
    return proc_env


def _decode_partial(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def exec_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path | str,
    env: dict[str, str | None] | None = None,
    stdin_text: str | None = None,
    timeout: float = 30.0,
) -> CLIResult:
    """Execute a subprocess safely with UTF-8 stdio, isolated env, and timeout."""
    cmd_list = [str(part) for part in cmd]
    proc_env = build_clean_env(env)
    try:
        completed = subprocess.run(
            cmd_list,
            cwd=str(cwd),
            env=proc_env,
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )
        return CLIResult(
            args=cmd_list,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CLIResult(
            args=cmd_list,
            returncode=-1,
            stdout=_decode_partial(exc.stdout),
            stderr=_decode_partial(exc.stderr),
            timed_out=True,
        )


def exec_cli(
    args: Sequence[str],
    *,
    cwd: Path | str,
    env: dict[str, str | None] | None = None,
    stdin_text: str | None = None,
    timeout: float = 30.0,
) -> CLIResult:
    """Invoke ``python -m sre_triage_agent <args>`` in an isolated subprocess."""
    return exec_subprocess(
        [sys.executable, "-m", "sre_triage_agent", *args],
        cwd=cwd,
        env=env,
        stdin_text=stdin_text,
        timeout=timeout,
    )


@pytest.fixture
def run_cli(tmp_path: Path) -> Callable[..., CLIResult]:
    """Pytest fixture invoking the agent CLI with default ``cwd=tmp_path``."""

    def _runner(
        args: Sequence[str],
        *,
        cwd: Path | str | None = None,
        env: dict[str, str | None] | None = None,
        stdin_text: str | None = None,
        timeout: float = 30.0,
    ) -> CLIResult:
        return exec_cli(
            args,
            cwd=cwd if cwd is not None else tmp_path,
            env=env,
            stdin_text=stdin_text,
            timeout=timeout,
        )

    return _runner


@pytest.fixture
def run_python(tmp_path: Path) -> Callable[..., CLIResult]:
    """Pytest fixture running Python subprocesses through the same harness mechanics."""

    def _runner(
        args: Sequence[str],
        *,
        cwd: Path | str | None = None,
        env: dict[str, str | None] | None = None,
        stdin_text: str | None = None,
        timeout: float = 30.0,
    ) -> CLIResult:
        return exec_subprocess(
            [sys.executable, *args],
            cwd=cwd if cwd is not None else tmp_path,
            env=env,
            stdin_text=stdin_text,
            timeout=timeout,
        )

    return _runner


# ---------------------------------------------------------------------------
# Loopback HTTP Mock Servers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapturedRequest:
    """Thread-safe record of an HTTP request received by a mock server."""

    method: str
    path: str
    query: str
    headers: dict[str, str]
    raw_body: bytes
    json_body: Any | None

    def get_header(self, name: str, default: str | None = None) -> str | None:
        """Case-insensitive header lookup."""
        target = name.lower()
        for key, value in self.headers.items():
            if key.lower() == target:
                return value
        return default


class _ThreadedServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


_RESERVED_ID_KEYS = frozenset(
    {
        "alert_id",
        "log_id",
        "runbook_id",
        "alert_url",
        "alert_api_type",
        "log_file",
        "incident_id",
        "step_id",
        "run_id",
        "decision_id",
        "outcome_id",
        "event_id",
        "schema_version",
        "str",
        "string",
        "int",
        "integer",
        "float",
        "bool",
        "boolean",
        "null",
        "none",
        "any",
        "optional",
        "required",
        "<alert_id>",
        "<log_id>",
        "<runbook_id>",
        "alert_ids",
        "log_ids",
        "runbook_ids",
        "log_entries",
        "log_level",
        "log_source",
        "log_lines",
        "alert_events",
        "alert_name",
        "alert_severity",
        "alert_status",
        "alert_count",
        "alert-only",
        "log-only",
    }
)


def _extract_ids_from_messages(messages: Any) -> tuple[list[str], list[str], list[str]]:
    """Extract ingested alert_id, log_id, and runbook_id tokens from LLM messages."""
    non_system_chunks: list[str] = []
    system_chunks: list[str] = []

    def _append_content(target: list[str], content: Any) -> None:
        if isinstance(content, str):
            target.append(content)
        elif isinstance(content, list):
            for part in content:
                target.append(json.dumps(part) if not isinstance(part, str) else part)

    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                target = system_chunks if msg.get("role") == "system" else non_system_chunks
                _append_content(target, msg.get("content", ""))
            elif isinstance(msg, str):
                non_system_chunks.append(msg)
    elif isinstance(messages, str):
        non_system_chunks.append(messages)

    def _scan_blob(blob: str) -> tuple[list[str], list[str], list[str]]:
        if not blob:
            return [], [], []

        def _collect(patterns: Sequence[str]) -> list[str]:
            found: list[str] = []
            seen: set[str] = set()
            for pat in patterns:
                for match in re.findall(pat, blob):
                    token = match.strip().strip("\"'`").rstrip(".,;:)\"'`")
                    if token and token.lower() not in _RESERVED_ID_KEYS and token not in seen:
                        seen.add(token)
                        found.append(token)
            return found

        alert_ids = _collect(
            (
                r'"alert_id"\s*:\s*"([^"]+)"',
                r"'alert_id'\s*:\s*'([^']+)'",
                r"""\balert_id[=:][ \t]*["'`]?([A-Za-z0-9_.:-]+)["'`]?""",
                r"\[alert:([A-Za-z0-9_.:-]+)\]",
                r"\b((?:alt|alert)[-_](?=[A-Za-z0-9_.:-]*\d)[A-Za-z0-9_.:-]+)\b",
            )
        )
        log_ids = _collect(
            (
                r'"log_id"\s*:\s*"([^"]+)"',
                r"'log_id'\s*:\s*'([^']+)'",
                r"""\blog_id[=:][ \t]*["'`]?([A-Za-z0-9_.:-]+)["'`]?""",
                r"\[log:([A-Za-z0-9_.:-]+)\]",
                r"\b(log[-_](?=[A-Za-z0-9_.:-]*\d)[A-Za-z0-9_.:-]+)\b",
            )
        )
        runbook_ids = _collect(
            (
                r'"runbook_id"\s*:\s*"([^"]+)"',
                r"'runbook_id'\s*:\s*'([^']+)'",
                r"""\brunbook_id[=:][ \t]*["'`]?([A-Za-z0-9_.:-]+)["'`]?""",
                r"\b(rb[-_][A-Za-z0-9_.:-]+)\b",
            )
        )
        return alert_ids, log_ids, runbook_ids

    if non_system_chunks:
        extracted = _scan_blob("\n".join(non_system_chunks))
        if any(extracted):
            return extracted

    if system_chunks:
        return _scan_blob("\n".join(system_chunks))

    return [], [], []


class _BaseMockServer:
    """Base ephemeral ``127.0.0.1:0`` mock HTTP server with cancelable delays."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.requests: list[CapturedRequest] = []
        self.status_code: int = 200
        self.delay_s: float = 0.0
        self.mode: str = "valid"
        self.raw_body_override: bytes | str | None = None
        self.response_factory: Callable[[CapturedRequest], Any] | None = None

        outer = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _handle(self, method: str) -> None:
                parsed = urlsplit(self.path)
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                raw_body = self.rfile.read(content_length) if content_length > 0 else b""
                json_body: Any | None = None
                if raw_body:
                    try:
                        json_body = json.loads(raw_body.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        json_body = None

                req = CapturedRequest(
                    method=method,
                    path=parsed.path,
                    query=parsed.query,
                    headers={k: v for k, v in self.headers.items()},
                    raw_body=raw_body,
                    json_body=json_body,
                )
                with outer._lock:
                    outer.requests.append(req)
                    delay = outer.delay_s

                if delay > 0:
                    if outer._stop_event.wait(delay):
                        self.close_connection = True
                        try:
                            shutdown_body = b'{"error": "server shutting down"}'
                            self.send_response(503)
                            self.send_header("Content-Type", "application/json; charset=utf-8")
                            self.send_header("Content-Length", str(len(shutdown_body)))
                            self.send_header("Connection", "close")
                            self.end_headers()
                            if method != "HEAD":
                                self.wfile.write(shutdown_body)
                            self.wfile.flush()
                        except OSError:
                            pass
                        return

                code, headers, body = outer._dispatch(req)
                try:
                    self.send_response(code)
                    for h_name, h_val in headers.items():
                        self.send_header(h_name, h_val)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    if method != "HEAD":
                        self.wfile.write(body)
                except OSError:
                    return

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

            def do_HEAD(self) -> None:
                self._handle("HEAD")

        self._server = _ThreadedServer(("127.0.0.1", 0), _Handler)
        self.port: int = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def configure(
        self,
        *,
        status_code: int | None = None,
        delay_s: float | None = None,
        mode: str | None = None,
        raw_body_override: bytes | str | None = ...,  # type: ignore[assignment]
        response_factory: Callable[[CapturedRequest], Any] | None = ...,  # type: ignore[assignment]
    ) -> None:
        with self._lock:
            if status_code is not None:
                self.status_code = status_code
            if delay_s is not None:
                self.delay_s = delay_s
            if mode is not None:
                self.mode = mode
            if raw_body_override is not ...:
                self.raw_body_override = raw_body_override
            if response_factory is not ...:
                self.response_factory = response_factory

    def clear_requests(self) -> None:
        with self._lock:
            self.requests.clear()

    def get_requests(self) -> list[CapturedRequest]:
        with self._lock:
            return list(self.requests)

    def _json_response(self, code: int, payload: Any) -> tuple[int, dict[str, str], bytes]:
        body = json.dumps(payload).encode("utf-8")
        return code, {"Content-Type": "application/json; charset=utf-8"}, body

    def _raw_response(self, code: int, raw: bytes | str) -> tuple[int, dict[str, str], bytes]:
        body = raw.encode("utf-8") if isinstance(raw, str) else raw
        return code, {"Content-Type": "application/json; charset=utf-8"}, body

    def _dispatch(self, req: CapturedRequest) -> tuple[int, dict[str, str], bytes]:
        raise NotImplementedError

    def close(self) -> None:
        self._stop_event.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    def __enter__(self) -> _BaseMockServer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class PrometheusMockServer(_BaseMockServer):
    """Mock Prometheus HTTP API serving ``GET /api/v1/alerts``."""

    DEFAULT_ALERTS: list[dict[str, Any]] = [
        {
            "labels": {
                "alertname": "PodOOMKilled",
                "severity": "critical",
                "service": "checkout-api",
                "namespace": "production",
                "pod": "checkout-api-7d9f8c-x4k2p",
            },
            "annotations": {
                "summary": "Pod checkout-api-7d9f8c-x4k2p OOMKilled (exit 137)",
                "description": "Container memory limit 512Mi exceeded.",
            },
            "state": "firing",
            "activeAt": "2026-11-25T12:00:00Z",
            "value": "1",
        }
    ]

    def __init__(self, alerts: list[dict[str, Any]] | None = None) -> None:
        self.alerts: list[dict[str, Any]] = (
            [dict(a) for a in alerts] if alerts is not None else [dict(a) for a in self.DEFAULT_ALERTS]
        )
        super().__init__()

    @property
    def alerts_url(self) -> str:
        return f"{self.base_url}/api/v1/alerts"

    def set_alerts(self, alerts: list[dict[str, Any]]) -> None:
        with self._lock:
            self.alerts = [dict(a) for a in alerts]

    def _dispatch(self, req: CapturedRequest) -> tuple[int, dict[str, str], bytes]:
        with self._lock:
            mode = self.mode
            status_code = self.status_code
            raw_override = self.raw_body_override
            factory = self.response_factory
            alerts_snapshot = list(self.alerts)

        if factory is not None:
            produced = factory(req)
            if isinstance(produced, (bytes, str)):
                return self._raw_response(status_code, produced)
            return self._json_response(status_code, produced)

        if mode == "fail_all":
            err_code = status_code if status_code >= 400 else 503
            return self._json_response(err_code, {"status": "error", "error": "mock failure"})

        if req.path == "/api/v1/alerts" and req.method == "GET":
            if raw_override is not None:
                return self._raw_response(status_code, raw_override)
            if status_code != 200:
                return self._json_response(
                    status_code,
                    {"status": "error", "errorType": "internal", "error": f"HTTP {status_code}"},
                )
            return self._json_response(
                200,
                {"status": "success", "data": {"alerts": alerts_snapshot}},
            )

        if req.path == "/api/v2/alerts":
            return self._json_response(404, {"status": "error", "error": "not found"})

        if req.path in ("/", "/-/healthy", "/-/ready"):
            return self._json_response(200, {"status": "ok"})

        return self._json_response(404, {"status": "error", "error": f"unknown path {req.path}"})


class AlertmanagerMockServer(_BaseMockServer):
    """Mock Alertmanager v2 HTTP API serving ``GET /api/v2/alerts``."""

    DEFAULT_ALERTS: list[dict[str, Any]] = [
        {
            "fingerprint": "am-fp-001",
            "labels": {
                "alertname": "APIDBPoolExhausted",
                "severity": "critical",
                "service": "orders-api",
                "namespace": "production",
                "instance": "orders-api-01:8080",
            },
            "annotations": {
                "summary": "Database connection pool exhausted on orders-api",
                "description": "HikariPool active connections 100/100.",
            },
            "startsAt": "2026-11-25T12:05:00Z",
            "endsAt": "0001-01-01T00:00:00Z",
            "status": {"state": "active", "silencedBy": [], "inhibitedBy": []},
        }
    ]

    def __init__(self, alerts: list[dict[str, Any]] | None = None) -> None:
        self.alerts: list[dict[str, Any]] = (
            [dict(a) for a in alerts] if alerts is not None else [dict(a) for a in self.DEFAULT_ALERTS]
        )
        super().__init__()

    @property
    def alerts_url(self) -> str:
        return f"{self.base_url}/api/v2/alerts"

    def set_alerts(self, alerts: list[dict[str, Any]]) -> None:
        with self._lock:
            self.alerts = [dict(a) for a in alerts]

    def _dispatch(self, req: CapturedRequest) -> tuple[int, dict[str, str], bytes]:
        with self._lock:
            mode = self.mode
            status_code = self.status_code
            raw_override = self.raw_body_override
            factory = self.response_factory
            alerts_snapshot = list(self.alerts)

        if factory is not None:
            produced = factory(req)
            if isinstance(produced, (bytes, str)):
                return self._raw_response(status_code, produced)
            return self._json_response(status_code, produced)

        if mode == "fail_all":
            err_code = status_code if status_code >= 400 else 503
            return self._json_response(err_code, {"error": "mock failure"})

        if req.path == "/api/v2/alerts" and req.method == "GET":
            if raw_override is not None:
                return self._raw_response(status_code, raw_override)
            if status_code != 200:
                return self._json_response(status_code, {"error": f"HTTP {status_code}"})
            return self._json_response(200, alerts_snapshot)

        if req.path == "/api/v1/alerts":
            return self._json_response(404, {"error": "not found"})

        if req.path in ("/", "/-/healthy", "/-/ready"):
            return self._json_response(200, {"status": "ok"})

        return self._json_response(404, {"error": f"unknown path {req.path}"})


class _BaseLLMMockServer(_BaseMockServer):
    """Shared prompt-aware response builder for Ollama and OpenAI-compatible mocks."""

    def __init__(self, models: list[str] | None = None) -> None:
        self.models: list[str] = list(models) if models is not None else ["llama3.1:8b", "qwen2.5:7b"]
        self.custom_hypothesis: str | None = None
        self.custom_summary: str | None = None
        self.custom_steps: list[dict[str, Any]] | None = None
        self.custom_command: str | None = None
        self.custom_target: str | None = None
        self.custom_step_title: str | None = None
        super().__init__()

    def _build_assistant_content(self, req: CapturedRequest, mode: str) -> str:
        if mode == "malformed":
            return "Here is the analysis: {not valid json at all :::"

        messages = req.json_body.get("messages", []) if isinstance(req.json_body, dict) else []
        alert_ids, log_ids, runbook_ids = _extract_ids_from_messages(messages)

        if mode == "ungrounded":
            evidence_refs = [
                {
                    "kind": "alert",
                    "ref_id": "nonexistent-fabricated-id-999",
                    "excerpt": "Fabricated alert excerpt",
                    "reason": "Hallucinated evidence ID for grounding check",
                }
            ]
        elif mode == "empty_refs":
            evidence_refs = []
        else:
            evidence_refs = []
            if alert_ids:
                evidence_refs.append(
                    {
                        "kind": "alert",
                        "ref_id": alert_ids[0],
                        "excerpt": "Alert evidence extracted from prompt",
                        "reason": "Alert directly identifies the active incident",
                    }
                )
            if log_ids:
                evidence_refs.append(
                    {
                        "kind": "log",
                        "ref_id": log_ids[0],
                        "excerpt": "Log evidence extracted from prompt",
                        "reason": "Log entry corroborates failure mode",
                    }
                )
            if not evidence_refs:
                evidence_refs.append(
                    {
                        "kind": "alert",
                        "ref_id": "alert-mock-001",
                        "excerpt": "Fallback mock alert evidence",
                        "reason": "Mock default evidence reference",
                    }
                )

        if self.custom_steps is not None:
            steps = []
            for idx, step in enumerate(self.custom_steps, start=1):
                item = dict(step)
                item.setdefault("step_id", f"step-{idx}")
                item.setdefault("runbook_id", runbook_ids[0] if runbook_ids else "rb-mock-001")
                item.setdefault("evidence_refs", evidence_refs)
                steps.append(item)
        else:
            steps = [
                {
                    "step_id": "step-1",
                    "runbook_id": runbook_ids[0] if runbook_ids else "rb-mock-001",
                    "title": self.custom_step_title or "Verify and remediate impacted workload",
                    "description": "Inspect resource pressure and apply safe mitigation.",
                    "risk_level": "low",
                    "command_preview": self.custom_command or "kubectl get pods -A",
                    "target_resource": self.custom_target or "deployment/checkout-api",
                    "expected_effect": "Confirms pod health and restores service stability.",
                    "diff_template": "--- before\n+++ after\n- memory: 512Mi\n+ memory: 1024Mi",
                    "evidence_refs": evidence_refs,
                }
            ]

        payload = {
            "severity": "high",
            "root_cause_hypothesis": (
                self.custom_hypothesis
                or "Mock LLM correlated alert and log signals to identify resource exhaustion."
            ),
            "summary": self.custom_summary or "Mock LLM triage summary with evidence-linked steps.",
            "remediation_steps": steps,
        }
        rendered = json.dumps(payload, indent=2)
        if mode == "fenced":
            return f"```json\n{rendered}\n```"
        return rendered


class OllamaMockServer(_BaseLLMMockServer):
    """Mock Ollama HTTP server supporting ``/api/chat`` and readiness probes."""

    def _dispatch(self, req: CapturedRequest) -> tuple[int, dict[str, str], bytes]:
        with self._lock:
            mode = self.mode
            status_code = self.status_code
            raw_override = self.raw_body_override
            factory = self.response_factory
            models = list(self.models)

        if mode == "fail_all":
            err_code = status_code if status_code >= 400 else 503
            return self._json_response(err_code, {"error": "mock ollama unavailable"})

        if req.method == "GET" and req.path in ("/api/tags", "/api/version", "/"):
            if status_code >= 500:
                return self._json_response(status_code, {"error": "mock ollama error"})
            if req.path == "/api/version":
                return self._json_response(200, {"version": "0.3.12"})
            if req.path == "/":
                return self._raw_response(200, "Ollama is running")
            return self._json_response(
                200,
                {"models": [{"name": m, "model": m} for m in models]},
            )

        if req.method == "POST" and req.path == "/api/chat":
            if factory is not None:
                produced = factory(req)
                if isinstance(produced, (bytes, str)):
                    return self._raw_response(status_code, produced)
                return self._json_response(status_code, produced)
            if raw_override is not None:
                return self._raw_response(status_code, raw_override)
            if status_code != 200:
                return self._json_response(status_code, {"error": f"HTTP {status_code}"})

            model = "llama3.1:8b"
            if isinstance(req.json_body, dict) and isinstance(req.json_body.get("model"), str):
                model = req.json_body["model"]

            content = self._build_assistant_content(req, mode)
            return self._json_response(
                200,
                {
                    "model": model,
                    "created_at": "2026-11-25T12:10:00Z",
                    "message": {"role": "assistant", "content": content},
                    "done": True,
                },
            )

        return self._json_response(404, {"error": f"unknown path {req.path}"})


class OpenAICompatMockServer(_BaseLLMMockServer):
    """Mock OpenAI-compatible HTTP server (vLLM / llama.cpp / LocalAI)."""

    def __init__(self, models: list[str] | None = None) -> None:
        super().__init__(models=models or ["qwen2.5:7b", "mistral:7b"])

    def _dispatch(self, req: CapturedRequest) -> tuple[int, dict[str, str], bytes]:
        with self._lock:
            mode = self.mode
            status_code = self.status_code
            raw_override = self.raw_body_override
            factory = self.response_factory
            models = list(self.models)

        if mode == "fail_all":
            err_code = status_code if status_code >= 400 else 503
            return self._json_response(err_code, {"error": {"message": "mock server unavailable"}})

        if req.method == "GET" and req.path in ("/v1/models", "/models", "/health", "/"):
            if status_code >= 500:
                return self._json_response(status_code, {"error": {"message": "mock server error"}})
            if req.path in ("/health", "/"):
                return self._json_response(200, {"status": "ok"})
            return self._json_response(
                200,
                {
                    "object": "list",
                    "data": [{"id": m, "object": "model", "owned_by": "local"} for m in models],
                },
            )

        if req.method == "POST" and req.path in ("/v1/chat/completions", "/chat/completions"):
            if factory is not None:
                produced = factory(req)
                if isinstance(produced, (bytes, str)):
                    return self._raw_response(status_code, produced)
                return self._json_response(status_code, produced)
            if raw_override is not None:
                return self._raw_response(status_code, raw_override)
            if status_code != 200:
                return self._json_response(
                    status_code,
                    {"error": {"message": f"HTTP {status_code}", "type": "server_error"}},
                )

            model = "qwen2.5:7b"
            if isinstance(req.json_body, dict) and isinstance(req.json_body.get("model"), str):
                model = req.json_body["model"]

            content = self._build_assistant_content(req, mode)
            return self._json_response(
                200,
                {
                    "id": "chatcmpl-mock-001",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        return self._json_response(404, {"error": {"message": f"unknown path {req.path}"}})


@pytest.fixture
def prometheus_server():
    with PrometheusMockServer() as srv:
        yield srv


@pytest.fixture
def alertmanager_server():
    with AlertmanagerMockServer() as srv:
        yield srv


@pytest.fixture
def ollama_server():
    with OllamaMockServer() as srv:
        yield srv


@pytest.fixture
def openai_compat_server():
    with OpenAICompatMockServer() as srv:
        yield srv


# ---------------------------------------------------------------------------
# Resilient Parsing, Config & Audit Helpers
# ---------------------------------------------------------------------------


def extract_list(
    payload: Any,
    candidate_keys: tuple[str, ...] = (
        "events",
        "items",
        "scenarios",
        "results",
        "data",
        "records",
    ),
) -> list[Any]:
    """Extract a list from either a bare JSON array or a wrapper object."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in candidate_keys:
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        list_values = [val for val in payload.values() if isinstance(val, list)]
        if len(list_values) == 1:
            return list_values[0]
    raise AssertionError(f"Could not extract list from JSON payload: {payload!r}")


def read_audit_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL audit log file into a list of parsed dicts."""
    assert path.exists(), f"Expected audit log file at {path}, but it does not exist."
    events: list[dict[str, Any]] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"Invalid JSON on line {idx} of {path}: {exc}\nline={stripped!r}") from exc
        assert isinstance(obj, dict), f"Audit line {idx} must be a JSON object, got {type(obj)}"
        events.append(obj)
    return events


def write_yaml_config(path: Path, data: dict[str, Any]) -> Path:
    """Serialize configuration safely to YAML using ``yaml.safe_dump``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def write_jsonl_log(path: Path, entries: Sequence[dict[str, Any]]) -> Path:
    """Write JSONL log records in UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def write_text_log(path: Path, lines: Sequence[str]) -> Path:
    """Write plain-text log lines in UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")
    return path


def unused_loopback_port() -> int:
    """Reserve and release an ephemeral localhost port to simulate connection refused."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def assert_valid_triage_json_payload(payload: dict[str, Any]) -> None:
    """Verify top-level ``triage --json`` contract and cross-object integrity."""
    for required_key in ("run_id", "results", "decisions", "dry_runs", "audit_log_path"):
        assert required_key in payload, f"Missing {required_key!r} in triage JSON: {payload.keys()}"

    assert isinstance(payload["run_id"], str) and payload["run_id"]
    assert isinstance(payload["results"], list)
    assert isinstance(payload["decisions"], list)
    assert isinstance(payload["dry_runs"], list)
    assert isinstance(payload["audit_log_path"], str) and payload["audit_log_path"]

    step_ids: set[str] = set()
    incident_ids: set[str] = set()
    for result in payload["results"]:
        assert result["run_id"] == payload["run_id"]
        incident_ids.add(result["incident_id"])
        assert result["severity"].lower() in VALID_SEVERITIES
        assert result["source_mode"] in {"bundled", "live"}
        assert isinstance(result["root_cause_hypothesis"], str) and result["root_cause_hypothesis"].strip()
        assert isinstance(result["summary"], str) and result["summary"].strip()
        assert isinstance(result["provider_used"], str) and result["provider_used"]
        assert isinstance(result["model_used"], str) and result["model_used"]
        assert isinstance(result["fallback_used"], bool)

        alert_ids = {a["alert_id"] for a in result["alerts"]}
        log_ids = {l["log_id"] for l in result["logs"]}
        valid_refs = alert_ids | log_ids

        assert len(result["remediation_steps"]) >= 1
        for step in result["remediation_steps"]:
            step_ids.add(step["step_id"])
            assert step["risk_level"] in {"low", "medium", "high"}
            assert isinstance(step["title"], str) and step["title"].strip()
            assert isinstance(step["command_preview"], str) and step["command_preview"].strip()
            assert isinstance(step["target_resource"], str) and step["target_resource"].strip()
            assert isinstance(step["expected_effect"], str) and step["expected_effect"].strip()
            assert len(step["evidence_refs"]) >= 1
            for ref in step["evidence_refs"]:
                assert ref["kind"] in {"alert", "log"}
                assert ref["ref_id"] in valid_refs, (
                    f"Ungrounded evidence ref {ref['ref_id']!r}; valid={sorted(valid_refs)}"
                )
                assert isinstance(ref["excerpt"], str) and ref["excerpt"].strip()
                assert isinstance(ref["reason"], str) and ref["reason"].strip()

    for decision in payload["decisions"]:
        assert decision["run_id"] == payload["run_id"]
        assert decision["incident_id"] in incident_ids
        assert decision["step_id"] in step_ids
        assert decision["decision"] in {"approved", "rejected", "skipped"}

    for dry_run in payload["dry_runs"]:
        assert dry_run["run_id"] == payload["run_id"]
        assert dry_run["incident_id"] in incident_ids
        assert dry_run["step_id"] in step_ids
        assert dry_run["executed_against_real_infra"] is False
        assert dry_run["simulated_status"] in {
            "SUCCESS_DRY_RUN",
            "SKIPPED",
            "WOULD_FAIL_PRECHECK",
        }


def assert_valid_audit_events(events: list[dict[str, Any]]) -> None:
    """Verify canonical ``AuditEvent`` schema on parsed JSONL rows."""
    valid_types = {
        "triage_run_started",
        "triage_completed",
        "remediation_suggested",
        "operator_decision",
        "dry_run_executed",
    }
    assert len(events) >= 1, "Expected at least 1 audit event"
    for ev in events:
        assert ev.get("schema_version") == "1.0"
        assert isinstance(ev.get("event_id"), str) and ev["event_id"]
        assert isinstance(ev.get("timestamp"), str) and "T" in ev["timestamp"]
        assert isinstance(ev.get("run_id"), str) and ev["run_id"]
        assert isinstance(ev.get("incident_id"), str)
        assert ev.get("event_type") in valid_types
        assert isinstance(ev.get("actor"), str) and ev["actor"]
        assert isinstance(ev.get("payload"), dict)
