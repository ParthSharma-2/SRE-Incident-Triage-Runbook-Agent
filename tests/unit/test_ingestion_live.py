"""Unit and loopback-HTTP tests for live log/alert ingestion and incident correlation."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from sre_triage_agent.ingestion import (
    FileLogSource,
    IngestionError,
    PrometheusAlertSource,
    correlate_incidents,
    parse_alerts_payload,
)
from sre_triage_agent.models import AlertEvent, LogEntry


class _LoopbackServer:
    """Ephemeral localhost HTTP server capturing requests for Prometheus/Alertmanager tests."""

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
def start_http_server():
    servers: list[_LoopbackServer] = []

    def _start(handler_fn: Callable[[str, dict[str, str]], tuple[int, Any, float]]) -> _LoopbackServer:
        srv = _LoopbackServer(handler_fn)
        servers.append(srv)
        return srv

    yield _start

    for srv in servers:
        srv.close()


def test_prometheus_and_alertmanager_live_http_ingestion(start_http_server) -> None:
    prom_payload = {
        "status": "success",
        "data": {
            "alerts": [
                {
                    "labels": {
                        "alertname": "KubePodOOMKilled",
                        "severity": "critical",
                        "service": "checkout-api",
                        "pod": "checkout-api-1",
                    },
                    "annotations": {"summary": "OOMKilled"},
                    "state": "firing",
                    "activeAt": "2026-11-25T10:15:42.123456789Z",
                },
                {
                    "labels": {"alertname": "ResolvedNoise", "severity": "info", "service": "checkout-api"},
                    "annotations": {"summary": "Old resolved alert"},
                    "state": "inactive",
                    "activeAt": "2026-11-25T09:00:00Z",
                },
            ]
        },
    }
    prom_srv = start_http_server(
        lambda path, _hdrs: (200, prom_payload, 0.0) if path == "/api/v1/alerts" else (404, {"error": "not found"}, 0.0)
    )

    alerts = PrometheusAlertSource(prom_srv.base_url, api_type="auto").read()
    assert len(alerts) == 1
    assert alerts[0].alertname == "KubePodOOMKilled"
    assert alerts[0].starts_at.tzinfo == timezone.utc
    assert alerts[0].starts_at.microsecond == 123456

    all_alerts = PrometheusAlertSource(
        f"{prom_srv.base_url}/api/v1/alerts",
        api_type="prometheus",
        include_resolved=True,
    ).read()
    assert len(all_alerts) == 2
    assert {a.status for a in all_alerts} == {"firing", "resolved"}

    am_payload = [
        {
            "labels": {
                "alertname": "ApiHigh5xxErrorRate",
                "severity": "critical",
                "service": "orders-service",
            },
            "annotations": {"summary": "5xx spike"},
            "startsAt": "2026-11-25T11:02:15Z",
            "status": {"state": "active"},
        },
        {
            "labels": {"alertname": "ResolvedApi", "severity": "warning", "service": "orders-service"},
            "annotations": {},
            "startsAt": "2026-11-25T08:00:00Z",
            "status": {"state": "resolved"},
        },
    ]
    am_srv = start_http_server(
        lambda path, _hdrs: (200, am_payload, 0.0) if path == "/api/v2/alerts" else (404, "not found", 0.0)
    )

    am_alerts = PrometheusAlertSource(am_srv.base_url, api_type="auto").read()
    assert len(am_alerts) == 1
    assert am_alerts[0].alertname == "ApiHigh5xxErrorRate"
    assert [req[0] for req in am_srv.requests] == ["/api/v1/alerts", "/api/v2/alerts"]

    am_explicit = PrometheusAlertSource(
        am_srv.base_url,
        api_type="alertmanager",
        include_resolved=True,
    ).read()
    assert len(am_explicit) == 2


def test_prometheus_alert_source_auth_headers_and_error_cases(start_http_server) -> None:
    srv = start_http_server(
        lambda path, _hdrs: (
            200,
            {"status": "success", "data": {"alerts": []}},
            0.0,
        )
    )
    PrometheusAlertSource(
        srv.base_url,
        auth_token="secret-token",
        headers={"X-Custom-Org": "sre-team"},
    ).read()
    _, captured_headers = srv.requests[0]
    assert captured_headers["authorization"] == "Bearer secret-token"
    assert captured_headers["x-custom-org"] == "sre-team"

    # 500 error in auto mode must fail fast without falling back to /api/v2/alerts
    err_srv = start_http_server(lambda _p, _h: (500, "internal failure", 0.0))
    with pytest.raises(IngestionError) as exc_info:
        PrometheusAlertSource(err_srv.base_url, api_type="auto").read()
    assert exc_info.value.status_code == 500
    assert len(err_srv.requests) == 1

    # 401/403 unauthorized
    unauth_srv = start_http_server(lambda _p, _h: (401, "unauthorized", 0.0))
    with pytest.raises(IngestionError) as exc_unauth:
        PrometheusAlertSource(unauth_srv.base_url, api_type="prometheus").read()
    assert exc_unauth.value.status_code == 401

    # Malformed JSON
    bad_json_srv = start_http_server(lambda _p, _h: (200, "{bad-json", 0.0))
    with pytest.raises(IngestionError):
        PrometheusAlertSource(bad_json_srv.base_url, api_type="prometheus").read()

    # Prometheus error envelope
    prom_err_srv = start_http_server(
        lambda _p, _h: (200, {"status": "error", "errorType": "bad_data", "error": "invalid"}, 0.0)
    )
    with pytest.raises(IngestionError, match="bad_data"):
        PrometheusAlertSource(prom_err_srv.base_url, api_type="prometheus").read()

    # Strict api_type mismatch
    mismatch_srv = start_http_server(lambda _p, _h: (200, [], 0.0))
    with pytest.raises(IngestionError):
        PrometheusAlertSource(mismatch_srv.base_url, api_type="prometheus").read()

    # Timeout
    slow_srv = start_http_server(
        lambda _p, _h: (200, {"status": "success", "data": {"alerts": []}}, 0.35)
    )
    with pytest.raises(IngestionError, match="Timed out"):
        PrometheusAlertSource(slow_srv.base_url, timeout_s=0.05).read()

    # Connection refused (use an already-closed server port)
    closed_srv = start_http_server(lambda _p, _h: (200, {}, 0.0))
    closed_url = closed_srv.base_url
    closed_srv.close()
    with pytest.raises(IngestionError):
        PrometheusAlertSource(closed_url, timeout_s=1.0).read()

    # Invalid URL / timeout validation
    with pytest.raises(IngestionError):
        PrometheusAlertSource("ftp://localhost:9090")
    with pytest.raises(IngestionError):
        PrometheusAlertSource("http://localhost:9090", timeout_s=0)


def test_file_log_source_formats_tail_cursor_truncation_and_utf8(tmp_path: Path) -> None:
    log_path = tmp_path / "mixed.log"
    content_bytes = (
        b'{"timestamp": "2026-11-25T10:00:00Z", "level": "WARN", "service": "api", "message": "mem 90%", "kubernetes": {"pod": "api-1"}}\n'
        b"\n"
        b'2026-11-25 10:00:05 INFO service=api pod=api-1 msg="steady state"\n'
        b"Nov 25 10:00:10 worker-01 kubelet[120]: ERROR host=worker-01 DiskPressure detected\n"
        b"plain line with invalid utf8 \xff\xfe and CRITICAL failure\n"
    )
    log_path.write_bytes(content_bytes)

    entries = FileLogSource(log_path, reference_year=2026).read()
    assert len(entries) == 4
    assert [e.line_number for e in entries] == [1, 3, 4, 5]
    assert entries[0].attributes["pod"] == "api-1"
    assert entries[1].service == "api"
    assert entries[1].message == "steady state"
    assert entries[2].service == "kubelet"
    assert entries[2].attributes["host"] == "worker-01"
    assert entries[2].timestamp == datetime(2026, 11, 25, 10, 0, 10, tzinfo=timezone.utc)
    assert entries[3].level == "CRITICAL"
    assert "\ufffd" in entries[3].message

    # tail_lines = 2 and 0
    tailed = FileLogSource(log_path, tail_lines=2, reference_year=2026).read()
    assert [e.line_number for e in tailed] == [4, 5]
    assert FileLogSource(log_path, tail_lines=0).read() == []

    # Incremental cursor read + append + truncation recovery
    src = FileLogSource(log_path, reference_year=2026)
    batch1 = src.read_batch(cursor=0)
    assert len(batch1.entries) == 4
    assert batch1.next_cursor == len(content_bytes)

    with open(log_path, "ab") as f:
        f.write(b"2026-11-25T10:01:00Z ERROR service=api appended error\n")

    batch2 = src.read_batch(cursor=batch1.next_cursor)
    assert len(batch2.entries) == 1
    assert batch2.entries[0].line_number == 6
    assert "appended error" in batch2.entries[0].message

    # Truncate smaller than previous cursor -> resets to 0
    log_path.write_text("2026-11-25T10:02:00Z INFO service=api rotated\n", encoding="utf-8")
    batch3 = src.read_batch(cursor=batch2.next_cursor)
    assert len(batch3.entries) == 1
    assert batch3.entries[0].line_number == 1

    # Invalid JSONL handling
    jsonl_path = tmp_path / "bad.log.jsonl"
    jsonl_path.write_text("{not-json\n", encoding="utf-8")
    lenient = FileLogSource(jsonl_path, strict_json=False).read()
    assert lenient[0].attributes.get("parse_warning") == "invalid_json"
    with pytest.raises(IngestionError):
        FileLogSource(jsonl_path, strict_json=True).read()

    # Negative / missing / directory errors
    with pytest.raises(IngestionError):
        FileLogSource(log_path, tail_lines=-1).read()
    with pytest.raises(IngestionError):
        FileLogSource(log_path).read_batch(cursor=-1)
    with pytest.raises(IngestionError):
        FileLogSource(tmp_path / "missing.log").read()
    with pytest.raises(IngestionError):
        FileLogSource(tmp_path).read()


def test_correlate_incidents_edge_cases() -> None:
    assert correlate_incidents([], []) == []

    t0 = datetime(2026, 11, 25, 12, 0, 0, tzinfo=timezone.utc)

    alert_a = AlertEvent(
        alert_id="alert-a",
        alertname="KubePodOOMKilled",
        status="firing",
        severity="critical",
        service="checkout-api",
        instance="worker-node-01:9100",
        labels={"pod": "checkout-api-1", "node": "worker-node-01"},
        starts_at=t0,
    )
    alert_b = AlertEvent(
        alert_id="alert-b",
        alertname="ApiHigh5xxErrorRate",
        status="firing",
        severity="high",
        service="orders-service",
        labels={"pod": "orders-1"},
        starts_at=t0 + timedelta(seconds=30),
    )
    alert_far = AlertEvent(
        alert_id="alert-far",
        alertname="KubePodOOMKilled",
        status="firing",
        severity="warning",
        service="checkout-api",
        starts_at=t0 + timedelta(hours=2),
    )

    # Alert-only
    alert_only = correlate_incidents([alert_a], [], source_mode="live")
    assert len(alert_only) == 1
    assert alert_only[0].logs == []
    assert alert_only[0].service == "checkout-api"

    # Log-only (with naive datetime to verify UTC normalization)
    naive_log = LogEntry(
        log_id="log-naive",
        timestamp=datetime(2026, 11, 25, 12, 1, 0),
        level="ERROR",
        service="billing-api",
        message="Database error",
        source="billing.log",
        line_number=1,
    )
    log_only = correlate_incidents([], [naive_log], source_mode="live")
    assert len(log_only) == 1
    assert log_only[0].alerts == []
    assert log_only[0].severity == "high"
    assert log_only[0].created_at.tzinfo == timezone.utc

    # Pod/host token correlation when log service == "unknown" + multi-service & time separation
    unknown_svc_log = LogEntry(
        log_id="log-token",
        timestamp=t0 + timedelta(seconds=20),
        level="CRITICAL",
        service="unknown",
        message="kernel OOM",
        source="syslog",
        line_number=10,
        attributes={"host": "worker-node-01"},
    )
    bundles = correlate_incidents(
        [alert_a, alert_b, alert_far],
        [unknown_svc_log, naive_log],
        source_mode="live",
        time_window_seconds=900,
    )
    assert len(bundles) == 4
    checkout_primary = next(b for b in bundles if b.service == "checkout-api" and b.severity == "critical")
    assert [entry.log_id for entry in checkout_primary.logs] == ["log-token"]


def test_whitespace_service_normalization_in_alerts_logs_and_correlation(tmp_path: Path) -> None:
    # Alerts with whitespace-only `service` must fall back to subsequent labels or "unknown"
    alerts = parse_alerts_payload(
        [
            {
                "labels": {"alertname": "AlertFallback", "service": "   ", "app": "real-app", "pod": "pod-1"},
                "startsAt": "2026-11-25T13:00:00Z",
                "status": {"state": "active"},
            },
            {
                "labels": {"alertname": "AlertBlankA", "service": "   ", "pod": "pod-a"},
                "startsAt": "2026-11-25T13:00:10Z",
                "status": {"state": "active"},
            },
            {
                "labels": {"alertname": "AlertBlankB", "service": "\t", "pod": "pod-b"},
                "startsAt": "2026-11-25T13:00:20Z",
                "status": {"state": "active"},
            },
        ]
    )
    assert alerts[0].service == "real-app"
    assert alerts[1].service == "unknown"
    assert alerts[2].service == "unknown"

    # Unrelated alerts with blank/"unknown" services and distinct pods must not merge
    alert_bundles = correlate_incidents([alerts[1], alerts[2]], [], source_mode="live")
    assert len(alert_bundles) == 2

    # Direct AlertEvent/LogEntry objects constructed with whitespace service also stay separated
    t0 = datetime(2026, 11, 25, 13, 0, 0, tzinfo=timezone.utc)
    raw_alert_a = AlertEvent(
        alert_id="a-blank-1",
        alertname="A1",
        status="firing",
        severity="warning",
        service="   ",
        labels={"pod": "p1"},
        starts_at=t0,
    )
    raw_alert_b = AlertEvent(
        alert_id="a-blank-2",
        alertname="A2",
        status="firing",
        severity="warning",
        service="\t",
        labels={"pod": "p2"},
        starts_at=t0,
    )
    raw_log = LogEntry(
        log_id="l-blank-1",
        timestamp=t0,
        level="ERROR",
        service="  ",
        message="unrelated error",
        source="app.log",
        line_number=1,
        attributes={"pod": "p3"},
    )
    separated = correlate_incidents([raw_alert_a, raw_alert_b], [raw_log], source_mode="live")
    assert len(separated) == 3
    assert all(b.service == "unknown" for b in separated)

    # Text logs with whitespace-only kv `service` fall back cleanly and collapse double spaces
    log_path = tmp_path / "whitespace_service.log"
    log_path.write_text(
        '2026-11-25T10:00:00Z ERROR service="   " app=real-svc msg="fail"\n'
        '2026-11-25T10:00:05Z WARN service="  " Eviction detected on /var/lib/containerd\n'
        '2026-11-25T10:00:10Z WARN service="  " Eviction manager detected availablefs=4.1% on /var/lib/containerd\n',
        encoding="utf-8",
    )
    entries = FileLogSource(log_path, default_service="   ").read()
    assert entries[0].service == "real-svc"
    assert entries[1].service == "unknown"
    assert entries[2].message == "Eviction manager detected on /var/lib/containerd"
