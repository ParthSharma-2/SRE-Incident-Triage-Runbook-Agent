"""Unit and CLI tests for F13 Click + Rich CLI, config precedence, cp1252 safety, and docs."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from rich.console import Console

from sre_triage_agent.audit import AuditLogger
from sre_triage_agent.cli import _render_audit_table, _render_scripted_summary, main
from sre_triage_agent.config import load_config
from sre_triage_agent.llm import DeterministicFallbackProvider
from sre_triage_agent.triage import TriageEngine


@pytest.fixture(autouse=True)
def _clean_sre_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "SRE_AGENT_LLM_PROVIDER",
        "SRE_AGENT_LLM_BASE_URL",
        "SRE_AGENT_LLM_MODEL",
        "SRE_AGENT_LLM_API_KEY",
        "SRE_AGENT_LLM_TIMEOUT_SECONDS",
        "SRE_AGENT_LLM_ALLOW_FALLBACK",
        "SRE_AGENT_PROMETHEUS_URL",
        "SRE_AGENT_ALERT_API_TYPE",
        "SRE_AGENT_LOG_FILE",
        "SRE_AGENT_TAIL_LINES",
        "SRE_AGENT_INCLUDE_RESOLVED",
        "SRE_AGENT_AUDIT_LOG_PATH",
        "SRE_AGENT_OPERATOR",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def http_server_factory():
    servers: list[ThreadingHTTPServer] = []

    def _start(routes: dict[str, tuple[int, dict[str, Any]]]) -> str:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def _handle(self) -> None:
                status, body = routes.get(self.path, (404, {"error": "not found"}))
                raw = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                self._handle()

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                if length:
                    _ = self.rfile.read(length)
                self._handle()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        host, port = server.server_address
        return f"http://{host}:{port}"

    yield _start

    for srv in servers:
        srv.shutdown()
        srv.server_close()


def test_config_example_yaml_loads_cleanly() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_config(repo_root / "config.example.yaml")
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.base_url == "http://localhost:11434"
    assert cfg.audit.audit_log_path == "./audit_log.jsonl"


def test_scenarios_list_table_and_json() -> None:
    runner = CliRunner()

    res_table = runner.invoke(main, ["scenarios", "list"])
    assert res_table.exit_code == 0
    for name in ("pod_oom_killed", "api_5xx_db_pool", "disk_pressure", "tls_cert_expiring"):
        assert name in res_table.stdout

    res_json = runner.invoke(main, ["scenarios", "list", "--json"])
    assert res_json.exit_code == 0
    payload = json.loads(res_json.stdout)
    assert len(payload["scenarios"]) == 4


def test_triage_scripted_json_and_audit_inspection(tmp_path: Path) -> None:
    runner = CliRunner()
    audit_path = tmp_path / "cli_audit.jsonl"

    res = runner.invoke(
        main,
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            "approve: safe inspect,reject: hold memory bump,skip",
            "--operator",
            "cli-tester",
            "--audit-log",
            str(audit_path),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stderr
    assert res.stderr == ""
    data = json.loads(res.stdout)
    assert data["aborted"] is False
    assert len(data["results"]) == 1
    assert len(data["decisions"]) == 3
    assert [d["decision"] for d in data["decisions"]] == ["approved", "rejected", "skipped"]
    assert len(data["dry_runs"]) == 1
    assert data["dry_runs"][0]["executed_against_real_infra"] is False
    assert Path(data["audit_log_path"]) == audit_path

    incident_id = data["results"][0]["incident_id"]
    run_id = data["run_id"]

    # audit list with decision filter
    list_res = runner.invoke(
        main,
        [
            "audit",
            "list",
            "--audit-log",
            str(audit_path),
            "--run-id",
            run_id,
            "--decision",
            "approved",
            "--json",
        ],
    )
    assert list_res.exit_code == 0
    list_data = json.loads(list_res.stdout)
    assert list_data["count"] == 1
    assert list_data["events"][0]["event_type"] == "operator_decision"

    # audit show for the incident
    show_res = runner.invoke(
        main,
        ["audit", "show", incident_id, "--audit-log", str(audit_path), "--json"],
    )
    assert show_res.exit_code == 0
    show_data = json.loads(show_res.stdout)
    event_types = {ev["event_type"] for ev in show_data["events"]}
    assert event_types == {
        "triage_run_started",
        "triage_completed",
        "remediation_suggested",
        "operator_decision",
        "dry_run_executed",
    }

    # audit show for missing incident -> exit 1
    missing_res = runner.invoke(
        main,
        ["audit", "show", "inc-missing", "--audit-log", str(audit_path)],
    )
    assert missing_res.exit_code == 1


def test_triage_interactive_stdin_and_quit_exits_zero(tmp_path: Path) -> None:
    runner = CliRunner()
    audit_path = tmp_path / "interactive_audit.jsonl"

    res = runner.invoke(
        main,
        [
            "triage",
            "--scenario",
            "api_5xx_db_pool",
            "--provider",
            "deterministic",
            "--audit-log",
            str(audit_path),
        ],
        input="a\nr: high risk\nq\n",
    )
    assert res.exit_code == 0, res.stderr
    assert "[HIGH RISK]" in res.stdout or "Audit log written" in res.stdout

    events = AuditLogger(audit_path).read_events()
    decisions = [e for e in events if e.event_type == "operator_decision"]
    dry_runs = [e for e in events if e.event_type == "dry_run_executed"]
    assert len(decisions) == 2
    assert len(dry_runs) == 1


def test_triage_all_scenarios_and_env_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    env_audit = tmp_path / "from_env.jsonl"
    monkeypatch.setenv("SRE_AGENT_LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("SRE_AGENT_AUDIT_LOG_PATH", str(env_audit))
    monkeypatch.setenv("SRE_AGENT_OPERATOR", "env-sre")

    res = runner.invoke(
        main,
        [
            "triage",
            "--all-scenarios",
            "--decisions",
            "approve",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stderr
    data = json.loads(res.stdout)
    assert len(data["results"]) == 4
    assert Path(data["audit_log_path"]) == env_audit
    assert data["decisions"][0]["operator"] == "env-sre"


def test_triage_live_mode_with_prometheus_and_openai_compat_server(
    tmp_path: Path,
    http_server_factory,
) -> None:
    log_file = tmp_path / "live.log"
    log_file.write_text(
        "2026-11-26T01:00:00Z ERROR payment-api Database timeout on orders_db\n",
        encoding="utf-8",
    )
    audit_path = tmp_path / "live_audit.jsonl"

    base_url = http_server_factory(
        {
            "/api/v1/alerts": (
                200,
                {
                    "status": "success",
                    "data": {
                        "alerts": [
                            {
                                "labels": {
                                    "alertname": "ApiHigh5xxErrorRate",
                                    "severity": "critical",
                                    "service": "payment-api",
                                    "namespace": "prod",
                                },
                                "annotations": {"summary": "HTTP 503 spike on payment-api"},
                                "state": "firing",
                                "activeAt": "2026-11-26T01:00:00Z",
                            }
                        ]
                    },
                },
            ),
            "/v1/chat/completions": (
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "severity": "critical",
                                        "root_cause_hypothesis": "DB pool timeout in payment-api",
                                        "summary": "Inspect DB pool and service pods",
                                        "remediation_steps": [
                                            {
                                                "step_id": "step-inspect-payment",
                                                "runbook_id": "api_db_pool_exhaustion",
                                                "title": "Inspect payment-api pods",
                                                "description": "Check pod status",
                                                "risk_level": "low",
                                                "command_preview": "kubectl get pods -n prod -l app=payment-api",
                                                "target_resource": "deployment/payment-api",
                                                "expected_effect": "Verify pod health",
                                                "diff_template": "",
                                                "evidence_refs": [
                                                    {
                                                        "kind": "alert",
                                                        "ref_id": "alert-placeholder",
                                                        "excerpt": "HTTP 503 spike",
                                                        "reason": "Critical alert",
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                )
                            }
                        }
                    ]
                },
            ),
        }
    )

    # First run with deterministic provider to confirm live log + Prometheus HTTP ingestion
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "triage",
            "--log-file",
            str(log_file),
            "--alert-url",
            f"{base_url}/api/v1/alerts",
            "--provider",
            "deterministic",
            "--decisions",
            "approve,skip,skip",
            "--audit-log",
            str(audit_path),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stderr
    data = json.loads(res.stdout)
    assert len(data["results"]) == 1
    assert data["results"][0]["source_mode"] == "live"
    assert len(data["dry_runs"]) == 1

    # Verify strict fallback failure returns exit 1 when endpoint returns ungrounded alert_id
    res_strict = runner.invoke(
        main,
        [
            "triage",
            "--log-file",
            str(log_file),
            "--alert-url",
            f"{base_url}/api/v1/alerts",
            "--provider",
            "openai_compatible",
            "--base-url",
            base_url,
            "--model",
            "local-llm",
            "--no-allow-fallback",
            "--audit-log",
            str(audit_path),
            "--json",
        ],
    )
    assert res_strict.exit_code == 1


def test_providers_check_healthy_and_unhealthy(http_server_factory) -> None:
    runner = CliRunner()

    res_det = runner.invoke(
        main,
        ["providers", "check", "--provider", "deterministic", "--json"],
    )
    assert res_det.exit_code == 0
    assert json.loads(res_det.stdout)["healthy"] is True

    healthy_url = http_server_factory(
        {"/api/tags": (200, {"models": [{"name": "llama3.1:8b"}]})}
    )
    res_ollama = runner.invoke(
        main,
        [
            "providers",
            "check",
            "--provider",
            "ollama",
            "--base-url",
            healthy_url,
            "--model",
            "llama3.1:8b",
        ],
    )
    assert res_ollama.exit_code == 0
    assert "[OK]" in res_ollama.stdout

    unhealthy_url = http_server_factory({"/api/tags": (503, {"error": "down"})})
    res_fail = runner.invoke(
        main,
        [
            "providers",
            "check",
            "--provider",
            "ollama",
            "--base-url",
            unhealthy_url,
            "--model",
            "llama3.1:8b",
            "--json",
        ],
    )
    assert res_fail.exit_code == 1
    assert json.loads(res_fail.stdout)["healthy"] is False


@pytest.mark.parametrize(
    "args",
    [
        ["triage"],
        ["triage", "--scenario", "pod_oom_killed", "--all-scenarios"],
        ["triage", "--scenario", "pod_oom_killed", "--log-file", "app.log"],
        ["triage", "--scenario", "pod_oom_killed", "--decisions", "invalid_choice"],
        ["triage", "--scenario", "pod_oom_killed", "--timeout", "0"],
        ["triage", "--scenario", "pod_oom_killed", "--tail-lines", "0"],
        ["triage", "--scenario", "pod_oom_killed", "--tail-lines", "-1"],
        ["providers", "check", "--timeout", "-2"],
    ],
)
def test_cli_usage_errors_return_exit_code_2(args: list[str]) -> None:
    runner = CliRunner()
    res = runner.invoke(main, args)
    assert res.exit_code == 2


def test_cli_operational_errors_return_exit_code_1(tmp_path: Path) -> None:
    runner = CliRunner()

    res_missing_scenario = runner.invoke(
        main,
        ["triage", "--scenario", "no_such_scenario", "--provider", "deterministic"],
    )
    assert res_missing_scenario.exit_code == 1

    res_missing_log = runner.invoke(
        main,
        [
            "triage",
            "--log-file",
            str(tmp_path / "missing.log"),
            "--provider",
            "deterministic",
        ],
    )
    assert res_missing_log.exit_code == 1


def test_rich_output_is_windows_cp1252_safe(tmp_path: Path) -> None:
    raw_buf = BytesIO()
    cp1252_stream = TextIOWrapper(raw_buf, encoding="cp1252", errors="strict")
    console = Console(file=cp1252_stream, force_terminal=False, no_color=True, safe_box=True)

    engine = TriageEngine(provider=DeterministicFallbackProvider())
    result = engine.triage_scenario("pod_oom_killed", run_id="run-cp1252")
    audit_logger = AuditLogger(tmp_path / "cp1252.jsonl")
    audit_logger.log_triage_completed(result)

    from sre_triage_agent.approval import ScriptedDecisionHandler, process_remediations

    summary = process_remediations(
        result,
        handler=ScriptedDecisionHandler("approve,reject,skip"),
        audit_logger=audit_logger,
    )
    _render_scripted_summary(console, [result], summary, audit_logger.path)
    _render_audit_table(console, "Audit CP1252", audit_logger.read_events())
    cp1252_stream.flush()
    assert b"pod_oom_killed" in raw_buf.getvalue() or b"run-cp1252" in raw_buf.getvalue()


def test_module_and_console_script_help_subprocesses() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "sre_triage_agent", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    for cmd in ("triage", "scenarios", "audit", "providers"):
        assert cmd in proc.stdout

    console_script = shutil.which("sre-agent")
    if console_script is None:
        candidate = Path(sys.executable).with_name("sre-agent.exe")
        if candidate.is_file():
            console_script = str(candidate)
    assert console_script is not None, "sre-agent console script not found on PATH or next to Python executable"
    script_proc = subprocess.run(
        [console_script, "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert script_proc.returncode == 0
    for cmd in ("triage", "scenarios", "audit", "providers"):
        assert cmd in script_proc.stdout


def test_triage_interactive_json_routes_prompts_to_stderr_keeping_stdout_clean(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    audit_path = tmp_path / "interactive_json.jsonl"

    res = runner.invoke(
        main,
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--audit-log",
            str(audit_path),
            "--json",
        ],
        input="a\nq\n",
    )
    assert res.exit_code == 0, res.stderr
    assert "Incident Triage" in res.stderr
    data = json.loads(res.stdout)
    assert data["aborted"] is True
    assert len(data["decisions"]) == 1
    assert data["decisions"][0]["decision"] == "approved"
    assert len(data["dry_runs"]) == 1


def test_triage_non_json_scripted_summary_output(tmp_path: Path) -> None:
    runner = CliRunner()
    audit_path = tmp_path / "scripted_table.jsonl"

    res = runner.invoke(
        main,
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            "approve,skip,skip",
            "--audit-log",
            str(audit_path),
        ],
    )
    assert res.exit_code == 0, res.stderr
    assert "Remediation Steps" in res.stdout
    assert "Audit log written" in res.stdout


def test_triage_live_repeatable_log_files_alertmanager_tail_lines_and_include_resolved(
    tmp_path: Path,
    http_server_factory,
) -> None:
    log_a = tmp_path / "a.log"
    log_a.write_text(
        "\n".join(
            [
                "2026-11-26T01:00:00Z INFO checkout-api Ignored line 1",
                "2026-11-26T01:00:01Z INFO checkout-api Ignored line 2",
                "2026-11-26T01:00:02Z ERROR checkout-api OOMKilled exit code 137 on pod web-1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log_b = tmp_path / "b.log"
    log_b.write_text(
        "2026-11-26T01:00:03Z ERROR checkout-api Container restarted after OOMKilled\n",
        encoding="utf-8",
    )
    audit_path = tmp_path / "live_am.jsonl"

    am_url = http_server_factory(
        {
            "/api/v2/alerts": (
                200,
                [
                    {
                        "labels": {
                            "alertname": "KubePodOOMKilled",
                            "severity": "critical",
                            "service": "checkout-api",
                            "namespace": "prod",
                        },
                        "annotations": {"summary": "OOMKilled firing"},
                        "status": {"state": "active"},
                        "startsAt": "2026-11-26T01:00:00Z",
                    },
                    {
                        "labels": {
                            "alertname": "KubePodCrashLoopBackOff",
                            "severity": "warning",
                            "service": "checkout-api",
                            "namespace": "prod",
                        },
                        "annotations": {"summary": "Resolved restart alert"},
                        "status": {"state": "resolved"},
                        "startsAt": "2026-11-26T00:55:00Z",
                    },
                ],
            )
        }
    )

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "triage",
            "--log-file",
            str(log_a),
            "--log-file",
            str(log_b),
            "--tail-lines",
            "1",
            "--alert-url",
            f"{am_url}/api/v2/alerts",
            "--alert-api-type",
            "alertmanager",
            "--include-resolved",
            "--provider",
            "deterministic",
            "--decisions",
            "approve",
            "--audit-log",
            str(audit_path),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stderr
    data = json.loads(res.stdout)
    assert len(data["results"]) == 1
    result_item = data["results"][0]
    assert result_item["source_mode"] == "live"
    assert len(result_item["logs"]) == 2
    assert {a["status"] for a in result_item["alerts"]} == {"firing", "resolved"}


def test_triage_config_yaml_precedence_and_missing_config_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yaml_audit = tmp_path / "yaml_audit.jsonl"
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: ollama",
                "  model: yaml-model",
                "audit:",
                f"  audit_log_path: {yaml_audit.as_posix()}",
                "operator: yaml-operator",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Env overrides YAML provider; CLI --operator overrides both env and YAML
    monkeypatch.setenv("SRE_AGENT_LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("SRE_AGENT_OPERATOR", "env-operator")

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--config",
            str(cfg_path),
            "--operator",
            "cli-operator",
            "--decisions",
            "approve",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["results"][0]["provider_used"] == "deterministic"
    assert data["decisions"][0]["operator"] == "cli-operator"
    assert Path(data["audit_log_path"]) == yaml_audit

    missing_res = runner.invoke(
        main,
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--config",
            str(tmp_path / "does_not_exist.yaml"),
        ],
    )
    assert missing_res.exit_code == 1


def test_triage_allow_fallback_recovery_and_api_key_header(tmp_path: Path) -> None:
    captured_auth: list[str | None] = []

    class AuthCaptureHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                _ = self.rfile.read(length)
            captured_auth.append(self.headers.get("Authorization"))
            raw = json.dumps({"error": "upstream unavailable"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), AuthCaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        base_url = f"http://{host}:{port}"
        audit_path = tmp_path / "fallback_audit.jsonl"
        runner = CliRunner()
        res = runner.invoke(
            main,
            [
                "triage",
                "--scenario",
                "pod_oom_killed",
                "--provider",
                "openai_compatible",
                "--base-url",
                base_url,
                "--model",
                "local-qwen",
                "--api-key",
                "secret-local-token",
                "--allow-fallback",
                "--decisions",
                "approve",
                "--audit-log",
                str(audit_path),
                "--json",
            ],
        )
        assert res.exit_code == 0, res.stderr
        data = json.loads(res.stdout)
        assert data["results"][0]["fallback_used"] is True
        assert data["results"][0]["provider_used"] == "deterministic"
        assert captured_auth == ["Bearer secret-local-token"]
    finally:
        server.shutdown()
        server.server_close()


def test_providers_check_openai_compatible_variants(http_server_factory) -> None:
    runner = CliRunner()

    healthy_url = http_server_factory(
        {"/v1/models": (200, {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]})}
    )
    res_ok = runner.invoke(
        main,
        [
            "providers",
            "check",
            "--provider",
            "openai_compatible",
            "--base-url",
            healthy_url,
            "--model",
            "Qwen/Qwen2.5-7B-Instruct",
            "--api-key",
            "tok",
            "--json",
        ],
    )
    assert res_ok.exit_code == 0
    assert json.loads(res_ok.stdout)["healthy"] is True

    res_missing_model = runner.invoke(
        main,
        [
            "providers",
            "check",
            "--provider",
            "openai_compatible",
            "--base-url",
            healthy_url,
            "--model",
            "missing-model",
            "--json",
        ],
    )
    assert res_missing_model.exit_code == 1
    assert json.loads(res_missing_model.stdout)["healthy"] is False

    err_url = http_server_factory({"/v1/models": (503, {"error": "down"})})
    res_err = runner.invoke(
        main,
        [
            "providers",
            "check",
            "--provider",
            "openai_compatible",
            "--base-url",
            err_url,
            "--model",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
    )
    assert res_err.exit_code == 1


def test_audit_list_and_show_filters_tables_and_empty_states(tmp_path: Path) -> None:
    runner = CliRunner()
    audit_path = tmp_path / "audit_filters.jsonl"

    triage_res = runner.invoke(
        main,
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            "approve,reject: too risky,skip",
            "--audit-log",
            str(audit_path),
            "--json",
        ],
    )
    assert triage_res.exit_code == 0
    incident_id = json.loads(triage_res.stdout)["results"][0]["incident_id"]

    rejected_res = runner.invoke(
        main,
        [
            "audit",
            "list",
            "--audit-log",
            str(audit_path),
            "--incident-id",
            incident_id,
            "--event-type",
            "operator_decision",
            "--decision",
            "rejected",
            "--json",
        ],
    )
    assert rejected_res.exit_code == 0
    rejected_data = json.loads(rejected_res.stdout)
    assert rejected_data["count"] == 1
    assert rejected_data["events"][0]["payload"]["decision"] == "rejected"

    skipped_res = runner.invoke(
        main,
        [
            "audit",
            "list",
            "--audit-log",
            str(audit_path),
            "--decision",
            "skipped",
            "--json",
        ],
    )
    assert skipped_res.exit_code == 0
    assert json.loads(skipped_res.stdout)["count"] == 1

    table_list = runner.invoke(
        main,
        ["audit", "list", "--audit-log", str(audit_path)],
    )
    assert table_list.exit_code == 0
    assert "Audit Events" in table_list.stdout
    assert incident_id[:12] in table_list.stdout

    empty_list = runner.invoke(
        main,
        ["audit", "list", "--audit-log", str(audit_path), "--incident-id", "inc-none"],
    )
    assert empty_list.exit_code == 0
    assert "No matching audit events found." in empty_list.stdout

    show_table = runner.invoke(
        main,
        ["audit", "show", incident_id, "--audit-log", str(audit_path)],
    )
    assert show_table.exit_code == 0
    assert "Incident Lifecycle" in show_table.stdout

    show_missing_json = runner.invoke(
        main,
        ["audit", "show", "inc-missing", "--audit-log", str(audit_path), "--json"],
    )
    assert show_missing_json.exit_code == 1
    assert json.loads(show_missing_json.stdout)["count"] == 0
