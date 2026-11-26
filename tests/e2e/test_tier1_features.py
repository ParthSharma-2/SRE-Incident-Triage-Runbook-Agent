"""Tier 1 E2E Tests: Happy-path feature coverage across E1–E6 (34 tests)."""

from __future__ import annotations

from pathlib import Path

from conftest import (
    CANONICAL_SCENARIOS,
    AlertmanagerMockServer,
    OllamaMockServer,
    OpenAICompatMockServer,
    PrometheusMockServer,
    assert_valid_audit_events,
    assert_valid_triage_json_payload,
    extract_list,
    read_audit_jsonl,
    write_jsonl_log,
    write_text_log,
    write_yaml_config,
)


# ===========================================================================
# E1: Bundled Sample Scenario Triage & Discovery (6 tests)
# ===========================================================================


def test_t1_e1_scenarios_list_human_output(run_cli) -> None:
    """scenarios list outputs all four bundled canonical scenario names."""
    res = run_cli(["scenarios", "list"])
    assert res.returncode == 0, res.combined_output
    for name in CANONICAL_SCENARIOS:
        assert name in res.stdout


def test_t1_e1_scenarios_list_json_schema(run_cli) -> None:
    """scenarios list --json returns structured metadata for all 4 scenarios."""
    res = run_cli(["scenarios", "list", "--json"])
    assert res.returncode == 0, res.combined_output
    items = extract_list(res.parse_json(), ("scenarios", "items", "results", "data"))
    names = {item["name"] for item in items}
    assert set(CANONICAL_SCENARIOS).issubset(names)
    for item in items:
        assert isinstance(item.get("description"), str) and item["description"].strip()
        alert_count = item.get("alert_count", item.get("alerts_count"))
        log_count = item.get("log_count", item.get("logs_count"))
        assert isinstance(alert_count, int) and alert_count >= 1
        assert isinstance(log_count, int) and log_count >= 1


def test_t1_e1_triage_scenario_pod_oom_killed(run_cli, tmp_path: Path) -> None:
    """Bundled pod_oom_killed scenario produces valid grounded triage JSON."""
    audit_path = tmp_path / "audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["scenario_name"] == "pod_oom_killed"
    assert result["source_mode"] == "bundled"
    assert len(result["alerts"]) >= 1
    assert len(result["logs"]) >= 1


def test_t1_e1_triage_scenario_api_5xx_db_pool(run_cli, tmp_path: Path) -> None:
    """Bundled api_5xx_db_pool scenario ingests plain-text logs and JSON alerts."""
    audit_path = tmp_path / "audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "api_5xx_db_pool",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["scenario_name"] == "api_5xx_db_pool"
    assert result["source_mode"] == "bundled"
    assert len(result["alerts"]) >= 1
    assert len(result["logs"]) >= 1


def test_t1_e1_triage_scenario_disk_pressure(run_cli, tmp_path: Path) -> None:
    """Bundled disk_pressure scenario produces grounded remediation steps."""
    audit_path = tmp_path / "audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "disk_pressure",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["scenario_name"] == "disk_pressure"
    assert result["source_mode"] == "bundled"
    assert len(result["remediation_steps"]) >= 1


def test_t1_e1_triage_all_scenarios_batch(run_cli, tmp_path: Path) -> None:
    """--all-scenarios triages all 4 bundled scenarios under a shared run_id."""
    audit_path = tmp_path / "audit.jsonl"
    decisions = ",".join(["approve"] * 20)
    res = run_cli(
        [
            "triage",
            "--all-scenarios",
            "--provider",
            "deterministic",
            "--decisions",
            decisions,
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    scenario_names = {r["scenario_name"] for r in payload["results"]}
    assert set(CANONICAL_SCENARIOS).issubset(scenario_names)
    incident_ids = [r["incident_id"] for r in payload["results"]]
    assert len(set(incident_ids)) == len(payload["results"])


# ===========================================================================
# E2: Live Log & Prometheus / Alertmanager Ingestion (6 tests)
# ===========================================================================


def test_t1_e2_live_jsonl_log_and_prometheus_alerts(
    run_cli,
    prometheus_server: PrometheusMockServer,
    tmp_path: Path,
) -> None:
    """Live JSONL log + Prometheus /api/v1/alerts are ingested and surfaced in triage."""
    log_path = write_jsonl_log(
        tmp_path / "app.log.jsonl",
        [
            {
                "timestamp": "2026-11-25T12:00:05Z",
                "level": "ERROR",
                "service": "checkout-api",
                "message": "Container killed due to memory OOM exit code 137",
            }
        ],
    )
    audit_path = tmp_path / "audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            prometheus_server.alerts_url,
            "--alert-api-type",
            "prometheus",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["source_mode"] == "live"
    assert any(a["alertname"] == "PodOOMKilled" for a in result["alerts"])
    assert any("OOM" in entry["message"] for entry in result["logs"])
    reqs = prometheus_server.get_requests()
    assert any(r.method == "GET" and r.path == "/api/v1/alerts" for r in reqs)


def test_t1_e2_live_plain_log_and_alertmanager_alerts(
    run_cli,
    alertmanager_server: AlertmanagerMockServer,
    tmp_path: Path,
) -> None:
    """Live plain-text log + Alertmanager /api/v2/alerts are ingested end-to-end."""
    log_path = write_text_log(
        tmp_path / "orders.log",
        [
            "2026-11-25T12:05:01Z ERROR orders-api HikariPool-1 - Connection is not available, request timed out",
            "2026-11-25T12:05:02Z ERROR orders-api HTTP 503 returned to client on /checkout",
        ],
    )
    audit_path = tmp_path / "audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            alertmanager_server.alerts_url,
            "--alert-api-type",
            "alertmanager",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["source_mode"] == "live"
    assert any(a["alertname"] == "APIDBPoolExhausted" for a in result["alerts"])
    assert any("HikariPool" in entry["message"] for entry in result["logs"])
    reqs = alertmanager_server.get_requests()
    assert any(r.method == "GET" and r.path == "/api/v2/alerts" for r in reqs)


def test_t1_e2_alert_api_type_auto_detects_prometheus(
    run_cli,
    prometheus_server: PrometheusMockServer,
    tmp_path: Path,
) -> None:
    """--alert-api-type auto discovers Prometheus /api/v1/alerts from base URL."""
    log_path = write_text_log(
        tmp_path / "svc.log",
        ["2026-11-25T12:00:10Z ERROR checkout-api OutOfMemoryError in worker"],
    )
    res = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            prometheus_server.base_url,
            "--alert-api-type",
            "auto",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    assert any(a["alertname"] == "PodOOMKilled" for a in payload["results"][0]["alerts"])


def test_t1_e2_alert_api_type_auto_detects_alertmanager(
    run_cli,
    alertmanager_server: AlertmanagerMockServer,
    tmp_path: Path,
) -> None:
    """--alert-api-type auto discovers Alertmanager /api/v2/alerts from base URL."""
    log_path = write_text_log(
        tmp_path / "svc.log",
        ["2026-11-25T12:05:10Z ERROR orders-api remaining connection slots are reserved"],
    )
    res = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            alertmanager_server.base_url,
            "--alert-api-type",
            "auto",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    assert any(a["alertname"] == "APIDBPoolExhausted" for a in payload["results"][0]["alerts"])


def test_t1_e2_live_env_log_file_and_prometheus_url(
    run_cli,
    prometheus_server: PrometheusMockServer,
    tmp_path: Path,
) -> None:
    """SRE_AGENT_LOG_FILE and SRE_AGENT_PROMETHEUS_URL configure live triage via env."""
    log_path = write_jsonl_log(
        tmp_path / "env_live.jsonl",
        [
            {
                "timestamp": "2026-11-25T12:01:00Z",
                "level": "CRITICAL",
                "service": "checkout-api",
                "message": "Env-configured log source OOMKill signal",
            }
        ],
    )
    res = run_cli(
        [
            "triage",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ],
        env={
            "SRE_AGENT_LOG_FILE": str(log_path),
            "SRE_AGENT_PROMETHEUS_URL": prometheus_server.alerts_url,
            "SRE_AGENT_ALERT_API_TYPE": "prometheus",
        },
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["source_mode"] == "live"
    assert any("Env-configured log source" in entry["message"] for entry in result["logs"])
    assert any(a["alertname"] == "PodOOMKilled" for a in result["alerts"])


def test_t1_e2_live_tail_lines_limits_ingested_logs(
    run_cli,
    prometheus_server: PrometheusMockServer,
    tmp_path: Path,
) -> None:
    """--tail-lines N keeps only the newest N lines from a larger log file."""
    lines = [
        f"2026-11-25T12:00:{idx:02d}Z ERROR checkout-api LOG_SEQ_{idx:02d} event"
        for idx in range(1, 21)
    ]
    log_path = write_text_log(tmp_path / "many.log", lines)
    res = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            prometheus_server.alerts_url,
            "--tail-lines",
            "5",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    logs = payload["results"][0]["logs"]
    assert 1 <= len(logs) <= 5
    joined = "\n".join(entry["message"] for entry in logs)
    assert "LOG_SEQ_20" in joined
    assert "LOG_SEQ_01" not in joined


# ===========================================================================
# E3: Evidence-Linked Runbook Remediation Suggestions (5 tests)
# ===========================================================================


def test_t1_e3_remediation_steps_contract_fields(run_cli, tmp_path: Path) -> None:
    """Every RemediationStep contains all required contract fields and valid risk_level."""
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    steps = payload["results"][0]["remediation_steps"]
    assert len(steps) >= 1
    for step in steps:
        assert step["step_id"]
        assert step["runbook_id"]
        assert step["risk_level"] in {"low", "medium", "high"}


def test_t1_e3_every_remediation_cites_real_ingested_evidence(run_cli, tmp_path: Path) -> None:
    """Every EvidenceRef.ref_id resolves to an actual ingested alert_id or log_id."""
    res = run_cli(
        [
            "triage",
            "--scenario",
            "api_5xx_db_pool",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    alert_ids = {a["alert_id"] for a in result["alerts"]}
    log_ids = {l["log_id"] for l in result["logs"]}
    for step in result["remediation_steps"]:
        for ref in step["evidence_refs"]:
            if ref["kind"] == "alert":
                assert ref["ref_id"] in alert_ids
            else:
                assert ref["ref_id"] in log_ids


def test_t1_e3_runbook_selection_matches_incident_domain(run_cli, tmp_path: Path) -> None:
    """Different incident scenarios match distinct domain-appropriate runbooks."""
    r1 = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit1.jsonl"),
            "--json",
        ]
    )
    r2 = run_cli(
        [
            "triage",
            "--scenario",
            "tls_cert_expiring",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit2.jsonl"),
            "--json",
        ]
    )
    assert r1.returncode == 0, r1.combined_output
    assert r2.returncode == 0, r2.combined_output
    rb1 = {s["runbook_id"] for s in r1.parse_json()["results"][0]["remediation_steps"]}
    rb2 = {s["runbook_id"] for s in r2.parse_json()["results"][0]["remediation_steps"]}
    assert rb1 != rb2


def test_t1_e3_llm_fenced_json_parsed_and_grounded(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Markdown-fenced JSON from Ollama is stripped, parsed, and grounded without fallback."""
    ollama_server.configure(mode="fenced")
    ollama_server.custom_step_title = "Fenced JSON Remediation Step"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--model",
            "llama3.1:8b",
            "--no-allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["fallback_used"] is False
    assert any(s["title"] == "Fenced JSON Remediation Step" for s in result["remediation_steps"])


def test_t1_e3_correlated_live_incident_evidence_grounding(
    run_cli,
    prometheus_server: PrometheusMockServer,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Live correlated alerts + logs ground Ollama-generated remediation steps."""
    log_path = write_jsonl_log(
        tmp_path / "correlated.jsonl",
        [
            {
                "timestamp": "2026-11-25T12:00:08Z",
                "level": "ERROR",
                "service": "checkout-api",
                "message": "Memory cgroup out of memory: Killed process 42 (java)",
            }
        ],
    )
    res = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            prometheus_server.alerts_url,
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--model",
            "llama3.1:8b",
            "--no-allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["fallback_used"] is False
    assert len(result["alerts"]) >= 1 and len(result["logs"]) >= 1


# ===========================================================================
# E4: Interactive & Scripted Approval + Zero-Side-Effect Dry-Run (6 tests)
# ===========================================================================


def test_t1_e4_scripted_approve_executes_dry_run(run_cli, tmp_path: Path) -> None:
    """--decisions approve produces approved OperatorDecision and SUCCESS_DRY_RUN."""
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    assert payload["decisions"][0]["decision"] == "approved"
    assert len(payload["dry_runs"]) >= 1
    first_dr = payload["dry_runs"][0]
    assert first_dr["simulated_status"] == "SUCCESS_DRY_RUN"
    assert first_dr["executed_against_real_infra"] is False
    assert isinstance(first_dr["diff_preview"], str) and first_dr["diff_preview"].strip()
    assert isinstance(first_dr["simulated_stdout"], str) and first_dr["simulated_stdout"].strip()


def test_t1_e4_scripted_reject_skips_dry_run(run_cli, tmp_path: Path) -> None:
    """--decisions reject records rejected decision and does not execute SUCCESS_DRY_RUN."""
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["reject"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    assert all(d["decision"] == "rejected" for d in payload["decisions"])
    assert all(dr["simulated_status"] != "SUCCESS_DRY_RUN" for dr in payload["dry_runs"])


def test_t1_e4_scripted_skip_records_skipped_decision(run_cli, tmp_path: Path) -> None:
    """--decisions skip records skipped decisions without running SUCCESS_DRY_RUN."""
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["skip"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    assert all(d["decision"] == "skipped" for d in payload["decisions"])
    assert all(dr["simulated_status"] != "SUCCESS_DRY_RUN" for dr in payload["dry_runs"])


def test_t1_e4_interactive_stdin_approve_with_reason(run_cli, tmp_path: Path) -> None:
    """Interactive stdin approval records operator, approved decision, and dry-run in audit."""
    audit_path = tmp_path / "interactive_approve.jsonl"
    stdin_payload = "a\napproved for maintenance\n" + "a\n\n" * 10
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--operator",
            "bob",
            "--audit-log",
            str(audit_path),
        ],
        stdin_text=stdin_payload,
    )
    assert res.returncode == 0, res.combined_output
    events = read_audit_jsonl(audit_path)
    assert_valid_audit_events(events)
    decision_events = [e for e in events if e["event_type"] == "operator_decision"]
    assert len(decision_events) >= 1
    assert any(e["payload"].get("decision") == "approved" for e in decision_events)
    assert any(e["actor"] == "bob" or e["payload"].get("operator") == "bob" for e in decision_events)
    dry_events = [e for e in events if e["event_type"] == "dry_run_executed"]
    assert len(dry_events) >= 1
    assert all(e["payload"].get("executed_against_real_infra") is False for e in dry_events)


def test_t1_e4_interactive_stdin_reject_with_reason(run_cli, tmp_path: Path) -> None:
    """Interactive stdin rejection records rejected decision and avoids SUCCESS_DRY_RUN."""
    audit_path = tmp_path / "interactive_reject.jsonl"
    stdin_payload = "r\nwaiting on DBA\n" + "s\n\n" * 10
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--operator",
            "alice",
            "--audit-log",
            str(audit_path),
        ],
        stdin_text=stdin_payload,
    )
    assert res.returncode == 0, res.combined_output
    events = read_audit_jsonl(audit_path)
    assert_valid_audit_events(events)
    decision_events = [e for e in events if e["event_type"] == "operator_decision"]
    assert len(decision_events) >= 1
    rejected = [e for e in decision_events if e["payload"].get("decision") == "rejected"]
    assert len(rejected) >= 1
    assert any(e["actor"] == "alice" or e["payload"].get("operator") == "alice" for e in rejected)
    dry_events = [e for e in events if e["event_type"] == "dry_run_executed"]
    assert all(e["payload"].get("simulated_status") != "SUCCESS_DRY_RUN" for e in dry_events)


def test_t1_e4_dry_run_never_mutates_filesystem(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Approved dry-run simulates commands without touching or deleting local files."""
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("do-not-delete", encoding="utf-8")
    not_created = tmp_path / "should_not_exist.txt"

    ollama_server.custom_command = f"rm -f {sentinel} && touch {not_created}"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--no-allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    assert sentinel.read_text(encoding="utf-8") == "do-not-delete"
    assert not not_created.exists()
    assert all(dr["executed_against_real_infra"] is False for dr in payload["dry_runs"])


# ===========================================================================
# E5: Open-Source LLM Provider Configuration & Fallback (6 tests)
# ===========================================================================


def test_t1_e5_ollama_provider_request_and_model(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Ollama provider sends POST /api/chat with configured model and no fallback."""
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--model",
            "llama3.1:8b",
            "--no-allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["provider_used"] == "ollama"
    assert result["model_used"] == "llama3.1:8b"
    assert result["fallback_used"] is False
    chat_reqs = [r for r in ollama_server.get_requests() if r.method == "POST" and r.path == "/api/chat"]
    assert len(chat_reqs) >= 1
    assert chat_reqs[0].json_body["model"] == "llama3.1:8b"


def test_t1_e5_openai_compatible_provider_request_and_api_key(
    run_cli,
    openai_compat_server: OpenAICompatMockServer,
    tmp_path: Path,
) -> None:
    """OpenAI-compatible provider sends Bearer token and model to /v1/chat/completions."""
    res = run_cli(
        [
            "triage",
            "--scenario",
            "api_5xx_db_pool",
            "--provider",
            "openai_compatible",
            "--base-url",
            openai_compat_server.base_url,
            "--model",
            "qwen2.5:7b",
            "--api-key",
            "vllm-secret",
            "--no-allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["provider_used"] == "openai_compatible"
    assert result["fallback_used"] is False
    post_reqs = [
        r
        for r in openai_compat_server.get_requests()
        if r.method == "POST" and r.path in ("/v1/chat/completions", "/chat/completions")
    ]
    assert len(post_reqs) >= 1
    assert post_reqs[0].json_body["model"] == "qwen2.5:7b"
    assert post_reqs[0].get_header("Authorization") == "Bearer vllm-secret"


def test_t1_e5_deterministic_provider_zero_network_calls(run_cli, tmp_path: Path) -> None:
    """deterministic provider succeeds completely offline."""
    res = run_cli(
        [
            "triage",
            "--scenario",
            "disk_pressure",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    assert payload["results"][0]["provider_used"] == "deterministic"


def test_t1_e5_providers_check_ollama_and_openai_compat(
    run_cli,
    ollama_server: OllamaMockServer,
    openai_compat_server: OpenAICompatMockServer,
) -> None:
    """providers check --json reports healthy connectivity for both open-source providers."""
    res_ollama = run_cli(
        [
            "providers",
            "check",
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--model",
            "llama3.1:8b",
            "--json",
        ]
    )
    assert res_ollama.returncode == 0, res_ollama.combined_output
    ollama_json = res_ollama.parse_json()
    assert isinstance(ollama_json, dict)

    res_oai = run_cli(
        [
            "providers",
            "check",
            "--provider",
            "openai_compatible",
            "--base-url",
            openai_compat_server.base_url,
            "--model",
            "qwen2.5:7b",
            "--json",
        ]
    )
    assert res_oai.returncode == 0, res_oai.combined_output
    oai_json = res_oai.parse_json()
    assert isinstance(oai_json, dict)


def test_t1_e5_env_vars_configure_provider_and_audit_path(
    run_cli,
    openai_compat_server: OpenAICompatMockServer,
    tmp_path: Path,
) -> None:
    """SRE_AGENT_* environment variables configure LLM provider, model, key, and audit path."""
    env_audit = tmp_path / "env_audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--no-allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--json",
        ],
        env={
            "SRE_AGENT_LLM_PROVIDER": "openai_compatible",
            "SRE_AGENT_LLM_BASE_URL": openai_compat_server.base_url,
            "SRE_AGENT_LLM_MODEL": "mistral:7b",
            "SRE_AGENT_LLM_API_KEY": "env-secret-key",
            "SRE_AGENT_AUDIT_LOG_PATH": str(env_audit),
        },
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    assert env_audit.exists()
    post_reqs = [r for r in openai_compat_server.get_requests() if r.method == "POST"]
    assert post_reqs[0].json_body["model"] == "mistral:7b"
    assert post_reqs[0].get_header("Authorization") == "Bearer env-secret-key"


def test_t1_e5_yaml_config_file_configures_triage(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """--config <yaml> configures LLM provider, model, and audit path."""
    yaml_audit = tmp_path / "yaml_audit.jsonl"
    cfg_path = write_yaml_config(
        tmp_path / "agent.yaml",
        {
            "llm_provider": "ollama",
            "llm_base_url": ollama_server.base_url,
            "llm_model": "qwen2.5:7b",
            "llm_allow_fallback": False,
            "audit_log_path": str(yaml_audit),
        },
    )
    res = run_cli(
        [
            "triage",
            "--config",
            str(cfg_path),
            "--scenario",
            "pod_oom_killed",
            "--decisions",
            ",".join(["approve"] * 8),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    assert yaml_audit.exists()
    chat_reqs = [r for r in ollama_server.get_requests() if r.method == "POST"]
    assert chat_reqs[0].json_body["model"] == "qwen2.5:7b"


# ===========================================================================
# E6: Persistent Append-Only Audit Logging & Inspection (5 tests)
# ===========================================================================


def test_t1_e6_audit_log_jsonl_schema_and_lifecycle(run_cli, tmp_path: Path) -> None:
    """Single approved triage records full AuditEvent lifecycle in UTF-8 JSONL."""
    audit_path = tmp_path / "lifecycle_audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    events = read_audit_jsonl(audit_path)
    assert_valid_audit_events(events)
    event_types = {e["event_type"] for e in events}
    expected_types = {
        "triage_run_started",
        "triage_completed",
        "remediation_suggested",
        "operator_decision",
        "dry_run_executed",
    }
    assert expected_types.issubset(event_types)


def test_t1_e6_audit_log_append_only_across_runs(run_cli, tmp_path: Path) -> None:
    """Sequential runs append to the same audit file without modifying prior lines."""
    audit_path = tmp_path / "append_only.jsonl"
    r1 = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert r1.returncode == 0, r1.combined_output
    before_lines = audit_path.read_text(encoding="utf-8").splitlines()

    r2 = run_cli(
        [
            "triage",
            "--scenario",
            "disk_pressure",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert r2.returncode == 0, r2.combined_output
    after_lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(after_lines) > len(before_lines)
    assert after_lines[: len(before_lines)] == before_lines

    run_ids = {ev["run_id"] for ev in read_audit_jsonl(audit_path)}
    assert {r1.parse_json()["run_id"], r2.parse_json()["run_id"]}.issubset(run_ids)


def test_t1_e6_audit_list_filter_by_incident_and_run_id(run_cli, tmp_path: Path) -> None:
    """audit list filters accurately by --incident-id and --run-id."""
    audit_path = tmp_path / "filter_ids.jsonl"
    r1 = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    r2 = run_cli(
        [
            "triage",
            "--scenario",
            "tls_cert_expiring",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert r1.returncode == 0 and r2.returncode == 0
    inc1 = r1.parse_json()["results"][0]["incident_id"]
    run2 = r2.parse_json()["run_id"]

    by_inc = run_cli(
        ["audit", "list", "--audit-log", str(audit_path), "--incident-id", inc1, "--json"]
    )
    assert by_inc.returncode == 0, by_inc.combined_output
    inc_events = extract_list(by_inc.parse_json())
    assert len(inc_events) >= 1
    assert all(e["incident_id"] == inc1 for e in inc_events)

    by_run = run_cli(
        ["audit", "list", "--audit-log", str(audit_path), "--run-id", run2, "--json"]
    )
    assert by_run.returncode == 0, by_run.combined_output
    run_events = extract_list(by_run.parse_json())
    assert len(run_events) >= 1
    assert all(e["run_id"] == run2 for e in run_events)


def test_t1_e6_audit_list_filter_by_event_type_and_decision(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """audit list filters accurately by --event-type and --decision."""
    ollama_server.custom_steps = [
        {
            "step_id": "step-1",
            "title": "Inspect pod memory usage",
            "description": "Gather container memory stats.",
            "risk_level": "low",
            "command_preview": "kubectl top pod checkout-api",
            "target_resource": "pod/checkout-api",
            "expected_effect": "Confirms memory pressure.",
            "diff_template": "--- before\n+++ after\n# read-only check",
        },
        {
            "step_id": "step-2",
            "title": "Increase pod memory limit",
            "description": "Patch deployment memory limits.",
            "risk_level": "medium",
            "command_preview": "kubectl set resources deployment/checkout-api --limits=memory=1Gi",
            "target_resource": "deployment/checkout-api",
            "expected_effect": "Prevents repeat OOMKilled.",
            "diff_template": "--- before\n+++ after\n- memory: 512Mi\n+ memory: 1Gi",
        },
        {
            "step_id": "step-3",
            "title": "Restart deployment rollout",
            "description": "Perform rolling restart.",
            "risk_level": "high",
            "command_preview": "kubectl rollout restart deployment/checkout-api",
            "target_resource": "deployment/checkout-api",
            "expected_effect": "Restores healthy pods.",
            "diff_template": "--- before\n+++ after\n+ restartedAt: now",
        },
    ]
    audit_path = tmp_path / "filter_decision.jsonl"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--no-allow-fallback",
            "--decisions",
            "approve,reject,skip,approve,approve,approve,approve,approve",
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output

    q = run_cli(
        [
            "audit",
            "list",
            "--audit-log",
            str(audit_path),
            "--event-type",
            "operator_decision",
            "--decision",
            "rejected",
            "--json",
        ]
    )
    assert q.returncode == 0, q.combined_output
    events = extract_list(q.parse_json())
    assert len(events) >= 1
    for ev in events:
        assert ev["event_type"] == "operator_decision"
        assert ev["payload"].get("decision") == "rejected"


def test_t1_e6_audit_show_chronological_incident_lifecycle(run_cli, tmp_path: Path) -> None:
    """audit show <incident_id> returns that incident's events in chronological order."""
    audit_path = tmp_path / "show_lifecycle.jsonl"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "api_5xx_db_pool",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    inc_id = res.parse_json()["results"][0]["incident_id"]

    show = run_cli(["audit", "show", inc_id, "--audit-log", str(audit_path), "--json"])
    assert show.returncode == 0, show.combined_output
    events = extract_list(show.parse_json())
    assert len(events) >= 3
    assert all(e["incident_id"] == inc_id for e in events)
    timestamps = [e["timestamp"] for e in events]
    assert timestamps == sorted(timestamps)
