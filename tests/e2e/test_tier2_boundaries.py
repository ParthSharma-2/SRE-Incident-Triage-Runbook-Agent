"""Tier 2 E2E Tests: Boundaries, error handling, and security checks across E1–E6 (34 tests)."""

from __future__ import annotations

import json
import re
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
    unused_loopback_port,
    write_jsonl_log,
    write_text_log,
)


def _assert_no_traceback(output: str) -> None:
    assert "Traceback (most recent call last)" not in output, f"Unhandled traceback:\n{output}"


def _is_shell_token_sanitized(text: str, dangerous_fragment: str) -> bool:
    """Verify dangerous fragment is either absent or safely enclosed in single quotes."""
    if dangerous_fragment not in text:
        return True
    # Check that every occurrence of dangerous_fragment sits inside '...' single-quoted token
    quoted_spans = [m.span() for m in re.finditer(r"'[^']*'", text)]
    start = 0
    while True:
        idx = text.find(dangerous_fragment, start)
        if idx == -1:
            return True
        end = idx + len(dangerous_fragment)
        if not any(qs <= idx and end <= qe for qs, qe in quoted_spans):
            return False
        start = end


# ===========================================================================
# E1: Bundled Scenarios & CLI Boundaries (5 tests)
# ===========================================================================


def test_t2_e1_unknown_scenario_name_fails_cleanly(run_cli, tmp_path: Path) -> None:
    """Unknown --scenario exits non-zero (1 or 2) with clear message and no traceback."""
    res = run_cli(
        [
            "triage",
            "--scenario",
            "nonexistent_scenario_xyz",
            "--provider",
            "deterministic",
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
        ]
    )
    assert res.returncode in (1, 2)
    _assert_no_traceback(res.combined_output)
    assert res.combined_output.strip() != ""


def test_t2_e1_missing_all_input_sources_fails(run_cli, tmp_path: Path) -> None:
    """Running triage with neither sample scenario nor live source fails cleanly."""
    res = run_cli(
        [
            "triage",
            "--provider",
            "deterministic",
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
        ]
    )
    assert res.returncode in (1, 2)
    _assert_no_traceback(res.combined_output)
    assert res.combined_output.strip() != ""


def test_t2_e1_invalid_cli_flag_returns_exit_2(run_cli) -> None:
    """Unknown CLI flag exits with Click usage error code 2."""
    res = run_cli(["triage", "--totally-invalid-flag-xyz"])
    assert res.returncode == 2
    _assert_no_traceback(res.combined_output)


def test_t2_e1_scenarios_list_from_isolated_tmp_cwd(run_cli, tmp_path: Path) -> None:
    """scenarios list discovers bundled scenarios regardless of current working directory."""
    isolated = tmp_path / "empty_cwd"
    isolated.mkdir()
    res = run_cli(["scenarios", "list", "--json"], cwd=isolated)
    assert res.returncode == 0, res.combined_output
    items = extract_list(res.parse_json(), ("scenarios", "items", "results", "data"))
    names = {item["name"] for item in items}
    assert set(CANONICAL_SCENARIOS).issubset(names)


def test_t2_e1_missing_config_yaml_returns_exit_1(run_cli, tmp_path: Path) -> None:
    """Missing --config YAML file fails cleanly with exit code 1."""
    missing = tmp_path / "does_not_exist.yaml"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--config",
            str(missing),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
        ]
    )
    assert res.returncode == 1
    _assert_no_traceback(res.combined_output)


# ===========================================================================
# E2: Live Ingestion Boundaries (7 tests)
# ===========================================================================


def test_t2_e2_missing_log_file_returns_exit_1(run_cli, tmp_path: Path) -> None:
    """Nonexistent --log-file fails cleanly with exit code 1."""
    res = run_cli(
        [
            "triage",
            "--log-file",
            str(tmp_path / "missing.log"),
            "--provider",
            "deterministic",
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
        ]
    )
    assert res.returncode == 1
    _assert_no_traceback(res.combined_output)


def test_t2_e2_empty_log_file_alert_only_incident(
    run_cli,
    prometheus_server: PrometheusMockServer,
    tmp_path: Path,
) -> None:
    """0-byte log file with firing Prometheus alert succeeds as alert-only incident."""
    empty_log = tmp_path / "empty.log"
    empty_log.write_text("", encoding="utf-8")
    res = run_cli(
        [
            "triage",
            "--log-file",
            str(empty_log),
            "--alert-url",
            prometheus_server.alerts_url,
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
    assert result["logs"] == []
    assert len(result["alerts"]) >= 1


def test_t2_e2_log_only_no_alerts_handled_gracefully(
    run_cli,
    alertmanager_server: AlertmanagerMockServer,
    tmp_path: Path,
) -> None:
    """Error log with zero active alerts from Alertmanager succeeds as log-only incident."""
    alertmanager_server.set_alerts([])
    log_path = write_text_log(
        tmp_path / "log_only.log",
        [
            "2026-11-25T12:10:00Z ERROR payments-api Database connection refused on port 5432",
            "2026-11-25T12:10:01Z CRITICAL payments-api Circuit breaker opened for db-primary",
        ],
    )
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
            str(tmp_path / "audit.jsonl"),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["alerts"] == []
    assert len(result["logs"]) >= 1


def test_t2_e2_malformed_jsonl_and_invalid_utf8_bytes(
    run_cli,
    prometheus_server: PrometheusMockServer,
    tmp_path: Path,
) -> None:
    """Broken JSON lines and invalid UTF-8 bytes do not crash log ingestion."""
    mixed_log = tmp_path / "corrupt_mixed.jsonl"
    raw = (
        b"{not valid json\n"
        b"\n"
        b"\xff\xfe\xfd invalid utf8 bytes ERROR checkout-api bad bytes\n"
        + json.dumps(
            {
                "timestamp": "2026-11-25T12:00:15Z",
                "level": "ERROR",
                "service": "checkout-api",
                "message": "VALID_SURVIVING_JSONL_LINE OOMKilled",
            }
        ).encode("utf-8")
        + b"\n"
    )
    mixed_log.write_bytes(raw)
    res = run_cli(
        [
            "triage",
            "--log-file",
            str(mixed_log),
            "--alert-url",
            prometheus_server.alerts_url,
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
    _assert_no_traceback(res.combined_output)
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    messages = "\n".join(entry["message"] for entry in payload["results"][0]["logs"])
    assert "VALID_SURVIVING_JSONL_LINE" in messages


def test_t2_e2_prometheus_http_500_returns_exit_1(
    run_cli,
    prometheus_server: PrometheusMockServer,
    tmp_path: Path,
) -> None:
    """Prometheus endpoint HTTP 500 surfaces clean operational error exit code 1."""
    prometheus_server.configure(status_code=500)
    log_path = write_text_log(
        tmp_path / "app.log",
        ["2026-11-25T12:00:00Z ERROR checkout-api sample error"],
    )
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
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
        ]
    )
    assert res.returncode == 1
    _assert_no_traceback(res.combined_output)


def test_t2_e2_resolved_alerts_filtered_unless_include_resolved(
    run_cli,
    prometheus_server: PrometheusMockServer,
    tmp_path: Path,
) -> None:
    """Resolved/inactive alerts are excluded by default and included with --include-resolved."""
    prometheus_server.set_alerts(
        [
            {
                "labels": {
                    "alertname": "ActiveAlert",
                    "severity": "critical",
                    "service": "checkout-api",
                },
                "annotations": {"summary": "Still firing"},
                "state": "firing",
                "activeAt": "2026-11-25T12:00:00Z",
                "value": "1",
            },
            {
                "labels": {
                    "alertname": "ResolvedAlert",
                    "severity": "warning",
                    "service": "checkout-api",
                },
                "annotations": {"summary": "Already resolved"},
                "state": "inactive",
                "activeAt": "2026-11-25T11:00:00Z",
                "value": "0",
            },
        ]
    )
    log_path = write_text_log(
        tmp_path / "svc.log",
        ["2026-11-25T12:00:05Z ERROR checkout-api active failure"],
    )
    without_flag = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            prometheus_server.alerts_url,
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit1.jsonl"),
            "--json",
        ]
    )
    assert without_flag.returncode == 0, without_flag.combined_output
    default_names = {a["alertname"] for a in without_flag.parse_json()["results"][0]["alerts"]}
    assert "ActiveAlert" in default_names
    assert "ResolvedAlert" not in default_names

    with_flag = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            prometheus_server.alerts_url,
            "--include-resolved",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit2.jsonl"),
            "--json",
        ]
    )
    assert with_flag.returncode == 0, with_flag.combined_output
    all_names = {a["alertname"] for a in with_flag.parse_json()["results"][0]["alerts"]}
    assert {"ActiveAlert", "ResolvedAlert"}.issubset(all_names)


def test_t2_e2_tail_lines_boundary_exceeds_total_and_invalid(
    run_cli,
    prometheus_server: PrometheusMockServer,
    tmp_path: Path,
) -> None:
    """--tail-lines larger than file returns all lines; negative value fails cleanly."""
    log_path = write_text_log(
        tmp_path / "three.log",
        [
            "2026-11-25T12:00:01Z ERROR checkout-api LINE_1",
            "2026-11-25T12:00:02Z ERROR checkout-api LINE_2",
            "2026-11-25T12:00:03Z ERROR checkout-api LINE_3",
        ],
    )
    ok_res = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            prometheus_server.alerts_url,
            "--tail-lines",
            "1000",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit_ok.jsonl"),
            "--json",
        ]
    )
    assert ok_res.returncode == 0, ok_res.combined_output
    assert len(ok_res.parse_json()["results"][0]["logs"]) == 3

    bad_res = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            prometheus_server.alerts_url,
            "--tail-lines",
            "-1",
            "--provider",
            "deterministic",
            "--audit-log",
            str(tmp_path / "audit_bad.jsonl"),
        ]
    )
    assert bad_res.returncode in (1, 2)
    _assert_no_traceback(bad_res.combined_output)


# ===========================================================================
# E3: Evidence Grounding & Reasoning Boundaries (5 tests)
# ===========================================================================


def test_t2_e3_fabricated_evidence_ref_triggers_fallback(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Hallucinated evidence_refs ID triggers deterministic fallback when allowed."""
    ollama_server.configure(mode="ungrounded")
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--allow-fallback",
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
    assert result["fallback_used"] is True
    assert result["fallback_reason"]
    all_refs = [ref["ref_id"] for step in result["remediation_steps"] for ref in step["evidence_refs"]]
    assert "nonexistent-fabricated-id-999" not in all_refs


def test_t2_e3_fabricated_evidence_ref_strict_fails_exit_1(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Hallucinated evidence_refs ID fails with exit 1 under --no-allow-fallback."""
    ollama_server.configure(mode="ungrounded")
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
        ]
    )
    assert res.returncode == 1
    _assert_no_traceback(res.combined_output)


def test_t2_e3_empty_evidence_refs_strict_fails_exit_1(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Empty evidence_refs array violates grounding contract and fails under strict mode."""
    ollama_server.configure(mode="empty_refs")
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
        ]
    )
    assert res.returncode == 1
    _assert_no_traceback(res.combined_output)


def test_t2_e3_malformed_llm_json_fallback_vs_strict(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Malformed LLM response falls back when allowed and fails with exit 1 when strict."""
    ollama_server.configure(mode="malformed")
    fallback_res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit_fb.jsonl"),
            "--json",
        ]
    )
    assert fallback_res.returncode == 0, fallback_res.combined_output
    fb_payload = fallback_res.parse_json()
    assert_valid_triage_json_payload(fb_payload)
    assert fb_payload["results"][0]["fallback_used"] is True

    strict_res = run_cli(
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
            str(tmp_path / "audit_strict.jsonl"),
        ]
    )
    assert strict_res.returncode == 1
    _assert_no_traceback(strict_res.combined_output)


def test_t2_e3_unknown_alertname_produces_safe_fallback_triage(
    run_cli,
    prometheus_server: PrometheusMockServer,
    tmp_path: Path,
) -> None:
    """Unrecognized alertname still yields valid grounded generic triage without KeyError."""
    prometheus_server.set_alerts(
        [
            {
                "labels": {
                    "alertname": "CustomUnmatchedAlert99",
                    "severity": "warning",
                    "service": "telemetry-proxy",
                },
                "annotations": {"summary": "Unusual custom metric spike"},
                "state": "firing",
                "activeAt": "2026-11-25T12:20:00Z",
                "value": "42",
            }
        ]
    )
    log_path = write_text_log(
        tmp_path / "custom.log",
        ["2026-11-25T12:20:05Z ERROR telemetry-proxy unexpected upstream latency spike"],
    )
    res = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            prometheus_server.alerts_url,
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


# ===========================================================================
# E4: Dry-Run Sanitization & Interactive Approval Boundaries (6 tests)
# ===========================================================================


def test_t2_e4_dry_run_sanitizes_shell_metacharacters(
    run_cli,
    alertmanager_server: AlertmanagerMockServer,
    tmp_path: Path,
) -> None:
    """Shell injection payloads in alert labels are sanitized and never executed."""
    alertmanager_server.set_alerts(
        [
            {
                "fingerprint": "fp-inj",
                "labels": {
                    "alertname": "PodOOMKilled",
                    "severity": "critical",
                    "service": "api && touch canary_pwned_svc.txt",
                    "pod": "web-0; touch canary_pwned_pod.txt # $(id)",
                    "namespace": "prod",
                },
                "annotations": {"summary": "Injection attempt in labels"},
                "startsAt": "2026-11-25T12:00:00Z",
                "status": {"state": "active"},
            }
        ]
    )
    res = run_cli(
        [
            "triage",
            "--alert-url",
            alertmanager_server.alerts_url,
            "--alert-api-type",
            "alertmanager",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ],
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    assert not (tmp_path / "canary_pwned_svc.txt").exists()
    assert not (tmp_path / "canary_pwned_pod.txt").exists()

    for dr in payload["dry_runs"]:
        assert dr["executed_against_real_infra"] is False
        assert _is_shell_token_sanitized(dr["command_preview"], "; touch canary_pwned_pod.txt")
        assert _is_shell_token_sanitized(dr["command_preview"], "&& touch canary_pwned_svc.txt")


def test_t2_e4_interactive_quit_mid_flow(run_cli, tmp_path: Path) -> None:
    """Interactive [q]uit stops processing remaining steps cleanly without traceback."""
    audit_path = tmp_path / "quit_audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--audit-log",
            str(audit_path),
        ],
        stdin_text="q\n",
        timeout=15.0,
    )
    assert res.returncode == 0, res.combined_output
    assert res.timed_out is False
    _assert_no_traceback(res.combined_output)
    events = read_audit_jsonl(audit_path)
    assert_valid_audit_events(events)
    dry_runs = [e for e in events if e["event_type"] == "dry_run_executed"]
    assert all(e["payload"].get("simulated_status") != "SUCCESS_DRY_RUN" for e in dry_runs)


def test_t2_e4_interactive_eof_aborts_gracefully(run_cli, tmp_path: Path) -> None:
    """Immediate EOF on stdin does not hang or crash with unhandled EOFError traceback."""
    audit_path = tmp_path / "eof_audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--audit-log",
            str(audit_path),
        ],
        stdin_text="",
        timeout=15.0,
    )
    assert res.timed_out is False
    assert res.returncode in (0, 1)
    _assert_no_traceback(res.combined_output)
    if audit_path.exists() and audit_path.read_text(encoding="utf-8").strip():
        events = read_audit_jsonl(audit_path)
        assert_valid_audit_events(events)


def test_t2_e4_interactive_invalid_input_reprompts_then_approves(
    run_cli,
    tmp_path: Path,
) -> None:
    """Invalid interactive inputs re-prompt until valid [a]pprove is received."""
    audit_path = tmp_path / "reprompt_audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--audit-log",
            str(audit_path),
        ],
        stdin_text="zzz\n?\na\n\n" + "a\n\n" * 10,
        timeout=15.0,
    )
    assert res.returncode == 0, res.combined_output
    _assert_no_traceback(res.combined_output)
    events = read_audit_jsonl(audit_path)
    assert_valid_audit_events(events)
    decisions = [e for e in events if e["event_type"] == "operator_decision"]
    assert any(e["payload"].get("decision") == "approved" for e in decisions)


def test_t2_e4_invalid_scripted_decision_token_fails(run_cli, tmp_path: Path) -> None:
    """Invalid --decisions token fails with exit 1 or 2 and executes zero dry runs."""
    audit_path = tmp_path / "bad_decisions.jsonl"
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            "bogus_cmd",
            "--audit-log",
            str(audit_path),
        ]
    )
    assert res.returncode in (1, 2)
    _assert_no_traceback(res.combined_output)
    if audit_path.exists() and audit_path.read_text(encoding="utf-8").strip():
        events = read_audit_jsonl(audit_path)
        assert not any(
            e["event_type"] == "dry_run_executed"
            and e["payload"].get("simulated_status") == "SUCCESS_DRY_RUN"
            for e in events
        )


def test_t2_e4_scripted_decisions_length_mismatch(run_cli, tmp_path: Path) -> None:
    """Underflow or overflow in --decisions list never raises IndexError traceback."""
    underflow = run_cli(
        [
            "triage",
            "--all-scenarios",
            "--provider",
            "deterministic",
            "--decisions",
            "approve",
            "--audit-log",
            str(tmp_path / "underflow.jsonl"),
        ]
    )
    assert underflow.returncode in (0, 1, 2)
    _assert_no_traceback(underflow.combined_output)

    overflow = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--decisions",
            ",".join(["approve"] * 25),
            "--audit-log",
            str(tmp_path / "overflow.jsonl"),
            "--json",
        ]
    )
    assert overflow.returncode in (0, 1, 2)
    _assert_no_traceback(overflow.combined_output)


# ===========================================================================
# E5: LLM Provider Failures & Fallback Boundaries (6 tests)
# ===========================================================================


def test_t2_e5_ollama_http_500_falls_back_to_deterministic(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Ollama HTTP 500 transparently falls back to deterministic provider when allowed."""
    ollama_server.configure(status_code=500)
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--allow-fallback",
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
    assert result["fallback_used"] is True
    assert result["fallback_reason"]
    assert result["provider_used"] == "deterministic"


def test_t2_e5_ollama_http_500_strict_no_fallback_exits_1(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Ollama HTTP 500 with --no-allow-fallback exits with code 1."""
    ollama_server.configure(status_code=500)
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
        ]
    )
    assert res.returncode == 1
    _assert_no_traceback(res.combined_output)


def test_t2_e5_ollama_timeout_triggers_fallback(
    run_cli,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Slow Ollama response exceeding --timeout falls back to deterministic provider."""
    ollama_server.configure(delay_s=2.0)
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--timeout",
            "0.3",
            "--allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--json",
        ],
        timeout=15.0,
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["fallback_used"] is True
    assert result["fallback_reason"]


def test_t2_e5_openai_compat_timeout_strict_exits_1(
    run_cli,
    openai_compat_server: OpenAICompatMockServer,
    tmp_path: Path,
) -> None:
    """Slow OpenAI-compatible response exceeding --timeout exits 1 under strict mode."""
    openai_compat_server.configure(delay_s=2.0)
    res = run_cli(
        [
            "triage",
            "--scenario",
            "api_5xx_db_pool",
            "--provider",
            "openai_compatible",
            "--base-url",
            openai_compat_server.base_url,
            "--timeout",
            "0.3",
            "--no-allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
        ],
        timeout=15.0,
    )
    assert res.returncode == 1
    _assert_no_traceback(res.combined_output)


def test_t2_e5_unreachable_provider_strict_exits_1(run_cli, tmp_path: Path) -> None:
    """Unreachable loopback port with --no-allow-fallback fails cleanly with exit 1."""
    closed_port = unused_loopback_port()
    res = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "ollama",
            "--base-url",
            f"http://127.0.0.1:{closed_port}",
            "--timeout",
            "1.0",
            "--no-allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
        ]
    )
    assert res.returncode == 1
    _assert_no_traceback(res.combined_output)


def test_t2_e5_providers_check_unhealthy_exits_1(
    run_cli,
    ollama_server: OllamaMockServer,
) -> None:
    """providers check exits 1 when mock provider fails all health probes."""
    ollama_server.configure(mode="fail_all")
    res = run_cli(
        [
            "providers",
            "check",
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--json",
        ]
    )
    assert res.returncode == 1
    _assert_no_traceback(res.combined_output)


# ===========================================================================
# E6: Audit Log Boundaries & Windows UTF-8 Roundtrip (5 tests)
# ===========================================================================


def test_t2_e6_audit_list_missing_file_handled_cleanly(run_cli, tmp_path: Path) -> None:
    """audit list on missing audit log handles cleanly without FileNotFoundError traceback."""
    res = run_cli(
        ["audit", "list", "--audit-log", str(tmp_path / "missing_audit.jsonl"), "--json"]
    )
    assert res.returncode in (0, 1)
    _assert_no_traceback(res.combined_output)
    if res.returncode == 0:
        assert extract_list(res.parse_json()) == []


def test_t2_e6_audit_show_unknown_incident_id(run_cli, tmp_path: Path) -> None:
    """audit show for unknown incident_id handles cleanly without traceback."""
    audit_path = tmp_path / "audit.jsonl"
    seed = run_cli(
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
    assert seed.returncode == 0, seed.combined_output

    show = run_cli(
        ["audit", "show", "inc-does-not-exist-999", "--audit-log", str(audit_path), "--json"]
    )
    assert show.returncode in (0, 1)
    _assert_no_traceback(show.combined_output)
    if show.returncode == 0:
        assert extract_list(show.parse_json()) == []


def test_t2_e6_audit_list_corrupt_jsonl_line(run_cli, tmp_path: Path) -> None:
    """Corrupt line in audit JSONL does not crash audit list with unhandled JSONDecodeError."""
    audit_path = tmp_path / "corrupt_audit.jsonl"
    seed = run_cli(
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
    assert seed.returncode == 0, seed.combined_output
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write("{corrupt truncated json line\n")

    res = run_cli(["audit", "list", "--audit-log", str(audit_path), "--json"])
    assert res.returncode in (0, 1)
    _assert_no_traceback(res.combined_output)
    if res.returncode == 0:
        events = extract_list(res.parse_json())
        assert len(events) >= 1


def test_t2_e6_audit_log_creates_missing_parent_dirs(run_cli, tmp_path: Path) -> None:
    """--audit-log in nonexistent nested directories creates parent dirs automatically."""
    nested = tmp_path / "deep" / "nested" / "dir" / "audit.jsonl"
    assert not nested.parent.exists()
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
            str(nested),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    assert nested.exists()
    assert_valid_audit_events(read_audit_jsonl(nested))


def test_t2_e6_unicode_logs_roundtrip_audit_on_windows(
    run_cli,
    prometheus_server: PrometheusMockServer,
    tmp_path: Path,
) -> None:
    """Unicode characters survive ingestion, triage, audit persistence, and audit list on Windows."""
    unicode_msg = "Check failed ✓ café 日本語 — status"
    log_path = write_jsonl_log(
        tmp_path / "unicode.jsonl",
        [
            {
                "timestamp": "2026-11-25T12:00:12Z",
                "level": "ERROR",
                "service": "checkout-api",
                "message": unicode_msg,
            }
        ],
    )
    audit_path = tmp_path / "unicode_audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--log-file",
            str(log_path),
            "--alert-url",
            prometheus_server.alerts_url,
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
    assert any(unicode_msg in entry["message"] for entry in payload["results"][0]["logs"])

    audit_res = run_cli(["audit", "list", "--audit-log", str(audit_path), "--json"])
    assert audit_res.returncode == 0, audit_res.combined_output
    assert_valid_audit_events(read_audit_jsonl(audit_path))
