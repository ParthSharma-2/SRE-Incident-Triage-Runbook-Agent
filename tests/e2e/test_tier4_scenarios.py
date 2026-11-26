"""Tier 4 E2E Tests: 7 realistic multi-step DevOps/SRE end-to-end workflows."""

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


def test_t4_s1_k8s_oomkilled_cascade(
    run_cli,
    prometheus_server: PrometheusMockServer,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Scenario 1: K8s OOMKilled cascade -> fenced Ollama -> interactive approve/reject -> audit show."""
    prometheus_server.set_alerts(
        [
            {
                "labels": {
                    "alertname": "PodOOMKilled",
                    "severity": "critical",
                    "service": "checkout-api",
                    "namespace": "production",
                    "pod": "checkout-api-7d9f8c-x4k2p",
                },
                "annotations": {
                    "summary": "Pod checkout-api-7d9f8c-x4k2p terminated with OOMKilled (exit_code=137)"
                },
                "state": "firing",
                "activeAt": "2026-11-25T12:00:00Z",
                "value": "1",
            }
        ]
    )
    log_path = write_jsonl_log(
        tmp_path / "k8s_oom.jsonl",
        [
            {
                "timestamp": "2026-11-25T12:00:01Z",
                "level": "WARN",
                "service": "checkout-api",
                "message": "JVM heap usage at 96% on pod checkout-api-7d9f8c-x4k2p",
            },
            {
                "timestamp": "2026-11-25T12:00:04Z",
                "level": "CRITICAL",
                "service": "checkout-api",
                "message": "Container checkout-api killed by cgroup OOM killer exit_code=137",
            },
        ],
    )
    ollama_server.configure(mode="fenced")
    ollama_server.custom_steps = [
        {
            "step_id": "step-1",
            "title": "Capture pod diagnostics and memory profile",
            "description": "Inspect OOM events and container limit configuration.",
            "risk_level": "low",
            "command_preview": "kubectl describe pod checkout-api-7d9f8c-x4k2p -n production",
            "target_resource": "pod/checkout-api-7d9f8c-x4k2p",
            "expected_effect": "Confirms OOM termination cause without mutating workload.",
            "diff_template": "--- before\n+++ after\n# diagnostic read-only",
        },
        {
            "step_id": "step-2",
            "title": "Patch deployment memory limit to 2Gi",
            "description": "Raise memory limit and trigger rolling restart.",
            "risk_level": "high",
            "command_preview": "kubectl set resources deployment/checkout-api --limits=memory=2Gi",
            "target_resource": "deployment/checkout-api",
            "expected_effect": "Prevents repeat OOMKilled crashes.",
            "diff_template": "--- before\n+++ after\n- memory: 512Mi\n+ memory: 2Gi",
        },
    ]

    audit_path = tmp_path / "s1_audit.jsonl"
    stdin_script = "a\nApproved diagnostic step\nr\nWait for heap dump\n" + "s\n\n" * 5
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
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--model",
            "llama3.1:8b",
            "--no-allow-fallback",
            "--operator",
            "oncall-sre",
            "--audit-log",
            str(audit_path),
        ],
        stdin_text=stdin_script,
        timeout=20.0,
    )
    assert res.returncode == 0, res.combined_output

    events = read_audit_jsonl(audit_path)
    assert_valid_audit_events(events)
    decisions = [e for e in events if e["event_type"] == "operator_decision"]
    assert any(e["payload"].get("decision") == "approved" for e in decisions)
    assert any(e["payload"].get("decision") == "rejected" for e in decisions)

    dry_runs = [e for e in events if e["event_type"] == "dry_run_executed"]
    assert len(dry_runs) >= 1
    assert all(e["payload"].get("executed_against_real_infra") is False for e in dry_runs)

    completed = [e for e in events if e["event_type"] == "triage_completed"]
    assert len(completed) == 1
    if "fallback_used" in completed[0]["payload"]:
        assert completed[0]["payload"]["fallback_used"] is False

    inc_id = completed[0]["incident_id"]
    show = run_cli(["audit", "show", inc_id, "--audit-log", str(audit_path), "--json"])
    assert show.returncode == 0, show.combined_output
    shown_events = extract_list(show.parse_json())
    assert len(shown_events) >= 4
    timestamps = [e["timestamp"] for e in shown_events]
    assert timestamps == sorted(timestamps)


def test_t4_s2_api_5xx_db_pool_exhaustion_vllm(
    run_cli,
    alertmanager_server: AlertmanagerMockServer,
    openai_compat_server: OpenAICompatMockServer,
    tmp_path: Path,
) -> None:
    """Scenario 2: API 5xx + DB pool exhaustion -> Alertmanager v2 -> vLLM -> mixed decisions."""
    alertmanager_server.set_alerts(
        [
            {
                "fingerprint": "fp-db-active",
                "labels": {
                    "alertname": "APIDBPoolExhausted",
                    "severity": "critical",
                    "service": "orders-api",
                    "namespace": "production",
                },
                "annotations": {"summary": "HikariPool active connections 100/100"},
                "startsAt": "2026-11-25T12:05:00Z",
                "status": {"state": "active"},
            },
            {
                "fingerprint": "fp-db-suppressed",
                "labels": {
                    "alertname": "SuppressedOldAlert",
                    "severity": "warning",
                    "service": "orders-api",
                },
                "annotations": {"summary": "Silenced maintenance alert"},
                "startsAt": "2026-11-25T11:00:00Z",
                "status": {"state": "suppressed"},
            },
        ]
    )
    log_path = write_text_log(
        tmp_path / "orders_api.log",
        [
            "2026-11-25T12:05:01Z ERROR orders-api FATAL: remaining connection slots are reserved",
            "2026-11-25T12:05:02Z ERROR orders-api HikariPool-1 - Connection is not available after 30000ms",
        ],
    )
    openai_compat_server.custom_steps = [
        {
            "step_id": "step-1",
            "title": "Inspect active PostgreSQL connections",
            "description": "Query pg_stat_activity for idle-in-transaction sessions.",
            "risk_level": "low",
            "command_preview": "psql -c 'SELECT state, count(*) FROM pg_stat_activity GROUP BY 1'",
            "target_resource": "postgres/orders-db",
            "expected_effect": "Identifies leaked connection holders.",
            "diff_template": "--- before\n+++ after\n# diagnostic query",
        },
        {
            "step_id": "step-2",
            "title": "Terminate long-running idle transactions",
            "description": "Kill connections idle > 5m.",
            "risk_level": "medium",
            "command_preview": "psql -c 'SELECT pg_terminate_backend(pid)'",
            "target_resource": "postgres/orders-db",
            "expected_effect": "Frees blocked pool slots.",
            "diff_template": "--- before\n+++ after\n- idle_conns: 40\n+ idle_conns: 0",
        },
        {
            "step_id": "step-3",
            "title": "Bounce orders-api deployment",
            "description": "Rolling restart orders-api pods.",
            "risk_level": "high",
            "command_preview": "kubectl rollout restart deployment/orders-api",
            "target_resource": "deployment/orders-api",
            "expected_effect": "Resets client connection pools.",
            "diff_template": "--- before\n+++ after\n+ restartedAt: now",
        },
    ]

    audit_path = tmp_path / "s2_audit.jsonl"
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
            "openai_compatible",
            "--base-url",
            openai_compat_server.base_url,
            "--model",
            "qwen2.5:7b",
            "--api-key",
            "local-vllm-token",
            "--no-allow-fallback",
            "--decisions",
            "approve,reject,skip",
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)

    post_reqs = [r for r in openai_compat_server.get_requests() if r.method == "POST"]
    assert len(post_reqs) >= 1
    assert post_reqs[0].get_header("Authorization") == "Bearer local-vllm-token"

    # Suppressed alert should not appear as firing unless --include-resolved was requested
    firing_alerts = [
        a for a in payload["results"][0]["alerts"] if a["status"] == "firing"
    ]
    assert all(a["alertname"] != "SuppressedOldAlert" for a in firing_alerts)

    decision_map = {d["step_id"]: d["decision"] for d in payload["decisions"]}
    assert set(decision_map.values()) == {"approved", "rejected", "skipped"}
    success_dry_runs = [
        dr for dr in payload["dry_runs"] if dr["simulated_status"] == "SUCCESS_DRY_RUN"
    ]
    assert len(success_dry_runs) == 1
    assert decision_map[success_dry_runs[0]["step_id"]] == "approved"

    rej_query = run_cli(
        [
            "audit",
            "list",
            "--audit-log",
            str(audit_path),
            "--decision",
            "rejected",
            "--json",
        ]
    )
    assert rej_query.returncode == 0, rej_query.combined_output
    rej_events = extract_list(rej_query.parse_json())
    assert len(rej_events) == 1
    assert rej_events[0]["payload"]["decision"] == "rejected"


def test_t4_s3_disk_pressure_yaml_config_timeout_fallback(
    run_cli,
    prometheus_server: PrometheusMockServer,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Scenario 3: Disk pressure -> YAML config -> slow Ollama timeout -> deterministic fallback."""
    prometheus_server.set_alerts(
        [
            {
                "labels": {
                    "alertname": "HostDiskPressure",
                    "severity": "critical",
                    "service": "node-exporter",
                    "instance": "worker-node-03",
                },
                "annotations": {"summary": "/var/log disk usage above 95%"},
                "state": "firing",
                "activeAt": "2026-11-25T12:15:00Z",
                "value": "96.4",
            }
        ]
    )
    combined_log = write_text_log(
        tmp_path / "host_and_app.log",
        [
            "2026-11-25T12:14:58Z ERROR node-exporter No space left on device /var/log",
            "2026-11-25T12:15:01Z CRITICAL billing-worker Failed writing checkpoint: ENOSPC",
        ],
    )
    ollama_server.configure(delay_s=2.5)
    yaml_audit = tmp_path / "yaml_configured_audit.jsonl"
    cfg_path = write_yaml_config(
        tmp_path / "disk_triage.yaml",
        {
            "llm_provider": "ollama",
            "llm_base_url": ollama_server.base_url,
            "llm_model": "llama3.1:8b",
            "llm_timeout_seconds": 0.4,
            "llm_allow_fallback": True,
            "audit_log_path": str(yaml_audit),
        },
    )

    res = run_cli(
        [
            "triage",
            "--config",
            str(cfg_path),
            "--log-file",
            str(combined_log),
            "--alert-url",
            prometheus_server.alerts_url,
            "--decisions",
            ",".join(["approve"] * 8),
            "--json",
        ],
        timeout=15.0,
    )
    assert res.returncode == 0, res.combined_output
    assert res.timed_out is False
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["fallback_used"] is True
    assert result["fallback_reason"]
    assert result["provider_used"] == "deterministic"
    assert yaml_audit.exists()
    assert_valid_audit_events(read_audit_jsonl(yaml_audit))


def test_t4_s4_bundled_all_scenarios_batch_triage(run_cli, tmp_path: Path) -> None:
    """Scenario 4: Offline --all-scenarios batch with mixed interactive stdin and audit check."""
    audit_path = tmp_path / "s4_batch_audit.jsonl"
    mixed_stdin = "a\n\nr\n\ns\n\n" * 10
    res = run_cli(
        [
            "triage",
            "--all-scenarios",
            "--provider",
            "deterministic",
            "--operator",
            "batch-sre",
            "--audit-log",
            str(audit_path),
        ],
        stdin_text=mixed_stdin,
        timeout=20.0,
    )
    assert res.returncode == 0, res.combined_output
    events = read_audit_jsonl(audit_path)
    assert_valid_audit_events(events)

    run_ids = {e["run_id"] for e in events}
    assert len(run_ids) == 1
    run_id = next(iter(run_ids))

    completed = [e for e in events if e["event_type"] == "triage_completed"]
    assert len(completed) == len(CANONICAL_SCENARIOS)
    incident_ids = {e["incident_id"] for e in completed}
    assert len(incident_ids) == len(CANONICAL_SCENARIOS)

    list_res = run_cli(
        ["audit", "list", "--audit-log", str(audit_path), "--run-id", run_id, "--json"]
    )
    assert list_res.returncode == 0, list_res.combined_output
    listed = extract_list(list_res.parse_json())
    assert len(listed) == len(events)
    event_types = {e["event_type"] for e in listed}
    assert {"triage_completed", "remediation_suggested", "operator_decision"}.issubset(
        event_types
    )


def test_t4_s5_alert_only_injection_hardened_strict(
    run_cli,
    alertmanager_server: AlertmanagerMockServer,
    ollama_server: OllamaMockServer,
    tmp_path: Path,
) -> None:
    """Scenario 5: Alertmanager-only injection payload under strict Ollama mode + canary file."""
    canary = tmp_path / "canary.txt"
    canary.write_text("intact", encoding="utf-8")

    alertmanager_server.set_alerts(
        [
            {
                "fingerprint": "fp-inj-s5",
                "labels": {
                    "alertname": "PodOOMKilled",
                    "severity": "critical",
                    "service": "checkout-api",
                    "pod": "web-01; touch pwned_pod.txt",
                    "namespace": "prod && touch pwned_ns.txt",
                },
                "annotations": {"summary": "OOM with hostile label characters"},
                "startsAt": "2026-11-25T12:25:00Z",
                "status": {"state": "active"},
            }
        ]
    )
    ollama_server.custom_command = (
        "kubectl delete pod 'web-01; touch pwned_pod.txt' -n 'prod && touch pwned_ns.txt'"
    )

    audit_path = tmp_path / "s5_audit.jsonl"
    res = run_cli(
        [
            "triage",
            "--alert-url",
            alertmanager_server.alerts_url,
            "--alert-api-type",
            "alertmanager",
            "--provider",
            "ollama",
            "--base-url",
            ollama_server.base_url,
            "--no-allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ],
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.combined_output
    payload = res.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert result["fallback_used"] is False
    assert result["logs"] == []
    for step in result["remediation_steps"]:
        assert all(ref["kind"] == "alert" for ref in step["evidence_refs"])

    assert canary.read_text(encoding="utf-8") == "intact"
    assert not (tmp_path / "pwned_pod.txt").exists()
    assert not (tmp_path / "pwned_ns.txt").exists()
    assert all(dr["executed_against_real_infra"] is False for dr in payload["dry_runs"])


def test_t4_s6_env_driven_pipeline_tail_and_resolved(
    run_cli,
    prometheus_server: PrometheusMockServer,
    openai_compat_server: OpenAICompatMockServer,
    tmp_path: Path,
) -> None:
    """Scenario 6: Full SRE_AGENT_* env pipeline + providers check + --tail-lines + --include-resolved."""
    prometheus_server.set_alerts(
        [
            {
                "labels": {
                    "alertname": "HighLatency",
                    "severity": "warning",
                    "service": "gateway",
                },
                "annotations": {"summary": "Active latency spike"},
                "state": "firing",
                "activeAt": "2026-11-25T12:30:00Z",
                "value": "1",
            },
            {
                "labels": {
                    "alertname": "RecoveredNode",
                    "severity": "info",
                    "service": "gateway",
                },
                "annotations": {"summary": "Previously resolved alert"},
                "state": "inactive",
                "activeAt": "2026-11-25T11:30:00Z",
                "value": "0",
            },
        ]
    )
    lines = [
        f"2026-11-25T12:00:{i % 60:02d}Z INFO gateway STALE_MARKER_LINE_{i}"
        for i in range(1, 476)
    ] + [
        f"2026-11-25T12:30:{i % 60:02d}Z ERROR gateway RECENT_BURST_{i}"
        for i in range(476, 501)
    ]
    big_log = write_text_log(tmp_path / "gateway_500.log", lines)
    env_audit = tmp_path / "s6_env_audit.jsonl"

    pipeline_env = {
        "SRE_AGENT_LLM_PROVIDER": "openai_compatible",
        "SRE_AGENT_LLM_BASE_URL": openai_compat_server.base_url,
        "SRE_AGENT_LLM_MODEL": "qwen2.5:7b",
        "SRE_AGENT_LLM_API_KEY": "ci-pipe-key",
        "SRE_AGENT_PROMETHEUS_URL": prometheus_server.alerts_url,
        "SRE_AGENT_ALERT_API_TYPE": "prometheus",
        "SRE_AGENT_AUDIT_LOG_PATH": str(env_audit),
    }

    chk = run_cli(["providers", "check", "--json"], env=pipeline_env)
    assert chk.returncode == 0, chk.combined_output
    assert isinstance(chk.parse_json(), dict)

    tri = run_cli(
        [
            "triage",
            "--log-file",
            str(big_log),
            "--tail-lines",
            "25",
            "--include-resolved",
            "--no-allow-fallback",
            "--decisions",
            ",".join(["approve"] * 8),
            "--json",
        ],
        env=pipeline_env,
    )
    assert tri.returncode == 0, tri.combined_output
    payload = tri.parse_json()
    assert_valid_triage_json_payload(payload)
    result = payload["results"][0]
    assert 1 <= len(result["logs"]) <= 25
    joined_msgs = "\n".join(entry["message"] for entry in result["logs"])
    assert "RECENT_BURST_495" in joined_msgs
    assert "STALE_MARKER_LINE_1" not in joined_msgs
    assert {a["alertname"] for a in result["alerts"]} >= {"HighLatency", "RecoveredNode"}

    aud = run_cli(["audit", "list", "--json"], env=pipeline_env)
    assert aud.returncode == 0, aud.combined_output
    assert len(extract_list(aud.parse_json())) >= 4


def test_t4_s7_multi_run_persistent_audit_and_quit(run_cli, tmp_path: Path) -> None:
    """Scenario 7: Sequential runs on shared audit log (Run 1 interactive quit -> Run 2 full approval)."""
    audit_path = tmp_path / "shared_multi_run.jsonl"

    # Run 1: approve step 1 then quit mid-flow
    r1 = run_cli(
        [
            "triage",
            "--scenario",
            "pod_oom_killed",
            "--provider",
            "deterministic",
            "--operator",
            "shift-lead-1",
            "--audit-log",
            str(audit_path),
        ],
        stdin_text="a\n\nq\n",
        timeout=15.0,
    )
    assert r1.returncode == 0, r1.combined_output
    events_after_r1 = read_audit_jsonl(audit_path)
    assert_valid_audit_events(events_after_r1)
    run1_id = events_after_r1[0]["run_id"]
    inc1_id = next(e["incident_id"] for e in events_after_r1 if e.get("incident_id"))

    # Run 2: full scripted approval appending to the same audit file
    r2 = run_cli(
        [
            "triage",
            "--scenario",
            "api_5xx_db_pool",
            "--provider",
            "deterministic",
            "--operator",
            "shift-lead-2",
            "--decisions",
            ",".join(["approve"] * 8),
            "--audit-log",
            str(audit_path),
            "--json",
        ]
    )
    assert r2.returncode == 0, r2.combined_output
    r2_payload = r2.parse_json()
    assert_valid_triage_json_payload(r2_payload)
    run2_id = r2_payload["run_id"]
    inc2_id = r2_payload["results"][0]["incident_id"]
    assert run1_id != run2_id
    assert inc1_id != inc2_id

    all_events = read_audit_jsonl(audit_path)
    assert len(all_events) > len(events_after_r1)
    assert all_events[: len(events_after_r1)] == events_after_r1

    show1 = run_cli(["audit", "show", inc1_id, "--audit-log", str(audit_path), "--json"])
    show2 = run_cli(["audit", "show", inc2_id, "--audit-log", str(audit_path), "--json"])
    assert show1.returncode == 0 and show2.returncode == 0
    s1_events = extract_list(show1.parse_json())
    s2_events = extract_list(show2.parse_json())
    assert all(e["incident_id"] == inc1_id for e in s1_events)
    assert all(e["incident_id"] == inc2_id for e in s2_events)
    assert [e["timestamp"] for e in s1_events] == sorted(e["timestamp"] for e in s1_events)
    assert [e["timestamp"] for e in s2_events] == sorted(e["timestamp"] for e in s2_events)
