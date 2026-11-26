"""Tier 3 E2E Tests: 16-case GF(4) Orthogonal Array OA(16) pairwise suite (100% pair coverage)."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import (
    AlertmanagerMockServer,
    OllamaMockServer,
    OpenAICompatMockServer,
    PrometheusMockServer,
    assert_valid_audit_events,
    assert_valid_triage_json_payload,
    extract_list,
    read_audit_jsonl,
    write_jsonl_log,
)

THREE_MOCK_STEPS = [
    {
        "step_id": "step-1",
        "title": "Inspect target resource state",
        "description": "Collect current resource utilization and pod status.",
        "risk_level": "low",
        "command_preview": "kubectl describe pod target-pod",
        "target_resource": "pod/target-pod",
        "expected_effect": "Captures diagnostic events.",
        "diff_template": "--- before\n+++ after\n# read-only inspect",
    },
    {
        "step_id": "step-2",
        "title": "Adjust service resource limits",
        "description": "Increase container memory or pool threshold.",
        "risk_level": "medium",
        "command_preview": "kubectl set resources deployment/target --limits=memory=1Gi",
        "target_resource": "deployment/target",
        "expected_effect": "Prevents recurring OOM/exhaustion.",
        "diff_template": "--- before\n+++ after\n- memory: 512Mi\n+ memory: 1Gi",
    },
    {
        "step_id": "step-3",
        "title": "Rollout restart workload",
        "description": "Gracefully cycle pods to pick up updated limits.",
        "risk_level": "high",
        "command_preview": "kubectl rollout restart deployment/target",
        "target_resource": "deployment/target",
        "expected_effect": "Restores clean healthy replicas.",
        "diff_template": "--- before\n+++ after\n+ restartedAt: now",
    },
]

OA16_MATRIX: list[tuple[str, str | None, str, str, str]] = [
    ("bundled_single", "pod_oom_killed", "deterministic", "scripted_approve", "file_jsonl"),
    ("bundled_single", "api_5xx_db_pool", "ollama_mock", "scripted_mixed", "audit_list_filtered"),
    ("bundled_single", "disk_pressure", "openai_compat_mock", "interactive_approve", "audit_show_incident"),
    ("bundled_single", "tls_cert_expiring", "ollama_fallback", "interactive_quit", "file_jsonl"),
    ("bundled_all", None, "deterministic", "scripted_mixed", "audit_show_incident"),
    ("bundled_all", None, "ollama_mock", "scripted_approve", "file_jsonl"),
    ("bundled_all", None, "openai_compat_mock", "interactive_quit", "file_jsonl"),
    ("bundled_all", None, "ollama_fallback", "interactive_approve", "audit_list_filtered"),
    ("live_prometheus", None, "deterministic", "interactive_approve", "file_jsonl"),
    ("live_prometheus", None, "ollama_mock", "interactive_quit", "audit_show_incident"),
    ("live_prometheus", None, "openai_compat_mock", "scripted_approve", "audit_list_filtered"),
    ("live_prometheus", None, "ollama_fallback", "scripted_mixed", "file_jsonl"),
    ("live_alertmanager", None, "deterministic", "interactive_quit", "audit_list_filtered"),
    ("live_alertmanager", None, "ollama_mock", "interactive_approve", "file_jsonl"),
    ("live_alertmanager", None, "openai_compat_mock", "scripted_mixed", "file_jsonl"),
    ("live_alertmanager", None, "ollama_fallback", "scripted_approve", "audit_show_incident"),
]

OA16_IDS = [
    f"oa16_{idx:02d}_{ing}_{prov}_{appr}_{aud}"
    for idx, (ing, _sc, prov, appr, aud) in enumerate(OA16_MATRIX, start=1)
]


@pytest.mark.parametrize(
    ("ingestion_mode", "scenario_name", "provider_mode", "approval_mode", "audit_mode"),
    OA16_MATRIX,
    ids=OA16_IDS,
)
def test_t3_pairwise_oa16_matrix(
    ingestion_mode: str,
    scenario_name: str | None,
    provider_mode: str,
    approval_mode: str,
    audit_mode: str,
    run_cli,
    prometheus_server: PrometheusMockServer,
    alertmanager_server: AlertmanagerMockServer,
    ollama_server: OllamaMockServer,
    openai_compat_server: OpenAICompatMockServer,
    tmp_path: Path,
) -> None:
    """Execute 16-row GF(4) orthogonal array covering 100% of 2-way feature pairs."""
    audit_path = tmp_path / "oa16_audit.jsonl"
    args: list[str] = ["triage", "--audit-log", str(audit_path)]

    # --- Factor A: Ingestion ---
    if ingestion_mode == "bundled_single":
        assert scenario_name is not None
        args.extend(["--scenario", scenario_name])
    elif ingestion_mode == "bundled_all":
        args.append("--all-scenarios")
    elif ingestion_mode == "live_prometheus":
        log_path = write_jsonl_log(
            tmp_path / "live_prom.jsonl",
            [
                {
                    "timestamp": "2026-11-25T12:00:05Z",
                    "level": "ERROR",
                    "service": "checkout-api",
                    "message": "Pod checkout-api-7d9f8c-x4k2p OOMKilled exit 137",
                }
            ],
        )
        args.extend(
            [
                "--log-file",
                str(log_path),
                "--alert-url",
                prometheus_server.alerts_url,
                "--alert-api-type",
                "prometheus",
            ]
        )
    elif ingestion_mode == "live_alertmanager":
        log_path = write_jsonl_log(
            tmp_path / "live_am.jsonl",
            [
                {
                    "timestamp": "2026-11-25T12:05:05Z",
                    "level": "ERROR",
                    "service": "orders-api",
                    "message": "HikariPool-1 connection timeout after 30000ms",
                }
            ],
        )
        args.extend(
            [
                "--log-file",
                str(log_path),
                "--alert-url",
                alertmanager_server.alerts_url,
                "--alert-api-type",
                "alertmanager",
            ]
        )

    # --- Factor B: LLM Provider ---
    if provider_mode == "deterministic":
        args.extend(["--provider", "deterministic"])
    elif provider_mode == "ollama_mock":
        ollama_server.custom_steps = [dict(s) for s in THREE_MOCK_STEPS]
        args.extend(
            [
                "--provider",
                "ollama",
                "--base-url",
                ollama_server.base_url,
                "--model",
                "llama3.1:8b",
                "--no-allow-fallback",
            ]
        )
    elif provider_mode == "openai_compat_mock":
        openai_compat_server.custom_steps = [dict(s) for s in THREE_MOCK_STEPS]
        args.extend(
            [
                "--provider",
                "openai_compatible",
                "--base-url",
                openai_compat_server.base_url,
                "--model",
                "qwen2.5:7b",
                "--api-key",
                "oa16-local-key",
                "--no-allow-fallback",
            ]
        )
    elif provider_mode == "ollama_fallback":
        ollama_server.configure(status_code=500)
        args.extend(
            [
                "--provider",
                "ollama",
                "--base-url",
                ollama_server.base_url,
                "--model",
                "llama3.1:8b",
                "--allow-fallback",
            ]
        )

    # --- Factor C: Approval ---
    is_interactive = approval_mode.startswith("interactive_")
    stdin_text: str | None = None
    token_budget = 20 if ingestion_mode == "bundled_all" else 8
    if approval_mode == "scripted_approve":
        args.extend(["--decisions", ",".join(["approve"] * token_budget), "--json"])
    elif approval_mode == "scripted_mixed":
        mixed_tokens = "approve,reject,skip," + ",".join(["approve"] * (token_budget - 3))
        args.extend(["--decisions", mixed_tokens, "--json"])
    elif approval_mode == "interactive_approve":
        stdin_text = "a\n\n" * 18
    elif approval_mode == "interactive_quit":
        stdin_text = "a\n\nq\n"

    res = run_cli(args, stdin_text=stdin_text, timeout=20.0)
    assert res.returncode == 0, res.combined_output
    assert res.timed_out is False

    # Verify HTTP server interactions where expected
    if ingestion_mode == "live_prometheus":
        assert any(r.path == "/api/v1/alerts" for r in prometheus_server.get_requests())
    elif ingestion_mode == "live_alertmanager":
        assert any(r.path == "/api/v2/alerts" for r in alertmanager_server.get_requests())

    if provider_mode == "ollama_mock":
        assert any(r.path == "/api/chat" for r in ollama_server.get_requests())
    elif provider_mode == "openai_compat_mock":
        oai_posts = [r for r in openai_compat_server.get_requests() if r.method == "POST"]
        assert len(oai_posts) >= 1
        assert oai_posts[0].get_header("Authorization") == "Bearer oa16-local-key"
    elif provider_mode == "ollama_fallback":
        assert any(r.path == "/api/chat" for r in ollama_server.get_requests())

    # Validate triage output and audit log
    events = read_audit_jsonl(audit_path)
    assert_valid_audit_events(events)
    run_id = events[0]["run_id"]
    incident_ids = [e["incident_id"] for e in events if e.get("incident_id")]
    assert len(incident_ids) >= 1

    if not is_interactive:
        payload = res.parse_json()
        assert_valid_triage_json_payload(payload)
        run_id = payload["run_id"]
        if ingestion_mode == "bundled_all":
            assert len(payload["results"]) == 4
        if provider_mode == "ollama_fallback":
            assert all(r["fallback_used"] is True for r in payload["results"])
        if approval_mode == "scripted_mixed":
            recorded = {d["decision"] for d in payload["decisions"]}
            assert {"approved", "rejected", "skipped"}.issubset(recorded)
    else:
        decision_values = [
            e["payload"].get("decision")
            for e in events
            if e["event_type"] == "operator_decision"
        ]
        assert "approved" in decision_values
        if approval_mode == "interactive_approve":
            assert any(e["event_type"] == "dry_run_executed" for e in events)

    # --- Factor D: Audit Inspection ---
    if audit_mode == "file_jsonl":
        if ingestion_mode == "bundled_all":
            unique_incidents = {
                e["incident_id"]
                for e in events
                if e["event_type"] == "triage_completed" and e.get("incident_id")
            }
            assert len(unique_incidents) >= 1
    elif audit_mode == "audit_list_filtered":
        list_args = ["audit", "list", "--audit-log", str(audit_path), "--run-id", run_id, "--json"]
        if approval_mode == "scripted_mixed":
            # Verify decision filtering works cleanly across OA16 rows
            if provider_mode == "ollama_mock":
                list_args.extend(["--decision", "rejected"])
            else:
                list_args.extend(["--decision", "skipped"])
        elif approval_mode == "interactive_approve":
            list_args.extend(["--event-type", "dry_run_executed"])

        list_res = run_cli(list_args)
        assert list_res.returncode == 0, list_res.combined_output
        listed = extract_list(list_res.parse_json())
        assert len(listed) >= 1
        assert all(ev["run_id"] == run_id for ev in listed)
    elif audit_mode == "audit_show_incident":
        target_inc = incident_ids[0]
        show_res = run_cli(["audit", "show", target_inc, "--audit-log", str(audit_path), "--json"])
        assert show_res.returncode == 0, show_res.combined_output
        shown = extract_list(show_res.parse_json())
        assert len(shown) >= 2
        assert all(ev["incident_id"] == target_inc for ev in shown)
        timestamps = [ev["timestamp"] for ev in shown]
        assert timestamps == sorted(timestamps)
