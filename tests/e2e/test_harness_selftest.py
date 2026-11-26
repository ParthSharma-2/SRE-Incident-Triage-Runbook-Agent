"""Self-tests verifying the opaque-box E2E test harness before/alongside agent code."""

from __future__ import annotations

import json
import os
import threading
import time
from itertools import combinations
from pathlib import Path

import httpx
import pytest
import yaml

from conftest import (
    CANONICAL_SCENARIOS,
    PROJECT_ROOT,
    SRC_ROOT,
    AlertmanagerMockServer,
    OllamaMockServer,
    OpenAICompatMockServer,
    PrometheusMockServer,
    _extract_ids_from_messages,
    build_clean_env,
    extract_list,
    read_audit_jsonl,
    unused_loopback_port,
    write_yaml_config,
)
from test_tier3_combinations import OA16_MATRIX


def test_selftest_prometheus_mock_server_protocol_and_modes(
    prometheus_server: PrometheusMockServer,
) -> None:
    """Verify Prometheus mock endpoints, request capture, 500 error, and delay."""
    resp = httpx.get(prometheus_server.alerts_url, timeout=5.0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert len(body["data"]["alerts"]) == 1
    assert body["data"]["alerts"][0]["labels"]["alertname"] == "PodOOMKilled"

    # Alertmanager path returns 404 so auto-detection differentiates Prometheus
    am_resp = httpx.get(f"{prometheus_server.base_url}/api/v2/alerts", timeout=5.0)
    assert am_resp.status_code == 404

    prometheus_server.configure(status_code=500)
    err_resp = httpx.get(prometheus_server.alerts_url, timeout=5.0)
    assert err_resp.status_code == 500

    prometheus_server.configure(status_code=200, delay_s=0.15)
    t0 = time.monotonic()
    slow_resp = httpx.get(prometheus_server.alerts_url, timeout=5.0)
    elapsed = time.monotonic() - t0
    assert slow_resp.status_code == 200
    assert elapsed >= 0.12

    reqs = prometheus_server.get_requests()
    assert len(reqs) == 4
    assert reqs[0].method == "GET"
    assert reqs[0].path == "/api/v1/alerts"

    # Rapid teardown during active delay_s=10.0 must disconnect HTTP/1.1 keep-alive client in < 1.5s
    for srv_cls, path in (
        (PrometheusMockServer, "/api/v1/alerts"),
        (OpenAICompatMockServer, "/v1/models"),
    ):
        with srv_cls() as slow_srv:
            slow_srv.configure(delay_s=10.0)
            client_done = threading.Event()
            client_outcomes: list[tuple[str, float]] = []

            with httpx.Client(verify=False, trust_env=False, timeout=10.0) as client:
                def _fetch(
                    url: str = f"{slow_srv.base_url}{path}",
                    http_client: httpx.Client = client,
                    done_evt: threading.Event = client_done,
                    outcomes: list[tuple[str, float]] = client_outcomes,
                ) -> None:
                    t_req = time.monotonic()
                    try:
                        resp = http_client.get(url)
                        outcomes.append((f"status:{resp.status_code}", time.monotonic() - t_req))
                    except Exception as exc:  # pragma: no cover - timing-dependent disconnect
                        outcomes.append((type(exc).__name__, time.monotonic() - t_req))
                    finally:
                        done_evt.set()

                t_worker = threading.Thread(target=_fetch, daemon=True)
                t_worker.start()
                try:
                    deadline = time.monotonic() + 2.0
                    while not slow_srv.get_requests() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    assert slow_srv.get_requests(), "In-flight request did not reach mock server"

                    t_close = time.monotonic()
                    slow_srv.close()
                    close_elapsed = time.monotonic() - t_close
                    assert client_done.wait(timeout=1.5), (
                        "Client hung on HTTP/1.1 keep-alive during teardown"
                    )
                finally:
                    slow_srv.close()
                    t_worker.join(timeout=5.0)

            assert close_elapsed < 1.5
            assert client_outcomes and client_outcomes[0][1] < 1.5
            assert client_outcomes[0][0] != "ReadTimeout"


def test_selftest_alertmanager_mock_server_protocol_and_modes(
    alertmanager_server: AlertmanagerMockServer,
) -> None:
    """Verify Alertmanager v2 mock endpoints, custom alerts, and header capture."""
    alertmanager_server.set_alerts(
        [
            {
                "fingerprint": "fp-active",
                "labels": {"alertname": "High5xx", "service": "api", "pod": "api-0; id"},
                "annotations": {"summary": "Active 5xx"},
                "startsAt": "2026-11-25T12:00:00Z",
                "status": {"state": "active"},
            },
            {
                "fingerprint": "fp-suppressed",
                "labels": {"alertname": "ResolvedWarn", "service": "api"},
                "annotations": {"summary": "Suppressed"},
                "startsAt": "2026-11-25T11:00:00Z",
                "status": {"state": "suppressed"},
            },
        ]
    )

    resp = httpx.get(
        alertmanager_server.alerts_url,
        headers={"X-Custom-Probe": "e2e-check"},
        timeout=5.0,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["labels"]["pod"] == "api-0; id"
    assert data[1]["status"]["state"] == "suppressed"

    prom_resp = httpx.get(f"{alertmanager_server.base_url}/api/v1/alerts", timeout=5.0)
    assert prom_resp.status_code == 404

    reqs = alertmanager_server.get_requests()
    assert reqs[0].get_header("x-custom-probe") == "e2e-check"


def test_selftest_ollama_mock_server_probes_modes_and_dynamic_ids(
    ollama_server: OllamaMockServer,
) -> None:
    """Verify Ollama probes, prompt-aware dynamic ID grounding, and response modes."""
    assert httpx.get(f"{ollama_server.base_url}/api/tags", timeout=5.0).status_code == 200
    assert httpx.get(f"{ollama_server.base_url}/api/version", timeout=5.0).status_code == 200
    assert httpx.get(f"{ollama_server.base_url}/", timeout=5.0).status_code == 200

    prompt_payload = {
        "model": "llama3.1:8b",
        "messages": [
            {
                "role": "user",
                "content": (
                    'Analyze incident with "alert_id": "alt_9911aa" '
                    'and log_id=log_4455bb using runbook_id=rb-oom-01.'
                ),
            }
        ],
    }

    # Valid mode extracts dynamic IDs
    chat_resp = httpx.post(f"{ollama_server.base_url}/api/chat", json=prompt_payload, timeout=5.0)
    assert chat_resp.status_code == 200
    content_obj = json.loads(chat_resp.json()["message"]["content"])
    refs = content_obj["remediation_steps"][0]["evidence_refs"]
    assert {r["ref_id"] for r in refs} == {"alt_9911aa", "log_4455bb"}
    assert content_obj["remediation_steps"][0]["runbook_id"] == "rb-oom-01"

    # Fenced mode wraps valid JSON in markdown code fence
    ollama_server.configure(mode="fenced")
    fenced_content = httpx.post(
        f"{ollama_server.base_url}/api/chat", json=prompt_payload, timeout=5.0
    ).json()["message"]["content"]
    assert fenced_content.startswith("```json")
    assert "alt_9911aa" in fenced_content

    # Malformed mode emits unparseable text
    ollama_server.configure(mode="malformed")
    malformed_content = httpx.post(
        f"{ollama_server.base_url}/api/chat", json=prompt_payload, timeout=5.0
    ).json()["message"]["content"]
    with pytest.raises(json.JSONDecodeError):
        json.loads(malformed_content)

    # fail_all fails probes too
    ollama_server.configure(mode="fail_all")
    assert httpx.get(f"{ollama_server.base_url}/api/tags", timeout=5.0).status_code == 503

    # Reserved schema keys in system prompt must never leak as extracted IDs, and quoted YAML/MD works
    alert_only_msgs = [
        {"role": "system", "content": "Cite alert_id or log_id and runbook_id in JSON."},
        {"role": "user", "content": '{"alert_id": "alt-999"}'},
    ]
    assert _extract_ids_from_messages(alert_only_msgs) == (["alt-999"], [], [])

    yaml_md_msgs = [
        {"role": "user", "content": 'alert_id: "alt-123"\nlog_id: \'log-456\'\nrunbook_id: "rb-789"'}
    ]
    assert _extract_ids_from_messages(yaml_md_msgs) == (["alt-123"], ["log-456"], ["rb-789"])

    ollama_server.configure(mode="valid")
    alert_only_resp = httpx.post(
        f"{ollama_server.base_url}/api/chat",
        json={"model": "llama3.1:8b", "messages": alert_only_msgs},
        timeout=5.0,
    )
    assert alert_only_resp.status_code == 200
    alert_only_obj = json.loads(alert_only_resp.json()["message"]["content"])
    alert_only_refs = alert_only_obj["remediation_steps"][0]["evidence_refs"]
    assert len(alert_only_refs) == 1
    assert alert_only_refs[0]["kind"] == "alert"
    assert alert_only_refs[0]["ref_id"] == "alt-999"

    # System schema types/placeholders/prose must never pollute alert-only user payloads
    schema_prose_msgs = [
        {
            "role": "system",
            "content": (
                'Return JSON with "alert_id": "string", log_id: str, <log_id>, '
                "log_ids, log_entries, log_level, alert_name, alert-only, alert_events, "
                "alert_description, log_timestamp."
            ),
        },
        {"role": "user", "content": '{"alert_id": "alt-real-01"}'},
    ]
    assert _extract_ids_from_messages(schema_prose_msgs) == (["alt-real-01"], [], [])

    # Even if schema prose appears in a non-system prompt, reserved tokens and non-numeric
    # unkeyed snake_case prose must be ignored, while explicit keyed non-numeric IDs match.
    inline_prose_msgs = [
        {
            "role": "user",
            "content": (
                '"alert_id": "string", log_id: str, "<log_id>", log_ids, log_entries, '
                "log_level, alert_name, alert-only, alert_events, alert_description, "
                'log_timestamp, {"alert_id": "alt-real-01"}'
            ),
        }
    ]
    assert _extract_ids_from_messages(inline_prose_msgs) == (["alt-real-01"], [], [])
    assert _extract_ids_from_messages(
        [{"role": "user", "content": "alert_id=custom-nonnumeric [log:syslog-main]"}]
    ) == (["custom-nonnumeric"], ["syslog-main"], [])

    schema_chat_resp = httpx.post(
        f"{ollama_server.base_url}/api/chat",
        json={"model": "llama3.1:8b", "messages": schema_prose_msgs},
        timeout=5.0,
    )
    assert schema_chat_resp.status_code == 200
    schema_chat_obj = json.loads(schema_chat_resp.json()["message"]["content"])
    schema_refs = schema_chat_obj["remediation_steps"][0]["evidence_refs"]
    assert len(schema_refs) == 1
    assert schema_refs[0]["kind"] == "alert"
    assert schema_refs[0]["ref_id"] == "alt-real-01"


def test_selftest_openai_compat_mock_server_auth_probes_and_errors(
    openai_compat_server: OpenAICompatMockServer,
) -> None:
    """Verify OpenAI-compatible probes, Bearer auth capture, and error modes."""
    assert httpx.get(f"{openai_compat_server.base_url}/v1/models", timeout=5.0).status_code == 200
    assert httpx.get(f"{openai_compat_server.base_url}/models", timeout=5.0).status_code == 200
    assert httpx.get(f"{openai_compat_server.base_url}/health", timeout=5.0).status_code == 200

    resp = httpx.post(
        f"{openai_compat_server.base_url}/v1/chat/completions",
        headers={"Authorization": "Bearer secret-local-key"},
        json={
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": '{"alert_id": "alert-xyz", "log_id": "log-123"}'}],
        },
        timeout=5.0,
    )
    assert resp.status_code == 200
    choice_content = json.loads(resp.json()["choices"][0]["message"]["content"])
    refs = choice_content["remediation_steps"][0]["evidence_refs"]
    assert {r["ref_id"] for r in refs} == {"alert-xyz", "log-123"}

    post_reqs = [r for r in openai_compat_server.get_requests() if r.method == "POST"]
    assert len(post_reqs) == 1
    assert post_reqs[0].get_header("Authorization") == "Bearer secret-local-key"

    openai_compat_server.configure(mode="ungrounded")
    ungrounded = httpx.post(
        f"{openai_compat_server.base_url}/v1/chat/completions",
        json={"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "alert_id=a1"}]},
        timeout=5.0,
    )
    ungrounded_refs = json.loads(ungrounded.json()["choices"][0]["message"]["content"])[
        "remediation_steps"
    ][0]["evidence_refs"]
    assert ungrounded_refs[0]["ref_id"] == "nonexistent-fabricated-id-999"


def test_selftest_subprocess_env_isolation_utf8_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    run_python,
    tmp_path: Path,
) -> None:
    """Verify run_python/build_clean_env isolates SRE_AGENT_*, handles UTF-8, stdin, and timeout."""
    monkeypatch.setenv("SRE_AGENT_LEAKED_VAR", "should_be_stripped")
    cleaned = build_clean_env({"CUSTOM_KEEP": "yes"})
    assert "SRE_AGENT_LEAKED_VAR" not in cleaned
    assert cleaned["CUSTOM_KEEP"] == "yes"
    assert cleaned["PYTHONUTF8"] == "1"
    assert str(SRC_ROOT) in cleaned["PYTHONPATH"].split(os.pathsep)

    # UTF-8 + stdin + CWD with spaces/tmp_path + PYTHONPATH verification
    spaced_dir = tmp_path / "dir with spaces"
    spaced_dir.mkdir()
    script = (
        "import json, os, sys; "
        "data = sys.stdin.read().strip(); "
        "print(json.dumps({'cwd': os.getcwd(), 'stdin': data, "
        "'leaked': 'SRE_AGENT_LEAKED_VAR' in os.environ, "
        "'src_in_pp': sys.argv[1] in os.environ.get('PYTHONPATH', ''), "
        "'unicode': '✓ → 日本語'}))"
    )
    res = run_python(
        ["-c", script, str(SRC_ROOT)],
        cwd=spaced_dir,
        stdin_text="approve ✓\n",
    )
    assert res.returncode == 0
    assert res.timed_out is False
    parsed = res.parse_json()
    assert Path(parsed["cwd"]).resolve() == spaced_dir.resolve()
    assert parsed["stdin"] == "approve ✓"
    assert parsed["leaked"] is False
    assert parsed["src_in_pp"] is True
    assert parsed["unicode"] == "✓ → 日本語"

    # Timeout handling does not hang or raise unhandled error
    timeout_res = run_python(["-c", "import time; time.sleep(2.0)"], timeout=0.2)
    assert timeout_res.timed_out is True
    assert timeout_res.returncode == -1


def test_selftest_helpers_extract_list_audit_yaml_and_port(tmp_path: Path) -> None:
    """Verify extract_list, read_audit_jsonl, write_yaml_config, and unused_loopback_port."""
    assert extract_list([{"name": "a"}]) == [{"name": "a"}]
    assert extract_list({"scenarios": [{"name": "b"}]}) == [{"name": "b"}]
    assert extract_list({"events": [{"event_id": "e1"}]}) == [{"event_id": "e1"}]
    assert extract_list({"custom_key": [1, 2, 3]}) == [1, 2, 3]
    with pytest.raises(AssertionError):
        extract_list({"only_str": "nope"})

    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(
        json.dumps({"event_id": "1", "msg": "✓"}) + "\n\n" + json.dumps({"event_id": "2"}) + "\n",
        encoding="utf-8",
    )
    rows = read_audit_jsonl(audit_path)
    assert [r["event_id"] for r in rows] == ["1", "2"]
    assert rows[0]["msg"] == "✓"

    cfg_path = write_yaml_config(
        tmp_path / "nested" / "config.yaml",
        {"log_file": str(tmp_path / "User\\logs\\app.log"), "llm_provider": "ollama"},
    )
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert loaded["llm_provider"] == "ollama"
    assert "User\\logs\\app.log" in loaded["log_file"]

    port = unused_loopback_port()
    assert 1 <= port <= 65535
    assert PROJECT_ROOT.exists()

    # Regression check: OA16_MATRIX must cover 84/84 (100%) 2-way pairs across all 4 factors
    assert len(OA16_MATRIX) == 16
    factor_indices = (0, 2, 3, 4)
    total_covered = 0
    total_expected = 0
    for i, j in combinations(factor_indices, 2):
        levels_i = {row[i] for row in OA16_MATRIX}
        levels_j = {row[j] for row in OA16_MATRIX}
        observed_pairs = {(row[i], row[j]) for row in OA16_MATRIX}
        expected_count = len(levels_i) * len(levels_j)
        assert len(observed_pairs) == expected_count, (
            f"OA16 missing pairs for factor ({i}, {j}): "
            f"{len(observed_pairs)}/{expected_count}"
        )
        total_covered += len(observed_pairs)
        total_expected += expected_count
    assert total_expected == 84
    assert total_covered == 84
    bundled_single_scenarios = [row[1] for row in OA16_MATRIX if row[0] == "bundled_single"]
    assert set(bundled_single_scenarios) == set(CANONICAL_SCENARIOS)
