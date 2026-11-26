"""Unit tests for F12 persistent append-only JSONL AuditLogger."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sre_triage_agent.audit import AuditError, AuditLogger, AuditStore
from sre_triage_agent.llm import DeterministicFallbackProvider
from sre_triage_agent.models import (
    AuditEvent,
    DryRunOutcome,
    OperatorDecision,
)
from sre_triage_agent.triage import TriageEngine


def test_audit_logger_creates_nested_dirs_and_records_all_five_event_types(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "nested" / "dir" / "audit_log.jsonl"
    logger = AuditLogger(log_path)

    engine = TriageEngine(provider=DeterministicFallbackProvider())
    result = engine.triage_scenario("pod_oom_killed", run_id="run-audit-001")
    step = result.remediation_steps[0]

    e1 = logger.log_triage_run_started(
        run_id=result.run_id,
        incident_id=result.incident_id,
        payload={"mode": "bundled"},
    )
    e2 = logger.log_triage_completed(result)
    e3 = logger.log_remediation_suggested(result, step)

    decision = OperatorDecision(
        decision_id="dec-1",
        run_id=result.run_id,
        incident_id=result.incident_id,
        step_id=step.step_id,
        decision="approved",
        operator="alice",
        reason="looks safe",
    )
    e4 = logger.log_operator_decision(decision)

    outcome = DryRunOutcome(
        outcome_id="dryrun-1",
        run_id=result.run_id,
        incident_id=result.incident_id,
        step_id=step.step_id,
        command_preview=step.command_preview,
        diff_preview="--- a\n+++ b",
        simulated_stdout="[DRY-RUN] ok",
        simulated_status="SUCCESS_DRY_RUN",
        executed_against_real_infra=False,
    )
    e5 = logger.log_dry_run_executed(outcome)

    assert log_path.is_file()
    raw_bytes = log_path.read_bytes()
    assert b"\r\n" not in raw_bytes
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5

    # Re-open in a fresh AuditLogger instance to verify append-only persistence
    reader = AuditLogger(log_path)
    events = reader.read_events()
    assert [ev.event_type for ev in events] == [
        "triage_run_started",
        "triage_completed",
        "remediation_suggested",
        "operator_decision",
        "dry_run_executed",
    ]
    for ev, expected in zip(events, [e1, e2, e3, e4, e5], strict=True):
        assert ev.schema_version == "1.0"
        assert ev.event_id == expected.event_id
        assert ev.timestamp.tzinfo is not None
        assert ev.run_id == "run-audit-001"
        assert ev.incident_id == result.incident_id

    assert events[-1].payload["executed_against_real_infra"] is False


def test_audit_logger_query_filters_with_logical_and(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path)
    base_ts = datetime(2026, 11, 26, 1, 0, 0, tzinfo=timezone.utc)

    events = [
        AuditEvent(
            event_id="evt-1",
            timestamp=base_ts,
            run_id="run-a",
            incident_id="inc-1",
            event_type="operator_decision",
            actor="alice",
            payload={"decision": "approved", "step_id": "s1"},
        ),
        AuditEvent(
            event_id="evt-2",
            timestamp=base_ts + timedelta(seconds=1),
            run_id="run-a",
            incident_id="inc-1",
            event_type="operator_decision",
            actor="alice",
            payload={"decision": "rejected", "step_id": "s2"},
        ),
        AuditEvent(
            event_id="evt-3",
            timestamp=base_ts + timedelta(seconds=2),
            run_id="run-b",
            incident_id="inc-2",
            event_type="operator_decision",
            actor="bob",
            payload={"decision": "approved", "step_id": "s3"},
        ),
        AuditEvent(
            event_id="evt-4",
            timestamp=base_ts + timedelta(seconds=3),
            run_id="run-a",
            incident_id="inc-1",
            event_type="dry_run_executed",
            actor="dry-run-executor",
            payload={"step_id": "s1", "executed_against_real_infra": False},
        ),
    ]
    for ev in events:
        logger.log_event(ev)

    assert [e.event_id for e in logger.query(incident_id="inc-1")] == [
        "evt-1",
        "evt-2",
        "evt-4",
    ]
    assert [e.event_id for e in logger.query(run_id="run-b")] == ["evt-3"]
    assert [e.event_id for e in logger.query(event_type="dry_run_executed")] == ["evt-4"]
    assert [e.event_id for e in logger.query(decision="approved")] == ["evt-1", "evt-3"]
    assert [
        e.event_id
        for e in logger.query(incident_id="inc-1", run_id="run-a", decision="rejected")
    ] == ["evt-2"]


def test_audit_logger_tolerates_corrupt_lines_non_utf8_and_missing_trailing_newline(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "corrupt.jsonl"
    valid_event = AuditEvent(
        event_id="evt-valid-1",
        timestamp=datetime(2026, 11, 26, 2, 0, tzinfo=timezone.utc),
        run_id="run-1",
        incident_id="inc-1",
        event_type="triage_run_started",
        actor="tester",
        payload={},
    )
    # Write corrupt lines, blank line, non-UTF8 bytes, and a valid JSON line WITHOUT trailing newline
    raw_bytes = (
        b"{broken json\n"
        b"\n"
        b'{"event_id": "missing-fields"}\n'
        b"\xff\xfeinvalid-utf8\n"
        + valid_event.model_dump_json().encode("utf-8")
    )
    log_path.write_bytes(raw_bytes)

    logger = AuditLogger(log_path)
    # Appending another event must isolate it on a new line even though file lacked trailing \n
    second_event = AuditEvent(
        event_id="evt-valid-2",
        timestamp=datetime(2026, 11, 26, 2, 1, tzinfo=timezone.utc),
        run_id="run-1",
        incident_id="inc-1",
        event_type="triage_completed",
        actor="tester",
        payload={},
    )
    logger.log_event(second_event)

    read_res = logger.read_all_with_diagnostics()
    assert [ev.event_id for ev in read_res.events] == ["evt-valid-1", "evt-valid-2"]
    assert len(read_res.corrupt_lines) == 3
    assert [c.line_number for c in read_res.corrupt_lines] == [1, 3, 4]
    assert len(logger.last_corrupt_lines) == 3


def test_audit_logger_missing_file_returns_empty_and_directory_raises(
    tmp_path: Path,
) -> None:
    missing = AuditLogger(tmp_path / "does_not_exist.jsonl")
    assert missing.read_events() == []
    assert missing.query(incident_id="inc-x") == []

    dir_logger = AuditLogger(tmp_path)
    with pytest.raises(AuditError):
        dir_logger.read_events()
    with pytest.raises(AuditError):
        dir_logger.log_triage_run_started(run_id="r", incident_id="i")


def test_audit_logger_thread_safe_concurrent_appends(tmp_path: Path) -> None:
    log_path = tmp_path / "concurrent.jsonl"
    logger = AuditLogger(log_path)

    def _write_one(idx: int) -> None:
        ev = AuditEvent(
            event_id=f"evt-thread-{idx}",
            run_id="run-conc",
            incident_id="inc-conc",
            event_type="triage_run_started",
            actor="worker",
            payload={"idx": idx},
        )
        logger.log_event(ev)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_write_one, range(40)))

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40
    for line in lines:
        json.loads(line)
    events = logger.read_events()
    assert len(events) == 40


def test_audit_store_alias_append_and_show_incident_chronological_sort(
    tmp_path: Path,
) -> None:
    assert AuditStore is AuditLogger
    log_path = tmp_path / "chronological.jsonl"
    store = AuditStore(log_path)
    base_ts = datetime(2026, 11, 26, 5, 0, 0, tzinfo=timezone.utc)

    # Append out-of-order timestamps + an unrelated incident
    store.append(
        AuditEvent(
            event_id="evt-late",
            timestamp=base_ts + timedelta(seconds=30),
            run_id="run-1",
            incident_id="inc-target",
            event_type="dry_run_executed",
            actor="alice",
            payload={"executed_against_real_infra": False},
        )
    )
    store.append(
        AuditEvent(
            event_id="evt-other",
            timestamp=base_ts + timedelta(seconds=10),
            run_id="run-1",
            incident_id="inc-other",
            event_type="triage_run_started",
            actor="sre-triage-agent",
            payload={},
        )
    )
    store.append(
        AuditEvent(
            event_id="evt-early",
            timestamp=base_ts,
            run_id="run-1",
            incident_id="inc-target",
            event_type="triage_run_started",
            actor="sre-triage-agent",
            payload={},
        )
    )
    store.append(
        AuditEvent(
            event_id="evt-mid",
            timestamp=base_ts + timedelta(seconds=15),
            run_id="run-1",
            incident_id="inc-target",
            event_type="operator_decision",
            actor="alice",
            payload={"decision": "approved"},
        )
    )

    ordered = store.show_incident("inc-target")
    assert [ev.event_id for ev in ordered] == ["evt-early", "evt-mid", "evt-late"]
