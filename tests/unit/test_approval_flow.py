"""Unit tests for F11 interactive and scripted operator approval flows."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from sre_triage_agent.approval import (
    ActionValue,
    ApprovalError,
    ApprovalSummary,
    ApprovalWorkflow,
    DecisionAction,
    DecisionHandler,
    DecisionValue,
    InteractivePromptHandler,
    ParsedDecision,
    PromptDecision,
    ScriptedDecisionHandler,
    parse_decision_input,
    parse_scripted_decisions,
    process_remediations,
)
from sre_triage_agent.audit import AuditLogger
from sre_triage_agent.executor import DryRunExecutor
from sre_triage_agent.llm import DeterministicFallbackProvider
from sre_triage_agent.triage import TriageEngine


def _sample_triage_result(scenario: str = "api_5xx_db_pool"):
    engine = TriageEngine(provider=DeterministicFallbackProvider())
    return engine.triage_scenario(scenario, run_id="run-approval-test")


def test_parse_decision_input_supports_short_long_and_inline_reasons() -> None:
    assert parse_decision_input("a").action == "approved"
    assert parse_decision_input("approve").action == "approved"
    d = parse_decision_input("r: waiting for DBA window")
    assert d.action == "rejected"
    assert d.reason == "waiting for DBA window"

    d_space = parse_decision_input("skip defer to next shift")
    assert d_space.action == "skipped"
    assert d_space.reason == "defer to next shift"

    assert parse_decision_input("q").action == "quit"

    with pytest.raises(ApprovalError):
        parse_decision_input("")
    with pytest.raises(ApprovalError):
        parse_decision_input("maybe")


def test_scripted_decision_handler_validation_underflow_and_overflow() -> None:
    with pytest.raises(ApprovalError):
        ScriptedDecisionHandler("")
    with pytest.raises(ApprovalError):
        ScriptedDecisionHandler("approve,,skip")
    with pytest.raises(ApprovalError):
        ScriptedDecisionHandler("approve,bogus")

    result = _sample_triage_result("pod_oom_killed")
    assert len(result.remediation_steps) == 3

    # Underflow defaults to skip
    underflow_handler = ScriptedDecisionHandler("approve")
    summary = process_remediations(result, handler=underflow_handler, operator="alice")
    assert [d.decision for d in summary.decisions] == ["approved", "skipped", "skipped"]
    assert len(summary.dry_runs) == 1
    assert underflow_handler.underflow_count == 2
    assert summary.aborted is False

    # Underflow error mode
    strict_handler = ScriptedDecisionHandler("approve", on_underflow="error")
    with pytest.raises(ApprovalError):
        process_remediations(result, handler=strict_handler)

    # Overflow ignores extra tokens cleanly
    overflow_handler = ScriptedDecisionHandler("a,r,s,a,r")
    summary_overflow = process_remediations(result, handler=overflow_handler)
    assert [d.decision for d in summary_overflow.decisions] == [
        "approved",
        "rejected",
        "skipped",
    ]
    assert overflow_handler.remaining == 2


def test_process_remediations_approve_executes_reject_skips_and_quit_halts(
    tmp_path: Path,
) -> None:
    result = _sample_triage_result("api_5xx_db_pool")
    audit_path = tmp_path / "audit.jsonl"
    audit_logger = AuditLogger(audit_path)

    # Approve step 1, reject step 2, quit before step 3
    handler = ScriptedDecisionHandler("approve: inspect first,reject: too risky,quit")
    summary = process_remediations(
        result,
        handler=handler,
        executor=DryRunExecutor(),
        audit_logger=audit_logger,
        operator="oncall-sre",
    )

    assert summary.aborted is True
    assert [d.decision for d in summary.decisions] == ["approved", "rejected"]
    assert summary.decisions[0].reason == "inspect first"
    assert summary.decisions[1].reason == "too risky"
    assert len(summary.dry_runs) == 1
    assert summary.dry_runs[0].step_id == result.remediation_steps[0].step_id
    assert summary.dry_runs[0].executed_against_real_infra is False

    events = audit_logger.read_events()
    assert [ev.event_type for ev in events] == [
        "operator_decision",
        "dry_run_executed",
        "operator_decision",
    ]


def test_interactive_prompt_handler_reprompts_invalid_displays_risk_and_handles_eof(
    tmp_path: Path,
) -> None:
    result = _sample_triage_result("api_5xx_db_pool")
    # Step 3 in api_5xx_db_pool has risk_level == "high"
    assert any(step.risk_level == "high" for step in result.remediation_steps)

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, safe_box=True)
    responses = iter(["invalid-choice", "a", "r: manual block", EOFError()])

    def fake_input(prompt: str) -> str:
        nxt = next(responses)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    handler = InteractivePromptHandler(console=console, input_fn=fake_input)
    audit_logger = AuditLogger(tmp_path / "interactive.jsonl")
    summary = process_remediations(
        result,
        handler=handler,
        audit_logger=audit_logger,
        operator="bob",
    )

    output = buf.getvalue()
    assert "[WARN]" in output
    assert "[HIGH RISK]" in output
    assert "[ABORTED]" in output
    assert summary.aborted is True
    assert [d.decision for d in summary.decisions] == ["approved", "rejected"]
    assert len(summary.dry_runs) == 1


def test_interactive_prompt_handler_ask_reason_prompts_second_line() -> None:
    result = _sample_triage_result("pod_oom_killed")
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, safe_box=True)
    responses = iter(["a", "safe diagnostic", "q"])

    handler = InteractivePromptHandler(
        console=console,
        input_fn=lambda _: next(responses),
        ask_reason=True,
    )
    summary = process_remediations(result, handler=handler)
    assert summary.aborted is True
    assert len(summary.decisions) == 1
    assert summary.decisions[0].reason == "safe diagnostic"


def test_approval_types_protocols_and_parse_scripted_decisions() -> None:
    assert ParsedDecision is PromptDecision
    assert DecisionValue is not None
    assert ActionValue is DecisionAction

    default_summary = ApprovalSummary()
    assert default_summary.run_id == ""
    assert default_summary.events_logged == 0

    assert parse_scripted_decisions("a, reject: risky, skip, q") == [
        "approved",
        "rejected",
        "skipped",
        "quit",
    ]
    with pytest.raises(ApprovalError):
        parse_scripted_decisions("")
    with pytest.raises(ApprovalError):
        parse_scripted_decisions("a,,s")

    assert isinstance(InteractivePromptHandler(), DecisionHandler)
    assert isinstance(ScriptedDecisionHandler("a"), DecisionHandler)


def test_scripted_handler_sequence_underflow_quit_index_and_overflow() -> None:
    result = _sample_triage_result("pod_oom_killed")
    assert len(result.remediation_steps) == 3

    seq_handler = ScriptedDecisionHandler(["approve", "reject: risk"], on_underflow="quit")
    summary = process_remediations(result, handler=seq_handler, operator="alice")
    assert summary.run_id == result.run_id
    assert summary.aborted is True
    assert seq_handler.index == 2
    assert seq_handler.underflow_count == 1
    assert seq_handler.overflow_count == 0
    assert [d.decision for d in summary.decisions] == ["approved", "rejected"]
    assert summary.events_logged == 0  # no audit logger passed

    over_handler = ScriptedDecisionHandler(["a", "s", "r", "a", "s"])
    over_summary = process_remediations(result, handler=over_handler)
    assert over_summary.aborted is False
    assert over_handler.index == 3
    assert over_handler.overflow_count == 2


def test_interactive_rich_panels_and_dry_run_actor_attribution(tmp_path: Path) -> None:
    result = _sample_triage_result("api_5xx_db_pool")
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, safe_box=True)
    responses = iter(["a: safe check", "s", "r: too risky"])

    handler = InteractivePromptHandler(console=console, input_fn=lambda _: next(responses))
    audit_path = tmp_path / "panels.jsonl"
    audit_logger = AuditLogger(audit_path)
    summary = process_remediations(
        result,
        handler=handler,
        audit_logger=audit_logger,
        operator="oncall-alice",
    )

    output = buf.getvalue()
    assert output.count(f"Incident Triage: {result.incident_id}") == 1
    assert "[LOW RISK]" in output
    assert "[MEDIUM RISK]" in output
    assert "[HIGH RISK]" in output
    assert "Evidence" in output
    assert "[DRY-RUN]" in output
    assert summary.run_id == result.run_id
    assert summary.events_logged == 4

    events = audit_logger.read_events()
    dry_run_events = [ev for ev in events if ev.event_type == "dry_run_executed"]
    assert len(dry_run_events) == 1
    assert dry_run_events[0].actor == "oncall-alice"
    assert [ev.event_type for ev in events] == [
        "operator_decision",
        "dry_run_executed",
        "operator_decision",
        "operator_decision",
    ]


def test_interactive_lookahead_and_stdin_stream_with_ask_reason() -> None:
    result = _sample_triage_result("pod_oom_killed")
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, safe_box=True)
    stdin_stream = StringIO("a\nr: high risk\nq\n")

    handler = InteractivePromptHandler(
        console=console,
        stdin=stdin_stream,
        ask_reason=True,
    )
    summary = process_remediations(result, handler=handler, operator="stream-op")
    assert summary.aborted is True
    assert [d.decision for d in summary.decisions] == ["approved", "rejected"]
    assert summary.decisions[0].reason is None
    assert summary.decisions[1].reason == "high risk"
    assert "[ABORTED]" in buf.getvalue()

    # EOF on explicit stdin when empty
    eof_buf = StringIO()
    eof_handler = InteractivePromptHandler(
        console=Console(file=eof_buf, force_terminal=False, no_color=True, safe_box=True),
        stdin=StringIO(""),
    )
    eof_summary = process_remediations(result, handler=eof_handler)
    assert eof_summary.aborted is True
    assert eof_summary.decisions == []


def test_interactive_keyboard_interrupt_on_primary_and_reason_prompts() -> None:
    result = _sample_triage_result("pod_oom_killed")
    buf1 = StringIO()

    def _interrupt_primary(_: str) -> str:
        raise KeyboardInterrupt()

    h1 = InteractivePromptHandler(
        console=Console(file=buf1, force_terminal=False, no_color=True, safe_box=True),
        input_fn=_interrupt_primary,
    )
    s1 = process_remediations(result, handler=h1)
    assert s1.aborted is True
    assert s1.decisions == []
    assert "[ABORTED]" in buf1.getvalue()

    buf2 = StringIO()
    calls = iter(["a", KeyboardInterrupt()])

    def _interrupt_reason(_: str) -> str:
        nxt = next(calls)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    h2 = InteractivePromptHandler(
        console=Console(file=buf2, force_terminal=False, no_color=True, safe_box=True),
        input_fn=_interrupt_reason,
        ask_reason=True,
    )
    s2 = process_remediations(result, handler=h2)
    assert s2.aborted is True
    assert s2.decisions == []
    assert "[ABORTED]" in buf2.getvalue()


def test_approval_workflow_scripted_and_interactive(tmp_path: Path) -> None:
    result = _sample_triage_result("pod_oom_killed")
    audit_logger = AuditLogger(tmp_path / "workflow_audit.jsonl")

    wf_scripted = ApprovalWorkflow(operator="wf-scripted")
    summary_s = wf_scripted.run_scripted(
        [result],
        ["approve", "reject: not now", "skip"],
        audit_logger=audit_logger,
    )
    assert summary_s.aborted is False
    assert len(summary_s.decisions) == 3
    assert len(summary_s.dry_runs) == 1

    out_stream = StringIO()
    wf_interactive = ApprovalWorkflow(
        operator="wf-interactive",
        stdin=StringIO("a\nq\n"),
        stdout=out_stream,
    )
    summary_i = wf_interactive.run_interactive([result], ask_reason=True)
    assert summary_i.aborted is True
    assert len(summary_i.decisions) == 1
    assert "Incident Triage" in out_stream.getvalue()
