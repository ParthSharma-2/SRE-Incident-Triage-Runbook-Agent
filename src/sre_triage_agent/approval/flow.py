"""Operator approval flow (interactive Rich prompts and scripted automation) with audit integration."""

from __future__ import annotations

import sys
import uuid
from collections.abc import Callable, Sequence
from typing import Literal, Protocol, TextIO, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sre_triage_agent.audit import AuditLogger
from sre_triage_agent.executor import DryRunExecutor
from sre_triage_agent.models import (
    DryRunOutcome,
    OperatorDecision,
    RemediationStep,
    TriageResult,
    utc_now,
)


class ApprovalError(ValueError):
    """Raised when scripted or interactive approval configuration or input is invalid."""


DecisionValue = Literal["approved", "rejected", "skipped"]
DecisionAction = Literal["approved", "rejected", "skipped", "quit"]
ActionValue = DecisionAction

_DECISION_ALIASES: dict[str, DecisionAction] = {
    "a": "approved",
    "approve": "approved",
    "approved": "approved",
    "r": "rejected",
    "reject": "rejected",
    "rejected": "rejected",
    "s": "skipped",
    "skip": "skipped",
    "skipped": "skipped",
    "q": "quit",
    "quit": "quit",
}


class PromptDecision(BaseModel):
    """Normalized operator decision produced by a DecisionHandler."""

    model_config = ConfigDict(frozen=True)

    action: DecisionAction
    reason: str | None = None


ParsedDecision = PromptDecision


class ApprovalSummary(BaseModel):
    """Aggregated outcome of processing remediation steps across triage results."""

    model_config = ConfigDict(frozen=True)

    run_id: str = ""
    decisions: list[OperatorDecision] = Field(default_factory=list)
    dry_runs: list[DryRunOutcome] = Field(default_factory=list)
    aborted: bool = False
    events_logged: int = 0


@runtime_checkable
class DecisionHandler(Protocol):
    """Protocol for interactive or scripted operator remediation decisions."""

    def decide(
        self,
        step: RemediationStep,
        triage_result: TriageResult,
        *,
        step_index: int = 1,
        total_steps: int = 1,
    ) -> PromptDecision:
        """Return the operator's decision for the given remediation step."""


def parse_decision_input(raw: str) -> PromptDecision:
    """Parse an operator choice string such as 'a', 'reject: risky', or 'skip'."""
    if not isinstance(raw, str):
        raise ApprovalError("Decision input must be a string")
    stripped = raw.strip()
    if not stripped:
        raise ApprovalError(
            "Empty decision input; expected [a]pprove, [r]eject, [s]kip, or [q]uit"
        )

    note: str | None = None
    if ":" in stripped:
        token, _, raw_note = stripped.partition(":")
        choice_key = token.strip().lower()
        note = raw_note.strip() or None
    else:
        parts = stripped.split(None, 1)
        first_key = parts[0].strip().lower()
        if len(parts) == 2 and first_key in _DECISION_ALIASES:
            choice_key = first_key
            note = parts[1].strip() or None
        else:
            choice_key = stripped.lower()

    action = _DECISION_ALIASES.get(choice_key)
    if action is None:
        raise ApprovalError(
            f"Invalid decision '{stripped}'; expected [a]pprove, [r]eject, [s]kip, or [q]uit"
        )
    return PromptDecision(action=action, reason=note)


def parse_scripted_decisions(raw: str) -> list[DecisionAction]:
    """Parse a comma-separated --decisions string into validated action values."""
    if not isinstance(raw, str) or not raw.strip():
        raise ApprovalError("--decisions must not be empty")
    raw_parts = raw.split(",")
    parsed: list[DecisionAction] = []
    for part in raw_parts:
        token = part.strip()
        if not token:
            raise ApprovalError(
                f"Empty decision token in '{raw}'. Expected comma-separated approve/reject/skip/quit"
            )
        parsed.append(parse_decision_input(token).action)
    return parsed


def _risk_badge(risk_level: str) -> Text:
    norm = risk_level.strip().lower()
    if norm == "high":
        return Text(" [HIGH RISK] ", style="bold white on red")
    if norm == "medium":
        return Text(" [MEDIUM RISK] ", style="bold black on yellow")
    return Text(" [LOW RISK] ", style="bold white on green")


class InteractivePromptHandler:
    """Rich interactive terminal approval prompt handler (Windows cp1252 & CliRunner safe)."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        input_fn: Callable[[str], str] | None = None,
        ask_reason: bool = False,
        stderr: bool = False,
        stdin: TextIO | None = None,
    ) -> None:
        self._console = console
        self._input_fn = input_fn
        self.ask_reason = ask_reason
        self._stderr = stderr
        self._stdin = stdin
        self._lookahead: str | None = None
        self._displayed_incidents: set[tuple[str, str]] = set()

    def _get_console(self) -> Console:
        if self._console is not None:
            return self._console
        return Console(stderr=self._stderr, safe_box=True)

    def _read_line(self, console: Console, prompt: str) -> str:
        if self._lookahead is not None:
            buffered = self._lookahead
            self._lookahead = None
            return buffered
        if self._input_fn is not None:
            return self._input_fn(prompt)
        console.print(prompt, end="", highlight=False, markup=False)
        stream = self._stdin if self._stdin is not None else sys.stdin
        line = stream.readline()
        if line == "":
            raise EOFError("EOF on stdin")
        return line.rstrip("\r\n")

    @staticmethod
    def _looks_like_next_decision(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if ":" in stripped:
            head = stripped.split(":", 1)[0].strip().lower()
            return head in _DECISION_ALIASES
        return stripped.lower() in _DECISION_ALIASES

    def begin_incident(self, triage_result: TriageResult) -> None:
        """Render the incident triage summary once per (run_id, incident_id)."""
        key = (triage_result.run_id, triage_result.incident_id)
        if key in self._displayed_incidents:
            return
        self._displayed_incidents.add(key)

        console = self._get_console()
        provider_info = f"{triage_result.provider_used}/{triage_result.model_used}"
        if triage_result.fallback_used:
            provider_info += " [FALLBACK]"

        table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
        )
        table.add_column("Field", style="bold cyan", no_wrap=True)
        table.add_column("Value")
        table.add_row("Run ID", triage_result.run_id)
        table.add_row("Incident ID", triage_result.incident_id)
        table.add_row("Severity", triage_result.severity.upper())
        table.add_row(
            "Source",
            f"{triage_result.source_mode}"
            + (f" ({triage_result.scenario_name})" if triage_result.scenario_name else ""),
        )
        table.add_row("Provider", provider_info)
        if triage_result.fallback_reason:
            table.add_row("Fallback Reason", triage_result.fallback_reason)
        table.add_row(
            "Signals",
            f"{len(triage_result.alerts)} alert(s), {len(triage_result.logs)} log(s), "
            f"{len(triage_result.remediation_steps)} remediation step(s)",
        )
        table.add_row("Root Cause", triage_result.root_cause_hypothesis)
        table.add_row("Summary", triage_result.summary)

        console.print(
            Panel(
                table,
                title=f"Incident Triage: {triage_result.incident_id}",
                border_style="blue",
                safe_box=True,
            )
        )

    def decide(
        self,
        step: RemediationStep,
        triage_result: TriageResult,
        *,
        step_index: int = 1,
        total_steps: int = 1,
    ) -> PromptDecision:
        """Display step details, evidence, and risk badge, then prompt for decision."""
        self.begin_incident(triage_result)
        console = self._get_console()

        header = Text()
        header.append(f"Step {step_index}/{total_steps}: {step.title} ", style="bold")
        header.append_text(_risk_badge(step.risk_level))

        detail = Table(show_header=False, box=None, padding=(0, 1))
        detail.add_column("Field", style="bold", no_wrap=True)
        detail.add_column("Value")
        detail.add_row("Step ID", f"{step.step_id} (runbook: {step.runbook_id})")
        detail.add_row("Risk Level", _risk_badge(step.risk_level))
        detail.add_row("Target", step.target_resource)
        detail.add_row("Description", step.description)
        detail.add_row("Command Preview", step.command_preview)
        detail.add_row("Expected Effect", step.expected_effect)
        if step.diff_template.strip():
            detail.add_row("Planned Diff", step.diff_template.strip())

        evidence_lines = [
            f"- [{ref.kind}] {ref.ref_id}: {ref.excerpt} ({ref.reason})"
            for ref in step.evidence_refs
        ]
        detail.add_row("Evidence", "\n".join(evidence_lines))

        border = "red" if step.risk_level == "high" else "yellow" if step.risk_level == "medium" else "cyan"
        console.print(
            Panel(
                detail,
                title=header,
                border_style=border,
                safe_box=True,
            )
        )

        prompt_text = "Decision [a]pprove / [r]eject / [s]kip / [q]uit (optional ': reason'): "
        while True:
            try:
                raw = self._read_line(console, prompt_text)
            except (EOFError, KeyboardInterrupt):
                console.print(
                    "[ABORTED] Operator input ended or interrupted; stopping remediation review."
                )
                return PromptDecision(action="quit", reason="EOF or operator interrupt")

            try:
                parsed = parse_decision_input(raw)
            except ApprovalError as exc:
                console.print(f"[WARN] {exc}")
                continue

            if parsed.action == "quit":
                console.print("[ABORTED] Operator quit approval flow.")
                return parsed

            if (
                self.ask_reason
                and parsed.action in {"approved", "rejected", "skipped"}
                and parsed.reason is None
            ):
                try:
                    note_raw = self._read_line(
                        console,
                        "Optional reason/note (press Enter to skip): ",
                    )
                except EOFError:
                    return parsed
                except KeyboardInterrupt:
                    console.print(
                        "[ABORTED] Operator interrupted reason prompt; stopping remediation review."
                    )
                    return PromptDecision(action="quit", reason="Operator interrupt")

                if self._looks_like_next_decision(note_raw):
                    self._lookahead = note_raw
                    return PromptDecision(action=parsed.action, reason=None)

                note = note_raw.strip() or None
                return PromptDecision(action=parsed.action, reason=note)

            return parsed

    def show_outcome(self, outcome: DryRunOutcome) -> None:
        """Render the simulated dry-run outcome cleanly using ASCII status tags."""
        console = self._get_console()
        status_tag = (
            "[OK] SUCCESS_DRY_RUN"
            if outcome.simulated_status == "SUCCESS_DRY_RUN"
            else f"[WARN] {outcome.simulated_status}"
        )
        body = (
            f"Outcome ID: {outcome.outcome_id}\n"
            f"Status: {status_tag}\n"
            f"Executed Against Real Infra: {outcome.executed_against_real_infra}\n"
            f"Command: {outcome.command_preview}\n"
            f"Diff:\n{outcome.diff_preview}\n"
            f"Simulated Stdout:\n{outcome.simulated_stdout}"
        )
        border = "green" if outcome.simulated_status == "SUCCESS_DRY_RUN" else "yellow"
        console.print(
            Panel(
                body,
                title=f"[DRY-RUN] Step {outcome.step_id}",
                border_style=border,
                safe_box=True,
            )
        )


class ScriptedDecisionHandler:
    """Deterministic scripted decision handler for CLI `--decisions` and automated tests."""

    def __init__(
        self,
        decisions: Sequence[str] | str,
        *,
        on_underflow: Literal["skip", "quit", "error"] = "skip",
    ) -> None:
        if isinstance(decisions, str):
            raw_tokens = [tok.strip() for tok in decisions.split(",")]
            if not decisions.strip() or any(not tok for tok in raw_tokens):
                raise ApprovalError(
                    "Scripted decisions string must be a non-empty comma-separated list"
                )
        else:
            raw_tokens = [str(tok).strip() for tok in decisions]
            if not raw_tokens or any(not tok for tok in raw_tokens):
                raise ApprovalError("Scripted decisions sequence must not be empty")

        self._decisions: list[PromptDecision] = [
            parse_decision_input(tok) for tok in raw_tokens
        ]
        self.on_underflow = on_underflow
        self._index: int = 0
        self.underflow_count: int = 0

    @property
    def index(self) -> int:
        return self._index

    @property
    def remaining(self) -> int:
        return max(0, len(self._decisions) - self._index)

    @property
    def overflow_count(self) -> int:
        return self.remaining

    def decide(
        self,
        step: RemediationStep,
        triage_result: TriageResult,
        *,
        step_index: int = 1,
        total_steps: int = 1,
    ) -> PromptDecision:
        if self._index < len(self._decisions):
            choice = self._decisions[self._index]
            self._index += 1
            return choice

        self.underflow_count += 1
        if self.on_underflow == "skip":
            return PromptDecision(
                action="skipped",
                reason="Scripted decisions exhausted (auto-skipped)",
            )
        if self.on_underflow == "quit":
            return PromptDecision(
                action="quit",
                reason="Scripted decisions exhausted",
            )
        raise ApprovalError(
            f"Scripted decisions exhausted at step '{step.step_id}' "
            f"(provided {len(self._decisions)})"
        )


def process_remediations(
    triage_results: TriageResult | Sequence[TriageResult],
    *,
    handler: DecisionHandler,
    executor: DryRunExecutor | None = None,
    audit_logger: AuditLogger | None = None,
    operator: str = "sre-operator",
) -> ApprovalSummary:
    """Process operator approvals, trigger DryRunExecutor on approvals, and log audit events."""
    results_list = (
        [triage_results]
        if isinstance(triage_results, TriageResult)
        else list(triage_results)
    )
    run_id = results_list[0].run_id if results_list else f"run-{uuid.uuid4().hex[:12]}"
    effective_executor = executor if executor is not None else DryRunExecutor()

    decisions: list[OperatorDecision] = []
    dry_runs: list[DryRunOutcome] = []
    aborted = False
    events_logged = 0

    for result in results_list:
        if hasattr(handler, "begin_incident"):
            handler.begin_incident(result)

        total_steps = len(result.remediation_steps)
        for idx, step in enumerate(result.remediation_steps, start=1):
            choice = handler.decide(
                step,
                result,
                step_index=idx,
                total_steps=total_steps,
            )
            if choice.action == "quit":
                aborted = True
                break

            decision_record = OperatorDecision(
                decision_id=f"dec-{uuid.uuid4().hex[:12]}",
                run_id=result.run_id,
                incident_id=result.incident_id,
                step_id=step.step_id,
                decision=choice.action,
                operator=operator,
                reason=choice.reason,
                decided_at=utc_now(),
            )
            decisions.append(decision_record)
            if audit_logger is not None:
                audit_logger.log_operator_decision(decision_record)
                events_logged += 1

            if choice.action == "approved":
                outcome = effective_executor.simulate(step, result)
                dry_runs.append(outcome)
                if audit_logger is not None:
                    audit_logger.log_dry_run_executed(outcome, actor=operator)
                    events_logged += 1
                if hasattr(handler, "show_outcome"):
                    handler.show_outcome(outcome)

        if aborted:
            break

    return ApprovalSummary(
        run_id=run_id,
        decisions=decisions,
        dry_runs=dry_runs,
        aborted=aborted,
        events_logged=events_logged,
    )


class ApprovalWorkflow:
    """High-level wrapper for scripted or interactive approval flows."""

    def __init__(
        self,
        *,
        operator: str = "sre-operator",
        executor: DryRunExecutor | None = None,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.operator = operator
        self.executor = executor if executor is not None else DryRunExecutor()
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout

    def run_scripted(
        self,
        results: TriageResult | Sequence[TriageResult],
        decisions: Sequence[str] | str,
        *,
        audit_logger: AuditLogger | None = None,
        on_underflow: Literal["skip", "quit", "error"] = "skip",
    ) -> ApprovalSummary:
        handler = ScriptedDecisionHandler(decisions, on_underflow=on_underflow)
        return process_remediations(
            results,
            handler=handler,
            executor=self.executor,
            audit_logger=audit_logger,
            operator=self.operator,
        )

    def run_interactive(
        self,
        results: TriageResult | Sequence[TriageResult],
        *,
        audit_logger: AuditLogger | None = None,
        ask_reason: bool = False,
    ) -> ApprovalSummary:
        console = Console(
            file=self.stdout,
            force_terminal=False,
            no_color=True,
            safe_box=True,
        )
        handler = InteractivePromptHandler(
            console=console,
            ask_reason=ask_reason,
            stdin=self.stdin,
        )
        return process_remediations(
            results,
            handler=handler,
            executor=self.executor,
            audit_logger=audit_logger,
            operator=self.operator,
        )
