"""Thread-safe, append-only UTF-8 LF JSONL audit log store."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sre_triage_agent.models import (
    AuditEvent,
    DryRunOutcome,
    OperatorDecision,
    RemediationStep,
    TriageResult,
    utc_now,
)


class AuditError(OSError):
    """Raised when reading or writing the persistent audit log fails."""


@dataclass(frozen=True)
class CorruptLine:
    """Diagnostic record for a malformed or non-UTF8 line encountered in JSONL."""

    line_number: int
    raw_line: str
    error: str


@dataclass(frozen=True)
class AuditReadResult:
    """Result of reading an audit JSONL file with corrupt-line diagnostics."""

    events: list[AuditEvent]
    corrupt_lines: list[CorruptLine]


def _new_event_id() -> str:
    return f"evt-{uuid.uuid4().hex[:12]}"


class AuditLogger:
    """Thread-safe, append-only UTF-8 LF JSONL audit log reader and writer."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.last_corrupt_lines: list[CorruptLine] = []

    def log_event(self, event: AuditEvent) -> AuditEvent:
        """Append a single AuditEvent as a UTF-8 LF JSONL line."""
        self.append_many([event])
        return event

    def append(self, event: AuditEvent) -> AuditEvent:
        """Alias for log_event."""
        return self.log_event(event)

    def append_many(self, events: Sequence[AuditEvent]) -> None:
        """Append multiple AuditEvent records atomically under a lock using LF endings."""
        if not events:
            return
        if self.path.exists() and self.path.is_dir():
            raise AuditError(f"Audit log path is a directory: {self.path}")
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                needs_leading_newline = False
                if self.path.exists() and self.path.stat().st_size > 0:
                    with self.path.open("rb") as rf:
                        rf.seek(-1, 2)
                        needs_leading_newline = rf.read(1) != b"\n"

                with self.path.open("ab") as fh:
                    if needs_leading_newline:
                        fh.write(b"\n")
                    for event in events:
                        fh.write(event.model_dump_json().encode("utf-8"))
                        fh.write(b"\n")
                    fh.flush()
        except OSError as exc:
            raise AuditError(f"Failed to write audit log '{self.path}': {exc}") from exc

    def log_triage_run_started(
        self,
        *,
        run_id: str,
        incident_id: str,
        actor: str = "sre-triage-agent",
        payload: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Record a triage_run_started AuditEvent."""
        return self.log_event(
            AuditEvent(
                event_id=_new_event_id(),
                timestamp=utc_now(),
                run_id=run_id,
                incident_id=incident_id,
                event_type="triage_run_started",
                actor=actor,
                payload=dict(payload or {}),
            )
        )

    def log_triage_completed(
        self,
        result: TriageResult,
        *,
        actor: str = "sre-triage-agent",
    ) -> AuditEvent:
        """Record a triage_completed AuditEvent for a TriageResult."""
        return self.log_event(
            AuditEvent(
                event_id=_new_event_id(),
                timestamp=utc_now(),
                run_id=result.run_id,
                incident_id=result.incident_id,
                event_type="triage_completed",
                actor=actor,
                payload={
                    "scenario_name": result.scenario_name,
                    "source_mode": result.source_mode,
                    "severity": result.severity,
                    "provider_used": result.provider_used,
                    "model_used": result.model_used,
                    "fallback_used": result.fallback_used,
                    "fallback_reason": result.fallback_reason,
                    "summary": result.summary,
                    "root_cause_hypothesis": result.root_cause_hypothesis,
                },
            )
        )

    def log_remediation_suggested(
        self,
        result: TriageResult,
        step: RemediationStep,
        *,
        actor: str = "sre-triage-agent",
    ) -> AuditEvent:
        """Record a remediation_suggested AuditEvent for a RemediationStep."""
        return self.log_event(
            AuditEvent(
                event_id=_new_event_id(),
                timestamp=utc_now(),
                run_id=result.run_id,
                incident_id=result.incident_id,
                event_type="remediation_suggested",
                actor=actor,
                payload={
                    "step_id": step.step_id,
                    "runbook_id": step.runbook_id,
                    "title": step.title,
                    "risk_level": step.risk_level,
                    "command_preview": step.command_preview,
                    "target_resource": step.target_resource,
                    "evidence_refs": [ref.model_dump(mode="json") for ref in step.evidence_refs],
                },
            )
        )

    def log_operator_decision(self, decision: OperatorDecision) -> AuditEvent:
        """Record an operator_decision AuditEvent."""
        return self.log_event(
            AuditEvent(
                event_id=_new_event_id(),
                timestamp=decision.decided_at,
                run_id=decision.run_id,
                incident_id=decision.incident_id,
                event_type="operator_decision",
                actor=decision.operator,
                payload={
                    "decision_id": decision.decision_id,
                    "step_id": decision.step_id,
                    "decision": decision.decision,
                    "operator": decision.operator,
                    "reason": decision.reason,
                },
            )
        )

    def log_dry_run_executed(
        self,
        outcome: DryRunOutcome,
        *,
        actor: str = "dry-run-executor",
    ) -> AuditEvent:
        """Record a dry_run_executed AuditEvent."""
        return self.log_event(
            AuditEvent(
                event_id=_new_event_id(),
                timestamp=outcome.executed_at,
                run_id=outcome.run_id,
                incident_id=outcome.incident_id,
                event_type="dry_run_executed",
                actor=actor,
                payload={
                    "outcome_id": outcome.outcome_id,
                    "step_id": outcome.step_id,
                    "command_preview": outcome.command_preview,
                    "diff_preview": outcome.diff_preview,
                    "simulated_stdout": outcome.simulated_stdout,
                    "simulated_status": outcome.simulated_status,
                    "executed_against_real_infra": outcome.executed_against_real_infra,
                },
            )
        )

    def read_all_with_diagnostics(self) -> AuditReadResult:
        """Read all valid AuditEvents while recording corrupt or non-UTF8 lines."""
        with self._lock:
            if not self.path.exists():
                self.last_corrupt_lines = []
                return AuditReadResult(events=[], corrupt_lines=[])
            if self.path.is_dir():
                raise AuditError(f"Audit log path is a directory: {self.path}")
            try:
                raw_bytes = self.path.read_bytes()
            except OSError as exc:
                raise AuditError(f"Failed to read audit log '{self.path}': {exc}") from exc

        events: list[AuditEvent] = []
        corrupt: list[CorruptLine] = []

        for line_number, raw_line_bytes in enumerate(raw_bytes.splitlines(), start=1):
            if not raw_line_bytes.strip():
                continue
            try:
                text = raw_line_bytes.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                corrupt.append(
                    CorruptLine(
                        line_number=line_number,
                        raw_line=raw_line_bytes.decode("utf-8", errors="replace"),
                        error=f"UTF-8 decode error: {exc}",
                    )
                )
                continue
            try:
                events.append(AuditEvent.model_validate_json(text))
            except (ValidationError, ValueError) as exc:
                corrupt.append(
                    CorruptLine(
                        line_number=line_number,
                        raw_line=text,
                        error=str(exc),
                    )
                )

        self.last_corrupt_lines = corrupt
        return AuditReadResult(events=events, corrupt_lines=corrupt)

    def read_events(
        self,
        *,
        incident_id: str | None = None,
        run_id: str | None = None,
        event_type: str | None = None,
        decision: str | None = None,
    ) -> list[AuditEvent]:
        """Read and filter AuditEvents with logical AND across provided filters."""
        events = self.read_all_with_diagnostics().events
        return self._filter_events(
            events,
            incident_id=incident_id,
            run_id=run_id,
            event_type=event_type,
            decision=decision,
        )

    def query(
        self,
        *,
        incident_id: str | None = None,
        run_id: str | None = None,
        event_type: str | None = None,
        decision: str | None = None,
    ) -> list[AuditEvent]:
        """Query AuditEvents with logical AND across provided filters."""
        return self.read_events(
            incident_id=incident_id,
            run_id=run_id,
            event_type=event_type,
            decision=decision,
        )

    def show_incident(self, incident_id: str) -> list[AuditEvent]:
        """Return all events for an incident sorted chronologically by timestamp."""
        return sorted(self.read_events(incident_id=incident_id), key=lambda ev: ev.timestamp)

    @staticmethod
    def _filter_events(
        events: Sequence[AuditEvent],
        *,
        incident_id: str | None = None,
        run_id: str | None = None,
        event_type: str | None = None,
        decision: str | None = None,
    ) -> list[AuditEvent]:
        filtered: list[AuditEvent] = []
        for event in events:
            if incident_id is not None and event.incident_id != incident_id:
                continue
            if run_id is not None and event.run_id != run_id:
                continue
            if event_type is not None and event.event_type != event_type:
                continue
            if decision is not None and event.payload.get("decision") != decision:
                continue
            filtered.append(event)
        return filtered


AuditStore = AuditLogger
