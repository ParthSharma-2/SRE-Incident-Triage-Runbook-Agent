"""Canonical immutable Pydantic v2 domain models for sre_triage_agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    """Return the current UTC-aware timestamp."""
    return datetime.now(timezone.utc)


def _ensure_utc(value: datetime) -> datetime:
    """Normalize naive or offset-aware datetimes to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]


class LogEntry(BaseModel):
    """Normalized log entry ingested from bundled or live log files."""

    model_config = ConfigDict(frozen=True)

    log_id: str
    timestamp: UtcDatetime
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
    service: str
    message: str
    source: str
    line_number: int = Field(ge=0)
    attributes: dict[str, str] = Field(default_factory=dict)


class AlertEvent(BaseModel):
    """Normalized alert ingested from Prometheus or Alertmanager payloads."""

    model_config = ConfigDict(frozen=True)

    alert_id: str
    alertname: str
    status: Literal["firing", "resolved", "pending"]
    severity: str
    service: str
    namespace: str | None = None
    instance: str | None = None
    summary: str = ""
    description: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    starts_at: UtcDatetime


class EvidenceRef(BaseModel):
    """Citation linking a remediation step to an ingested alert or log entry."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["alert", "log"]
    ref_id: str
    excerpt: str
    reason: str


class IncidentBundle(BaseModel):
    """Correlated group of alerts and logs representing an operational incident."""

    model_config = ConfigDict(frozen=True)

    incident_id: str
    title: str
    service: str
    severity: str
    scenario_name: str | None = None
    source_mode: Literal["bundled", "live"]
    alerts: list[AlertEvent] = Field(default_factory=list)
    logs: list[LogEntry] = Field(default_factory=list)
    created_at: UtcDatetime = Field(default_factory=utc_now)


class RunbookStepTemplate(BaseModel):
    """Parameterized remediation step inside a YAML runbook template."""

    model_config = ConfigDict(frozen=True)

    step_id: str
    title: str
    description: str
    risk_level: Literal["low", "medium", "high"]
    command_template: str
    target_resource_template: str = "{service}"
    expected_effect: str = ""
    diff_template: str = ""
    preconditions: list[str] = Field(default_factory=list)
    verification_hint: str = ""
    simulated_stdout_template: str = ""


class RunbookTemplate(BaseModel):
    """Runbook definition loaded from YAML templates."""

    model_config = ConfigDict(frozen=True)

    runbook_id: str
    title: str
    description: str
    match_alertnames: list[str] = Field(default_factory=list)
    match_keywords: list[str] = Field(default_factory=list)
    severity_hint: str = "warning"
    steps: list[RunbookStepTemplate] = Field(default_factory=list)


class RemediationStep(BaseModel):
    """Evidence-backed remediation step suggested by the triage engine."""

    model_config = ConfigDict(frozen=True)

    step_id: str
    runbook_id: str
    title: str
    description: str
    risk_level: Literal["low", "medium", "high"]
    command_preview: str
    target_resource: str
    expected_effect: str
    diff_template: str = ""
    evidence_refs: list[EvidenceRef] = Field(..., min_length=1)

    @field_validator("evidence_refs")
    @classmethod
    def _validate_non_empty_evidence(cls, value: list[EvidenceRef]) -> list[EvidenceRef]:
        if not value:
            raise ValueError("evidence_refs must contain at least one EvidenceRef")
        return value


class TriageResult(BaseModel):
    """Structured triage output for an incident bundle."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    run_id: str
    incident_id: str
    scenario_name: str | None = None
    source_mode: Literal["bundled", "live"]
    severity: str
    root_cause_hypothesis: str
    summary: str
    provider_used: str
    model_used: str
    fallback_used: bool
    fallback_reason: str | None = None
    alerts: list[AlertEvent] = Field(default_factory=list)
    logs: list[LogEntry] = Field(default_factory=list)
    remediation_steps: list[RemediationStep] = Field(default_factory=list)
    created_at: UtcDatetime = Field(default_factory=utc_now)


class OperatorDecision(BaseModel):
    """Operator approval, rejection, or skip record for a remediation step."""

    model_config = ConfigDict(frozen=True)

    decision_id: str
    run_id: str
    incident_id: str
    step_id: str
    decision: Literal["approved", "rejected", "skipped"]
    operator: str
    reason: str | None = None
    decided_at: UtcDatetime = Field(default_factory=utc_now)


class DryRunOutcome(BaseModel):
    """Simulated dry-run execution outcome (never touches real infrastructure)."""

    model_config = ConfigDict(frozen=True)

    outcome_id: str
    run_id: str
    incident_id: str
    step_id: str
    command_preview: str
    diff_preview: str
    simulated_stdout: str
    simulated_status: Literal["SUCCESS_DRY_RUN", "SKIPPED", "WOULD_FAIL_PRECHECK"]
    executed_against_real_infra: Literal[False] = False
    executed_at: UtcDatetime = Field(default_factory=utc_now)


class AuditEvent(BaseModel):
    """Append-only audit log event tracking the full incident lifecycle."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    event_id: str
    timestamp: UtcDatetime = Field(default_factory=utc_now)
    run_id: str
    incident_id: str
    event_type: Literal[
        "triage_run_started",
        "triage_completed",
        "remediation_suggested",
        "operator_decision",
        "dry_run_executed",
    ]
    actor: str
    payload: dict[str, Any] = Field(default_factory=dict)
