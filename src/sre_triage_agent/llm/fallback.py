"""Deterministic rule-based fallback, testable mock provider, resilient wrapper, and factory."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import httpx

from sre_triage_agent.config import LLMConfig
from sre_triage_agent.llm.base import (
    LLMConfigError,
    LLMProvider,
    LLMProviderError,
    LLMTriageResponse,
    ProviderHealth,
    extract_json_object,
    normalize_severity,
    validate_and_ground_response,
    validate_evidence_grounding,
)
from sre_triage_agent.llm.ollama import OllamaProvider
from sre_triage_agent.llm.openai_compat import OpenAICompatibleProvider
from sre_triage_agent.models import (
    AlertEvent,
    EvidenceRef,
    IncidentBundle,
    LogEntry,
    RemediationStep,
    RunbookTemplate,
)
from sre_triage_agent.runbooks.catalog import (
    _keyword_matches,
    extract_incident_context,
    render_template,
)

_ALERT_SEVERITY_RANK: dict[str, int] = {
    "critical": 5,
    "high": 4,
    "error": 4,
    "warning": 3,
    "warn": 3,
    "medium": 2,
    "info": 1,
    "low": 0,
}

_LOG_LEVEL_RANK: dict[str, int] = {
    "CRITICAL": 4,
    "ERROR": 3,
    "WARN": 2,
    "INFO": 1,
    "DEBUG": 0,
}


def _rank_alerts(alerts: Sequence[AlertEvent]) -> list[AlertEvent]:
    return sorted(
        alerts,
        key=lambda a: (
            -_ALERT_SEVERITY_RANK.get(a.severity.strip().lower(), 1),
            a.starts_at,
            a.alert_id,
        ),
    )


def _rank_logs(logs: Sequence[LogEntry]) -> list[LogEntry]:
    return sorted(
        logs,
        key=lambda entry: (
            -_LOG_LEVEL_RANK.get(entry.level, 0),
            entry.timestamp,
            entry.line_number,
            entry.log_id,
        ),
    )


def _alert_to_evidence(alert: AlertEvent) -> EvidenceRef:
    excerpt = (
        alert.summary.strip()
        or alert.description.strip()
        or f"{alert.alertname} ({alert.severity}) on {alert.service}"
    )[:240]
    reason = f"Alert {alert.alertname} ({alert.severity}, status={alert.status}) on {alert.service}"
    return EvidenceRef(kind="alert", ref_id=alert.alert_id, excerpt=excerpt, reason=reason)


def _log_to_evidence(entry: LogEntry) -> EvidenceRef:
    excerpt = f"[{entry.level}] {entry.message.strip()}"[:240]
    reason = f"Log line {entry.line_number} from {entry.source} ({entry.level})"
    return EvidenceRef(kind="log", ref_id=entry.log_id, excerpt=excerpt, reason=reason)


def _select_step_evidence(
    ranked_alerts: Sequence[AlertEvent],
    ranked_logs: Sequence[LogEntry],
    runbook: RunbookTemplate | None = None,
) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []

    if runbook is not None:
        alert_patterns = {name.strip().lower() for name in runbook.match_alertnames if name.strip()}
        kw_patterns = [kw.strip().lower() for kw in runbook.match_keywords if kw.strip()]
        matching_alerts = [
            alert for alert in ranked_alerts if alert.alertname.strip().lower() in alert_patterns
        ]
        chosen_alerts = (matching_alerts or list(ranked_alerts))[:2]

        matching_logs = [
            entry
            for entry in ranked_logs
            if any(
                _keyword_matches(
                    kw,
                    f"{entry.message} {' '.join(entry.attributes.values())}".lower(),
                )
                for kw in kw_patterns
            )
        ]
        chosen_logs = (matching_logs or list(ranked_logs))[:2]
    else:
        chosen_alerts = list(ranked_alerts)[:2]
        chosen_logs = list(ranked_logs)[:2]

    for alert in chosen_alerts:
        refs.append(_alert_to_evidence(alert))
    for entry in chosen_logs:
        refs.append(_log_to_evidence(entry))

    if not refs and ranked_alerts:
        refs.append(_alert_to_evidence(ranked_alerts[0]))
    if not refs and ranked_logs:
        refs.append(_log_to_evidence(ranked_logs[0]))
    return refs


class DeterministicFallbackProvider:
    """Offline deterministic rule-based triage provider backed by matched runbooks."""

    provider_name: str = "deterministic"
    model_name: str = "rule-based"

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(
            provider=self.provider_name,
            model=self.model_name,
            base_url=None,
            healthy=True,
            model_available=True,
            detail="Deterministic offline rule-based triage provider is always ready",
            latency_ms=0.0,
        )

    def triage(
        self,
        bundle: IncidentBundle,
        runbooks: Sequence[RunbookTemplate],
    ) -> LLMTriageResponse:
        severity = normalize_severity(bundle.severity, "warning")
        if not bundle.alerts and not bundle.logs:
            return LLMTriageResponse(
                severity=severity,
                root_cause_hypothesis="Insufficient alert and log evidence to determine root cause.",
                summary=(
                    f"Incident {bundle.incident_id} for {bundle.service} has no alerts or log entries."
                ),
                remediation_steps=[],
                provider_used=self.provider_name,
                model_used=self.model_name,
                fallback_used=False,
                fallback_reason=None,
            )

        ctx = extract_incident_context(bundle)
        ranked_alerts = _rank_alerts(bundle.alerts)
        ranked_logs = _rank_logs(bundle.logs)

        steps: list[RemediationStep] = []
        runbooks_with_steps = [rb for rb in runbooks if rb.steps]
        if runbooks_with_steps:
            primary_rb = runbooks_with_steps[0]
            for rb in runbooks_with_steps[:2]:
                step_refs = _select_step_evidence(ranked_alerts, ranked_logs, rb)
                for step_tmpl in rb.steps:
                    if len(steps) >= 5:
                        break
                    steps.append(
                        RemediationStep(
                            step_id=step_tmpl.step_id,
                            runbook_id=rb.runbook_id,
                            title=render_template(step_tmpl.title, ctx),
                            description=render_template(step_tmpl.description, ctx),
                            risk_level=step_tmpl.risk_level,
                            command_preview=render_template(step_tmpl.command_template, ctx),
                            target_resource=render_template(
                                step_tmpl.target_resource_template, ctx
                            ),
                            expected_effect=render_template(step_tmpl.expected_effect, ctx),
                            diff_template=render_template(step_tmpl.diff_template, ctx),
                            evidence_refs=step_refs,
                        )
                    )
                if len(steps) >= 5:
                    break

            top_alert_note = (
                f" Triggered by {ranked_alerts[0].alertname}." if ranked_alerts else ""
            )
            top_log_note = (
                f" Key log signal: {ranked_logs[0].message.strip()}" if ranked_logs else ""
            )
            hypothesis = (
                f"{render_template(primary_rb.title, ctx)}: "
                f"{render_template(primary_rb.description, ctx)}{top_alert_note}{top_log_note}"
            )
            summary = (
                f"Matched runbook '{primary_rb.runbook_id}' for {ctx['service']} "
                f"across {len(bundle.alerts)} alert(s) and {len(bundle.logs)} log(s); "
                f"generated {len(steps)} evidence-linked remediation step(s)."
            )
        else:
            generic_refs = _select_step_evidence(ranked_alerts, ranked_logs, None)
            steps = [
                RemediationStep(
                    step_id="generic-inspect-service-status",
                    runbook_id="generic_triage",
                    title=f"Inspect {ctx['service']} workload status and recent logs",
                    description=(
                        f"Check pod health and recent error logs for {ctx['service']} in "
                        f"namespace {ctx['namespace']}."
                    ),
                    risk_level="low",
                    command_preview=(
                        f"kubectl get pods -n {ctx['namespace']} -l app={ctx['service']} && "
                        f"kubectl logs -n {ctx['namespace']} -l app={ctx['service']} --tail=100"
                    ),
                    target_resource=f"service/{ctx['service']}",
                    expected_effect=(
                        f"Collects live status and recent error logs for {ctx['service']} without "
                        "modifying infrastructure."
                    ),
                    diff_template="",
                    evidence_refs=generic_refs,
                ),
                RemediationStep(
                    step_id="generic-verify-host-events",
                    runbook_id="generic_triage",
                    title=f"Inspect recent events and resource health on {ctx['node']}",
                    description=(
                        f"Describe pod {ctx['pod']} and list recent events in namespace "
                        f"{ctx['namespace']}."
                    ),
                    risk_level="low",
                    command_preview=(
                        f"kubectl describe pod {ctx['pod']} -n {ctx['namespace']} && "
                        f"kubectl get events -n {ctx['namespace']} --sort-by=.lastTimestamp | tail -n 20"
                    ),
                    target_resource=f"pod/{ctx['pod']}",
                    expected_effect=(
                        f"Surfaces scheduling, probe, or dependency events for {ctx['service']}."
                    ),
                    diff_template="",
                    evidence_refs=generic_refs,
                ),
            ]
            top_signal = (
                ranked_alerts[0].summary or ranked_alerts[0].alertname
                if ranked_alerts
                else ranked_logs[0].message.strip()
            )
            hypothesis = (
                f"Operational anomaly detected on {ctx['service']} ({severity}): {top_signal}"
            )
            if runbooks:
                summary = (
                    f"Matched runbook(s) for {ctx['service']} contained no remediation steps; "
                    f"generated {len(steps)} safe diagnostic step(s) linked to observed evidence."
                )
            else:
                summary = (
                    f"No specialized runbook matched {ctx['service']}; generated {len(steps)} "
                    "safe diagnostic step(s) linked to observed evidence."
                )

        return LLMTriageResponse(
            severity=severity,
            root_cause_hypothesis=hypothesis,
            summary=summary,
            remediation_steps=steps,
            provider_used=self.provider_name,
            model_used=self.model_name,
            fallback_used=False,
            fallback_reason=None,
        )


class MockLLMProvider:
    """Testable in-memory LLMProvider supporting scripted responses, JSON strings, handlers, or errors."""

    def __init__(
        self,
        *,
        response: LLMTriageResponse | dict[str, Any] | str | None = None,
        handler: Callable[
            [IncidentBundle, Sequence[RunbookTemplate]],
            LLMTriageResponse | dict[str, Any] | str,
        ]
        | None = None,
        error: Exception | None = None,
        provider_name: str = "mock",
        model_name: str = "mock-model",
        healthy: bool = True,
    ) -> None:
        self.response = response
        self.handler = handler
        self.error = error
        self.provider_name = provider_name
        self.model_name = model_name
        self.healthy = healthy
        self.calls: list[tuple[IncidentBundle, list[RunbookTemplate]]] = []

    def triage(
        self,
        bundle: IncidentBundle,
        runbooks: Sequence[RunbookTemplate],
    ) -> LLMTriageResponse:
        self.calls.append((bundle, list(runbooks)))

        if self.error is not None:
            raise self.error

        raw: LLMTriageResponse | dict[str, Any] | str | None
        if self.handler is not None:
            raw = self.handler(bundle, runbooks)
        else:
            raw = self.response

        if raw is None:
            base = DeterministicFallbackProvider().triage(bundle, runbooks)
            return base.model_copy(
                update={
                    "provider_used": self.provider_name,
                    "model_used": self.model_name,
                    "fallback_used": False,
                    "fallback_reason": None,
                }
            )

        if isinstance(raw, LLMTriageResponse):
            return raw

        if isinstance(raw, str):
            parsed = extract_json_object(raw, provider=self.provider_name, model=self.model_name)
            return validate_and_ground_response(
                parsed,
                bundle,
                provider=self.provider_name,
                model=self.model_name,
            )

        if isinstance(raw, dict):
            return validate_and_ground_response(
                raw,
                bundle,
                provider=self.provider_name,
                model=self.model_name,
            )

        raise TypeError(f"Unsupported MockLLMProvider response type: {type(raw).__name__}")

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(
            provider=self.provider_name,
            model=self.model_name,
            base_url=None,
            healthy=self.healthy,
            model_available=self.healthy,
            detail="Mock LLM provider healthy" if self.healthy else "Mock LLM provider unhealthy",
            latency_ms=0.0,
        )


class ResilientLLMProvider:
    """Wrapper that optionally falls back to DeterministicFallbackProvider on LLMProviderError."""

    def __init__(
        self,
        primary: LLMProvider,
        *,
        fallback: LLMProvider | None = None,
        allow_fallback: bool = True,
    ) -> None:
        self.primary = primary
        self.fallback = fallback if fallback is not None else DeterministicFallbackProvider()
        self.allow_fallback = allow_fallback
        self.provider_name = primary.provider_name
        self.model_name = primary.model_name

    def triage(
        self,
        bundle: IncidentBundle,
        runbooks: Sequence[RunbookTemplate],
    ) -> LLMTriageResponse:
        try:
            response = self.primary.triage(bundle, runbooks)
            validated_steps = validate_evidence_grounding(
                response.remediation_steps,
                bundle,
                provider=response.provider_used,
                model=response.model_used,
            )
            if validated_steps != response.remediation_steps:
                return response.model_copy(update={"remediation_steps": validated_steps})
            return response
        except LLMProviderError as exc:
            if not self.allow_fallback:
                raise
            fallback_resp = self.fallback.triage(bundle, runbooks)
            validated_fallback_steps = validate_evidence_grounding(
                fallback_resp.remediation_steps,
                bundle,
                provider=fallback_resp.provider_used,
                model=fallback_resp.model_used,
            )
            return fallback_resp.model_copy(
                update={
                    "remediation_steps": validated_fallback_steps,
                    "fallback_used": True,
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                }
            )

    def healthcheck(self) -> ProviderHealth:
        return self.primary.healthcheck()


def build_llm_provider(
    config: LLMConfig,
    *,
    client: httpx.Client | None = None,
    transport: httpx.BaseTransport | None = None,
    wrap_resilient: bool = True,
) -> LLMProvider:
    """Construct the configured open-source or deterministic LLMProvider from LLMConfig."""
    if config.provider == "deterministic":
        return DeterministicFallbackProvider()

    primary: LLMProvider
    if config.provider == "ollama":
        primary = OllamaProvider(
            base_url=config.base_url,
            model=config.model,
            timeout_s=config.timeout_seconds,
            client=client,
            transport=transport,
        )
    elif config.provider == "openai_compatible":
        primary = OpenAICompatibleProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout_s=config.timeout_seconds,
            client=client,
            transport=transport,
        )
    else:
        raise LLMConfigError(
            f"Unsupported LLM provider: {config.provider!r}",
            provider=str(config.provider),
        )

    if wrap_resilient:
        return ResilientLLMProvider(
            primary,
            fallback=DeterministicFallbackProvider(),
            allow_fallback=config.allow_fallback,
        )
    return primary
