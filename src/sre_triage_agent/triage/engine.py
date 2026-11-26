"""End-to-end TriageEngine orchestrating ingestion, runbook matching, LLM reasoning, and grounding."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

import httpx

from sre_triage_agent.config import AgentConfig, ConfigError, load_config
from sre_triage_agent.ingestion import (
    AlertSource,
    FileLogSource,
    LogSource,
    PrometheusAlertSource,
    correlate_incidents,
    list_bundled_scenarios,
    load_bundled_scenario,
)
from sre_triage_agent.ingestion.alerts import AlertApiType
from sre_triage_agent.llm import (
    DeterministicFallbackProvider,
    LLMProvider,
    LLMProviderError,
    LLMTriageResponse,
    build_llm_provider,
    validate_evidence_grounding,
)
from sre_triage_agent.models import (
    AlertEvent,
    IncidentBundle,
    LogEntry,
    TriageResult,
    utc_now,
)
from sre_triage_agent.runbooks import RunbookCatalog


class TriageError(Exception):
    """Raised when triage orchestration fails outside config/ingestion/runbook errors."""


def _new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


class TriageEngine:
    """Orchestrates bundled or live incident ingestion, runbook matching, LLM triage, and grounding."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        provider: LLMProvider | None = None,
        catalog: RunbookCatalog | None = None,
    ) -> None:
        self.config = config if config is not None else load_config()
        self.catalog = catalog if catalog is not None else RunbookCatalog.load_default()
        self.provider = provider if provider is not None else build_llm_provider(self.config.llm)

    def triage_bundle(
        self,
        incident: IncidentBundle,
        *,
        run_id: str | None = None,
    ) -> TriageResult:
        """Triage a single correlated IncidentBundle and verify strict evidence grounding."""
        effective_run_id = run_id or _new_run_id()
        candidates = self.catalog.match_candidates(incident)

        response: LLMTriageResponse
        try:
            response = self.provider.triage(incident, candidates)
            validated_steps = validate_evidence_grounding(
                response.remediation_steps,
                incident,
                provider=response.provider_used,
                model=response.model_used,
            )
        except LLMProviderError as exc:
            if (
                not self.config.llm.allow_fallback
                or isinstance(self.provider, DeterministicFallbackProvider)
            ):
                raise
            fallback_provider = DeterministicFallbackProvider()
            fallback_resp = fallback_provider.triage(incident, candidates)
            validated_steps = validate_evidence_grounding(
                fallback_resp.remediation_steps,
                incident,
                provider=fallback_resp.provider_used,
                model=fallback_resp.model_used,
            )
            response = fallback_resp.model_copy(
                update={
                    "fallback_used": True,
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                }
            )

        normalized_severity = (
            response.severity.strip().lower()
            or incident.severity.strip().lower()
            or "warning"
        )

        return TriageResult(
            run_id=effective_run_id,
            incident_id=incident.incident_id,
            scenario_name=incident.scenario_name,
            source_mode=incident.source_mode,
            severity=normalized_severity,
            root_cause_hypothesis=response.root_cause_hypothesis,
            summary=response.summary,
            provider_used=response.provider_used,
            model_used=response.model_used,
            fallback_used=response.fallback_used,
            fallback_reason=response.fallback_reason,
            alerts=list(incident.alerts),
            logs=list(incident.logs),
            remediation_steps=validated_steps,
            created_at=utc_now(),
        )

    def triage_scenario(
        self,
        scenario_name: str,
        *,
        include_resolved: bool | None = None,
        scenarios_dir: Path | None = None,
        run_id: str | None = None,
    ) -> TriageResult:
        """Load and triage a single bundled scenario by name."""
        eff_include_resolved = (
            self.config.ingestion.include_resolved
            if include_resolved is None
            else include_resolved
        )
        bundle = load_bundled_scenario(
            scenario_name,
            include_resolved=eff_include_resolved,
            scenarios_dir=scenarios_dir,
        )
        return self.triage_bundle(bundle, run_id=run_id)

    def triage_all_scenarios(
        self,
        *,
        include_resolved: bool | None = None,
        scenarios_dir: Path | None = None,
        run_id: str | None = None,
    ) -> list[TriageResult]:
        """Load and triage all bundled scenarios under a shared batch run_id."""
        batch_run_id = run_id or _new_run_id()
        scenarios = list_bundled_scenarios(scenarios_dir=scenarios_dir)
        return [
            self.triage_scenario(
                info.name,
                include_resolved=include_resolved,
                scenarios_dir=scenarios_dir,
                run_id=batch_run_id,
            )
            for info in scenarios
        ]

    def triage_live(
        self,
        *,
        log_files: Sequence[str | Path] | None = None,
        alert_url: str | None = None,
        alert_api_type: AlertApiType | None = None,
        tail_lines: int | None = None,
        include_resolved: bool | None = None,
        time_window_seconds: int = 900,
        run_id: str | None = None,
        alert_headers: dict[str, str] | None = None,
        alert_auth_token: str | None = None,
        alert_client: httpx.Client | None = None,
        log_sources: Sequence[LogSource] | None = None,
        alert_source: AlertSource | None = None,
    ) -> list[TriageResult]:
        """Ingest live log files and/or Prometheus/Alertmanager alerts, correlate, and triage."""
        resolved_log_sources = list(log_sources) if log_sources is not None else []
        if log_files is not None:
            resolved_log_files = [Path(p) for p in log_files if str(p).strip()]
        elif (
            not resolved_log_sources
            and self.config.ingestion.log_file
            and self.config.ingestion.log_file.strip()
        ):
            resolved_log_files = [Path(self.config.ingestion.log_file.strip())]
        else:
            resolved_log_files = []

        resolved_alert_url = (
            alert_url
            if alert_url is not None
            else (self.config.ingestion.prometheus_url if alert_source is None else None)
        )
        if isinstance(resolved_alert_url, str) and not resolved_alert_url.strip():
            resolved_alert_url = None

        if (
            not resolved_log_sources
            and not resolved_log_files
            and alert_source is None
            and not resolved_alert_url
        ):
            raise ConfigError("Live triage requires at least one log file or alert URL")

        eff_api_type: AlertApiType = (
            alert_api_type if alert_api_type is not None else self.config.ingestion.alert_api_type
        )
        eff_tail_lines = (
            tail_lines if tail_lines is not None else self.config.ingestion.tail_lines
        )
        eff_include_resolved = (
            include_resolved
            if include_resolved is not None
            else self.config.ingestion.include_resolved
        )

        alerts: list[AlertEvent] = []
        if alert_source is not None:
            alerts.extend(alert_source.read())
        elif resolved_alert_url is not None:
            alerts.extend(
                PrometheusAlertSource(
                    resolved_alert_url,
                    api_type=eff_api_type,
                    include_resolved=eff_include_resolved,
                    headers=alert_headers,
                    auth_token=alert_auth_token,
                    client=alert_client,
                ).read()
            )

        logs: list[LogEntry] = []
        for src in resolved_log_sources:
            logs.extend(src.read())
        for log_path in resolved_log_files:
            logs.extend(FileLogSource(log_path, tail_lines=eff_tail_lines).read())

        bundles = correlate_incidents(
            alerts,
            logs,
            source_mode="live",
            time_window_seconds=time_window_seconds,
        )
        if not bundles:
            return []

        batch_run_id = run_id or _new_run_id()
        return [self.triage_bundle(bundle, run_id=batch_run_id) for bundle in bundles]
