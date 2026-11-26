"""Click + Rich CLI for the DevOps SRE Operational Incident Triage & Runbook Agent."""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import click
import httpx
from rich.console import Console
from rich.table import Table

from sre_triage_agent.approval import (
    ApprovalError,
    ApprovalSummary,
    InteractivePromptHandler,
    ScriptedDecisionHandler,
    process_remediations,
)
from sre_triage_agent.audit import AuditError, AuditLogger
from sre_triage_agent.config import AgentConfig, ConfigError, load_config
from sre_triage_agent.executor import DryRunExecutor
from sre_triage_agent.ingestion import IngestionError, list_bundled_scenarios
from sre_triage_agent.llm import LLMProviderError, build_llm_provider
from sre_triage_agent.models import AuditEvent, TriageResult
from sre_triage_agent.runbooks import RunbookError
from sre_triage_agent.triage import TriageEngine, TriageError


def _configure_stdio_utf8() -> None:
    """Ensure stdout/stderr handle UTF-8 cleanly on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def _emit_json(data: Any) -> None:
    click.echo(json.dumps(data, indent=2, ensure_ascii=False))


def _build_overrides(
    *,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    allow_fallback: bool | None = None,
    alert_url: str | None = None,
    alert_api_type: str | None = None,
    log_files: Sequence[str] = (),
    tail_lines: int | None = None,
    include_resolved: bool | None = None,
    audit_log: str | None = None,
    operator: str | None = None,
) -> dict[str, Any]:
    llm_overrides: dict[str, Any] = {}
    if provider is not None:
        llm_overrides["provider"] = provider
    if base_url is not None:
        llm_overrides["base_url"] = base_url
    if model is not None:
        llm_overrides["model"] = model
    if api_key is not None:
        llm_overrides["api_key"] = api_key
    if timeout is not None:
        llm_overrides["timeout_seconds"] = timeout
    if allow_fallback is not None:
        llm_overrides["allow_fallback"] = allow_fallback

    ingestion_overrides: dict[str, Any] = {}
    if alert_url is not None:
        ingestion_overrides["prometheus_url"] = alert_url
    if alert_api_type is not None:
        ingestion_overrides["alert_api_type"] = alert_api_type
    if log_files:
        ingestion_overrides["log_file"] = log_files[0]
    if tail_lines is not None:
        ingestion_overrides["tail_lines"] = tail_lines
    if include_resolved is not None:
        ingestion_overrides["include_resolved"] = include_resolved

    audit_overrides: dict[str, Any] = {}
    if audit_log is not None:
        audit_overrides["audit_log_path"] = audit_log

    overrides: dict[str, Any] = {}
    if llm_overrides:
        overrides["llm"] = llm_overrides
    if ingestion_overrides:
        overrides["ingestion"] = ingestion_overrides
    if audit_overrides:
        overrides["audit"] = audit_overrides
    if operator is not None:
        overrides["operator"] = operator
    return overrides


def _record_triage_events(audit_logger: AuditLogger, results: Sequence[TriageResult]) -> None:
    for result in results:
        audit_logger.log_triage_run_started(
            run_id=result.run_id,
            incident_id=result.incident_id,
            payload={
                "scenario_name": result.scenario_name,
                "source_mode": result.source_mode,
                "alert_count": len(result.alerts),
                "log_count": len(result.logs),
            },
        )
        audit_logger.log_triage_completed(result)
        for step in result.remediation_steps:
            audit_logger.log_remediation_suggested(result, step)


def _render_scripted_summary(
    console: Console,
    results: Sequence[TriageResult],
    summary: ApprovalSummary,
    audit_path: Path | str,
) -> None:
    """Render an ASCII/cp1252-safe Rich summary of triage results and dry-run decisions."""
    for result in results:
        console.print(
            f"[Incident] {result.incident_id} | scenario={result.scenario_name or 'live'} "
            f"| severity={result.severity} | provider={result.provider_used} ({result.model_used})"
        )
        console.print(f"  Summary: {result.summary}")
        console.print(f"  Root cause: {result.root_cause_hypothesis}")

        table = Table(title=f"Remediation Steps ({result.incident_id})", safe_box=True)
        table.add_column("Step ID")
        table.add_column("Risk")
        table.add_column("Title")
        table.add_column("Target")
        for step in result.remediation_steps:
            table.add_row(step.step_id, step.risk_level, step.title, step.target_resource)
        console.print(table)

    console.print(
        f"Audit log written: {audit_path} "
        f"(decisions={len(summary.decisions)}, dry_runs={len(summary.dry_runs)}, aborted={summary.aborted})"
    )


def _render_audit_table(
    console: Console,
    title: str,
    events: Sequence[AuditEvent],
) -> None:
    """Render an ASCII/cp1252-safe Rich table of AuditEvents."""
    table = Table(title=title, safe_box=True)
    table.add_column("Timestamp")
    table.add_column("Run ID")
    table.add_column("Incident ID")
    table.add_column("Event Type")
    table.add_column("Actor")
    for ev in events:
        table.add_row(
            ev.timestamp.isoformat(),
            ev.run_id,
            ev.incident_id,
            ev.event_type,
            ev.actor,
        )
    console.print(table)


@click.group()
def main() -> None:
    """DevOps/SRE Operational Incident Triage & Runbook Agent."""
    _configure_stdio_utf8()


@main.group()
def scenarios() -> None:
    """Discover bundled incident scenarios."""


@scenarios.command("list")
@click.option("--json", "json_output", is_flag=True, help="Output structured JSON.")
def scenarios_list(json_output: bool) -> None:
    """List bundled sample incident scenarios."""
    try:
        items = list_bundled_scenarios()
    except IngestionError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if json_output:
        _emit_json({"scenarios": [item.model_dump(mode="json") for item in items]})
        return

    console = Console(safe_box=True)
    table = Table(title="Bundled Incident Scenarios", safe_box=True)
    table.add_column("Scenario")
    table.add_column("Alerts", justify="right")
    table.add_column("Logs", justify="right")
    table.add_column("Description")
    for item in items:
        table.add_row(item.name, str(item.alert_count), str(item.log_count), item.description)
    console.print(table)


@main.group()
def providers() -> None:
    """Inspect and health-check open-source LLM providers."""


@providers.command("check")
@click.option(
    "--provider",
    type=click.Choice(["ollama", "openai_compatible", "deterministic"], case_sensitive=False),
    default=None,
    help="LLM provider type.",
)
@click.option("--base-url", default=None, help="LLM provider base URL.")
@click.option("--model", default=None, help="LLM model name.")
@click.option("--api-key", default=None, help="Optional API key for OpenAI-compatible servers.")
@click.option("--timeout", type=float, default=None, help="HTTP timeout in seconds.")
@click.option("--config", "config_path", type=click.Path(), default=None, help="YAML config path.")
@click.option("--json", "json_output", is_flag=True, help="Output structured JSON.")
def providers_check(
    provider: str | None,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    timeout: float | None,
    config_path: str | None,
    json_output: bool,
) -> None:
    """Check connectivity and model readiness for the configured LLM provider."""
    if timeout is not None and timeout <= 0:
        raise click.BadParameter("--timeout must be greater than 0", param_hint="--timeout")

    try:
        overrides = _build_overrides(
            provider=provider.lower() if provider else None,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
        cfg = load_config(config_path=config_path, overrides=overrides)
        llm_provider = build_llm_provider(cfg.llm)
        health = llm_provider.healthcheck()
    except (ConfigError, LLMProviderError, httpx.HTTPError, OSError) as exc:
        if json_output:
            _emit_json({"healthy": False, "detail": str(exc)})
        else:
            click.echo(f"Provider check failed: {exc}", err=True)
        sys.exit(1)

    if json_output:
        _emit_json(health.model_dump(mode="json"))
    else:
        status = "[OK]" if health.healthy else "[UNHEALTHY]"
        click.echo(f"{status} provider={health.provider} model={health.model}: {health.detail}")

    if not health.healthy:
        sys.exit(1)


@main.command("triage")
@click.option("--scenario", default=None, help="Bundled scenario name to triage.")
@click.option("--all-scenarios", is_flag=True, help="Triage all bundled scenarios.")
@click.option("--log-file", "log_files", multiple=True, help="Live log file path(s).")
@click.option("--alert-url", default=None, help="Live Prometheus or Alertmanager alerts URL.")
@click.option(
    "--alert-api-type",
    type=click.Choice(["auto", "prometheus", "alertmanager"], case_sensitive=False),
    default=None,
    help="Alert API payload format.",
)
@click.option("--tail-lines", type=int, default=None, help="Tail last N lines of live log files.")
@click.option(
    "--include-resolved/--no-include-resolved",
    default=None,
    help="Include resolved alerts during ingestion.",
)
@click.option(
    "--provider",
    type=click.Choice(["ollama", "openai_compatible", "deterministic"], case_sensitive=False),
    default=None,
    help="LLM provider to use for triage.",
)
@click.option("--base-url", default=None, help="Open-source LLM endpoint base URL.")
@click.option("--model", default=None, help="Open-source LLM model identifier.")
@click.option("--api-key", default=None, help="Optional API key for OpenAI-compatible endpoints.")
@click.option("--timeout", type=float, default=None, help="LLM request timeout in seconds.")
@click.option(
    "--allow-fallback/--no-allow-fallback",
    default=None,
    help="Allow deterministic fallback if LLM provider fails.",
)
@click.option(
    "--decisions",
    default=None,
    help="Comma-separated scripted decisions (approve, reject, skip, quit).",
)
@click.option("--operator", default=None, help="Operator identity recorded in audit logs.")
@click.option("--audit-log", default=None, help="Path to persistent JSONL audit log.")
@click.option("--config", "config_path", type=click.Path(), default=None, help="YAML config path.")
@click.option("--json", "json_output", is_flag=True, help="Output structured JSON.")
def triage_cmd(
    scenario: str | None,
    all_scenarios: bool,
    log_files: tuple[str, ...],
    alert_url: str | None,
    alert_api_type: str | None,
    tail_lines: int | None,
    include_resolved: bool | None,
    provider: str | None,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    timeout: float | None,
    allow_fallback: bool | None,
    decisions: str | None,
    operator: str | None,
    audit_log: str | None,
    config_path: str | None,
    json_output: bool,
) -> None:
    """Ingest alerts and logs, triage incidents, and run approval + dry-run workflow."""
    if timeout is not None and timeout <= 0:
        raise click.BadParameter("--timeout must be greater than 0", param_hint="--timeout")
    if tail_lines is not None and tail_lines < 1:
        raise click.BadParameter("--tail-lines must be at least 1", param_hint="--tail-lines")

    if scenario and all_scenarios:
        raise click.UsageError("Cannot combine --scenario and --all-scenarios")

    if (scenario or all_scenarios) and (log_files or alert_url):
        raise click.UsageError(
            "Cannot combine bundled --scenario/--all-scenarios with live --log-file/--alert-url."
        )

    scripted_handler: ScriptedDecisionHandler | None = None
    if decisions is not None:
        try:
            scripted_handler = ScriptedDecisionHandler(decisions)
        except ApprovalError as exc:
            raise click.BadParameter(str(exc), param_hint="--decisions") from exc

    try:
        overrides = _build_overrides(
            provider=provider.lower() if provider else None,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            allow_fallback=allow_fallback,
            alert_url=alert_url,
            alert_api_type=alert_api_type.lower() if alert_api_type else None,
            log_files=log_files,
            tail_lines=tail_lines,
            include_resolved=include_resolved,
            audit_log=audit_log,
            operator=operator,
        )
        cfg: AgentConfig = load_config(config_path=config_path, overrides=overrides)
    except ConfigError as exc:
        click.echo(f"Configuration error: {exc}", err=True)
        sys.exit(1)

    has_live_source = bool(
        log_files
        or alert_url
        or (cfg.ingestion.log_file and cfg.ingestion.log_file.strip())
        or (cfg.ingestion.prometheus_url and cfg.ingestion.prometheus_url.strip())
    )
    if not scenario and not all_scenarios and not has_live_source:
        raise click.UsageError(
            "Provide --scenario, --all-scenarios, or a live source (--log-file / --alert-url)."
        )

    run_id = _new_run_id()
    try:
        engine = TriageEngine(cfg)
        if all_scenarios:
            results = engine.triage_all_scenarios(run_id=run_id)
        elif scenario:
            results = [engine.triage_scenario(scenario, run_id=run_id)]
        else:
            results = engine.triage_live(
                log_files=list(log_files) if log_files else None,
                alert_url=alert_url,
                run_id=run_id,
            )
    except (
        ConfigError,
        IngestionError,
        RunbookError,
        LLMProviderError,
        TriageError,
        httpx.HTTPError,
        OSError,
    ) as exc:
        click.echo(f"Triage failed: {exc}", err=True)
        sys.exit(1)

    audit_logger = AuditLogger(cfg.audit.audit_log_path)
    try:
        _record_triage_events(audit_logger, results)
    except AuditError as exc:
        click.echo(f"Audit log error: {exc}", err=True)
        sys.exit(1)

    prompt_console = Console(
        file=sys.stderr if json_output else sys.stdout,
        force_terminal=False,
        no_color=True,
        safe_box=True,
    )
    handler = (
        scripted_handler
        if scripted_handler is not None
        else InteractivePromptHandler(console=prompt_console, ask_reason=True)
    )

    try:
        summary = process_remediations(
            results,
            handler=handler,
            executor=DryRunExecutor(),
            audit_logger=audit_logger,
            operator=cfg.operator,
        )
    except (AuditError, ApprovalError) as exc:
        click.echo(f"Approval/Audit error: {exc}", err=True)
        sys.exit(1)

    if json_output:
        _emit_json(
            {
                "run_id": run_id,
                "aborted": summary.aborted,
                "results": [r.model_dump(mode="json") for r in results],
                "decisions": [d.model_dump(mode="json") for d in summary.decisions],
                "dry_runs": [dr.model_dump(mode="json") for dr in summary.dry_runs],
                "audit_log_path": str(Path(cfg.audit.audit_log_path)),
            }
        )
    else:
        _render_scripted_summary(
            Console(force_terminal=False, no_color=True, safe_box=True),
            results,
            summary,
            Path(cfg.audit.audit_log_path),
        )


@main.group()
def audit() -> None:
    """Inspect persistent JSONL audit logs."""


@audit.command("list")
@click.option("--audit-log", default=None, help="Path to JSONL audit log.")
@click.option("--incident-id", default=None, help="Filter by incident ID.")
@click.option("--run-id", default=None, help="Filter by run ID.")
@click.option("--event-type", default=None, help="Filter by event_type.")
@click.option(
    "--decision",
    default=None,
    help="Filter by operator decision (approved/rejected/skipped).",
)
@click.option("--config", "config_path", type=click.Path(), default=None, help="YAML config path.")
@click.option("--json", "json_output", is_flag=True, help="Output structured JSON.")
def audit_list(
    audit_log: str | None,
    incident_id: str | None,
    run_id: str | None,
    event_type: str | None,
    decision: str | None,
    config_path: str | None,
    json_output: bool,
) -> None:
    """List audit events with optional filtering."""
    try:
        cfg = load_config(
            config_path=config_path,
            overrides=_build_overrides(audit_log=audit_log),
        )
        audit_logger = AuditLogger(cfg.audit.audit_log_path)
        events = audit_logger.query(
            incident_id=incident_id,
            run_id=run_id,
            event_type=event_type,
            decision=decision,
        )
    except (ConfigError, AuditError) as exc:
        click.echo(f"Audit error: {exc}", err=True)
        sys.exit(1)

    if json_output:
        _emit_json({"count": len(events), "events": [ev.model_dump(mode="json") for ev in events]})
        return

    if not events:
        click.echo("No matching audit events found.")
        return

    _render_audit_table(
        Console(force_terminal=False, no_color=True, safe_box=True),
        f"Audit Events ({cfg.audit.audit_log_path})",
        events,
    )


@audit.command("show")
@click.argument("incident_id")
@click.option("--audit-log", default=None, help="Path to JSONL audit log.")
@click.option("--config", "config_path", type=click.Path(), default=None, help="YAML config path.")
@click.option("--json", "json_output", is_flag=True, help="Output structured JSON.")
def audit_show(
    incident_id: str,
    audit_log: str | None,
    config_path: str | None,
    json_output: bool,
) -> None:
    """Show chronological audit lifecycle for a specific incident_id."""
    try:
        cfg = load_config(
            config_path=config_path,
            overrides=_build_overrides(audit_log=audit_log),
        )
        audit_logger = AuditLogger(cfg.audit.audit_log_path)
        events = audit_logger.show_incident(incident_id)
    except (ConfigError, AuditError) as exc:
        click.echo(f"Audit error: {exc}", err=True)
        sys.exit(1)

    if json_output:
        _emit_json(
            {
                "incident_id": incident_id,
                "count": len(events),
                "events": [ev.model_dump(mode="json") for ev in events],
            }
        )
        if not events:
            sys.exit(1)
        return

    if not events:
        click.echo(f"No audit events found for incident '{incident_id}'.", err=True)
        sys.exit(1)

    _render_audit_table(
        Console(force_terminal=False, no_color=True, safe_box=True),
        f"Incident Lifecycle ({incident_id})",
        events,
    )
