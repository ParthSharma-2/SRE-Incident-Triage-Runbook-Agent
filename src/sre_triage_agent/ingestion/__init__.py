"""Ingestion subpackage for bundled and live logs, alerts, and incident correlation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class IngestionError(Exception):
    """Typed error raised when log/alert ingestion or scenario loading fails."""

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        status_code: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.source = source
        self.status_code = status_code
        self.cause = cause


from sre_triage_agent.ingestion.alerts import (  # noqa: E402
    AlertSource,
    BundledAlertSource,
    PrometheusAlertSource,
    load_bundled_alerts,
    parse_alerts_payload,
)
from sre_triage_agent.ingestion.correlator import correlate_incidents  # noqa: E402
from sre_triage_agent.ingestion.logs import (  # noqa: E402
    BundledLogSource,
    FileLogSource,
    LogReadResult,
    LogSource,
)
from sre_triage_agent.models import IncidentBundle  # noqa: E402

SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "pod_oom_killed": "Kubernetes checkout-api pod terminated with OOMKilled (exit code 137) and CrashLoopBackOff.",
    "api_5xx_db_pool": "orders-service HTTP 503 burst caused by PostgreSQL connection pool exhaustion.",
    "disk_pressure": "worker-node-03 filesystem /var/lib/containerd exhausted, triggering KubeNodeDiskPressure.",
    "tls_cert_expiring": "Ingress TLS certificate for api.prod.example.com expiring within 48h due to ACME HTTP-01 403.",
}


class ScenarioInfo(BaseModel):
    """Metadata summary for a bundled incident scenario."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    alerts_path: Path
    log_path: Path
    alert_count: int
    log_count: int


def resolve_scenarios_dir(scenarios_dir: Path | None = None) -> Path:
    """Locate the bundled samples/scenarios directory."""
    if scenarios_dir is not None:
        resolved = Path(scenarios_dir)
        if not resolved.is_dir():
            raise IngestionError(
                f"Scenarios directory not found: {resolved}",
                source=str(resolved),
            )
        return resolved

    candidates = [
        Path(__file__).resolve().parents[3] / "samples" / "scenarios",
        Path.cwd() / "samples" / "scenarios",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise IngestionError(
        "Unable to locate bundled samples/scenarios directory",
        source=str(candidates[0]),
    )


def find_scenario_log_file(scenario_dir: Path) -> Path:
    """Locate the log fixture file inside a scenario directory."""
    candidates = sorted(
        p
        for p in scenario_dir.iterdir()
        if p.is_file() and (p.name.endswith(".log.jsonl") or p.name.endswith(".log"))
    )
    if not candidates:
        raise IngestionError(
            f"No log file (*.log or *.log.jsonl) found in scenario directory: {scenario_dir}",
            source=str(scenario_dir),
        )
    return candidates[0]


def _extract_scenario_description(name: str, alerts_path: Path) -> str:
    if name in SCENARIO_DESCRIPTIONS:
        return SCENARIO_DESCRIPTIONS[name]
    try:
        raw = json.loads(alerts_path.read_text(encoding="utf-8"))
        alerts = (
            raw.get("data", {}).get("alerts", [])
            if isinstance(raw, dict)
            else raw
            if isinstance(raw, list)
            else []
        )
        if alerts and isinstance(alerts[0], dict):
            annotations = alerts[0].get("annotations") or {}
            summary = annotations.get("summary") or annotations.get("description")
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return f"Bundled SRE incident scenario: {name}"


def list_bundled_scenarios(scenarios_dir: Path | None = None) -> list[ScenarioInfo]:
    """List all bundled incident scenarios with alert and log counts."""
    base_dir = resolve_scenarios_dir(scenarios_dir)
    scenarios: list[ScenarioInfo] = []
    for entry in sorted(base_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        alerts_path = entry / "alerts.json"
        if not alerts_path.is_file():
            continue
        log_path = find_scenario_log_file(entry)
        alerts = BundledAlertSource(
            entry.name,
            include_resolved=True,
            scenarios_dir=base_dir,
        ).read()
        logs = BundledLogSource(entry.name, scenarios_dir=base_dir).read()
        scenarios.append(
            ScenarioInfo(
                name=entry.name,
                description=_extract_scenario_description(entry.name, alerts_path),
                alerts_path=alerts_path,
                log_path=log_path,
                alert_count=len(alerts),
                log_count=len(logs),
            )
        )
    return scenarios


def load_bundled_scenario(
    scenario_name: str,
    *,
    include_resolved: bool = False,
    scenarios_dir: Path | None = None,
) -> IncidentBundle:
    """Load and correlate a single bundled scenario into an IncidentBundle."""
    base_dir = resolve_scenarios_dir(scenarios_dir)
    scenario_dir = base_dir / scenario_name
    if not scenario_dir.is_dir():
        raise IngestionError(
            f"Unknown bundled scenario: '{scenario_name}'",
            source=str(scenario_dir),
        )

    alerts = BundledAlertSource(
        scenario_name,
        include_resolved=include_resolved,
        scenarios_dir=base_dir,
    ).read()
    logs = BundledLogSource(scenario_name, scenarios_dir=base_dir).read()

    bundles = correlate_incidents(
        alerts,
        logs,
        source_mode="bundled",
        scenario_name=scenario_name,
    )
    if not bundles:
        raise IngestionError(
            f"Bundled scenario '{scenario_name}' produced no alerts or logs",
            source=str(scenario_dir),
        )
    return bundles[0]


__all__ = [
    "AlertSource",
    "BundledAlertSource",
    "BundledLogSource",
    "FileLogSource",
    "IngestionError",
    "LogReadResult",
    "LogSource",
    "PrometheusAlertSource",
    "SCENARIO_DESCRIPTIONS",
    "ScenarioInfo",
    "correlate_incidents",
    "find_scenario_log_file",
    "list_bundled_scenarios",
    "load_bundled_alerts",
    "load_bundled_scenario",
    "parse_alerts_payload",
    "resolve_scenarios_dir",
]
