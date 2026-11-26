"""Prometheus and Alertmanager alert ingestion (bundled fixtures and live HTTP)."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse, urlunparse

import httpx

from sre_triage_agent.ingestion import IngestionError
from sre_triage_agent.models import AlertEvent, utc_now

AlertStatus = Literal["firing", "resolved", "pending"]
AlertApiType = Literal["auto", "prometheus", "alertmanager"]


class AlertSource(Protocol):
    """Protocol for alert sources returning normalized AlertEvent objects."""

    def read(self) -> list[AlertEvent]:
        """Read normalized alert events."""
        ...


def _parse_alert_datetime(raw: Any) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        return utc_now()
    candidate = raw.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    match = re.match(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(\.\d+)?(.*)$", candidate)
    if match:
        base, frac, tz_part = match.groups()
        if frac and len(frac) > 7:
            frac = frac[:7]
        candidate = f"{base}{frac or ''}{tz_part}"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return utc_now()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_prometheus_state(raw_state: Any) -> AlertStatus:
    state = str(raw_state or "firing").strip().lower()
    if state in {"firing", "active"}:
        return "firing"
    if state == "pending":
        return "pending"
    if state in {"inactive", "resolved"}:
        return "resolved"
    return "firing"


def _normalize_alertmanager_state(raw_status: Any) -> AlertStatus:
    if isinstance(raw_status, dict):
        state = str(raw_status.get("state", "active")).strip().lower()
    else:
        state = str(raw_status or "active").strip().lower()
    if state in {"active", "firing"}:
        return "firing"
    if state in {"unprocessed", "pending"}:
        return "pending"
    if state in {"resolved", "inactive", "suppressed"}:
        return "resolved"
    return "firing"


def _first_non_empty(*candidates: str | None, default: str = "unknown") -> str:
    for candidate in candidates:
        if candidate is not None:
            stripped = candidate.strip()
            if stripped:
                return stripped
    return default


def _normalize_alert_record(
    raw_alert: dict[str, Any],
    *,
    format_kind: Literal["prometheus", "alertmanager"],
    source: str,
) -> AlertEvent:
    raw_labels = raw_alert.get("labels")
    if not isinstance(raw_labels, dict):
        raise IngestionError(
            "Alert payload item missing 'labels' object",
            source=source,
        )
    labels = {str(k): str(v) for k, v in raw_labels.items() if v is not None}

    raw_annotations = raw_alert.get("annotations") or {}
    if not isinstance(raw_annotations, dict):
        raw_annotations = {}
    annotations = {str(k): str(v) for k, v in raw_annotations.items() if v is not None}

    alertname = labels.get("alertname", "").strip()
    if not alertname:
        raise IngestionError(
            "Alert payload item missing non-empty 'alertname' label",
            source=source,
        )

    if format_kind == "prometheus":
        status = _normalize_prometheus_state(raw_alert.get("state"))
        starts_at = _parse_alert_datetime(raw_alert.get("activeAt") or raw_alert.get("startsAt"))
    else:
        status = _normalize_alertmanager_state(raw_alert.get("status"))
        starts_at = _parse_alert_datetime(raw_alert.get("startsAt") or raw_alert.get("updatedAt"))

    severity = labels.get("severity", "warning").strip().lower() or "warning"
    service = _first_non_empty(
        labels.get("service"),
        labels.get("app"),
        labels.get("job"),
        labels.get("component"),
        labels.get("container"),
        labels.get("instance"),
        default="unknown",
    )
    namespace = labels.get("namespace")
    instance = labels.get("instance") or labels.get("node") or labels.get("host")
    summary = annotations.get("summary", "").strip()
    description = annotations.get("description", "").strip()

    canonical_key = f"{alertname}|{sorted(labels.items())}|{starts_at.isoformat()}"
    alert_id = f"alert-{hashlib.sha256(canonical_key.encode('utf-8')).hexdigest()[:12]}"

    return AlertEvent(
        alert_id=alert_id,
        alertname=alertname,
        status=status,
        severity=severity,
        service=service,
        namespace=namespace,
        instance=instance,
        summary=summary,
        description=description,
        labels=labels,
        annotations=annotations,
        starts_at=starts_at,
    )


def parse_alerts_payload(
    payload: Any,
    *,
    include_resolved: bool = False,
    api_type: AlertApiType = "auto",
    source: str = "",
) -> list[AlertEvent]:
    """Parse Prometheus /api/v1/alerts or Alertmanager /api/v2/alerts JSON payload."""
    format_kind: Literal["prometheus", "alertmanager"]
    raw_items: list[Any]

    if isinstance(payload, dict):
        if api_type == "alertmanager":
            raise IngestionError(
                "Expected Alertmanager array payload, received object envelope",
                source=source,
            )
        status_value = payload.get("status")
        if status_value == "error":
            err_type = payload.get("errorType", "unknown_error")
            err_msg = payload.get("error", "Prometheus API returned error")
            raise IngestionError(
                f"Prometheus API error ({err_type}): {err_msg}",
                source=source,
            )
        data = payload.get("data")
        if status_value != "success" or not isinstance(data, dict) or not isinstance(data.get("alerts"), list):
            raise IngestionError(
                "Malformed Prometheus /api/v1/alerts payload: expected {'status': 'success', 'data': {'alerts': [...]}}",
                source=source,
            )
        format_kind = "prometheus"
        raw_items = data["alerts"]
    elif isinstance(payload, list):
        if api_type == "prometheus":
            raise IngestionError(
                "Expected Prometheus /api/v1/alerts object envelope, received array",
                source=source,
            )
        format_kind = "alertmanager"
        raw_items = payload
    else:
        raise IngestionError(
            f"Unsupported alerts payload type: {type(payload).__name__}",
            source=source,
        )

    events: list[AlertEvent] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise IngestionError(
                "Alert entry must be a JSON object",
                source=source,
            )
        event = _normalize_alert_record(item, format_kind=format_kind, source=source)
        if not include_resolved and event.status == "resolved":
            continue
        events.append(event)
    return events


def load_bundled_alerts(
    scenario_name: str,
    *,
    include_resolved: bool = False,
    scenarios_dir: Path | None = None,
) -> list[AlertEvent]:
    """Load and parse alerts.json for a named bundled scenario."""
    return BundledAlertSource(
        scenario_name,
        include_resolved=include_resolved,
        scenarios_dir=scenarios_dir,
    ).read()


class BundledAlertSource:
    """Reads alerts.json from a named scenario under samples/scenarios/."""

    def __init__(
        self,
        scenario_name: str,
        *,
        include_resolved: bool = False,
        scenarios_dir: Path | None = None,
    ) -> None:
        from sre_triage_agent.ingestion import resolve_scenarios_dir

        self.scenario_name = scenario_name
        self.include_resolved = include_resolved
        base_dir = resolve_scenarios_dir(scenarios_dir)
        scenario_dir = base_dir / scenario_name
        if not scenario_dir.is_dir():
            raise IngestionError(
                f"Unknown bundled scenario: '{scenario_name}'",
                source=str(scenario_dir),
            )
        self.alerts_path = scenario_dir / "alerts.json"
        if not self.alerts_path.is_file():
            raise IngestionError(
                f"Scenario alerts.json not found: {self.alerts_path}",
                source=str(self.alerts_path),
            )

    def read(self) -> list[AlertEvent]:
        """Load and parse alerts for this bundled scenario."""
        source_label = f"bundled:{self.scenario_name}/{self.alerts_path.name}"
        try:
            raw_text = self.alerts_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise IngestionError(
                f"Failed to read bundled alerts {self.alerts_path}: {exc}",
                source=str(self.alerts_path),
                cause=exc,
            ) from exc

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise IngestionError(
                f"Invalid JSON in bundled alerts {self.alerts_path}: {exc}",
                source=source_label,
                cause=exc,
            ) from exc

        return parse_alerts_payload(
            payload,
            include_resolved=self.include_resolved,
            api_type="auto",
            source=source_label,
        )


def _append_endpoint_suffix(base_url: str, suffix: str) -> str:
    parsed = urlparse(base_url)
    clean_path = parsed.path.rstrip("/")
    if clean_path.endswith(suffix):
        new_path = clean_path
    else:
        new_path = f"{clean_path}{suffix}"
    return urlunparse(parsed._replace(path=new_path))


class PrometheusAlertSource:
    """Queries live Prometheus (/api/v1/alerts) or Alertmanager (/api/v2/alerts) endpoints."""

    def __init__(
        self,
        url: str,
        *,
        api_type: AlertApiType = "auto",
        include_resolved: bool = False,
        timeout_s: float = 10.0,
        headers: dict[str, str] | None = None,
        auth_token: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise IngestionError(
                f"Invalid alert endpoint URL: '{url}' (expected http:// or https://)",
                source=url,
            )
        if timeout_s <= 0:
            raise IngestionError(
                f"timeout_s must be positive, got {timeout_s}",
                source=url,
            )

        self.url = url.strip()
        self.api_type = api_type
        self.include_resolved = include_resolved
        self.timeout_s = timeout_s
        self.headers = dict(headers or {})
        self.auth_token = auth_token
        self._injected_client = client

    def _build_headers(self) -> dict[str, str]:
        req_headers = dict(self.headers)
        if self.auth_token and not any(k.lower() == "authorization" for k in req_headers):
            req_headers["Authorization"] = f"Bearer {self.auth_token}"
        return req_headers

    def _resolve_candidate_endpoints(self) -> list[tuple[str, AlertApiType]]:
        parsed_path = urlparse(self.url).path.rstrip("/")
        if self.api_type == "prometheus":
            return [(_append_endpoint_suffix(self.url, "/api/v1/alerts"), "prometheus")]
        if self.api_type == "alertmanager":
            return [(_append_endpoint_suffix(self.url, "/api/v2/alerts"), "alertmanager")]

        if parsed_path.endswith("/api/v1/alerts"):
            return [(self.url, "prometheus")]
        if parsed_path.endswith("/api/v2/alerts"):
            return [(self.url, "alertmanager")]

        return [
            (_append_endpoint_suffix(self.url, "/api/v1/alerts"), "auto"),
            (_append_endpoint_suffix(self.url, "/api/v2/alerts"), "auto"),
        ]

    def read(self) -> list[AlertEvent]:
        """Fetch and normalize alerts from the configured Prometheus/Alertmanager endpoint."""
        candidates = self._resolve_candidate_endpoints()
        req_headers = self._build_headers()
        own_client = self._injected_client is None
        client = self._injected_client or httpx.Client(timeout=self.timeout_s)

        try:
            for idx, (endpoint, expected_api_type) in enumerate(candidates):
                has_fallback = idx + 1 < len(candidates)
                try:
                    response = client.get(endpoint, headers=req_headers, timeout=self.timeout_s)
                except httpx.TimeoutException as exc:
                    raise IngestionError(
                        f"Timed out querying alerts endpoint {endpoint} after {self.timeout_s}s",
                        source=endpoint,
                        cause=exc,
                    ) from exc
                except httpx.RequestError as exc:
                    raise IngestionError(
                        f"Network error querying alerts endpoint {endpoint}: {exc}",
                        source=endpoint,
                        cause=exc,
                    ) from exc

                if response.status_code == 404 and has_fallback:
                    continue

                if response.status_code < 200 or response.status_code >= 300:
                    raise IngestionError(
                        f"HTTP {response.status_code} from alerts endpoint {endpoint}",
                        source=endpoint,
                        status_code=response.status_code,
                    )

                try:
                    payload = response.json()
                except ValueError as exc:
                    raise IngestionError(
                        f"Invalid JSON response from alerts endpoint {endpoint}: {exc}",
                        source=endpoint,
                        status_code=response.status_code,
                        cause=exc,
                    ) from exc

                return parse_alerts_payload(
                    payload,
                    include_resolved=self.include_resolved,
                    api_type=expected_api_type,
                    source=endpoint,
                )

            raise IngestionError(
                f"Alerts endpoint returned 404 for all candidate API paths under {self.url}",
                source=self.url,
                status_code=404,
            )
        finally:
            if own_client:
                client.close()
