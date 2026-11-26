"""Incident correlation grouping alerts and logs into IncidentBundle objects."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, Sequence

from sre_triage_agent.models import AlertEvent, IncidentBundle, LogEntry, utc_now

_SEVERITY_RANK: dict[str, int] = {
    "critical": 5,
    "high": 4,
    "error": 4,
    "warning": 3,
    "warn": 3,
    "medium": 2,
    "info": 1,
    "low": 0,
}

_LOG_LEVEL_TO_SEVERITY: dict[str, str] = {
    "CRITICAL": "critical",
    "ERROR": "high",
    "WARN": "warning",
    "INFO": "info",
    "DEBUG": "info",
}

_TOKEN_KEYS = ("pod", "container", "node", "host", "instance")


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _extract_host_token(raw: str | None) -> set[str]:
    if not raw:
        return set()
    cleaned = raw.strip().lower()
    if not cleaned:
        return set()
    tokens = {cleaned}
    if ":" in cleaned:
        host_part = cleaned.split(":", 1)[0].strip()
        if host_part:
            tokens.add(host_part)
        if "." in host_part:
            short_host = host_part.split(".", 1)[0].strip()
            if short_host:
                tokens.add(short_host)
    elif "." in cleaned:
        short_host = cleaned.split(".", 1)[0].strip()
        if short_host:
            tokens.add(short_host)
    return tokens


def _alert_tokens(alert: AlertEvent) -> set[str]:
    tokens: set[str] = set()
    for key in _TOKEN_KEYS:
        tokens.update(_extract_host_token(alert.labels.get(key)))
    tokens.update(_extract_host_token(alert.instance))
    return tokens


def _log_tokens(entry: LogEntry) -> set[str]:
    tokens: set[str] = set()
    for key in _TOKEN_KEYS:
        tokens.update(_extract_host_token(entry.attributes.get(key)))
    return tokens


def _normalize_service(raw: str | None) -> str:
    if not raw:
        return "unknown"
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() == "unknown":
        return "unknown"
    return cleaned


def _same_known_service(service_a: str, service_b: str) -> bool:
    norm_a = _normalize_service(service_a)
    norm_b = _normalize_service(service_b)
    return norm_a != "unknown" and norm_b != "unknown" and norm_a.lower() == norm_b.lower()


def _severity_rank(severity: str) -> int:
    return _SEVERITY_RANK.get(severity.strip().lower(), 1)


@dataclass
class _IncidentCluster:
    service: str
    alerts: list[AlertEvent] = field(default_factory=list)
    logs: list[LogEntry] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)
    min_ts: datetime | None = None
    max_ts: datetime | None = None

    def add_alert(self, alert: AlertEvent) -> None:
        ts = _ensure_utc(alert.starts_at)
        self.alerts.append(alert)
        self.tokens.update(_alert_tokens(alert))
        norm_alert_svc = _normalize_service(alert.service)
        if _normalize_service(self.service) == "unknown" and norm_alert_svc != "unknown":
            self.service = norm_alert_svc
        self._extend_bounds(ts)

    def add_log(self, entry: LogEntry) -> None:
        ts = _ensure_utc(entry.timestamp)
        self.logs.append(entry)
        self.tokens.update(_log_tokens(entry))
        norm_entry_svc = _normalize_service(entry.service)
        if _normalize_service(self.service) == "unknown" and norm_entry_svc != "unknown":
            self.service = norm_entry_svc
        self._extend_bounds(ts)

    def _extend_bounds(self, ts: datetime) -> None:
        if self.min_ts is None or ts < self.min_ts:
            self.min_ts = ts
        if self.max_ts is None or ts > self.max_ts:
            self.max_ts = ts

    def within_window(self, ts: datetime, window: timedelta) -> bool:
        if self.min_ts is None or self.max_ts is None:
            return True
        return (self.min_ts - window) <= ts <= (self.max_ts + window)

    def midpoint(self) -> datetime:
        if self.min_ts is None or self.max_ts is None:
            return utc_now()
        delta = self.max_ts - self.min_ts
        return self.min_ts + delta / 2


def correlate_incidents(
    alerts: Sequence[AlertEvent],
    logs: Sequence[LogEntry],
    *,
    source_mode: Literal["bundled", "live"] = "live",
    scenario_name: str | None = None,
    time_window_seconds: int = 900,
    now: datetime | None = None,
) -> list[IncidentBundle]:
    """Correlate AlertEvent and LogEntry streams into deterministic IncidentBundles."""
    if not alerts and not logs:
        return []

    window = timedelta(seconds=max(0, time_window_seconds))
    sorted_alerts = sorted(
        alerts,
        key=lambda a: (_ensure_utc(a.starts_at), a.alert_id),
    )
    sorted_logs = sorted(
        logs,
        key=lambda entry: (_ensure_utc(entry.timestamp), entry.line_number, entry.log_id),
    )

    clusters: list[_IncidentCluster] = []

    for alert in sorted_alerts:
        alert_ts = _ensure_utc(alert.starts_at)
        tokens = _alert_tokens(alert)
        matched_cluster: _IncidentCluster | None = None

        for cluster in clusters:
            same_service = _same_known_service(alert.service, cluster.service)
            shared_token = bool(tokens & cluster.tokens)
            allow_resolved_merge = same_service and (
                alert.status == "resolved"
                or all(existing.status == "resolved" for existing in cluster.alerts)
            )
            if not allow_resolved_merge and not cluster.within_window(alert_ts, window):
                continue
            if same_service or shared_token:
                matched_cluster = cluster
                break

        if matched_cluster is None:
            matched_cluster = _IncidentCluster(service=_normalize_service(alert.service))
            clusters.append(matched_cluster)
        matched_cluster.add_alert(alert)

    unmatched_logs: list[LogEntry] = []
    for entry in sorted_logs:
        entry_ts = _ensure_utc(entry.timestamp)
        entry_tokens = _log_tokens(entry)
        best_idx: int | None = None
        best_key: tuple[int, float, int] | None = None

        for idx, cluster in enumerate(clusters):
            if not cluster.within_window(entry_ts, window):
                continue
            same_service = _same_known_service(entry.service, cluster.service)
            token_overlap = len(entry_tokens & cluster.tokens)
            service_mention = (
                _normalize_service(entry.service) == "unknown"
                and _normalize_service(cluster.service) != "unknown"
                and cluster.service.lower() in entry.message.lower()
            )
            score = (3 if same_service else 0) + (2 * token_overlap) + (1 if service_mention else 0)
            if score <= 0:
                continue

            distance_s = abs((entry_ts - cluster.midpoint()).total_seconds())
            candidate_key = (-score, distance_s, idx)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_idx = idx

        if best_idx is not None:
            clusters[best_idx].add_log(entry)
        else:
            unmatched_logs.append(entry)

    log_clusters: list[_IncidentCluster] = []
    for entry in unmatched_logs:
        entry_ts = _ensure_utc(entry.timestamp)
        entry_tokens = _log_tokens(entry)
        matched_log_cluster: _IncidentCluster | None = None

        for cluster in log_clusters:
            if not cluster.within_window(entry_ts, window):
                continue
            same_service = _same_known_service(entry.service, cluster.service)
            shared_token = bool(entry_tokens & cluster.tokens)
            if same_service or shared_token:
                matched_log_cluster = cluster
                break

        if matched_log_cluster is None:
            matched_log_cluster = _IncidentCluster(service=_normalize_service(entry.service))
            log_clusters.append(matched_log_cluster)
        matched_log_cluster.add_log(entry)

    all_clusters = clusters + log_clusters
    bundles: list[IncidentBundle] = []

    for cluster in all_clusters:
        service = _normalize_service(cluster.service)
        if cluster.alerts:
            primary_alert = max(
                cluster.alerts,
                key=lambda a: (_severity_rank(a.severity), -_ensure_utc(a.starts_at).timestamp()),
            )
            severity = primary_alert.severity.lower()
            title = primary_alert.summary or f"{primary_alert.alertname} on {service}"
        else:
            max_log = max(
                cluster.logs,
                key=lambda entry: (
                    _severity_rank(_LOG_LEVEL_TO_SEVERITY.get(entry.level, "info")),
                    -_ensure_utc(entry.timestamp).timestamp(),
                ),
            )
            severity = _LOG_LEVEL_TO_SEVERITY.get(max_log.level, "info")
            title = f"Log anomaly on {service} ({max_log.level})"

        sorted_alert_ids = sorted(a.alert_id for a in cluster.alerts)
        sorted_log_ids = sorted(entry.log_id for entry in cluster.logs)
        raw_inc_key = f"{service}|{sorted_alert_ids}|{sorted_log_ids}"
        incident_id = f"inc-{hashlib.sha256(raw_inc_key.encode('utf-8')).hexdigest()[:10]}"
        created_at = _ensure_utc(now) if now is not None else (cluster.max_ts or utc_now())

        bundles.append(
            IncidentBundle(
                incident_id=incident_id,
                title=title,
                service=service,
                severity=severity,
                scenario_name=scenario_name,
                source_mode=source_mode,
                alerts=cluster.alerts,
                logs=cluster.logs,
                created_at=created_at,
            )
        )

    bundles.sort(
        key=lambda b: (
            -_severity_rank(b.severity),
            _ensure_utc(b.created_at),
            b.service,
            b.incident_id,
        )
    )
    return bundles
