"""YAML runbook catalog loader, deterministic matcher, and safe template helpers."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from sre_triage_agent.models import IncidentBundle, RunbookTemplate

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class RunbookError(ValueError):
    """Raised when loading or validating runbook templates fails."""


def resolve_runbooks_dir(runbooks_dir: Path | None = None) -> Path:
    """Locate the bundled samples/runbooks directory or validate an explicit path."""
    if runbooks_dir is not None:
        resolved = Path(runbooks_dir)
        if not resolved.is_dir():
            raise RunbookError(f"Runbooks directory not found: {resolved}")
        return resolved

    candidates = [
        Path(__file__).resolve().parents[3] / "samples" / "runbooks",
        Path.cwd() / "samples" / "runbooks",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise RunbookError(
        f"Unable to locate bundled samples/runbooks directory (checked {candidates[0]})"
    )


def _keyword_matches(keyword: str, searchable_text: str) -> bool:
    """Match a keyword case-insensitively with word boundaries on alphanumeric edges."""
    kw = keyword.strip().lower()
    if not kw or not searchable_text:
        return False
    prefix = r"(?<![a-z0-9])" if kw[0].isalnum() else ""
    suffix = r"(?![a-z0-9])" if kw[-1].isalnum() else ""
    pattern = f"{prefix}{re.escape(kw)}{suffix}"
    return re.search(pattern, searchable_text) is not None


def _build_searchable_text(incident: IncidentBundle) -> str:
    """Collect all lowercased text tokens from incident metadata, alerts, and logs."""
    parts: list[str] = [incident.title, incident.service]
    for alert in incident.alerts:
        parts.extend(
            [
                alert.alertname,
                alert.service,
                alert.summary,
                alert.description,
                alert.namespace or "",
                alert.instance or "",
            ]
        )
        parts.extend(alert.labels.values())
        parts.extend(alert.annotations.values())
    for entry in incident.logs:
        parts.extend([entry.service, entry.message])
        parts.extend(entry.attributes.values())
    return "\n".join(part.strip().lower() for part in parts if part and part.strip())


class RunbookCatalog:
    """Immutable catalog of RunbookTemplate objects with deterministic matching."""

    def __init__(self, runbooks: Sequence[RunbookTemplate] = ()) -> None:
        by_id: dict[str, RunbookTemplate] = {}
        ordered: list[RunbookTemplate] = []
        for rb in runbooks:
            if rb.runbook_id in by_id:
                raise RunbookError(f"Duplicate runbook_id '{rb.runbook_id}' in catalog")
            by_id[rb.runbook_id] = rb
            ordered.append(rb)
        self._runbooks: tuple[RunbookTemplate, ...] = tuple(ordered)
        self._by_id: dict[str, RunbookTemplate] = by_id

    @property
    def runbooks(self) -> tuple[RunbookTemplate, ...]:
        """Return all runbooks in catalog order."""
        return self._runbooks

    def get(self, runbook_id: str) -> RunbookTemplate | None:
        """Return a runbook by its ID, or None if not found."""
        return self._by_id.get(runbook_id)

    def __len__(self) -> int:
        return len(self._runbooks)

    def __iter__(self) -> Iterator[RunbookTemplate]:
        return iter(self._runbooks)

    @classmethod
    def load_default(cls, runbooks_dir: Path | None = None) -> RunbookCatalog:
        """Load the default bundled runbook catalog."""
        resolved_dir = resolve_runbooks_dir(runbooks_dir)
        catalog = cls.from_directory(resolved_dir)
        if len(catalog) == 0:
            raise RunbookError(f"No runbook YAML files found in {resolved_dir}")
        return catalog

    @classmethod
    def from_directory(cls, path: str | Path) -> RunbookCatalog:
        """Load all *.yaml / *.yml runbook templates from a directory."""
        dir_path = Path(path)
        if not dir_path.exists():
            raise RunbookError(f"Runbooks directory not found: {dir_path}")
        if not dir_path.is_dir():
            raise RunbookError(f"Runbooks path is not a directory: {dir_path}")

        yaml_files = sorted(
            (
                p
                for p in dir_path.iterdir()
                if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
            ),
            key=lambda p: p.name,
        )
        if not yaml_files:
            return cls([])

        loaded: list[RunbookTemplate] = []
        seen_ids: set[str] = set()
        for file_path in yaml_files:
            try:
                raw_text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise RunbookError(f"Failed to read runbook file {file_path}: {exc}") from exc

            try:
                data = yaml.safe_load(raw_text)
            except yaml.YAMLError as exc:
                raise RunbookError(f"Invalid YAML syntax in runbook {file_path}: {exc}") from exc

            if not isinstance(data, dict):
                raise RunbookError(
                    f"Runbook file {file_path} must contain a top-level YAML mapping"
                )

            try:
                template = RunbookTemplate.model_validate(data)
            except ValidationError as exc:
                raise RunbookError(
                    f"Runbook schema validation failed in {file_path}: {exc}"
                ) from exc

            if template.runbook_id in seen_ids:
                raise RunbookError(
                    f"Duplicate runbook_id '{template.runbook_id}' found in {file_path}"
                )
            seen_ids.add(template.runbook_id)
            loaded.append(template)

        return cls(loaded)

    def match_candidates(
        self,
        incident: IncidentBundle,
        *,
        limit: int | None = None,
    ) -> list[RunbookTemplate]:
        """Rank and return candidate runbooks matching alerts or log/title keywords."""
        if limit is not None and limit <= 0:
            return []
        if not self._runbooks:
            return []

        alertnames = {
            alert.alertname.strip().lower()
            for alert in incident.alerts
            if alert.alertname and alert.alertname.strip()
        }
        searchable_text = _build_searchable_text(incident)
        incident_severity = incident.severity.strip().lower()

        ranked: list[tuple[int, int, int, str, RunbookTemplate]] = []
        for rb in self._runbooks:
            unique_alert_patterns = {
                name.strip().lower() for name in rb.match_alertnames if name and name.strip()
            }
            unique_keywords = {
                kw.strip().lower() for kw in rb.match_keywords if kw and kw.strip()
            }

            matched_alerts = sum(1 for name in unique_alert_patterns if name in alertnames)
            matched_keywords = sum(
                1 for kw in unique_keywords if _keyword_matches(kw, searchable_text)
            )
            base_score = 100 * matched_alerts + 15 * matched_keywords
            if base_score == 0:
                continue

            severity_bonus = (
                2 if incident_severity and incident_severity == rb.severity_hint.strip().lower() else 0
            )
            total_score = base_score + severity_bonus
            ranked.append(
                (-total_score, -matched_alerts, -matched_keywords, rb.runbook_id, rb)
            )

        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        selected = [item[4] for item in ranked]
        if limit is not None:
            return selected[:limit]
        return selected


def _first_non_empty(candidates: Sequence[str | None]) -> str | None:
    for item in candidates:
        if item is not None:
            stripped = item.strip()
            if stripped:
                return stripped
    return None


def _strip_port(instance: str | None) -> str | None:
    if not instance:
        return None
    cleaned = instance.strip()
    if not cleaned:
        return None
    if cleaned.startswith("["):
        end_bracket = cleaned.find("]")
        if end_bracket != -1:
            host_part = cleaned[1:end_bracket].strip()
            return host_part or None
        return cleaned.lstrip("[").strip() or None
    if cleaned.count(":") == 1:
        host_part = cleaned.split(":", 1)[0].strip()
        return host_part or None
    return cleaned


def extract_incident_context(incident: IncidentBundle) -> dict[str, str]:
    """Extract stripped, non-empty template context variables from an IncidentBundle."""
    service_raw = incident.service.strip()
    service = service_raw if service_raw else "unknown"

    namespaces: list[str | None] = []
    pods: list[str | None] = []
    containers: list[str | None] = []
    nodes: list[str | None] = []
    hosts: list[str | None] = []
    instances: list[str | None] = []
    mountpoints: list[str | None] = []
    domains: list[str | None] = []
    db_hosts: list[str | None] = []

    for alert in incident.alerts:
        namespaces.extend([alert.namespace, alert.labels.get("namespace")])
        pods.append(alert.labels.get("pod"))
        containers.append(alert.labels.get("container"))
        nodes.extend([alert.labels.get("node"), alert.labels.get("host")])
        hosts.extend([alert.labels.get("host"), alert.labels.get("node")])
        instances.extend([alert.instance, alert.labels.get("instance")])
        mountpoints.extend([alert.labels.get("mountpoint"), alert.labels.get("filesystem")])
        domains.append(alert.labels.get("domain"))
        db_hosts.append(alert.labels.get("db_host"))

    for entry in incident.logs:
        attrs = entry.attributes
        namespaces.append(attrs.get("namespace"))
        pods.append(attrs.get("pod"))
        containers.append(attrs.get("container"))
        nodes.extend([attrs.get("node"), attrs.get("host")])
        hosts.extend([attrs.get("host"), attrs.get("node")])
        instances.append(attrs.get("instance"))
        mountpoints.extend([attrs.get("mountpoint"), attrs.get("filesystem")])
        domains.append(attrs.get("domain"))
        db_hosts.append(attrs.get("db_host"))

    resolved_instance_raw = _first_non_empty(instances)
    resolved_node = (
        _first_non_empty(nodes)
        or _strip_port(resolved_instance_raw)
        or service
    )
    resolved_host = _first_non_empty(hosts) or resolved_node
    resolved_instance = resolved_instance_raw or resolved_node

    return {
        "service": service,
        "namespace": _first_non_empty(namespaces) or "default",
        "pod": _first_non_empty(pods) or service,
        "container": _first_non_empty(containers) or service,
        "node": resolved_node,
        "host": resolved_host,
        "instance": resolved_instance,
        "mountpoint": _first_non_empty(mountpoints) or "/var/lib/containerd",
        "domain": _first_non_empty(domains) or f"{service}.example.com",
        "db_host": _first_non_empty(db_hosts) or f"{service}-db",
    }


def render_template(template: str, context: Mapping[str, str]) -> str:
    """Safely replace known {key} placeholders without raising on unknown/JSON braces."""
    if not template:
        return ""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in context:
            return str(context[key])
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)
