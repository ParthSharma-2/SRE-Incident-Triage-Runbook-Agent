"""Log ingestion for JSONL, structured key-value, syslog, and plain-text log files."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from sre_triage_agent.ingestion import IngestionError
from sre_triage_agent.models import LogEntry, utc_now

LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]

_ISO_PREFIX_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+(?P<rest>.*)$"
)
_SYSLOG_PREFIX_RE = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<proc>[^\s\[:]+)(?:\[(?P<pid>\d+)\])?:\s*(?P<rest>.*)$"
)
_BRACKET_LEVEL_RE = re.compile(
    r"^\[?(?P<level>CRITICAL|FATAL|PANIC|ERROR|ERR|WARN|WARNING|INFO|DEBUG)\]?(?::|\s+)(?P<rest>.*)$",
    re.IGNORECASE,
)
_KV_TOKEN_RE = re.compile(
    r"""(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)=(?:"(?P<qval>[^"]*)"|'(?P<sqval>[^']*)'|(?P<uval>\S+))"""
)
_BRACKET_SERVICE_RE = re.compile(r"^\[(?P<svc>[A-Za-z0-9._-]+)\]\s*(?P<rest>.*)$")

_MONTH_MAP: dict[str, int] = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


class LogSource(Protocol):
    """Protocol for log sources returning normalized LogEntry objects."""

    def read(self) -> list[LogEntry]:
        """Read normalized log entries."""
        ...


class LogReadResult(BaseModel):
    """Result of reading a log file, including byte-offset cursor for incremental reads."""

    model_config = ConfigDict(frozen=True)

    entries: list[LogEntry]
    next_cursor: int
    total_lines_scanned: int


def _first_non_empty(*candidates: str | None, default: str = "unknown") -> str:
    for candidate in candidates:
        if candidate is not None:
            stripped = candidate.strip()
            if stripped:
                return stripped
    return default


def _normalize_level(raw_level: str | None, message: str = "") -> LogLevel:
    if raw_level:
        upper = raw_level.strip().upper()
        if upper in {"CRITICAL", "FATAL", "PANIC", "CRIT"}:
            return "CRITICAL"
        if upper in {"ERROR", "ERR"}:
            return "ERROR"
        if upper in {"WARN", "WARNING"}:
            return "WARN"
        if upper == "DEBUG":
            return "DEBUG"
        if upper == "INFO":
            return "INFO"

    upper_msg = message.upper()
    if any(tok in upper_msg for tok in ("CRITICAL", "FATAL", "PANIC", "OOMKILLED")):
        return "CRITICAL"
    if any(tok in upper_msg for tok in ("ERROR", "EXCEPTION", "TIMEOUTERROR", "FAILED", "NO SPACE LEFT")):
        return "ERROR"
    if any(tok in upper_msg for tok in ("WARN", "WARNING", "DISKPRESSURE", "DEGRADED")):
        return "WARN"
    if "DEBUG" in upper_msg:
        return "DEBUG"
    return "INFO"


def _parse_iso_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    if "," in candidate and "T" not in candidate:
        candidate = candidate.replace(",", ".", 1)
    elif "," in candidate:
        candidate = candidate.replace(",", ".", 1)
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
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_k8s_attributes(k8s: Any, target: dict[str, str]) -> str | None:
    if not isinstance(k8s, dict):
        return None
    for src_key, dst_key in (
        ("pod", "pod"),
        ("pod_name", "pod"),
        ("namespace", "namespace"),
        ("namespace_name", "namespace"),
        ("node", "node"),
        ("node_name", "node"),
        ("host", "host"),
        ("container", "container"),
        ("container_name", "container"),
    ):
        val = k8s.get(src_key)
        if val is not None and str(val).strip():
            target.setdefault(dst_key, str(val).strip())
    container = k8s.get("container_name") or k8s.get("container") or k8s.get("app")
    if container is not None and str(container).strip():
        return str(container).strip()
    return None


def _make_log_id(source_label: str, line_number: int, raw_line: str) -> str:
    digest = hashlib.sha256(
        f"{source_label}:{line_number}:{raw_line}".encode("utf-8", errors="replace")
    ).hexdigest()[:12]
    return f"log-{digest}"


class FileLogSource:
    """Reads JSONL, structured key-value, syslog, and plain-text log files."""

    def __init__(
        self,
        path: str | Path,
        *,
        default_service: str = "unknown",
        source_name: str | None = None,
        tail_lines: int | None = None,
        strict_json: bool = False,
        reference_year: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.default_service = default_service
        self.source_name = source_name or str(self.path)
        self.tail_lines = tail_lines
        self.strict_json = strict_json
        self.reference_year = reference_year or datetime.now(timezone.utc).year
        self.last_cursor: int = 0

    def read(self) -> list[LogEntry]:
        """Read all (or tailed) log entries from the beginning of the file."""
        return self.read_batch(cursor=0).entries

    def read_batch(self, cursor: int = 0) -> LogReadResult:
        """Read log entries incrementally from byte-offset cursor with truncation recovery."""
        if self.tail_lines is not None and self.tail_lines < 0:
            raise IngestionError(
                f"tail_lines must be >= 0, got {self.tail_lines}",
                source=self.source_name,
            )
        if cursor < 0:
            raise IngestionError(
                f"cursor must be >= 0, got {cursor}",
                source=self.source_name,
            )
        if not self.path.exists():
            raise IngestionError(
                f"Log file not found: {self.path}",
                source=self.source_name,
            )
        if self.path.is_dir():
            raise IngestionError(
                f"Log path is a directory, expected a file: {self.path}",
                source=self.source_name,
            )

        try:
            with open(self.path, "rb") as handle:
                file_size = os.fstat(handle.fileno()).st_size
                effective_cursor = 0 if cursor > file_size else cursor

                starting_line_no = 1
                if effective_cursor > 0:
                    starting_line_no = self._count_newlines_before(handle, effective_cursor) + 1
                    handle.seek(effective_cursor)

                total_scanned = 0
                if self.tail_lines == 0:
                    for _ in handle:
                        total_scanned += 1
                    next_cursor = handle.tell()
                    self.last_cursor = next_cursor
                    return LogReadResult(
                        entries=[],
                        next_cursor=next_cursor,
                        total_lines_scanned=total_scanned,
                    )

                collected: deque[tuple[int, str]] | list[tuple[int, str]]
                if self.tail_lines is not None:
                    collected = deque(maxlen=self.tail_lines)
                else:
                    collected = []

                for offset_idx, raw_bytes in enumerate(handle):
                    line_no = starting_line_no + offset_idx
                    total_scanned += 1
                    decoded = raw_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not decoded.strip():
                        continue
                    collected.append((line_no, decoded))

                next_cursor = handle.tell()
                self.last_cursor = next_cursor
        except OSError as exc:
            raise IngestionError(
                f"Failed to read log file {self.path}: {exc}",
                source=self.source_name,
                cause=exc,
            ) from exc

        entries = [self._parse_line(line_no, text) for line_no, text in collected]
        return LogReadResult(
            entries=entries,
            next_cursor=next_cursor,
            total_lines_scanned=total_scanned,
        )

    @staticmethod
    def _count_newlines_before(handle: Any, byte_limit: int) -> int:
        handle.seek(0)
        remaining = byte_limit
        newlines = 0
        chunk_size = 65536
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            newlines += chunk.count(b"\n")
            remaining -= len(chunk)
        return newlines

    def _parse_line(self, line_number: int, line_text: str) -> LogEntry:
        stripped = line_text.strip()
        is_json_candidate = stripped.startswith("{") or self.path.name.endswith(".jsonl")
        extra_attrs: dict[str, str] = {}

        if is_json_candidate:
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if self.strict_json:
                    raise IngestionError(
                        f"Invalid JSON log line at {self.source_name}:{line_number}: {exc}",
                        source=self.source_name,
                        cause=exc,
                    ) from exc
                extra_attrs["parse_warning"] = "invalid_json"
            else:
                if isinstance(payload, dict):
                    return self._build_entry_from_json(line_number, line_text, payload)
                if self.strict_json:
                    raise IngestionError(
                        f"JSON log line at {self.source_name}:{line_number} must be an object",
                        source=self.source_name,
                    )
                extra_attrs["parse_warning"] = "non_object_json"

        return self._build_entry_from_text(line_number, line_text, extra_attrs)

    def _build_entry_from_json(
        self,
        line_number: int,
        raw_line: str,
        payload: dict[str, Any],
    ) -> LogEntry:
        ts: datetime | None = None
        for ts_key in ("timestamp", "ts", "time", "@timestamp", "created_at"):
            if ts_key in payload:
                ts = _parse_iso_timestamp(str(payload[ts_key]))
                if ts is not None:
                    break
        timestamp = ts or utc_now()

        attributes: dict[str, str] = {}
        k8s_service = _extract_k8s_attributes(payload.get("kubernetes"), attributes)

        excluded_keys = {
            "timestamp",
            "ts",
            "time",
            "@timestamp",
            "created_at",
            "level",
            "severity",
            "lvl",
            "message",
            "msg",
            "event",
            "log",
            "kubernetes",
        }
        for key, value in payload.items():
            if key in excluded_keys or value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                attributes[str(key)] = str(value)

        raw_message = None
        for msg_key in ("message", "msg", "event", "log"):
            val = payload.get(msg_key)
            if val is not None and str(val).strip():
                raw_message = str(val).strip()
                break
        message = raw_message if raw_message is not None else json.dumps(payload, sort_keys=True)

        raw_level = None
        for lvl_key in ("level", "severity", "lvl"):
            val = payload.get(lvl_key)
            if val is not None and str(val).strip():
                raw_level = str(val)
                break
        level = _normalize_level(raw_level, message)

        service = _first_non_empty(self.default_service, default="unknown")
        for svc_key in ("service", "app", "component", "logger", "container"):
            val = payload.get(svc_key)
            if val is not None and str(val).strip():
                service = str(val).strip()
                break
        else:
            if k8s_service:
                service = k8s_service

        return LogEntry(
            log_id=_make_log_id(self.source_name, line_number, raw_line),
            timestamp=timestamp,
            level=level,
            service=service,
            message=message,
            source=self.source_name,
            line_number=line_number,
            attributes=attributes,
        )

    def _build_entry_from_text(
        self,
        line_number: int,
        raw_line: str,
        initial_attrs: dict[str, str],
    ) -> LogEntry:
        attributes = dict(initial_attrs)
        timestamp: datetime | None = None
        rest = raw_line.strip()
        syslog_proc: str | None = None
        bracket_service: str | None = None

        iso_match = _ISO_PREFIX_RE.match(rest)
        if iso_match:
            parsed_ts = _parse_iso_timestamp(iso_match.group("ts"))
            if parsed_ts is not None:
                timestamp = parsed_ts
                rest = iso_match.group("rest").strip()
        else:
            syslog_match = _SYSLOG_PREFIX_RE.match(rest)
            if syslog_match:
                month = _MONTH_MAP.get(syslog_match.group("mon"))
                if month is not None:
                    day = int(syslog_match.group("day"))
                    hh, mm, ss = (int(part) for part in syslog_match.group("time").split(":"))
                    try:
                        timestamp = datetime(
                            self.reference_year,
                            month,
                            day,
                            hh,
                            mm,
                            ss,
                            tzinfo=timezone.utc,
                        )
                    except ValueError:
                        timestamp = None
                host = syslog_match.group("host").strip()
                syslog_proc = syslog_match.group("proc").strip()
                if host:
                    attributes.setdefault("host", host)
                if syslog_proc:
                    attributes.setdefault("process", syslog_proc)
                if syslog_match.group("pid"):
                    attributes.setdefault("pid", syslog_match.group("pid"))
                rest = syslog_match.group("rest").strip()

        raw_level: str | None = None
        lvl_match = _BRACKET_LEVEL_RE.match(rest)
        if lvl_match:
            raw_level = lvl_match.group("level")
            rest = lvl_match.group("rest").strip()

        svc_match = _BRACKET_SERVICE_RE.match(rest)
        if svc_match:
            bracket_service = svc_match.group("svc").strip()
            rest = svc_match.group("rest").strip()

        kv_matches = list(_KV_TOKEN_RE.finditer(rest))
        for match in kv_matches:
            key = match.group("key")
            value = (
                match.group("qval")
                if match.group("qval") is not None
                else match.group("sqval")
                if match.group("sqval") is not None
                else match.group("uval")
            )
            if value is not None:
                attributes[key] = value

        message = attributes.get("msg") or attributes.get("message")
        if not message:
            cleaned = " ".join(_KV_TOKEN_RE.sub("", rest).split())
            message = cleaned if cleaned else rest

        level = _normalize_level(raw_level or attributes.get("level"), f"{rest} {message}")
        service = _first_non_empty(
            attributes.get("service"),
            attributes.get("app"),
            attributes.get("component"),
            bracket_service,
            syslog_proc,
            self.default_service,
            default="unknown",
        )

        return LogEntry(
            log_id=_make_log_id(self.source_name, line_number, raw_line),
            timestamp=timestamp or utc_now(),
            level=level,
            service=service,
            message=message,
            source=self.source_name,
            line_number=line_number,
            attributes=attributes,
        )


class BundledLogSource:
    """Reads logs from a named scenario directory under samples/scenarios/."""

    def __init__(
        self,
        scenario_name: str,
        *,
        scenarios_dir: Path | None = None,
        tail_lines: int | None = None,
    ) -> None:
        from sre_triage_agent.ingestion import (
            find_scenario_log_file,
            resolve_scenarios_dir,
        )

        self.scenario_name = scenario_name
        base_dir = resolve_scenarios_dir(scenarios_dir)
        scenario_dir = base_dir / scenario_name
        if not scenario_dir.is_dir():
            raise IngestionError(
                f"Unknown bundled scenario: '{scenario_name}'",
                source=str(scenario_dir),
            )
        self.log_path = find_scenario_log_file(scenario_dir)
        self._file_source = FileLogSource(
            self.log_path,
            source_name=f"bundled:{scenario_name}/{self.log_path.name}",
            tail_lines=tail_lines,
        )

    def read(self) -> list[LogEntry]:
        """Read all normalized log entries for this bundled scenario."""
        return self._file_source.read()
