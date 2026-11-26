"""Shared LLM provider protocol, typed errors, prompt builder, JSON recovery, and grounding."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from sre_triage_agent.models import (
    AlertEvent,
    EvidenceRef,
    IncidentBundle,
    LogEntry,
    RemediationStep,
    RunbookTemplate,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*\r?\n?(.*?)\r?\n?```", re.IGNORECASE | re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"^```(?:json)?\s*\r?\n?", re.IGNORECASE)

_VALID_SEVERITIES = frozenset({"critical", "high", "warning", "medium", "info", "low"})
_SEVERITY_ALIASES: dict[str, str] = {
    "warn": "warning",
    "error": "high",
    "fatal": "critical",
    "err": "high",
}
_VALID_RISK_LEVELS = frozenset({"low", "medium", "high"})

_LOG_PRIORITY: dict[str, int] = {
    "CRITICAL": 4,
    "ERROR": 3,
    "WARN": 2,
    "INFO": 1,
    "DEBUG": 0,
}


class LLMProviderError(Exception):
    """Base typed exception for all LLM provider configuration, transport, parse, and grounding errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        cause: Exception | None = None,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.cause = cause
        self.raw_response = raw_response


class LLMConfigError(LLMProviderError):
    """Raised when LLM provider configuration (URL, model, timeout, provider type) is invalid."""


class LLMConnectionError(LLMProviderError):
    """Raised on network failures, timeouts, non-2xx HTTP status, or provider API error payloads."""


class LLMResponseParseError(LLMProviderError):
    """Raised when an LLM response is empty, non-JSON, or violates the expected triage schema."""


class LLMGroundingError(LLMProviderError):
    """Raised when remediation steps fail evidence grounding against the IncidentBundle."""


class LLMTriageResponse(BaseModel):
    """Normalized, evidence-grounded response returned by an LLMProvider."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    severity: str
    root_cause_hypothesis: str
    summary: str
    remediation_steps: list[RemediationStep] = Field(default_factory=list)
    provider_used: str
    model_used: str
    fallback_used: bool = False
    fallback_reason: str | None = None


class ProviderHealth(BaseModel):
    """Connectivity and model-readiness status for an LLMProvider."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    provider: str
    model: str
    base_url: str | None = None
    healthy: bool
    model_available: bool | None = None
    detail: str
    latency_ms: float | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol implemented by all open-source, fallback, and mock LLM providers."""

    provider_name: str
    model_name: str

    def triage(
        self,
        bundle: IncidentBundle,
        runbooks: Sequence[RunbookTemplate],
    ) -> LLMTriageResponse:
        """Analyze an IncidentBundle and candidate RunbookTemplates into grounded steps."""

    def healthcheck(self) -> ProviderHealth:
        """Check endpoint connectivity and configured model availability."""


def validate_base_url(base_url: str, *, provider: str) -> str:
    """Validate and normalize an HTTP/HTTPS base URL, raising LLMConfigError on failure."""
    if not isinstance(base_url, str):
        raise LLMConfigError(f"{provider} base_url must be a string", provider=provider)
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        raise LLMConfigError(f"{provider} base_url must not be empty", provider=provider)
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise LLMConfigError(
            f"Invalid {provider} base_url '{base_url}': expected http:// or https:// URL with host",
            provider=provider,
        )
    return cleaned


def validate_model_and_timeout(
    model: str,
    timeout_s: float,
    *,
    provider: str,
) -> tuple[str, float]:
    """Validate model name and positive finite timeout_s, raising LLMConfigError on failure."""
    if not isinstance(model, str) or not model.strip():
        raise LLMConfigError(f"{provider} model must be a non-empty string", provider=provider)
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or float(timeout_s) <= 0.0
    ):
        raise LLMConfigError(
            f"{provider} timeout_s must be a positive finite number, got {timeout_s!r}",
            provider=provider,
            model=model.strip(),
        )
    return model.strip(), float(timeout_s)


def _select_prioritized_logs(logs: Sequence[LogEntry], max_logs: int) -> list[LogEntry]:
    if max_logs <= 0 or not logs:
        return []
    if len(logs) <= max_logs:
        return sorted(logs, key=lambda entry: (entry.timestamp, entry.line_number, entry.log_id))

    ranked = sorted(
        logs,
        key=lambda entry: (
            -_LOG_PRIORITY.get(entry.level, 0),
            entry.timestamp,
            entry.line_number,
            entry.log_id,
        ),
    )
    selected = ranked[:max_logs]
    return sorted(selected, key=lambda entry: (entry.timestamp, entry.line_number, entry.log_id))


def build_triage_messages(
    bundle: IncidentBundle,
    runbooks: Sequence[RunbookTemplate],
    *,
    max_logs: int = 30,
) -> list[dict[str, str]]:
    """Build system + user chat messages compatible with Ollama and OpenAI-compatible APIs."""
    selected_logs = _select_prioritized_logs(bundle.logs, max_logs=max_logs)

    system_prompt = (
        "You are an expert DevOps/SRE incident triage assistant. "
        "Analyze the provided incident alerts, logs, and candidate runbooks, and respond with ONLY a single JSON object.\n"
        "Required top-level JSON keys:\n"
        '  - "severity": one of "critical", "high", "warning", "medium", "info", "low"\n'
        '  - "root_cause_hypothesis": concise technical explanation of the root cause\n'
        '  - "summary": concise operational summary of impact and remediation plan\n'
        '  - "remediation_steps": non-empty list of concrete steps (or [] only if both alerts and logs are empty)\n'
        "Each remediation_steps item MUST include:\n"
        '  - "step_id": string identifier\n'
        '  - "runbook_id": matching candidate runbook_id, or "ad_hoc_diagnostic" if none match\n'
        '  - "title": short action title\n'
        '  - "description": clear explanation of the step\n'
        '  - "risk_level": one of "low", "medium", "high"\n'
        '  - "command_preview": concrete command with service/namespace/pod/node/domain/mountpoint filled in\n'
        '  - "target_resource": concrete target resource identifier\n'
        '  - "expected_effect": expected operational outcome\n'
        '  - "diff_template": unified diff preview or empty string ""\n'
        '  - "evidence_refs": non-empty list of objects with keys {"kind": "alert"|"log", "ref_id": "...", "excerpt": "...", "reason": "..."}\n'
        "CRITICAL GROUNDING RULE: Every evidence_refs item MUST cite an exact alert_id from valid_alert_ids "
        '(when kind="alert") or an exact log_id from valid_log_ids (when kind="log"). '
        "NEVER fabricate or modify alert_id or log_id values. "
        "Do not wrap output in prose; return pure JSON only."
    )

    payload = {
        "incident": {
            "incident_id": bundle.incident_id,
            "title": bundle.title,
            "service": bundle.service,
            "severity": bundle.severity,
            "source_mode": bundle.source_mode,
            "scenario_name": bundle.scenario_name,
        },
        "valid_alert_ids": [alert.alert_id for alert in bundle.alerts],
        "alerts": [
            {
                "alert_id": alert.alert_id,
                "alertname": alert.alertname,
                "status": alert.status,
                "severity": alert.severity,
                "service": alert.service,
                "namespace": alert.namespace,
                "instance": alert.instance,
                "summary": alert.summary,
                "description": alert.description,
                "labels": alert.labels,
            }
            for alert in bundle.alerts
        ],
        "valid_log_ids": [entry.log_id for entry in selected_logs],
        "logs": [
            {
                "log_id": entry.log_id,
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level,
                "service": entry.service,
                "line_number": entry.line_number,
                "message": entry.message,
                "attributes": entry.attributes,
            }
            for entry in selected_logs
        ],
        "candidate_runbooks": [
            {
                "runbook_id": rb.runbook_id,
                "title": rb.title,
                "description": rb.description,
                "severity_hint": rb.severity_hint,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "title": step.title,
                        "description": step.description,
                        "risk_level": step.risk_level,
                        "command_template": step.command_template,
                        "target_resource_template": step.target_resource_template,
                        "expected_effect": step.expected_effect,
                        "diff_template": step.diff_template,
                    }
                    for step in rb.steps
                ],
            }
            for rb in runbooks
        ],
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _triage_payload_score(obj: dict[str, Any]) -> int:
    """Score how strongly a parsed JSON dict resembles a triage response payload."""
    score = 0
    if "remediation_steps" in obj:
        score += 2
    if "root_cause_hypothesis" in obj:
        score += 1
    if "summary" in obj:
        score += 1
    return score


def extract_json_object(
    raw_text: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Extract a top-level JSON object from raw LLM output, handling markdown fences and prose."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise LLMResponseParseError(
            "LLM returned an empty response",
            provider=provider,
            model=model,
            raw_response=raw_text if isinstance(raw_text, str) else None,
        )

    cleaned = raw_text.lstrip("\ufeff").strip()
    snippet = cleaned[:500]

    # 1. Direct JSON object parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Markdown code fences (closed or unclosed leading fence)
    candidates: list[str] = [match.group(1).strip() for match in _FENCE_RE.finditer(cleaned)]
    if _OPEN_FENCE_RE.match(cleaned):
        candidates.append(_OPEN_FENCE_RE.sub("", cleaned, count=1).strip())

    fenced_dicts: list[dict[str, Any]] = []
    for block in candidates:
        if not block:
            continue
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                fenced_dicts.append(parsed)
        except json.JSONDecodeError:
            pass

    if fenced_dicts:
        best_fenced = max(fenced_dicts, key=_triage_payload_score)
        if _triage_payload_score(best_fenced) >= 3:
            return best_fenced

    # 3. Balanced object scan via JSONDecoder.raw_decode
    decoder = json.JSONDecoder()
    scanned_dicts: list[dict[str, Any]] = []
    idx = 0
    length = len(cleaned)
    while idx < length:
        if cleaned[idx] != "{":
            idx += 1
            continue
        try:
            parsed, end_idx = decoder.raw_decode(cleaned, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(parsed, dict):
            scanned_dicts.append(parsed)
            idx = max(idx + 1, end_idx)
        else:
            idx += 1

    all_dicts = fenced_dicts + scanned_dicts
    if all_dicts:
        best = max(all_dicts, key=_triage_payload_score)
        if _triage_payload_score(best) > 0:
            return best
        return all_dicts[0]

    raise LLMResponseParseError(
        "Failed to extract a valid JSON object from LLM response",
        provider=provider,
        model=model,
        raw_response=snippet,
    )


def normalize_severity(raw_severity: Any, fallback_severity: str = "warning") -> str:
    if isinstance(raw_severity, str):
        candidate = raw_severity.strip().lower()
        candidate = _SEVERITY_ALIASES.get(candidate, candidate)
        if candidate in _VALID_SEVERITIES:
            return candidate
    fb = fallback_severity.strip().lower()
    fb = _SEVERITY_ALIASES.get(fb, fb)
    return fb if fb in _VALID_SEVERITIES else "warning"


_normalize_severity = normalize_severity


def _default_alert_excerpt(alert: AlertEvent) -> str:
    text = alert.summary.strip() or alert.description.strip()
    if text:
        return text[:240]
    return f"{alert.alertname} ({alert.severity}) on {alert.service}"


def _default_alert_reason(alert: AlertEvent) -> str:
    return f"Alert {alert.alertname} ({alert.severity}, status={alert.status}) on {alert.service}"


def _default_log_excerpt(entry: LogEntry) -> str:
    return f"[{entry.level}] {entry.message.strip()}"[:240]


def _default_log_reason(entry: LogEntry) -> str:
    return f"Log line {entry.line_number} from {entry.source} ({entry.level})"


def validate_and_ground_response(
    payload: dict[str, Any],
    bundle: IncidentBundle,
    *,
    provider: str,
    model: str,
) -> LLMTriageResponse:
    """Validate raw LLM JSON and strictly ground every EvidenceRef against IncidentBundle IDs."""
    if not isinstance(payload, dict):
        raise LLMResponseParseError(
            "LLM payload must be a JSON object",
            provider=provider,
            model=model,
        )

    raw_hypothesis = payload.get("root_cause_hypothesis")
    raw_summary = payload.get("summary")
    if not isinstance(raw_hypothesis, str) or not raw_hypothesis.strip():
        raise LLMResponseParseError(
            "LLM response missing non-empty 'root_cause_hypothesis'",
            provider=provider,
            model=model,
        )
    if not isinstance(raw_summary, str) or not raw_summary.strip():
        raise LLMResponseParseError(
            "LLM response missing non-empty 'summary'",
            provider=provider,
            model=model,
        )

    severity = _normalize_severity(payload.get("severity"), bundle.severity)
    raw_steps = payload.get("remediation_steps")
    if not isinstance(raw_steps, list):
        raise LLMResponseParseError(
            "LLM response 'remediation_steps' must be a JSON array",
            provider=provider,
            model=model,
        )

    alert_map = {alert.alert_id: alert for alert in bundle.alerts}
    log_map = {entry.log_id: entry for entry in bundle.logs}

    if not alert_map and not log_map:
        return LLMTriageResponse(
            severity=severity,
            root_cause_hypothesis=raw_hypothesis.strip(),
            summary=raw_summary.strip(),
            remediation_steps=[],
            provider_used=provider,
            model_used=model,
            fallback_used=False,
            fallback_reason=None,
        )

    if len(raw_steps) == 0:
        raise LLMResponseParseError(
            "LLM returned zero remediation_steps for non-empty incident",
            provider=provider,
            model=model,
        )

    grounded_steps: list[RemediationStep] = []
    for idx, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise LLMResponseParseError(
                f"remediation_steps[{idx - 1}] must be an object",
                provider=provider,
                model=model,
            )

        raw_title = raw_step.get("title")
        raw_cmd = raw_step.get("command_preview")
        if not isinstance(raw_title, str) or not raw_title.strip():
            raise LLMResponseParseError(
                f"remediation_steps[{idx - 1}] missing non-empty 'title'",
                provider=provider,
                model=model,
            )
        if not isinstance(raw_cmd, str) or not raw_cmd.strip():
            raise LLMResponseParseError(
                f"remediation_steps[{idx - 1}] missing non-empty 'command_preview'",
                provider=provider,
                model=model,
            )

        raw_risk = raw_step.get("risk_level")
        if not isinstance(raw_risk, str) or raw_risk.strip().lower() not in _VALID_RISK_LEVELS:
            raise LLMResponseParseError(
                f"remediation_steps[{idx - 1}] has invalid 'risk_level': {raw_risk!r}",
                provider=provider,
                model=model,
            )
        risk_level: Literal["low", "medium", "high"] = raw_risk.strip().lower()  # type: ignore[assignment]

        raw_refs = raw_step.get("evidence_refs", [])
        if not isinstance(raw_refs, list):
            raise LLMResponseParseError(
                f"remediation_steps[{idx - 1}].evidence_refs must be a list",
                provider=provider,
                model=model,
            )

        valid_refs: list[EvidenceRef] = []
        seen_keys: set[tuple[str, str]] = set()
        for raw_ref in raw_refs:
            if not isinstance(raw_ref, dict):
                continue
            kind = str(raw_ref.get("kind") or "").strip().lower()
            ref_id = str(raw_ref.get("ref_id") or "").strip()
            if not ref_id:
                continue

            if kind == "alert" and ref_id in alert_map:
                key = ("alert", ref_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                alert = alert_map[ref_id]
                excerpt = str(raw_ref.get("excerpt") or "").strip() or _default_alert_excerpt(alert)
                reason = str(raw_ref.get("reason") or "").strip() or _default_alert_reason(alert)
                valid_refs.append(
                    EvidenceRef(kind="alert", ref_id=ref_id, excerpt=excerpt, reason=reason)
                )
            elif kind == "log" and ref_id in log_map:
                key = ("log", ref_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                entry = log_map[ref_id]
                excerpt = str(raw_ref.get("excerpt") or "").strip() or _default_log_excerpt(entry)
                reason = str(raw_ref.get("reason") or "").strip() or _default_log_reason(entry)
                valid_refs.append(
                    EvidenceRef(kind="log", ref_id=ref_id, excerpt=excerpt, reason=reason)
                )

        if not valid_refs:
            # Drop ungrounded step instead of fabricating citations.
            continue

        title = raw_title.strip()
        command_preview = raw_cmd.strip()
        step_id = str(raw_step.get("step_id") or "").strip() or f"step-{idx}"
        runbook_id = str(raw_step.get("runbook_id") or "").strip() or "ad_hoc_diagnostic"
        description = str(raw_step.get("description") or "").strip() or title
        target_resource = str(raw_step.get("target_resource") or "").strip() or bundle.service
        expected_effect = str(raw_step.get("expected_effect") or "").strip() or f"Execute: {title}"
        diff_template = str(raw_step.get("diff_template") or "")

        grounded_steps.append(
            RemediationStep(
                step_id=step_id,
                runbook_id=runbook_id,
                title=title,
                description=description,
                risk_level=risk_level,
                command_preview=command_preview,
                target_resource=target_resource,
                expected_effect=expected_effect,
                diff_template=diff_template,
                evidence_refs=valid_refs,
            )
        )

    if not grounded_steps:
        raise LLMGroundingError(
            "All LLM remediation steps failed evidence grounding against IncidentBundle",
            provider=provider,
            model=model,
        )

    return LLMTriageResponse(
        severity=severity,
        root_cause_hypothesis=raw_hypothesis.strip(),
        summary=raw_summary.strip(),
        remediation_steps=grounded_steps,
        provider_used=provider,
        model_used=model,
        fallback_used=False,
        fallback_reason=None,
    )


def validate_evidence_grounding(
    steps: Sequence[RemediationStep],
    bundle: IncidentBundle,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> list[RemediationStep]:
    """Defense-in-depth verification that every RemediationStep cites real bundle IDs."""
    valid_alert_ids = {alert.alert_id for alert in bundle.alerts}
    valid_log_ids = {entry.log_id for entry in bundle.logs}

    if not valid_alert_ids and not valid_log_ids:
        if steps:
            raise LLMGroundingError(
                "Remediation steps cannot cite evidence on an empty IncidentBundle",
                provider=provider,
                model=model,
            )
        return []

    if not steps:
        raise LLMGroundingError(
            "No grounded remediation steps returned for non-empty IncidentBundle",
            provider=provider,
            model=model,
        )

    verified: list[RemediationStep] = []
    for step in steps:
        if not step.evidence_refs:
            raise LLMGroundingError(
                f"RemediationStep '{step.step_id}' has no evidence_refs",
                provider=provider,
                model=model,
            )
        for ref in step.evidence_refs:
            if ref.kind == "alert":
                if ref.ref_id not in valid_alert_ids:
                    raise LLMGroundingError(
                        f"Ungrounded alert evidence ref_id '{ref.ref_id}' in step '{step.step_id}'",
                        provider=provider,
                        model=model,
                    )
            elif ref.kind == "log":
                if ref.ref_id not in valid_log_ids:
                    raise LLMGroundingError(
                        f"Ungrounded log evidence ref_id '{ref.ref_id}' in step '{step.step_id}'",
                        provider=provider,
                        model=model,
                    )
            if not ref.excerpt.strip() or not ref.reason.strip():
                raise LLMGroundingError(
                    f"EvidenceRef '{ref.ref_id}' in step '{step.step_id}' has empty excerpt or reason",
                    provider=provider,
                    model=model,
                )
        verified.append(step)

    return verified
