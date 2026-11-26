"""Zero-side-effect simulated dry-run executor with shell injection sanitization.

Strictly pure string simulation: never imports subprocess, os, socket, or HTTP clients.
"""

from __future__ import annotations

import re
import shlex
import uuid
from typing import Any, Literal

from sre_triage_agent.models import DryRunOutcome, RemediationStep, TriageResult, utc_now

_UNSAFE_SHELL_RE = re.compile(r"(?:&&|\|\||<\(|>\(|[;|&`<>#\r\n$]|\$\()")
_UNSAFE_METADATA_RE = re.compile(r"[;|&`<>#\r\n'\"$]")
_UNSAFE_UNQUOTED_RE = re.compile(r"(?:&&|\|\||<\(|>\(|[;|&`<>#\r\n'\"$]|\$\()")
_UNRENDERED_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9._:@=-]+$")
_SAFE_ID_SLASH_RE = re.compile(r"^[a-zA-Z0-9._:@=/-]+$")


def sanitize_identifier(value: str, *, allow_slash: bool = False) -> tuple[str, bool]:
    """Validate and normalize a target resource or identifier without path traversal."""
    cleaned = value.strip()
    if not cleaned:
        return "", False
    if ".." in cleaned.split("/") or "/.." in cleaned or "../" in cleaned:
        return cleaned, False
    if _UNSAFE_METADATA_RE.search(cleaned):
        return cleaned, False
    pattern = _SAFE_ID_SLASH_RE if allow_slash else _SAFE_ID_RE
    if not pattern.match(cleaned):
        return cleaned, False
    return cleaned, True


def _quote_context_at(text: str, pos: int) -> Literal["none", "single", "double"]:
    """Determine quote context ('none', 'single', or 'double') at character index pos."""
    state: Literal["none", "single", "double"] = "none"
    i = 0
    while i < pos:
        ch = text[i]
        if state == "none":
            if ch == "\\" and i + 1 < pos:
                i += 1
            elif ch == "'":
                state = "single"
            elif ch == '"':
                state = "double"
        elif state == "single":
            if ch == "'":
                state = "none"
        elif state == "double":
            if ch == "\\" and i + 1 < pos:
                i += 1
            elif ch == '"':
                state = "none"
        i += 1
    return state


def _escape_double_quoted_value(raw_val: str) -> str:
    """Escape a raw metadata string for safe embedding inside POSIX double quotes."""
    return (
        raw_val.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("$", "\\$")
    )


def _escape_single_quoted_value(raw_val: str) -> str:
    """Escape embedded single quotes for safe embedding inside POSIX single quotes."""
    return raw_val.replace("'", "'\"'\"'")


def _sanitize_interpolated_values(
    command: str,
    unsafe_values: list[str],
) -> str:
    """Sanitize interpolated metadata values in a rendered command preview."""
    candidates = [val for val in sorted(set(unsafe_values), key=len, reverse=True) if val]
    if not candidates or not command:
        return command

    pattern = re.compile("|".join(re.escape(val) for val in candidates))
    out: list[str] = []
    pos = 0
    for match in pattern.finditer(command):
        if match.start() > pos:
            out.append(command[pos : match.start()])
        prefix = "".join(out)
        ctx = _quote_context_at(prefix, len(prefix))
        raw_val = match.group(0)
        if ctx == "double":
            out.append(_escape_double_quoted_value(raw_val))
        elif ctx == "single":
            out.append(_escape_single_quoted_value(raw_val))
        else:
            out.append(shlex.quote(raw_val))
        pos = match.end()

    if pos < len(command):
        out.append(command[pos:])
    return "".join(out)


def _is_unsafe_metadata(value: str) -> bool:
    return bool(value and (_UNSAFE_METADATA_RE.search(value) or ".." in value))


def _metadata_candidates(raw_val: str | None) -> list[str]:
    """Return unique non-empty raw, stripped, and port-stripped metadata candidates."""
    if not raw_val:
        return []
    candidates: list[str] = [raw_val]
    stripped = raw_val.strip()
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    if stripped:
        port_stripped: str | None = None
        if stripped.startswith("["):
            end_bracket = stripped.find("]")
            if end_bracket != -1:
                port_stripped = stripped[1:end_bracket].strip()
            else:
                port_stripped = stripped.lstrip("[").strip()
        elif stripped.count(":") == 1:
            port_stripped = stripped.split(":", 1)[0].strip()
        else:
            digit_port_stripped = re.sub(r":\d+$", "", stripped).strip()
            if digit_port_stripped != stripped:
                port_stripped = digit_port_stripped

        if port_stripped and port_stripped not in candidates:
            candidates.append(port_stripped)

    return candidates


def _has_dangerous_command_injection(command: str) -> bool:
    """Detect unquoted/dangerous shell injection tokens or unclosed quotes across any runbook."""
    if "\r" in command or "\n" in command:
        return True

    state: Literal["none", "single", "double"] = "none"
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if state == "none":
            if ch == "\\":
                if i + 1 >= n:
                    return True
                i += 2
                continue
            if ch == "'":
                state = "single"
            elif ch == '"':
                state = "double"
            elif ch in {";", "`", "$", "#"}:
                return True
            elif command[i : i + 2] in {"||", "<(", ">("}:
                return True
            elif ch == "&":
                if command[i : i + 2] == "&&":
                    i += 2
                    continue
                return True
        elif state == "single":
            if ch == "'":
                state = "none"
        elif state == "double":
            if ch == "\\":
                if i + 1 >= n:
                    return True
                i += 2
                continue
            if ch == '"':
                state = "none"
            elif ch in {"`", "$"}:
                return True
        i += 1

    return state != "none"


def sanitize_command_preview(command: str) -> str:
    """Escape or single-quote any unquoted shell metacharacters in a command preview."""
    if not command:
        return ""

    out: list[str] = []
    state: Literal["none", "single", "double"] = "none"
    i = 0
    n = len(command)
    seg_start = 0
    dq_inner: list[str] = []

    while i < n:
        ch = command[i]
        if state == "none":
            if ch == "\\":
                if i + 1 < n:
                    i += 2
                    continue
                i += 1
                continue
            if ch == "'":
                if i > seg_start:
                    out.append(_quote_unsafe_tail(command[seg_start:i]))
                state = "single"
                seg_start = i
            elif ch == '"':
                if i > seg_start:
                    out.append(_quote_unsafe_tail(command[seg_start:i]))
                state = "double"
                seg_start = i
                dq_inner = []
        elif state == "single":
            if ch == "'":
                sq_text = command[seg_start : i + 1].replace("\r", " ").replace("\n", " ")
                out.append(sq_text)
                state = "none"
                seg_start = i + 1
        elif state == "double":
            if ch == "\\":
                if i + 1 < n:
                    escaped_pair = command[i : i + 2].replace("\r", " ").replace("\n", " ")
                    dq_inner.append(escaped_pair)
                    i += 2
                    continue
                i += 1
                continue
            if ch == '"':
                out.append(f'"{"".join(dq_inner)}"')
                state = "none"
                seg_start = i + 1
            elif ch in ("`", "$"):
                dq_inner.append(f"\\{ch}")
            elif ch in ("\r", "\n"):
                dq_inner.append(" ")
            else:
                dq_inner.append(ch)
        i += 1

    if state == "none":
        if seg_start < n:
            out.append(_quote_unsafe_tail(command[seg_start:], at_eof=True))
    else:
        tail = command[seg_start:].replace("\r", " ").replace("\n", " ")
        out.append(shlex.quote(tail))

    return "".join(out)


def _has_trailing_unescaped_backslash(text: str) -> bool:
    count = 0
    for ch in reversed(text):
        if ch == "\\":
            count += 1
        else:
            break
    return count % 2 == 1


def _quote_unsafe_tail(segment: str, *, at_eof: bool = False) -> str:
    match = _UNSAFE_UNQUOTED_RE.search(segment)
    if match is None:
        if at_eof and _has_trailing_unescaped_backslash(segment):
            token_start = len(segment) - 1
            while token_start > 0 and not segment[token_start - 1].isspace():
                token_start -= 1
            safe_prefix = segment[:token_start]
            unsafe_tail = segment[token_start:].replace("\r", " ").replace("\n", " ").strip()
            return f"{safe_prefix}{shlex.quote(unsafe_tail)}"
        return segment
    idx = match.start()
    token_start = idx
    while token_start > 0 and not segment[token_start - 1].isspace():
        token_start -= 1
    safe_prefix = segment[:token_start]
    raw_tail = segment[token_start:]
    trailing_ws = " " if (not at_eof and raw_tail.endswith((" ", "\t"))) else ""
    unsafe_tail = raw_tail.replace("\r", " ").replace("\n", " ").strip()
    if not unsafe_tail:
        return safe_prefix + trailing_ws
    return f"{safe_prefix}{shlex.quote(unsafe_tail)}{trailing_ws}"


class DryRunExecutor:
    """Simulates execution of RemediationSteps without touching real infrastructure."""

    def __init__(
        self,
        *,
        failed_prechecks: dict[str, str] | None = None,
        simulated_state: dict[str, Any] | None = None,
    ) -> None:
        self.failed_prechecks = dict(failed_prechecks or {})
        self.simulated_state = dict(simulated_state or {})

    def simulate(
        self,
        step: RemediationStep,
        result: TriageResult | None = None,
        *,
        run_id: str | None = None,
        incident_id: str | None = None,
    ) -> DryRunOutcome:
        """Run prechecks and simulate command execution with zero real-infra side effects."""
        eff_run_id = run_id or (result.run_id if result is not None else "run-dryrun")
        eff_incident_id = incident_id or (
            result.incident_id if result is not None else "inc-dryrun"
        )

        raw_cmd = step.command_preview
        safe_target, target_ok = sanitize_identifier(step.target_resource, allow_slash=True)

        failure_reasons: list[str] = []
        if not raw_cmd.strip():
            failure_reasons.append("Empty command preview")
        if _UNRENDERED_PLACEHOLDER_RE.search(raw_cmd):
            failure_reasons.append("Unrendered template placeholder in command preview")
        if not target_ok:
            failure_reasons.append(f"Unsafe target_resource '{step.target_resource}'")

        unsafe_metadata: list[str] = []
        if result is not None:
            for alert in result.alerts:
                for label_key, label_val in alert.labels.items():
                    matched = [
                        cand
                        for cand in _metadata_candidates(label_val)
                        if _is_unsafe_metadata(cand) and cand in raw_cmd
                    ]
                    if matched:
                        unsafe_metadata.extend(matched)
                        failure_reasons.append(
                            f"Unsafe characters in alert label '{label_key}'"
                        )
                for field_name, field_val in (
                    ("service", alert.service),
                    ("namespace", alert.namespace),
                    ("instance", alert.instance),
                ):
                    matched = [
                        cand
                        for cand in _metadata_candidates(field_val)
                        if _is_unsafe_metadata(cand) and cand in raw_cmd
                    ]
                    if matched:
                        unsafe_metadata.extend(matched)
                        failure_reasons.append(
                            f"Unsafe characters in alert {field_name}"
                        )
            for log_entry in result.logs:
                matched_svc = [
                    cand
                    for cand in _metadata_candidates(log_entry.service)
                    if _is_unsafe_metadata(cand) and cand in raw_cmd
                ]
                if matched_svc:
                    unsafe_metadata.extend(matched_svc)
                    failure_reasons.append("Unsafe characters in log service")
                for attr_key, attr_val in log_entry.attributes.items():
                    matched_attr = [
                        cand
                        for cand in _metadata_candidates(attr_val)
                        if _is_unsafe_metadata(cand) and cand in raw_cmd
                    ]
                    if matched_attr:
                        unsafe_metadata.extend(matched_attr)
                        failure_reasons.append(
                            f"Unsafe characters in log attribute '{attr_key}'"
                        )

        safe_cmd = _sanitize_interpolated_values(raw_cmd, unsafe_metadata) if unsafe_metadata else raw_cmd

        if step.runbook_id == "ad_hoc_diagnostic":
            if _UNSAFE_SHELL_RE.search(raw_cmd) or _has_dangerous_command_injection(raw_cmd):
                failure_reasons.append("Unsafe shell metacharacters in ad-hoc command preview")
                if _UNSAFE_SHELL_RE.search(safe_cmd) or _has_dangerous_command_injection(safe_cmd):
                    safe_cmd = sanitize_command_preview(safe_cmd)
        elif _has_dangerous_command_injection(raw_cmd):
            failure_reasons.append("Unsafe shell metacharacters in command preview")
            if _has_dangerous_command_injection(safe_cmd):
                safe_cmd = sanitize_command_preview(safe_cmd)

        if step.step_id in self.failed_prechecks:
            failure_reasons.append(self.failed_prechecks[step.step_id])
        if self.simulated_state.get("reachable") is False:
            failure_reasons.append("Target cluster or host is unreachable in simulated state")

        diff_preview = step.diff_template.strip()
        if not diff_preview:
            diff_preview = (
                f"--- before ({safe_target or step.target_resource})\n"
                f"+++ after ({safe_target or step.target_resource}) [DRY-RUN]\n"
                f"+ {safe_cmd}"
            )

        if failure_reasons:
            reason_text = "; ".join(dict.fromkeys(failure_reasons))
            simulated_stdout = (
                f"[PRECHECK FAIL] Step {step.step_id} ({step.title}) blocked before dry-run: "
                f"{reason_text}\n"
                f"[DRY-RUN] Preview: {safe_cmd}\n"
                f"[DRY-RUN] executed_against_real_infra=False"
            )
            return DryRunOutcome(
                outcome_id=f"dr-{uuid.uuid4().hex[:12]}",
                run_id=eff_run_id,
                incident_id=eff_incident_id,
                step_id=step.step_id,
                command_preview=safe_cmd,
                diff_preview=diff_preview,
                simulated_stdout=simulated_stdout,
                simulated_status="WOULD_FAIL_PRECHECK",
                executed_against_real_infra=False,
                executed_at=utc_now(),
            )

        simulated_stdout = (
            f"[PRECHECK PASS] Target '{safe_target}' validated.\n"
            f"[DRY-RUN] Simulated step {step.step_id} ({step.title}): {safe_cmd}\n"
            f"[DRY-RUN] Expected effect: {step.expected_effect}\n"
            f"[DRY-RUN] executed_against_real_infra=False"
        )
        return DryRunOutcome(
            outcome_id=f"dr-{uuid.uuid4().hex[:12]}",
            run_id=eff_run_id,
            incident_id=eff_incident_id,
            step_id=step.step_id,
            command_preview=safe_cmd,
            diff_preview=diff_preview,
            simulated_stdout=simulated_stdout,
            simulated_status="SUCCESS_DRY_RUN",
            executed_against_real_infra=False,
            executed_at=utc_now(),
        )

    def execute(
        self,
        step: RemediationStep,
        result: TriageResult | None = None,
        *,
        run_id: str | None = None,
        incident_id: str | None = None,
    ) -> DryRunOutcome:
        """Alias for simulate()."""
        return self.simulate(step, result, run_id=run_id, incident_id=incident_id)

    def skip(
        self,
        step: RemediationStep,
        result: TriageResult | None = None,
        *,
        reason: str = "Skipped by operator",
        run_id: str | None = None,
        incident_id: str | None = None,
    ) -> DryRunOutcome:
        """Record a SKIPPED dry-run outcome without executing."""
        eff_run_id = run_id or (result.run_id if result is not None else "run-dryrun")
        eff_incident_id = incident_id or (
            result.incident_id if result is not None else "inc-dryrun"
        )
        safe_cmd = sanitize_command_preview(step.command_preview)
        return DryRunOutcome(
            outcome_id=f"dr-{uuid.uuid4().hex[:12]}",
            run_id=eff_run_id,
            incident_id=eff_incident_id,
            step_id=step.step_id,
            command_preview=safe_cmd,
            diff_preview=f"# SKIPPED: {reason}",
            simulated_stdout=(
                f"[SKIPPED] Step {step.step_id} ({step.title}) skipped: {reason}\n"
                f"[DRY-RUN] executed_against_real_infra=False"
            ),
            simulated_status="SKIPPED",
            executed_against_real_infra=False,
            executed_at=utc_now(),
        )
