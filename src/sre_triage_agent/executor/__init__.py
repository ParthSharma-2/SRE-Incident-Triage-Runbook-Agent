"""Zero-side-effect simulated dry-run execution package."""

from sre_triage_agent.executor.dry_run import (
    DryRunExecutor,
    sanitize_command_preview,
    sanitize_identifier,
)

__all__ = [
    "DryRunExecutor",
    "sanitize_command_preview",
    "sanitize_identifier",
]
