"""Unit tests for bundled YAML runbooks, RunbookCatalog matching, and template rendering."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sre_triage_agent.ingestion import load_bundled_scenario
from sre_triage_agent.models import (
    AlertEvent,
    IncidentBundle,
    LogEntry,
    RunbookStepTemplate,
    RunbookTemplate,
)
from sre_triage_agent.runbooks import (
    RunbookCatalog,
    RunbookError,
    extract_incident_context,
    render_template,
    resolve_runbooks_dir,
)

_UNRENDERED_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")


def test_load_default_catalog_validates_all_four_runbooks() -> None:
    runbooks_dir = resolve_runbooks_dir()
    assert runbooks_dir.is_dir()

    catalog = RunbookCatalog.load_default()
    ids = {rb.runbook_id for rb in catalog}
    assert ids == {
        "k8s_oom_killed",
        "api_db_pool_exhaustion",
        "host_disk_pressure",
        "tls_cert_renewal",
    }
    assert len(catalog) == 4

    for rb in catalog:
        assert rb.title.strip()
        assert rb.description.strip()
        assert len(rb.match_alertnames) >= 1
        assert len(rb.match_keywords) >= 1
        assert len(rb.steps) >= 3
        for step in rb.steps:
            assert step.risk_level in {"low", "medium", "high"}
            assert step.command_template.strip()
            assert catalog.get(rb.runbook_id) == rb


@pytest.mark.parametrize(
    ("scenario_name", "expected_runbook_id"),
    [
        ("pod_oom_killed", "k8s_oom_killed"),
        ("api_5xx_db_pool", "api_db_pool_exhaustion"),
        ("disk_pressure", "host_disk_pressure"),
        ("tls_cert_expiring", "tls_cert_renewal"),
    ],
)
def test_match_candidates_ranks_bundled_scenarios_first(
    scenario_name: str,
    expected_runbook_id: str,
) -> None:
    catalog = RunbookCatalog.load_default()
    bundle = load_bundled_scenario(scenario_name)

    candidates = catalog.match_candidates(bundle)
    assert len(candidates) >= 1
    assert candidates[0].runbook_id == expected_runbook_id


def test_match_candidates_alert_only_and_log_only_incidents() -> None:
    catalog = RunbookCatalog.load_default()
    now = datetime(2026, 11, 25, 14, 0, 0, tzinfo=timezone.utc)

    alert_only = IncidentBundle(
        incident_id="inc-alert-only",
        title="OOM alert",
        service="checkout-api",
        severity="critical",
        source_mode="live",
        alerts=[
            AlertEvent(
                alert_id="a1",
                alertname="kubepodoomkilled",
                status="firing",
                severity="critical",
                service="checkout-api",
                starts_at=now,
            )
        ],
        logs=[],
        created_at=now,
    )
    alert_matches = catalog.match_candidates(alert_only)
    assert alert_matches
    assert alert_matches[0].runbook_id == "k8s_oom_killed"

    log_only = IncidentBundle(
        incident_id="inc-log-only",
        title="Disk anomaly",
        service="storage-node",
        severity="high",
        source_mode="live",
        alerts=[],
        logs=[
            LogEntry(
                log_id="l1",
                timestamp=now,
                level="ERROR",
                service="storage-node",
                message="Kernel reported NO SPACE LEFT ON DEVICE under /var/lib/containerd",
                source="syslog",
                line_number=1,
            )
        ],
        created_at=now,
    )
    log_matches = catalog.match_candidates(log_only)
    assert log_matches
    assert log_matches[0].runbook_id == "host_disk_pressure"


def test_match_candidates_scoring_and_deterministic_tie_breaking() -> None:
    now = datetime(2026, 11, 25, 14, 0, 0, tzinfo=timezone.utc)
    rb_alert = RunbookTemplate(
        runbook_id="rb_alert",
        title="Alert match",
        description="Matches alertname",
        match_alertnames=["TargetAlert"],
        match_keywords=[],
        severity_hint="warning",
        steps=[],
    )
    rb_kw = RunbookTemplate(
        runbook_id="rb_keyword",
        title="Keyword match",
        description="Matches keyword",
        match_alertnames=[],
        match_keywords=["503", "timeout"],
        severity_hint="critical",
        steps=[],
    )
    rb_tie_b = RunbookTemplate(
        runbook_id="rb_tie_b",
        title="Tie B",
        description="Tie",
        match_alertnames=[],
        match_keywords=["503"],
        severity_hint="warning",
        steps=[],
    )
    rb_tie_a = RunbookTemplate(
        runbook_id="rb_tie_a",
        title="Tie A",
        description="Tie",
        match_alertnames=[],
        match_keywords=["503"],
        severity_hint="warning",
        steps=[],
    )
    catalog = RunbookCatalog([rb_tie_b, rb_kw, rb_alert, rb_tie_a])

    # Word boundary check: "15032" must NOT match keyword "503"
    boundary_bundle = IncidentBundle(
        incident_id="inc-boundary",
        title="Port 15032 check",
        service="api",
        severity="warning",
        source_mode="live",
        alerts=[],
        logs=[
            LogEntry(
                log_id="l-boundary",
                timestamp=now,
                level="WARN",
                service="api",
                message="Connection opened on port 15032",
                source="app.log",
                line_number=1,
            )
        ],
        created_at=now,
    )
    assert catalog.match_candidates(boundary_bundle) == []

    # Alertname match (100) outranks keyword matches (15 + 15 + 2 = 32), and ties sort by runbook_id
    ranked_bundle = IncidentBundle(
        incident_id="inc-ranked",
        title="HTTP 503 timeout",
        service="api",
        severity="critical",
        source_mode="live",
        alerts=[
            AlertEvent(
                alert_id="a1",
                alertname="TargetAlert",
                status="firing",
                severity="critical",
                service="api",
                starts_at=now,
            )
        ],
        logs=[
            LogEntry(
                log_id="l1",
                timestamp=now,
                level="ERROR",
                service="api",
                message="Returned HTTP 503 after timeout",
                source="app.log",
                line_number=2,
            )
        ],
        created_at=now,
    )
    ranked = catalog.match_candidates(ranked_bundle)
    assert [rb.runbook_id for rb in ranked] == [
        "rb_alert",
        "rb_keyword",
        "rb_tie_a",
        "rb_tie_b",
    ]
    assert [rb.runbook_id for rb in catalog.match_candidates(ranked_bundle, limit=2)] == [
        "rb_alert",
        "rb_keyword",
    ]
    assert catalog.match_candidates(ranked_bundle, limit=0) == []


def test_match_candidates_unknown_and_empty_incidents_return_empty() -> None:
    catalog = RunbookCatalog.load_default()
    now = datetime(2026, 11, 25, 14, 0, 0, tzinfo=timezone.utc)

    empty_bundle = IncidentBundle(
        incident_id="inc-empty",
        title="Empty",
        service="unknown",
        severity="critical",
        source_mode="live",
        alerts=[],
        logs=[],
        created_at=now,
    )
    assert catalog.match_candidates(empty_bundle) == []

    unrelated_bundle = IncidentBundle(
        incident_id="inc-unrelated",
        title="Cache hit ratio dipped slightly",
        service="recommendations-worker",
        severity="critical",
        source_mode="live",
        alerts=[
            AlertEvent(
                alert_id="a-unrel",
                alertname="RecommendationCacheMissElevated",
                status="firing",
                severity="critical",
                service="recommendations-worker",
                summary="Redis hit ratio fell to 82%",
                starts_at=now,
            )
        ],
        logs=[
            LogEntry(
                log_id="l-unrel",
                timestamp=now,
                level="WARN",
                service="recommendations-worker",
                message="Cold cache segment miss for user cohort B",
                source="worker.log",
                line_number=1,
            )
        ],
        created_at=now,
    )
    assert catalog.match_candidates(unrelated_bundle) == []


def test_from_directory_empty_and_yml_filtering(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert len(RunbookCatalog.from_directory(empty_dir)) == 0

    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()
    (mixed_dir / "README.md").write_text("# ignored", encoding="utf-8")
    (mixed_dir / "sample.yml").write_text(
        """
runbook_id: custom_yml_rb
title: Custom YAML Runbook
description: Loaded from .yml extension
match_alertnames:
  - CustomAlert
match_keywords:
  - custom_signal
severity_hint: warning
steps:
  - step_id: step-1
    title: Check custom service
    description: Check status
    risk_level: low
    command_template: "echo {service}"
""",
        encoding="utf-8",
    )
    catalog = RunbookCatalog.from_directory(mixed_dir)
    assert len(catalog) == 1
    assert catalog.get("custom_yml_rb") is not None


def test_from_directory_error_cases(tmp_path: Path) -> None:
    # Missing directory
    with pytest.raises(RunbookError):
        RunbookCatalog.from_directory(tmp_path / "does_not_exist")

    # File instead of directory
    file_path = tmp_path / "not_a_dir.yaml"
    file_path.write_text("runbook_id: x", encoding="utf-8")
    with pytest.raises(RunbookError):
        RunbookCatalog.from_directory(file_path)

    # Invalid YAML syntax
    bad_yaml_dir = tmp_path / "bad_yaml"
    bad_yaml_dir.mkdir()
    (bad_yaml_dir / "broken.yaml").write_text("runbook_id: [unclosed", encoding="utf-8")
    with pytest.raises(RunbookError, match="Invalid YAML"):
        RunbookCatalog.from_directory(bad_yaml_dir)

    # Non-UTF-8 bytes
    bad_utf8_dir = tmp_path / "bad_utf8"
    bad_utf8_dir.mkdir()
    (bad_utf8_dir / "bad.yaml").write_bytes(b"\xff\xfe\x00\x80")
    with pytest.raises(RunbookError, match="Failed to read"):
        RunbookCatalog.from_directory(bad_utf8_dir)

    # Empty YAML (None)
    none_yaml_dir = tmp_path / "none_yaml"
    none_yaml_dir.mkdir()
    (none_yaml_dir / "empty.yaml").write_text("", encoding="utf-8")
    with pytest.raises(RunbookError, match="top-level YAML mapping"):
        RunbookCatalog.from_directory(none_yaml_dir)

    # Non-mapping YAML (list)
    list_yaml_dir = tmp_path / "list_yaml"
    list_yaml_dir.mkdir()
    (list_yaml_dir / "list.yaml").write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(RunbookError, match="top-level YAML mapping"):
        RunbookCatalog.from_directory(list_yaml_dir)

    # Schema validation error (bad risk_level / missing runbook_id)
    bad_schema_dir = tmp_path / "bad_schema"
    bad_schema_dir.mkdir()
    (bad_schema_dir / "invalid.yaml").write_text(
        """
title: Missing ID
description: Invalid risk
steps:
  - step_id: s1
    title: t
    description: d
    risk_level: extreme
    command_template: "echo hi"
""",
        encoding="utf-8",
    )
    with pytest.raises(RunbookError, match="schema validation failed"):
        RunbookCatalog.from_directory(bad_schema_dir)

    # Duplicate runbook_id across files
    dup_dir = tmp_path / "dup"
    dup_dir.mkdir()
    content = "runbook_id: dup_id\ntitle: A\ndescription: B\n"
    (dup_dir / "a.yaml").write_text(content, encoding="utf-8")
    (dup_dir / "b.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(RunbookError, match="Duplicate runbook_id"):
        RunbookCatalog.from_directory(dup_dir)


@pytest.mark.parametrize(
    "scenario_name",
    ["pod_oom_killed", "api_5xx_db_pool", "disk_pressure", "tls_cert_expiring"],
)
def test_extract_incident_context_and_render_template_across_scenarios(
    scenario_name: str,
) -> None:
    catalog = RunbookCatalog.load_default()
    bundle = load_bundled_scenario(scenario_name)
    ctx = extract_incident_context(bundle)
    primary_rb = catalog.match_candidates(bundle)[0]

    for step in primary_rb.steps:
        rendered_cmd = render_template(step.command_template, ctx)
        rendered_target = render_template(step.target_resource_template, ctx)
        rendered_desc = render_template(step.description, ctx)
        assert not _UNRENDERED_PLACEHOLDER_RE.search(rendered_cmd)
        assert not _UNRENDERED_PLACEHOLDER_RE.search(rendered_target)
        assert not _UNRENDERED_PLACEHOLDER_RE.search(rendered_desc)


def test_extract_context_strips_whitespace_and_render_preserves_literal_braces() -> None:
    now = datetime(2026, 11, 25, 14, 0, 0, tzinfo=timezone.utc)
    bundle = IncidentBundle(
        incident_id="inc-ws",
        title="Whitespace labels",
        service="  checkout-api  ",
        severity="warning",
        source_mode="live",
        alerts=[
            AlertEvent(
                alert_id="a1",
                alertname="TestAlert",
                status="firing",
                severity="warning",
                service="checkout-api",
                namespace="   ",
                labels={"pod": "\t  ", "node": "  node-09  "},
                starts_at=now,
            )
        ],
        logs=[
            LogEntry(
                log_id="l1",
                timestamp=now,
                level="WARN",
                service="checkout-api",
                message="warn",
                source="app.log",
                line_number=1,
                attributes={"namespace": "  prod-ns ", "pod": "  pod-123  "},
            )
        ],
        created_at=now,
    )
    ctx = extract_incident_context(bundle)
    assert ctx["service"] == "checkout-api"
    assert ctx["namespace"] == "prod-ns"
    assert ctx["pod"] == "pod-123"
    assert ctx["node"] == "node-09"

    rendered = render_template('echo {service} {unknown_key} {"json": true}', ctx)
    assert rendered == 'echo checkout-api {unknown_key} {"json": true}'


@pytest.mark.parametrize(
    ("instance_value", "expected_node"),
    [
        ("[::1]:8080", "::1"),
        ("[2001:db8::1]:9100", "2001:db8::1"),
        ("[::1]", "::1"),
        ("::1", "::1"),
        ("2001:db8::1", "2001:db8::1"),
        ("10.0.1.5:8080", "10.0.1.5"),
        ("worker-node-01:9100", "worker-node-01"),
        (":8080", "checkout-api"),
        ("[]:8080", "checkout-api"),
    ],
)
def test_extract_incident_context_handles_ipv6_and_port_stripping(
    instance_value: str,
    expected_node: str,
) -> None:
    now = datetime(2026, 11, 25, 14, 0, 0, tzinfo=timezone.utc)
    bundle = IncidentBundle(
        incident_id="inc-ipv6",
        title="IPv6 instance test",
        service="checkout-api",
        severity="warning",
        source_mode="live",
        alerts=[
            AlertEvent(
                alert_id="a1",
                alertname="TestAlert",
                status="firing",
                severity="warning",
                service="checkout-api",
                instance=instance_value,
                starts_at=now,
            )
        ],
        logs=[],
        created_at=now,
    )
    ctx = extract_incident_context(bundle)
    assert ctx["node"] == expected_node
    assert ctx["host"] == expected_node
    assert ctx["instance"] == instance_value
