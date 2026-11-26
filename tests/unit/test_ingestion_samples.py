"""Unit tests for bundled sample scenario fixtures and sample ingestion."""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pytest

from sre_triage_agent.ingestion import (
    BundledAlertSource,
    BundledLogSource,
    IngestionError,
    correlate_incidents,
    list_bundled_scenarios,
    load_bundled_alerts,
    load_bundled_scenario,
)

EXPECTED_SCENARIOS = {
    "pod_oom_killed": {
        "service": "checkout-api",
        "min_alerts": 2,
        "min_logs": 10,
        "alertnames": {"KubePodOOMKilled", "KubePodCrashLoopBackOff"},
        "keywords": ("OOMKilled", "137"),
    },
    "api_5xx_db_pool": {
        "service": "orders-service",
        "min_alerts": 2,
        "min_logs": 10,
        "alertnames": {"ApiHigh5xxErrorRate", "DatabaseConnectionPoolExhausted"},
        "keywords": ("connection pool", "503"),
    },
    "disk_pressure": {
        "service": "storage-node",
        "min_alerts": 2,
        "min_logs": 10,
        "alertnames": {"NodeFilesystemAlmostOutOfSpace", "KubeNodeDiskPressure"},
        "keywords": ("No space left on device", "/var/lib/containerd"),
    },
    "tls_cert_expiring": {
        "service": "ingress-gateway",
        "min_alerts": 1,
        "min_logs": 10,
        "alertnames": {"TlsCertificateExpiringSoon"},
        "keywords": ("TLS", "api.prod.example.com"),
    },
}


def test_list_bundled_scenarios_discovers_all_four_scenarios() -> None:
    scenarios = list_bundled_scenarios()
    names = {s.name for s in scenarios}
    assert names == set(EXPECTED_SCENARIOS.keys())

    for info in scenarios:
        spec = EXPECTED_SCENARIOS[info.name]
        assert info.description.strip()
        assert info.alerts_path.is_file()
        assert info.log_path.is_file()
        assert info.alert_count >= spec["min_alerts"]  # type: ignore[operator]
        assert info.log_count >= spec["min_logs"]  # type: ignore[operator]


@pytest.mark.parametrize("scenario_name", sorted(EXPECTED_SCENARIOS.keys()))
def test_bundled_scenario_alerts_logs_and_correlation(scenario_name: str) -> None:
    spec = EXPECTED_SCENARIOS[scenario_name]

    alerts = BundledAlertSource(scenario_name).read()
    assert alerts == load_bundled_alerts(scenario_name)
    assert len(alerts) >= spec["min_alerts"]  # type: ignore[operator]
    assert {a.alertname for a in alerts} >= spec["alertnames"]  # type: ignore[operator]
    for alert in alerts:
        assert alert.alert_id.startswith("alert-")
        assert alert.starts_at.tzinfo == timezone.utc
        assert alert.service == spec["service"]

    logs = BundledLogSource(scenario_name).read()
    assert len(logs) >= spec["min_logs"]  # type: ignore[operator]
    line_numbers = [entry.line_number for entry in logs]
    assert line_numbers == list(range(1, len(logs) + 1))
    for entry in logs:
        assert entry.log_id.startswith("log-")
        assert entry.timestamp.tzinfo == timezone.utc
        assert entry.source.startswith(f"bundled:{scenario_name}/")

    bundle = load_bundled_scenario(scenario_name)
    assert bundle.source_mode == "bundled"
    assert bundle.scenario_name == scenario_name
    assert bundle.service == spec["service"]
    assert len(bundle.alerts) == len(alerts)
    assert len(bundle.logs) == len(logs)
    assert bundle.created_at.tzinfo == timezone.utc

    # Repeatability check on deterministic IDs
    bundle_again = load_bundled_scenario(scenario_name)
    assert bundle_again.incident_id == bundle.incident_id

    combined_text = " ".join(
        [a.summary + " " + a.description for a in bundle.alerts]
        + [entry.message for entry in bundle.logs]
    )
    for kw in spec["keywords"]:  # type: ignore[attr-defined]
        assert kw.lower() in combined_text.lower()


def test_correlate_all_four_scenarios_produces_four_distinct_bundles() -> None:
    all_alerts = []
    all_logs = []
    for scenario_name in sorted(EXPECTED_SCENARIOS.keys()):
        all_alerts.extend(BundledAlertSource(scenario_name).read())
        all_logs.extend(BundledLogSource(scenario_name).read())

    bundles = correlate_incidents(all_alerts, all_logs, source_mode="bundled")
    assert len(bundles) == 4
    services = {b.service for b in bundles}
    assert services == {spec["service"] for spec in EXPECTED_SCENARIOS.values()}
    assert sum(len(b.alerts) for b in bundles) == len(all_alerts)
    assert sum(len(b.logs) for b in bundles) == len(all_logs)


def test_unknown_scenario_and_missing_dir_raise_ingestion_error(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        BundledAlertSource("does_not_exist").read()

    with pytest.raises(IngestionError):
        BundledLogSource("does_not_exist").read()

    with pytest.raises(IngestionError):
        load_bundled_scenario("does_not_exist")

    with pytest.raises(IngestionError):
        list_bundled_scenarios(scenarios_dir=tmp_path / "nonexistent")


def test_bundled_alert_source_non_utf8_raises_ingestion_error(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "bad_utf8"
    scenario_dir.mkdir()
    alerts_path = scenario_dir / "alerts.json"
    alerts_path.write_bytes(b"\xff\xfe{}")

    with pytest.raises(IngestionError) as exc_info:
        BundledAlertSource("bad_utf8", scenarios_dir=tmp_path).read()

    assert exc_info.value.source == str(alerts_path)
    assert isinstance(exc_info.value.cause, UnicodeDecodeError)
