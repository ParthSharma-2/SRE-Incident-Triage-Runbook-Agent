"""Unit tests for domain models (models.py) and configuration loading (config.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

import sre_triage_agent
from sre_triage_agent.config import (
    AgentConfig,
    AuditConfig,
    ConfigError,
    IngestionConfig,
    LLMConfig,
    load_config,
    load_yaml_config,
)
from sre_triage_agent.models import (
    AlertEvent,
    AuditEvent,
    DryRunOutcome,
    EvidenceRef,
    IncidentBundle,
    LogEntry,
    OperatorDecision,
    RemediationStep,
    RunbookStepTemplate,
    RunbookTemplate,
    TriageResult,
    utc_now,
)

ALL_ENV_VARS = [
    "SRE_AGENT_LLM_PROVIDER",
    "SRE_AGENT_LLM_BASE_URL",
    "SRE_AGENT_LLM_MODEL",
    "SRE_AGENT_LLM_API_KEY",
    "SRE_AGENT_LLM_TIMEOUT_SECONDS",
    "SRE_AGENT_LLM_ALLOW_FALLBACK",
    "SRE_AGENT_PROMETHEUS_URL",
    "SRE_AGENT_ALERT_API_TYPE",
    "SRE_AGENT_LOG_FILE",
    "SRE_AGENT_TAIL_LINES",
    "SRE_AGENT_INCLUDE_RESOLVED",
    "SRE_AGENT_AUDIT_LOG_PATH",
    "SRE_AGENT_OPERATOR",
]


@pytest.fixture
def clean_sre_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all SRE_AGENT_* environment variables for deterministic config tests."""
    for name in ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _sample_log() -> LogEntry:
    return LogEntry(
        log_id="log-001",
        timestamp=datetime(2026, 11, 25, 10, 15, 0, tzinfo=timezone.utc),
        level="ERROR",
        service="checkout-api",
        message="OOMKilled exit code 137",
        source="app.log.jsonl",
        line_number=6,
    )


def _sample_alert() -> AlertEvent:
    return AlertEvent(
        alert_id="alert-001",
        alertname="KubePodOOMKilled",
        status="firing",
        severity="critical",
        service="checkout-api",
        namespace="production",
        instance="10.42.3.18:8080",
        summary="Pod OOMKilled",
        starts_at=datetime(2026, 11, 25, 10, 15, 42, tzinfo=timezone.utc),
    )


def _sample_evidence() -> EvidenceRef:
    return EvidenceRef(
        kind="alert",
        ref_id="alert-001",
        excerpt="KubePodOOMKilled firing",
        reason="Container exceeded 512Mi memory limit",
    )


def _sample_step() -> RemediationStep:
    return RemediationStep(
        step_id="step-1",
        runbook_id="k8s_oom_killed",
        title="Inspect memory usage and increase limit",
        description="Increase deployment memory limit from 512Mi to 1024Mi",
        risk_level="medium",
        command_preview="kubectl set resources deployment/checkout-api -n production --limits=memory=1024Mi",
        target_resource="deployment/checkout-api",
        expected_effect="Prevent OOMKilled terminations under peak load",
        diff_template="- memory: 512Mi\n+ memory: 1024Mi",
        evidence_refs=[_sample_evidence()],
    )


def test_package_version_and_exports() -> None:
    assert sre_triage_agent.__version__ == "0.1.0"
    assert sre_triage_agent.load_config is load_config


def test_all_domain_and_config_models_are_frozen() -> None:
    log = _sample_log()
    alert = _sample_alert()
    ev = _sample_evidence()
    bundle = IncidentBundle(
        incident_id="inc-001",
        title="OOMKilled",
        service="checkout-api",
        severity="critical",
        source_mode="bundled",
        alerts=[alert],
        logs=[log],
    )
    rb_step = RunbookStepTemplate(
        step_id="step-1",
        title="Patch limit",
        description="Increase memory",
        risk_level="medium",
        command_template="kubectl set resources deployment/{service}",
    )
    rb = RunbookTemplate(
        runbook_id="k8s_oom_killed",
        title="OOMKilled Runbook",
        description="Handle pod OOM",
        match_alertnames=["KubePodOOMKilled"],
        match_keywords=["OOMKilled"],
        steps=[rb_step],
    )
    step = _sample_step()
    triage = TriageResult(
        run_id="run-001",
        incident_id="inc-001",
        source_mode="bundled",
        severity="critical",
        root_cause_hypothesis="Memory limit too low",
        summary="checkout-api OOMKilled",
        provider_used="ollama",
        model_used="llama3.1:8b",
        fallback_used=False,
        alerts=[alert],
        logs=[log],
        remediation_steps=[step],
    )
    decision = OperatorDecision(
        decision_id="dec-001",
        run_id="run-001",
        incident_id="inc-001",
        step_id="step-1",
        decision="approved",
        operator="alice",
    )
    outcome = DryRunOutcome(
        outcome_id="out-001",
        run_id="run-001",
        incident_id="inc-001",
        step_id="step-1",
        command_preview=step.command_preview,
        diff_preview="- 512Mi\n+ 1024Mi",
        simulated_stdout="deployment.apps/checkout-api resource requirements updated (dry run)",
        simulated_status="SUCCESS_DRY_RUN",
    )
    audit = AuditEvent(
        event_id="evt-001",
        run_id="run-001",
        incident_id="inc-001",
        event_type="triage_completed",
        actor="sre-triage-agent",
        payload={"severity": "critical"},
    )

    instances_and_attrs: list[tuple[object, str, object]] = [
        (log, "level", "INFO"),
        (alert, "status", "resolved"),
        (ev, "kind", "log"),
        (bundle, "severity", "low"),
        (rb_step, "risk_level", "high"),
        (rb, "title", "changed"),
        (step, "title", "changed"),
        (triage, "fallback_used", True),
        (decision, "decision", "rejected"),
        (outcome, "simulated_status", "SKIPPED"),
        (audit, "actor", "bob"),
        (LLMConfig(), "model", "other"),
        (IngestionConfig(), "alert_api_type", "prometheus"),
        (AuditConfig(), "audit_log_path", "/tmp/other.jsonl"),
        (AgentConfig(), "operator", "other"),
    ]
    for model, attr, val in instances_and_attrs:
        with pytest.raises(ValidationError):
            setattr(model, attr, val)


def test_mutable_defaults_are_isolated_and_model_copy_does_not_mutate() -> None:
    log_a = _sample_log()
    log_b = _sample_log()
    assert log_a.attributes is not log_b.attributes

    alert_a = _sample_alert()
    alert_b = _sample_alert()
    assert alert_a.labels is not alert_b.labels
    assert alert_a.annotations is not alert_b.annotations

    copied = log_a.model_copy(update={"level": "CRITICAL"})
    assert copied.level == "CRITICAL"
    assert log_a.level == "ERROR"


def test_remediation_step_requires_non_empty_evidence_and_dry_run_cannot_be_real() -> None:
    with pytest.raises(ValidationError):
        RemediationStep(
            step_id="step-1",
            runbook_id="rb-1",
            title="Title",
            description="Desc",
            risk_level="low",
            command_preview="echo ok",
            target_resource="svc",
            expected_effect="none",
            evidence_refs=[],
        )

    assert len(_sample_step().evidence_refs) == 1

    outcome = DryRunOutcome(
        outcome_id="out-1",
        run_id="run-1",
        incident_id="inc-1",
        step_id="step-1",
        command_preview="echo ok",
        diff_preview="",
        simulated_stdout="ok",
        simulated_status="SUCCESS_DRY_RUN",
    )
    assert outcome.executed_against_real_infra is False

    with pytest.raises(ValidationError):
        DryRunOutcome(
            outcome_id="out-2",
            run_id="run-1",
            incident_id="inc-1",
            step_id="step-1",
            command_preview="echo ok",
            diff_preview="",
            simulated_stdout="ok",
            simulated_status="SUCCESS_DRY_RUN",
            executed_against_real_infra=True,  # type: ignore[arg-type]
        )


def test_invalid_literals_and_bounds_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        LogEntry(
            log_id="l1",
            timestamp=utc_now(),
            level="TRACE",  # type: ignore[arg-type]
            service="svc",
            message="msg",
            source="src",
            line_number=1,
        )
    with pytest.raises(ValidationError):
        LogEntry(
            log_id="l1",
            timestamp=utc_now(),
            level="INFO",
            service="svc",
            message="msg",
            source="src",
            line_number=-1,
        )
    with pytest.raises(ValidationError):
        AlertEvent(
            alert_id="a1",
            alertname="Test",
            status="silenced",  # type: ignore[arg-type]
            severity="warning",
            service="svc",
            starts_at=utc_now(),
        )
    with pytest.raises(ValidationError):
        EvidenceRef(kind="metric", ref_id="1", excerpt="x", reason="y")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        OperatorDecision(
            decision_id="d1",
            run_id="r1",
            incident_id="i1",
            step_id="s1",
            decision="deferred",  # type: ignore[arg-type]
            operator="op",
        )
    with pytest.raises(ValidationError):
        AuditEvent(
            event_id="e1",
            run_id="r1",
            incident_id="i1",
            event_type="unknown_event",  # type: ignore[arg-type]
            actor="agent",
        )
    with pytest.raises(ValidationError):
        LLMConfig(provider="openai")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        LLMConfig(timeout_seconds=0)
    with pytest.raises(ValidationError):
        LLMConfig(timeout_seconds=-5.0)
    with pytest.raises(ValidationError):
        IngestionConfig(tail_lines=0)
    with pytest.raises(ValidationError):
        IngestionConfig(alert_api_type="datadog")  # type: ignore[arg-type]


def test_utc_datetime_normalization_and_json_round_trip() -> None:
    naive = datetime(2026, 11, 25, 10, 0, 0)
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    offset_dt = datetime(2026, 11, 25, 15, 30, 0, tzinfo=ist_tz)

    entry_naive = LogEntry(
        log_id="l1",
        timestamp=naive,
        level="INFO",
        service="svc",
        message="naive",
        source="src",
        line_number=1,
    )
    entry_offset = LogEntry(
        log_id="l2",
        timestamp=offset_dt,
        level="INFO",
        service="svc",
        message="offset",
        source="src",
        line_number=2,
    )
    entry_z = AlertEvent(
        alert_id="a1",
        alertname="ZAlert",
        status="firing",
        severity="warning",
        service="svc",
        starts_at="2026-11-25T10:00:00Z",  # type: ignore[arg-type]
    )

    assert entry_naive.timestamp.tzinfo == timezone.utc
    assert entry_offset.timestamp == datetime(2026, 11, 25, 10, 0, 0, tzinfo=timezone.utc)
    assert entry_z.starts_at == datetime(2026, 11, 25, 10, 0, 0, tzinfo=timezone.utc)

    bundle = IncidentBundle(
        incident_id="inc-1",
        title="OOM",
        service="svc",
        severity="critical",
        source_mode="bundled",
        alerts=[entry_z],
        logs=[entry_naive],
    )
    assert IncidentBundle.model_validate_json(bundle.model_dump_json()) == bundle

    rb = RunbookTemplate(
        runbook_id="rb-1",
        title="Runbook",
        description="Desc",
        steps=[
            RunbookStepTemplate(
                step_id="s1",
                title="Step",
                description="Do",
                risk_level="low",
                command_template="echo {service}",
            )
        ],
    )
    assert RunbookTemplate.model_validate_json(rb.model_dump_json()) == rb

    triage = TriageResult(
        run_id="r1",
        incident_id="inc-1",
        source_mode="bundled",
        severity="critical",
        root_cause_hypothesis="OOM",
        summary="Summary",
        provider_used="deterministic",
        model_used="rule-based",
        fallback_used=True,
        fallback_reason="Ollama offline",
        alerts=[entry_z],
        logs=[entry_naive],
        remediation_steps=[_sample_step()],
    )
    assert TriageResult.model_validate_json(triage.model_dump_json()) == triage

    decision = OperatorDecision(
        decision_id="d1",
        run_id="r1",
        incident_id="inc-1",
        step_id="s1",
        decision="rejected",
        operator="alice",
        reason="Too risky",
    )
    assert OperatorDecision.model_validate_json(decision.model_dump_json()) == decision

    outcome = DryRunOutcome(
        outcome_id="o1",
        run_id="r1",
        incident_id="inc-1",
        step_id="s1",
        command_preview="echo ok",
        diff_preview="- a\n+ b",
        simulated_stdout="ok",
        simulated_status="SKIPPED",
    )
    assert DryRunOutcome.model_validate_json(outcome.model_dump_json()) == outcome

    audit = AuditEvent(
        event_id="e1",
        run_id="r1",
        incident_id="inc-1",
        event_type="operator_decision",
        actor="alice",
        payload={"decision": "rejected"},
    )
    assert AuditEvent.model_validate_json(audit.model_dump_json()) == audit


def test_config_defaults_and_all_env_vars(
    clean_sre_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_config()
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.base_url == "http://localhost:11434"
    assert cfg.llm.model == "llama3.1:8b"
    assert cfg.llm.api_key is None
    assert cfg.llm.timeout_seconds == 30.0
    assert cfg.llm.allow_fallback is True
    assert cfg.ingestion.prometheus_url is None
    assert cfg.ingestion.alert_api_type == "auto"
    assert cfg.ingestion.log_file is None
    assert cfg.ingestion.tail_lines is None
    assert cfg.ingestion.include_resolved is False
    assert cfg.audit.audit_log_path == "./audit_log.jsonl"
    assert cfg.operator == "sre-operator"

    monkeypatch.setenv("SRE_AGENT_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SRE_AGENT_LLM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("SRE_AGENT_LLM_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("SRE_AGENT_LLM_API_KEY", "local-token")
    monkeypatch.setenv("SRE_AGENT_LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("SRE_AGENT_LLM_ALLOW_FALLBACK", "false")
    monkeypatch.setenv("SRE_AGENT_PROMETHEUS_URL", "http://127.0.0.1:9090")
    monkeypatch.setenv("SRE_AGENT_ALERT_API_TYPE", "alertmanager")
    monkeypatch.setenv("SRE_AGENT_LOG_FILE", "/var/log/app.log")
    monkeypatch.setenv("SRE_AGENT_TAIL_LINES", "250")
    monkeypatch.setenv("SRE_AGENT_INCLUDE_RESOLVED", "true")
    monkeypatch.setenv("SRE_AGENT_AUDIT_LOG_PATH", "/var/tmp/audit.jsonl")
    monkeypatch.setenv("SRE_AGENT_OPERATOR", "oncall-alice")

    env_cfg = load_config()
    assert env_cfg.llm.provider == "openai_compatible"
    assert env_cfg.llm.base_url == "http://127.0.0.1:8000"
    assert env_cfg.llm.model == "qwen2.5:7b"
    assert env_cfg.llm.api_key == "local-token"
    assert env_cfg.llm.timeout_seconds == 12.5
    assert env_cfg.llm.allow_fallback is False
    assert env_cfg.ingestion.prometheus_url == "http://127.0.0.1:9090"
    assert env_cfg.ingestion.alert_api_type == "alertmanager"
    assert env_cfg.ingestion.log_file == "/var/log/app.log"
    assert env_cfg.ingestion.tail_lines == 250
    assert env_cfg.ingestion.include_resolved is True
    assert env_cfg.audit.audit_log_path == "/var/tmp/audit.jsonl"
    assert env_cfg.operator == "oncall-alice"

    monkeypatch.setenv("SRE_AGENT_LLM_API_KEY", "")
    assert load_config().llm.api_key is None


def test_yaml_loading_precedence_and_partial_overrides(
    clean_sre_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    yaml_file = tmp_path / "agent.yaml"
    yaml_file.write_text(
        """
llm:
  provider: openai_compatible
  base_url: http://yaml-host:8000
  model: mistral:7b
  timeout_seconds: 45.0
ingestion:
  prometheus_url: http://yaml-prom:9090
  alert_api_type: prometheus
audit:
  audit_log_path: ./yaml_audit.jsonl
operator: yaml-op
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("SRE_AGENT_LLM_BASE_URL", "http://env-host:11434")

    cfg = load_config(
        config_path=yaml_file,
        overrides={"llm": {"provider": "deterministic"}},
    )

    # overrides > env > YAML > defaults, without clobbering sibling fields in llm
    assert cfg.llm.provider == "deterministic"
    assert cfg.llm.base_url == "http://env-host:11434"
    assert cfg.llm.model == "mistral:7b"
    assert cfg.llm.timeout_seconds == 45.0
    assert cfg.llm.allow_fallback is True
    assert cfg.ingestion.prometheus_url == "http://yaml-prom:9090"
    assert cfg.ingestion.alert_api_type == "prometheus"
    assert cfg.audit.audit_log_path == "./yaml_audit.jsonl"
    assert cfg.operator == "yaml-op"


def test_yaml_empty_and_flat_aliases(clean_sre_env: None, tmp_path: Path) -> None:
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    assert load_yaml_config(empty_yaml) == {}
    assert load_config(config_path=empty_yaml).llm.provider == "ollama"

    flat_yaml = tmp_path / "flat.yaml"
    flat_yaml.write_text(
        "llm_provider: deterministic\naudit_log_path: ./custom_audit.jsonl\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path=flat_yaml)
    assert cfg.llm.provider == "deterministic"
    assert cfg.audit.audit_log_path == "./custom_audit.jsonl"


def test_yaml_errors_raise_config_error(clean_sre_env: None, tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError):
        load_config(config_path=missing)

    with pytest.raises(ConfigError):
        load_config(config_path=tmp_path)

    bad_syntax = tmp_path / "bad.yaml"
    bad_syntax.write_text("llm:\n  provider: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path=bad_syntax)

    non_mapping = tmp_path / "list.yaml"
    non_mapping.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path=non_mapping)

    bad_section = tmp_path / "bad_section.yaml"
    bad_section.write_text("llm: not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path=bad_section)

    invalid_values = tmp_path / "invalid_val.yaml"
    invalid_values.write_text("llm:\n  provider: proprietary\n  timeout_seconds: -5\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path=invalid_values)

    non_utf8 = tmp_path / "non_utf8.yaml"
    non_utf8.write_bytes(b"\xff\xfe\x80llm:\n  model: test\n")
    with pytest.raises(ConfigError, match="Failed to read configuration file"):
        load_config(config_path=non_utf8)
