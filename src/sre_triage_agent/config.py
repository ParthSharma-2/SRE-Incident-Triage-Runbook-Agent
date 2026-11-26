"""Configuration loading with pydantic-settings, YAML files, and SRE_AGENT_* env vars."""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class ConfigError(ValueError):
    """Raised when configuration loading or validation fails."""


class LLMConfig(BaseModel):
    """Open-source LLM provider configuration."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    provider: Literal["ollama", "openai_compatible", "deterministic"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    api_key: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    allow_fallback: bool = True


class IngestionConfig(BaseModel):
    """Live and sample log/alert ingestion configuration."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    prometheus_url: str | None = None
    alert_api_type: Literal["auto", "prometheus", "alertmanager"] = "auto"
    log_file: str | None = None
    tail_lines: int | None = Field(default=None, ge=1)
    include_resolved: bool = False


class AuditConfig(BaseModel):
    """Audit logging configuration."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    audit_log_path: str = "./audit_log.jsonl"


_ENV_FIELD_MAPPING: dict[str, tuple[str | None, str, bool]] = {
    "SRE_AGENT_LLM_PROVIDER": ("llm", "provider", False),
    "SRE_AGENT_LLM_BASE_URL": ("llm", "base_url", False),
    "SRE_AGENT_LLM_MODEL": ("llm", "model", False),
    "SRE_AGENT_LLM_API_KEY": ("llm", "api_key", True),
    "SRE_AGENT_LLM_TIMEOUT_SECONDS": ("llm", "timeout_seconds", False),
    "SRE_AGENT_LLM_ALLOW_FALLBACK": ("llm", "allow_fallback", False),
    "SRE_AGENT_PROMETHEUS_URL": ("ingestion", "prometheus_url", True),
    "SRE_AGENT_ALERT_API_TYPE": ("ingestion", "alert_api_type", False),
    "SRE_AGENT_LOG_FILE": ("ingestion", "log_file", True),
    "SRE_AGENT_TAIL_LINES": ("ingestion", "tail_lines", True),
    "SRE_AGENT_INCLUDE_RESOLVED": ("ingestion", "include_resolved", False),
    "SRE_AGENT_AUDIT_LOG_PATH": ("audit", "audit_log_path", False),
    "SRE_AGENT_OPERATOR": (None, "operator", False),
}

_YAML_FLAT_ALIASES: dict[str, tuple[str, str]] = {
    "llm_provider": ("llm", "provider"),
    "llm_base_url": ("llm", "base_url"),
    "llm_model": ("llm", "model"),
    "llm_api_key": ("llm", "api_key"),
    "llm_timeout_seconds": ("llm", "timeout_seconds"),
    "llm_allow_fallback": ("llm", "allow_fallback"),
    "prometheus_url": ("ingestion", "prometheus_url"),
    "alert_api_type": ("ingestion", "alert_api_type"),
    "log_file": ("ingestion", "log_file"),
    "tail_lines": ("ingestion", "tail_lines"),
    "include_resolved": ("ingestion", "include_resolved"),
    "audit_log_path": ("audit", "audit_log_path"),
}

_YAML_CONFIG_CTX: ContextVar[dict[str, Any] | None] = ContextVar(
    "_sre_agent_yaml_config",
    default=None,
)


def _lookup_env_case_insensitive(name: str) -> str | None:
    if name in os.environ:
        return os.environ[name]
    target = name.lower()
    for key, value in os.environ.items():
        if key.lower() == target:
            return value
    return None


class _MappedEnvSource(PydanticBaseSettingsSource):
    """Maps flat PROJECT.md SRE_AGENT_* env vars into nested config sections."""

    def get_field_value(
        self,
        field: FieldInfo,
        field_name: str,
    ) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for env_name, (section, target_field, empty_as_none) in _ENV_FIELD_MAPPING.items():
            raw = _lookup_env_case_insensitive(env_name)
            if raw is None:
                continue
            stripped = raw.strip()
            if stripped == "":
                if empty_as_none:
                    value: Any = None
                else:
                    continue
            else:
                value = stripped

            if section is None:
                resolved[target_field] = value
            else:
                bucket = resolved.setdefault(section, {})
                if isinstance(bucket, dict):
                    bucket[target_field] = value
        return resolved


class _YamlDictSource(PydanticBaseSettingsSource):
    """Supplies pre-loaded YAML mapping stored in the per-call ContextVar."""

    def __init__(self, settings_cls: type[BaseSettings], data: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self._data = data

    def get_field_value(
        self,
        field: FieldInfo,
        field_name: str,
    ) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)


class AgentConfig(BaseSettings):
    """Top-level application settings for sre_triage_agent."""

    model_config = SettingsConfigDict(
        env_prefix="SRE_AGENT_",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    operator: str = "sre-operator"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_data = _YAML_CONFIG_CTX.get() or {}
        return (
            init_settings,
            _MappedEnvSource(settings_cls),
            env_settings,
            _YamlDictSource(settings_cls, yaml_data),
            dotenv_settings,
            file_secret_settings,
        )


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load and normalize a YAML config file, raising ConfigError on failure."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    if config_path.is_dir():
        raise ConfigError(f"Configuration path is a directory: {config_path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Failed to read configuration file {config_path}: {exc}") from exc

    try:
        parsed = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML syntax in {config_path}: {exc}") from exc

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ConfigError(
            f"Configuration file {config_path} must contain a top-level YAML mapping"
        )

    normalized: dict[str, Any] = {}
    for section in ("llm", "ingestion", "audit"):
        if section in parsed:
            section_value = parsed[section]
            if section_value is None:
                normalized[section] = {}
            elif isinstance(section_value, dict):
                normalized[section] = dict(section_value)
            else:
                raise ConfigError(
                    f"Configuration section '{section}' in {config_path} must be a mapping"
                )

    if "operator" in parsed and parsed["operator"] is not None:
        normalized["operator"] = parsed["operator"]

    for flat_key, (section, target_field) in _YAML_FLAT_ALIASES.items():
        if flat_key in parsed:
            section_dict = normalized.setdefault(section, {})
            if target_field not in section_dict:
                section_dict[target_field] = parsed[flat_key]

    return normalized


def _prune_none_overrides(mapping: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            nested = _prune_none_overrides(value)
            if nested:
                cleaned[key] = nested
        elif value is not None:
            cleaned[key] = value
    return cleaned


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AgentConfig:
    """Load AgentConfig with precedence: overrides > env > YAML > defaults."""
    yaml_data = load_yaml_config(config_path) if config_path is not None else {}
    clean_overrides = _prune_none_overrides(overrides or {})
    token = _YAML_CONFIG_CTX.set(yaml_data)
    try:
        return AgentConfig(**clean_overrides)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration values: {exc}") from exc
    finally:
        _YAML_CONFIG_CTX.reset(token)
