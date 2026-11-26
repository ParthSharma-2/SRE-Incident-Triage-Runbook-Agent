"""Open-source Ollama HTTP adapter using /api/chat and /api/tags (zero proprietary SDKs)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from urllib.parse import urlparse

import httpx

from sre_triage_agent.llm.base import (
    LLMConnectionError,
    LLMResponseParseError,
    LLMTriageResponse,
    ProviderHealth,
    build_triage_messages,
    extract_json_object,
    validate_and_ground_response,
    validate_base_url,
    validate_model_and_timeout,
)
from sre_triage_agent.models import IncidentBundle, RunbookTemplate


def _resolve_ollama_urls(base_url: str) -> tuple[str, str]:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/api/chat"):
        prefix = base_url[: -len("/api/chat")]
        return base_url, f"{prefix}/api/tags"
    if path.endswith("/api"):
        return f"{base_url}/chat", f"{base_url}/tags"
    return f"{base_url}/api/chat", f"{base_url}/api/tags"


def _ollama_model_matches(configured_model: str, available_names: set[str]) -> bool:
    if configured_model in available_names:
        return True
    if ":" not in configured_model and f"{configured_model}:latest" in available_names:
        return True
    return False


class OllamaProvider:
    """Open-source Ollama provider communicating over httpx with /api/chat."""

    provider_name: str = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        *,
        timeout_s: float = 30.0,
        temperature: float = 0.1,
        headers: dict[str, str] | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = validate_base_url(base_url, provider=self.provider_name)
        validated_model, validated_timeout = validate_model_and_timeout(
            model, timeout_s, provider=self.provider_name
        )
        self.model = validated_model
        self.model_name = validated_model
        self.timeout_s = validated_timeout
        self.temperature = float(temperature)
        self._headers = dict(headers) if headers else {}
        self._injected_client = client
        self._transport = transport
        self._chat_url, self._tags_url = _resolve_ollama_urls(self.base_url)

    @contextmanager
    def _client_ctx(self) -> Iterator[httpx.Client]:
        if self._injected_client is not None:
            yield self._injected_client
            return
        with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
            yield client

    def triage(
        self,
        bundle: IncidentBundle,
        runbooks: Sequence[RunbookTemplate],
    ) -> LLMTriageResponse:
        messages = build_triage_messages(bundle, runbooks)
        request_body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }

        try:
            with self._client_ctx() as client:
                response = client.post(
                    self._chat_url,
                    json=request_body,
                    headers=self._headers or None,
                    timeout=self.timeout_s,
                )
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(
                f"Ollama request timed out after {self.timeout_s}s at {self._chat_url}",
                provider=self.provider_name,
                model=self.model,
                cause=exc,
            ) from exc
        except httpx.RequestError as exc:
            raise LLMConnectionError(
                f"Failed to connect to Ollama at {self._chat_url}: {exc}",
                provider=self.provider_name,
                model=self.model,
                cause=exc,
            ) from exc

        if response.status_code >= 400:
            raise LLMConnectionError(
                f"Ollama returned HTTP {response.status_code} from {self._chat_url}",
                provider=self.provider_name,
                model=self.model,
                status_code=response.status_code,
                raw_response=response.text[:500],
            )

        try:
            envelope = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseParseError(
                f"Ollama returned non-JSON envelope from {self._chat_url}: {exc}",
                provider=self.provider_name,
                model=self.model,
                status_code=response.status_code,
                cause=exc,
                raw_response=response.text[:500],
            ) from exc

        if not isinstance(envelope, dict):
            raise LLMResponseParseError(
                "Ollama envelope must be a JSON object",
                provider=self.provider_name,
                model=self.model,
                status_code=response.status_code,
                raw_response=response.text[:500],
            )

        if envelope.get("error"):
            raise LLMConnectionError(
                f"Ollama API error: {envelope['error']}",
                provider=self.provider_name,
                model=self.model,
                status_code=response.status_code,
                raw_response=response.text[:500],
            )

        message_obj = envelope.get("message")
        content = message_obj.get("content") if isinstance(message_obj, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseParseError(
                "Ollama response envelope missing non-empty 'message.content'",
                provider=self.provider_name,
                model=self.model,
                status_code=response.status_code,
                raw_response=response.text[:500],
            )

        parsed = extract_json_object(content, provider=self.provider_name, model=self.model)
        return validate_and_ground_response(
            parsed,
            bundle,
            provider=self.provider_name,
            model=self.model,
        )

    def healthcheck(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            with self._client_ctx() as client:
                response = client.get(
                    self._tags_url,
                    headers=self._headers or None,
                    timeout=self.timeout_s,
                )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return ProviderHealth(
                provider=self.provider_name,
                model=self.model,
                base_url=self.base_url,
                healthy=False,
                model_available=False,
                detail=f"Connection error probing {self._tags_url}: {exc}",
                latency_ms=elapsed_ms,
            )

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if response.status_code >= 400:
            return ProviderHealth(
                provider=self.provider_name,
                model=self.model,
                base_url=self.base_url,
                healthy=False,
                model_available=False,
                detail=f"HTTP {response.status_code} from {self._tags_url}",
                latency_ms=elapsed_ms,
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            return ProviderHealth(
                provider=self.provider_name,
                model=self.model,
                base_url=self.base_url,
                healthy=False,
                model_available=False,
                detail=f"Invalid JSON from {self._tags_url}: {exc}",
                latency_ms=elapsed_ms,
            )

        models_raw = data.get("models", []) if isinstance(data, dict) else []
        available_names: set[str] = set()
        if isinstance(models_raw, list):
            for item in models_raw:
                if isinstance(item, dict):
                    for key in ("name", "model"):
                        val = item.get(key)
                        if isinstance(val, str) and val.strip():
                            available_names.add(val.strip())

        model_available = _ollama_model_matches(self.model, available_names)
        detail = (
            f"Ollama reachable; model '{self.model}' available"
            if model_available
            else f"Ollama reachable at {self._tags_url}, but model '{self.model}' was not found"
        )
        return ProviderHealth(
            provider=self.provider_name,
            model=self.model,
            base_url=self.base_url,
            healthy=model_available,
            model_available=model_available,
            detail=detail,
            latency_ms=elapsed_ms,
        )
