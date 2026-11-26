"""Self-hosted OpenAI-compatible HTTP adapter (vLLM, llama.cpp, LocalAI, LM Studio)."""

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


def _resolve_openai_compat_urls(base_url: str) -> tuple[str, str]:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        prefix = base_url[: -len("/chat/completions")]
        return base_url, f"{prefix}/models"
    if path.endswith("/v1"):
        return f"{base_url}/chat/completions", f"{base_url}/models"
    return f"{base_url}/v1/chat/completions", f"{base_url}/v1/models"


class OpenAICompatibleProvider:
    """Self-hosted OpenAI-compatible /v1/chat/completions adapter over httpx."""

    provider_name: str = "openai_compatible"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "llama3.1:8b",
        *,
        api_key: str | None = None,
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
        self.api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.timeout_s = validated_timeout
        self.temperature = float(temperature)
        self._headers = self._build_headers(headers, self.api_key)
        self._injected_client = client
        self._transport = transport
        self._chat_url, self._models_url = _resolve_openai_compat_urls(self.base_url)

    @staticmethod
    def _build_headers(
        headers: dict[str, str] | None,
        api_key: str | None,
    ) -> dict[str, str]:
        merged = dict(headers) if headers else {}
        if api_key and not any(k.lower() == "authorization" for k in merged):
            merged["Authorization"] = f"Bearer {api_key}"
        return merged

    @contextmanager
    def _client_ctx(self) -> Iterator[httpx.Client]:
        if self._injected_client is not None:
            yield self._injected_client
            return
        with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
            yield client

    def _post_chat(self, payload: dict[str, object]) -> httpx.Response:
        try:
            with self._client_ctx() as client:
                return client.post(
                    self._chat_url,
                    json=payload,
                    headers=self._headers or None,
                    timeout=self.timeout_s,
                )
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(
                f"OpenAI-compatible request timed out after {self.timeout_s}s at {self._chat_url}",
                provider=self.provider_name,
                model=self.model,
                cause=exc,
            ) from exc
        except httpx.RequestError as exc:
            raise LLMConnectionError(
                f"Failed to connect to OpenAI-compatible endpoint {self._chat_url}: {exc}",
                provider=self.provider_name,
                model=self.model,
                cause=exc,
            ) from exc

    def triage(
        self,
        bundle: IncidentBundle,
        runbooks: Sequence[RunbookTemplate],
    ) -> LLMTriageResponse:
        messages = build_triage_messages(bundle, runbooks)
        request_body: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
        }

        response = self._post_chat(request_body)
        if response.status_code in (400, 422) and "response_format" in response.text.lower():
            fallback_body = {
                key: value for key, value in request_body.items() if key != "response_format"
            }
            response = self._post_chat(fallback_body)

        if response.status_code >= 400:
            raise LLMConnectionError(
                f"OpenAI-compatible endpoint returned HTTP {response.status_code} from {self._chat_url}",
                provider=self.provider_name,
                model=self.model,
                status_code=response.status_code,
                raw_response=response.text[:500],
            )

        try:
            envelope = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseParseError(
                f"OpenAI-compatible endpoint returned non-JSON envelope: {exc}",
                provider=self.provider_name,
                model=self.model,
                status_code=response.status_code,
                cause=exc,
                raw_response=response.text[:500],
            ) from exc

        if not isinstance(envelope, dict):
            raise LLMResponseParseError(
                "OpenAI-compatible response envelope must be a JSON object",
                provider=self.provider_name,
                model=self.model,
                status_code=response.status_code,
                raw_response=response.text[:500],
            )

        if envelope.get("error"):
            raise LLMConnectionError(
                f"OpenAI-compatible API error: {envelope['error']}",
                provider=self.provider_name,
                model=self.model,
                status_code=response.status_code,
                raw_response=response.text[:500],
            )

        choices = envelope.get("choices")
        content: str | None = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                content = msg["content"]

        if not content or not content.strip():
            raise LLMResponseParseError(
                "OpenAI-compatible response missing non-empty 'choices[0].message.content'",
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
                    self._models_url,
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
                detail=f"Connection error probing {self._models_url}: {exc}",
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
                detail=f"HTTP {response.status_code} from {self._models_url}",
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
                detail=f"Invalid JSON from {self._models_url}: {exc}",
                latency_ms=elapsed_ms,
            )

        models_list = data.get("data") if isinstance(data, dict) else None
        if isinstance(models_list, list) and len(models_list) > 0:
            available_ids = {
                item["id"].strip()
                for item in models_list
                if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
            }
            model_available = self.model in available_ids
            detail = (
                f"Endpoint reachable; model '{self.model}' available"
                if model_available
                else f"Endpoint reachable at {self._models_url}, but model '{self.model}' was not listed"
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

        return ProviderHealth(
            provider=self.provider_name,
            model=self.model,
            base_url=self.base_url,
            healthy=True,
            model_available=None,
            detail=f"Endpoint reachable at {self._models_url} (model list omitted by server)",
            latency_ms=elapsed_ms,
        )
