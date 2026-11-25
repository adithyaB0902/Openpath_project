"""Swappable LLM clients. Each has complete(system, user) -> text (expected to be a JSON object)."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

import requests


class LLMError(RuntimeError):
    """The LLM could not be reached or returned an error."""


class _Http:
    def __init__(self, session: Any = None, sleep: Callable[[float], None] = time.sleep,
                 timeout: int = 60, retries: int = 1):
        self.session = session or requests.Session()
        self.sleep, self.timeout, self.retries = sleep, timeout, retries

    def post(self, url: str, headers: dict, payload: dict) -> dict:
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt < self.retries:
                    self.sleep(2 * (attempt + 1))
                    continue
                raise LLMError(f"could not reach the LLM at {url} ({exc.__class__.__name__})") from exc
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                self.sleep(2 * (attempt + 1))
                continue
            if resp.status_code >= 400:
                raise LLMError(f"LLM request failed: HTTP {resp.status_code}: {(resp.text or '')[:200]}")
            try:
                return resp.json()
            except ValueError as exc:
                raise LLMError("LLM returned a non-JSON response") from exc
        raise LLMError("LLM request failed")  # pragma: no cover


class OllamaClient(_Http):
    """Local models via Ollama (https://ollama.com). No API key, no cost."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434", **kw):
        super().__init__(**kw)
        self.model, self.base_url = model, base_url.rstrip("/")

    def complete(self, system: str, user: str) -> str:
        data = self.post(f"{self.base_url}/api/chat", {}, {
            "model": self.model, "stream": False, "format": "json", "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("unexpected Ollama response shape") from exc


class OpenAICompatClient(_Http):
    """Any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str = "", **kw):
        super().__init__(**kw)
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key

    def complete(self, system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "temperature": 0.2,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        url = f"{self.base_url}/chat/completions"
        try:
            data = self.post(url, headers, {**payload, "response_format": {"type": "json_object"}})
        except LLMError as exc:
            if "HTTP 400" not in str(exc):  # some providers reject response_format; retry without it
                raise
            data = self.post(url, headers, payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected chat-completions response shape") from exc


class AnthropicClient(_Http):
    """Claude via the Anthropic Messages API (paid, needs an API key)."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.anthropic.com", **kw):
        super().__init__(**kw)
        self.model, self.api_key, self.base_url = model, api_key, base_url.rstrip("/")

    def complete(self, system: str, user: str) -> str:
        data = self.post(f"{self.base_url}/v1/messages",
                         {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                         {"model": self.model, "max_tokens": 1024, "temperature": 0.2, "system": system,
                          "messages": [{"role": "user", "content": user}]})
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError, AttributeError) as exc:
            raise LLMError("unexpected Anthropic response shape") from exc


class ScriptedLLM:
    """Test double: returns queued replies (str, dict, Exception, or callable(system, user))."""

    def __init__(self, replies: list):
        self.replies, self.calls = list(replies), []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.replies:
            raise LLMError("script exhausted")
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(system, user)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def get_llm(settings: Any, session: Any = None) -> Optional[Any]:
    """Build the configured client, or None to use the built-in question flow."""
    provider = (settings.llm_provider or "none").strip().lower()
    kw = {"session": session} if session else {}
    if provider in ("none", ""):
        return None
    if provider == "ollama":
        return OllamaClient(settings.llm_model or "llama3.2", settings.llm_base_url or "http://localhost:11434", **kw)
    if provider == "openai":
        if not settings.llm_model:
            raise ValueError("OPENPATH_LLM_MODEL is required for provider 'openai'")
        return OpenAICompatClient(settings.llm_model, settings.llm_base_url or "https://api.openai.com/v1",
                                  settings.llm_api_key, **kw)
    if provider == "anthropic":
        if not (settings.llm_model and settings.llm_api_key):
            raise ValueError("OPENPATH_LLM_MODEL and OPENPATH_LLM_API_KEY are required for provider 'anthropic'")
        return AnthropicClient(settings.llm_model, settings.llm_api_key, **kw)
    raise ValueError(f"unknown OPENPATH_LLM provider {provider!r} (use none, ollama, openai or anthropic)")
