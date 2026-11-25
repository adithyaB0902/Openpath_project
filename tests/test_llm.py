import json
from types import SimpleNamespace

import pytest
import requests

from openpath.interview import llm as L


class FakeResp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self.text = text if text is not None else json.dumps(body)
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, headers, json))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make(cls, responses, **kw):
    sleeps = []
    sess = FakeSession(responses)
    return cls(session=sess, sleep=sleeps.append, **kw), sess, sleeps


def test_ollama_request_and_parse():
    c, sess, _ = make(L.OllamaClient, [FakeResp(body={"message": {"content": '{"a": 1}'}})], model="llama3.2")
    assert c.complete("sys", "usr") == '{"a": 1}'
    url, _, payload = sess.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["format"] == "json" and payload["stream"] is False
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


def test_openai_compat_auth_and_response_format():
    c, sess, _ = make(L.OpenAICompatClient, [FakeResp(body={"choices": [{"message": {"content": "{}"}}]})],
                      model="m", base_url="https://api.example.com/v1/", api_key="secret-key")
    assert c.complete("s", "u") == "{}"
    url, headers, payload = sess.calls[0]
    assert url == "https://api.example.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret-key"
    assert payload["response_format"] == {"type": "json_object"}


def test_openai_compat_retries_without_response_format_on_400():
    c, sess, _ = make(L.OpenAICompatClient, [FakeResp(400, {"error": "unsupported"}),
                                             FakeResp(body={"choices": [{"message": {"content": "{}"}}]})], model="m")
    assert c.complete("s", "u") == "{}"
    assert "response_format" in sess.calls[0][2] and "response_format" not in sess.calls[1][2]


def test_anthropic_headers_and_text_blocks():
    body = {"content": [{"type": "text", "text": "{\"a\":"}, {"type": "text", "text": "1}"}]}
    c, sess, _ = make(L.AnthropicClient, [FakeResp(body=body)], model="claude-x", api_key="k")
    assert c.complete("sys", "usr") == '{"a":1}'
    url, headers, payload = sess.calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "k" and headers["anthropic-version"] == "2023-06-01"
    assert payload["system"] == "sys" and payload["messages"] == [{"role": "user", "content": "usr"}]


def test_retries_on_429_then_succeeds():
    c, sess, sleeps = make(L.OllamaClient, [FakeResp(429, {"e": 1}), FakeResp(body={"message": {"content": "ok"}})])
    assert c.complete("s", "u") == "ok" and sleeps == [2]


def test_error_does_not_leak_api_key():
    c, _, _ = make(L.OpenAICompatClient, [FakeResp(401, {"error": "bad key"})], model="m", api_key="super-secret")
    with pytest.raises(L.LLMError) as exc:
        c.complete("s", "u")
    assert "HTTP 401" in str(exc.value) and "super-secret" not in str(exc.value)


def test_network_error_becomes_llm_error():
    c, _, sleeps = make(L.OllamaClient, [requests.ConnectionError("boom"), requests.ConnectionError("boom")])
    with pytest.raises(L.LLMError, match="could not reach"):
        c.complete("s", "u")
    assert sleeps == [2]


def test_malformed_response_shape():
    c, _, _ = make(L.OllamaClient, [FakeResp(body={"unexpected": True})])
    with pytest.raises(L.LLMError, match="shape"):
        c.complete("s", "u")


def settings(**kw):
    base = dict(llm_provider="none", llm_model="", llm_base_url="", llm_api_key="")
    base.update(kw)
    return SimpleNamespace(**base)


def test_get_llm_providers():
    assert L.get_llm(settings()) is None
    assert isinstance(L.get_llm(settings(llm_provider="ollama")), L.OllamaClient)
    assert L.get_llm(settings(llm_provider="ollama")).model == "llama3.2"
    assert isinstance(L.get_llm(settings(llm_provider="openai", llm_model="m")), L.OpenAICompatClient)
    assert isinstance(L.get_llm(settings(llm_provider="anthropic", llm_model="m", llm_api_key="k")), L.AnthropicClient)


def test_get_llm_config_errors():
    with pytest.raises(ValueError, match="MODEL"):
        L.get_llm(settings(llm_provider="openai"))
    with pytest.raises(ValueError, match="API_KEY"):
        L.get_llm(settings(llm_provider="anthropic", llm_model="m"))
    with pytest.raises(ValueError, match="unknown"):
        L.get_llm(settings(llm_provider="bogus"))


def test_scripted_llm():
    s = L.ScriptedLLM([{"a": 1}, "text", L.LLMError("x"), lambda sys, usr: usr.upper()])
    assert json.loads(s.complete("s", "u")) == {"a": 1}
    assert s.complete("s", "u") == "text"
    with pytest.raises(L.LLMError):
        s.complete("s", "u")
    assert s.complete("s", "abc") == "ABC"
    with pytest.raises(L.LLMError, match="exhausted"):
        s.complete("s", "u")
