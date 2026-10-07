"""Offline self-check for contextualize._invoke_llm's error handling. No network: the LLM is stubbed."""
import sys
from pathlib import Path

import httpx
import openai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # not installed: package sits at the repo root

from r_doc_builder import contextualize


def test_invoke_llm_fails_fast_on_bad_requests():
    """A 400 is the request itself (gpt-6-luna with reasoning on rejects temperature 0): every
    chunk fails the same way, so stop the build with the message instead of retrying RETRIES
    times per chunk and silently embedding raw text. A content-filter 400 is about one chunk:
    that chunk gets raw text, without retries."""
    calls = []

    def rejecting(message):
        class Rejects:
            def invoke(self, prompt):
                calls.append(prompt)
                response = httpx.Response(400, request=httpx.Request("POST", "https://x/v1/chat/completions"))
                raise openai.BadRequestError(message, response=response, body=None)
        return lambda: Rejects()

    originals = (contextualize.get_llm, contextualize.time.sleep)
    contextualize.time.sleep = lambda seconds: None
    try:
        contextualize.get_llm = rejecting("Unsupported value: 'temperature' does not support 0.0 with this model.")
        try:
            contextualize._invoke_llm("p")
            raise AssertionError("a bad request must stop the build")
        except openai.BadRequestError as e:
            assert "temperature" in str(e)
        assert len(calls) == 1, "no retries"

        calls.clear()
        contextualize.get_llm = rejecting("The response was filtered: 'code': 'content_filter'")
        assert contextualize._invoke_llm("p") is None and len(calls) == 1, "this chunk only: raw text, no retries"
    finally:
        contextualize.get_llm, contextualize.time.sleep = originals


def test_reasoning_effort_settings():
    """COMPLETION_REASONING_EFFORT -> what get_llm sends; each rule is what the live API returned:
    unset: gpt-4o-mini rejects any reasoning_effort, even "none"
    none: gpt-6-luna with reasoning off takes temperature 0
    medium: a reasoning gpt-6-luna takes only its default temperature (plain calls: chat completions)"""
    import importlib
    import os

    from r_doc_builder import config

    keys = ("COMPLETION_REASONING_EFFORT", "COMPLETION_TEMPERATURE", "COMPLETION_MODEL_PROVIDER", "OPENAI_API_KEY")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update(COMPLETION_TEMPERATURE="0", COMPLETION_MODEL_PROVIDER="openai", OPENAI_API_KEY="test")
    try:
        for value, effort, temperature in [("", None, 0.0), ("none", "none", 0.0), ("medium", "medium", None)]:
            os.environ["COMPLETION_REASONING_EFFORT"] = value
            importlib.reload(config)
            params = config.get_llm()._default_params
            assert (params.get("reasoning_effort"), params.get("temperature")) == (effort, temperature), value
            assert ("reasoning_effort" in params) == (effort is not None), "unset is not sent at all"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(config)


def test_provider_selection_and_temperature_reach_every_client():
    """Which client the COMPLETION_* settings build, where it points, and that COMPLETION_TEMPERATURE
    reaches each one: a branch that dropped it would silently sample at the provider's default."""
    import importlib
    import os

    from r_doc_builder import config

    # empty, not unset: reload's load_dotenv would refill an unset one from .env
    base = {"COMPLETION_MODEL_PROVIDER": "", "COMPLETION_URL": "", "COMPLETION_REASONING_EFFORT": "",
            "COMPLETION_TEMPERATURE": "0.5", "COMPLETION_MODEL": "m", "OPENAI_API_KEY": "test"}
    cases = [  # env -> (client class, the endpoint attribute that must carry COMPLETION_URL)
        ({}, ("ChatOpenAI", None)),
        ({"COMPLETION_URL": "https://res.services.ai.azure.com/openai/v1"}, ("ChatOpenAI", "openai_api_base")),
        ({"COMPLETION_URL": "https://res.openai.azure.com"}, ("AzureChatOpenAI", "azure_endpoint")),
        ({"COMPLETION_URL": "http://gpu:8000/v1"}, ("ChatOpenAI", "openai_api_base")),  # vllm
        ({"COMPLETION_URL": "http://localhost:11434"}, ("ChatOllama", "base_url")),  # e.g. a kubectl tunnel
        ({"COMPLETION_URL": "http://gpu:8000/v1", "COMPLETION_MODEL_PROVIDER": "Ollama  # local"},
         ("ChatOllama", "base_url")),  # explicit provider wins; a trailing .env comment is dropped
    ]
    saved = {k: os.environ.get(k) for k in base}
    try:
        os.environ.update(base)
        importlib.reload(config)
        for env, (cls, endpoint) in cases:
            os.environ.update({**base, **env})
            config.get_llm.cache_clear()
            llm = config.get_llm()
            assert type(llm).__name__ == cls, (env, type(llm).__name__)
            assert llm.temperature == 0.5, env
            assert endpoint is None or getattr(llm, endpoint).rstrip("/") == env["COMPLETION_URL"], env
        os.environ["COMPLETION_MODEL_PROVIDER"] = "bedrock"
        config.get_llm.cache_clear()
        try:
            config.get_llm()
            raise AssertionError("an unknown provider must fail")
        except ValueError as e:
            assert "bedrock" in str(e)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(config)


if __name__ == "__main__":
    test_invoke_llm_fails_fast_on_bad_requests()
    test_reasoning_effort_settings()
    test_provider_selection_and_temperature_reach_every_client()
    print("ok")
