import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

import modules.groq_client as groq_client


def test_groq_client_is_cached_with_timeout_and_retries_disabled(monkeypatch):
    created = []

    def fake_groq(**kwargs):
        created.append(kwargs)
        return object()

    monkeypatch.setattr(groq_client, "_cached_client", None)
    monkeypatch.setattr(groq_client, "Groq", fake_groq)
    monkeypatch.setattr(
        groq_client, "get_groq_api_key", lambda: "test-key-not-a-secret"
    )

    client = groq_client.get_groq_client()

    assert groq_client.get_groq_client() is client
    assert created == [
        {
            "api_key": "test-key-not-a-secret",
            "timeout": 30.0,
            "max_retries": 0,
        }
    ]


def test_groq_client_reports_missing_api_key(monkeypatch):
    monkeypatch.setattr(groq_client, "_cached_client", None)
    monkeypatch.setattr(groq_client, "get_groq_api_key", lambda: None)

    with pytest.raises(ValueError, match="Groq API key not found"):
        groq_client.get_groq_client()


def test_groq_api_key_loads_from_streamlit_secrets(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    streamlit_module = ModuleType("streamlit")
    streamlit_module.__path__ = []
    streamlit_module.secrets = {
        "GROQ_API_KEY": "streamlit-test-key"
    }
    errors_module = ModuleType("streamlit.errors")
    errors_module.StreamlitSecretNotFoundError = type(
        "StreamlitSecretNotFoundError", (Exception,), {}
    )
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)
    monkeypatch.setitem(sys.modules, "streamlit.errors", errors_module)

    assert groq_client.get_groq_api_key() == "streamlit-test-key"


def test_completion_text_extracts_openai_compatible_content():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="  answer  ")
            )
        ]
    )

    assert groq_client.completion_text(response) == "answer"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (type("RateLimitError", (Exception,), {"status_code": 429})(), "rate limit"),
        (type("ServerError", (Exception,), {"status_code": 503})(), "temporarily unavailable"),
        (TimeoutError("secret details"), "timed out"),
    ],
)
def test_provider_errors_are_sanitized(error, message):
    result = groq_client.summarize_groq_error(error)

    assert message in result
    assert "secret details" not in result
