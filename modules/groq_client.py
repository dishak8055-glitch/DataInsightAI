"""Shared Groq configuration and cached client for all remote LLM features."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-20b")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

try:
    from groq import Groq
except ImportError:
    Groq = None

_cached_client = None


def get_groq_api_key():
    """Read the Groq key from the environment or Streamlit Secrets."""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.strip():
        return api_key.strip()

    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return None

    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
    except (StreamlitSecretNotFoundError, KeyError):
        return None
    return (
        secret_key.strip()
        if isinstance(secret_key, str) and secret_key.strip()
        else None
    )


def get_groq_client():
    """Return one cached Groq SDK client with retries disabled."""
    global _cached_client
    if _cached_client is not None:
        return _cached_client
    if Groq is None:
        raise RuntimeError("The groq package is not installed.")

    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError(
            "Groq API key not found. Set GROQ_API_KEY in the environment, "
            "the project .env file, or Streamlit Secrets."
        )

    _cached_client = Groq(
        api_key=api_key,
        timeout=30.0,
        max_retries=0,
    )
    return _cached_client


def reset_groq_client():
    """Clear the cached client; primarily useful for isolated tests."""
    global _cached_client
    _cached_client = None


def summarize_groq_error(error):
    """Map provider failures to actionable, key-safe messages."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if isinstance(error, ValueError) and "API key not found" in str(error):
        return (
            "Groq API key is not configured. Set GROQ_API_KEY in the environment "
            "or Streamlit Secrets."
        )
    if status == 429:
        return "Groq rate limit or quota reached. Wait before making another request."
    if status is not None and int(status) >= 500:
        return "Groq is temporarily unavailable. Try again later."
    if isinstance(error, (TimeoutError,)) or "timeout" in type(error).__name__.lower():
        return "The Groq request timed out. Try again later."
    if status in {401, 403}:
        return f"Groq rejected the request (HTTP {status}); verify the key and model access."
    if status is not None:
        return f"Groq rejected the request (HTTP {status}); verify model access and request settings."
    return f"Groq request failed ({type(error).__name__})."


def completion_text(response):
    """Extract non-empty assistant content from an OpenAI-compatible response."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Groq returned an invalid chat completion response.") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Groq returned an empty response.")
    return content.strip()
