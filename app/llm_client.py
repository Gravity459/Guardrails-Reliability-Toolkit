from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


_DEFAULT_ATTEMPTS = 5
_DEFAULT_INITIAL_DELAY = 1.0
_DEFAULT_MAX_DELAY = 60.0 
_DEFAULT_EXP_BASE = 2.0
_DEFAULT_JITTER = 1.0
_RETRY_STATUS_CODES = [429]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def build_retry_options(
    *,
    attempts: int | None = None,
    initial_delay: float | None = None,
    max_delay: float | None = None,
) -> types.HttpRetryOptions:
    return types.HttpRetryOptions(
        attempts=(
            attempts
            if attempts is not None
            else _env_int("LLM_RETRY_ATTEMPTS", _DEFAULT_ATTEMPTS)
        ),
        initial_delay=(
            initial_delay
            if initial_delay is not None
            else _env_float("LLM_RETRY_INITIAL_DELAY", _DEFAULT_INITIAL_DELAY)
        ),
        max_delay=(
            max_delay
            if max_delay is not None
            else _env_float("LLM_RETRY_MAX_DELAY", _DEFAULT_MAX_DELAY)
        ),
        exp_base=_DEFAULT_EXP_BASE,
        jitter=_DEFAULT_JITTER,
        http_status_codes=list(_RETRY_STATUS_CODES),
    )


def build_client(
    *,
    api_key: str | None = None,
    retry_options: types.HttpRetryOptions | None = None,
) -> genai.Client:
    """Create a Gemini client with bounded 429 retry handling enabled."""
    return genai.Client(
        api_key=os.getenv("LLM_API_KEY"),
        http_options=types.HttpOptions(
            retry_options=retry_options or build_retry_options(),
        ),
    )


client = build_client()


def call_llm(
    messages: list[dict], model: str = os.getenv("LLM_PRIMARY_MODEL"), **kwargs
) -> str:
    messages = messages or []

    contents = "\n\n".join(
        str(m["content"]) for m in messages if m.get("role") != "system"
    )

    response = client.models.generate_content(
        model=kwargs.get("model", model),
        contents=contents,
    )
    return response.text
