from __future__ import annotations

import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()


_DEFAULT_ATTEMPTS = 5
_DEFAULT_INITIAL_DELAY = 1.0
_DEFAULT_MAX_DELAY = 60.0

_RETRY_INFO_TYPE = "type.googleapis.com/google.rpc.RetryInfo"
_QUOTA_FAILURE_TYPE = "type.googleapis.com/google.rpc.QuotaFailure"
_DURATION_RE = re.compile(r"^(-?)(\d+(?:\.\d+)?)s$")
_MESSAGE_RETRY_RE = re.compile(r"retry in\s+(\d+(?:\.\d+)?)s", re.I)

_sleep = time.sleep


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


def _error_details_list(err: genai_errors.APIError) -> list:
    payload = err.details if isinstance(err.details, dict) else {}
    nested = payload.get("error", payload)
    if not isinstance(nested, dict):
        return []
    details = nested.get("details", [])
    return details if isinstance(details, list) else []


def _parse_duration_seconds(raw: object) -> float | None:
    if isinstance(raw, str):
        match = _DURATION_RE.fullmatch(raw.strip())
        if not match:
            return None
        sign = -1.0 if match.group(1) else 1.0
        return sign * float(match.group(2))
    if isinstance(raw, dict):
        seconds = float(raw.get("seconds") or 0)
        nanos = float(raw.get("nanos") or 0)
        return seconds + nanos / 1e9
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def parse_retry_delay_seconds(err: genai_errors.APIError) -> float | None:
    for detail in _error_details_list(err):
        if not isinstance(detail, dict):
            continue
        if detail.get("@type") != _RETRY_INFO_TYPE:
            continue
        parsed = _parse_duration_seconds(
            detail.get("retryDelay", detail.get("retry_delay"))
        )
        if parsed is not None:
            return parsed

    response = getattr(err, "response", None)
    headers = getattr(response, "headers", None) or {}
    header = None
    if hasattr(headers, "get"):
        header = headers.get("Retry-After") or headers.get("retry-after")
    if header is not None:
        try:
            return float(header)
        except (TypeError, ValueError):
            pass

    if err.message:
        match = _MESSAGE_RETRY_RE.search(err.message)
        if match:
            return float(match.group(1))
    return None


def is_daily_quota_error(err: genai_errors.APIError) -> bool:
    for detail in _error_details_list(err):
        if not isinstance(detail, dict):
            continue
        if detail.get("@type") != _QUOTA_FAILURE_TYPE:
            continue
        for violation in detail.get("violations") or []:
            if not isinstance(violation, dict):
                continue
            quota_id = str(violation.get("quotaId") or violation.get("quota_id") or "")
            if "PerDay" in quota_id:
                return True
    return False


def _next_wait(server_delay: float | None, last_wait: float | None) -> float:
    max_delay = _env_float("LLM_RETRY_MAX_DELAY", _DEFAULT_MAX_DELAY)
    if last_wait is None:
        delay = (
            server_delay
            if server_delay is not None
            else _env_float("LLM_RETRY_INITIAL_DELAY", _DEFAULT_INITIAL_DELAY)
        )
    else:
        delay = max(last_wait * 2, server_delay or 0.0)
    return min(max(delay, 0.0), max_delay)


def build_client() -> genai.Client:
    """Create a Gemini client with SDK auto-retry disabled."""
    return genai.Client(
        api_key=os.getenv("LLM_API_KEY"),
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
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

    attempts = _env_int("LLM_RETRY_ATTEMPTS", _DEFAULT_ATTEMPTS)
    last_wait: float | None = None
    last_error: genai_errors.ClientError | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=kwargs.get("model", model),
                contents=contents,
            )
            return response.text
        except genai_errors.ClientError as e:
            if e.code != 429:
                raise
            if is_daily_quota_error(e):
                raise
            last_error = e
            if attempt == attempts:
                raise
            delay = _next_wait(parse_retry_delay_seconds(e), last_wait)
            last_wait = delay
            print(f"429: waiting {delay}s (RetryInfo)", flush=True)
            _sleep(delay)

    if last_error is not None:
        raise last_error
    raise genai_errors.ClientError(
        429, {"message": "retries exhausted", "status": "RESOURCE_EXHAUSTED"}
    )
