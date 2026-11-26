"""Test setup: run with no real Gemini API key, no network.

1. app.llm_client builds a genai.Client at import time, which needs *some* key.
   Set a dummy one before any test imports `app`. Every test that reaches
   generate_content mocks it, so no request is sent.

"""

import os

os.environ.setdefault("LLM_API_KEY", "test-dummy-key")
os.environ.setdefault("LLM_PRIMARY_MODEL", "gemini-test")
# Run validators synchronously (no asyncio event loop) in tests.
os.environ.setdefault("GUARDRAILS_RUN_SYNC", "true")


def _disable_guardrails_telemetry() -> None:
    """Stop Guardrails sending anonymous metrics over the network during tests.

    Every Guard() re-reads ~/.guardrailsrc (metrics default to on), so force
    its telemetry setting off before `app` creates its guard.
    """
    from guardrails import Guard

    original = Guard._configure_hub_telemtry  # (sic) Guardrails' own spelling
    Guard._configure_hub_telemtry = lambda self, allow_metrics_collection=None: original(
        self, False
    )


_disable_guardrails_telemetry()
