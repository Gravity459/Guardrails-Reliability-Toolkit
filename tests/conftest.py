"""Test setup: run with no real Gemini API key, no Guardrails Hub token, no network.

1. app.llm_client builds a genai.Client at import time, which needs *some* key.
   Set a dummy one before any test imports `app`. Every test that reaches
   generate_content mocks it, so no request is sent.

2. app.schemas.guarded_schema imports ValidLength / ValidRange from
   guardrails.hub. Installing those needs a Guardrails Hub token. If they are
   not installed, register small stand-ins with the same names and arguments
   so `app` can be imported. Real validators are used when present.
"""

import os

os.environ.setdefault("LLM_API_KEY", "test-dummy-key")
os.environ.setdefault("LLM_PRIMARY_MODEL", "gemini-test")


def _install_hub_validator_stubs() -> None:
    import guardrails.hub as hub
    from guardrails import Validator, register_validator
    from guardrails.validator_base import FailResult, PassResult

    try:
        hub.ValidLength  # noqa: B018 - real validator installed
        hub.ValidRange  # noqa: B018
        return
    except ImportError:
        pass

    @register_validator("test-stub/valid_length", data_type=["string", "list"])
    class ValidLength(Validator):
        def __init__(self, min=None, max=None, on_fail=None):
            super().__init__(on_fail=on_fail, min=min, max=max)
            self._min, self._max = min, max

        def validate(self, value, metadata):
            n = len(value)
            if (self._min is not None and n < self._min) or (
                self._max is not None and n > self._max
            ):
                return FailResult(error_message=f"length {n} not in [{self._min}, {self._max}]")
            return PassResult()

    @register_validator("test-stub/valid_range", data_type=["integer", "float"])
    class ValidRange(Validator):
        def __init__(self, min=None, max=None, on_fail=None):
            super().__init__(on_fail=on_fail, min=min, max=max)
            self._min, self._max = min, max

        def validate(self, value, metadata):
            if (self._min is not None and value < self._min) or (
                self._max is not None and value > self._max
            ):
                return FailResult(error_message=f"{value} not in [{self._min}, {self._max}]")
            return PassResult()

    hub.ValidLength = ValidLength
    hub.ValidRange = ValidRange


_install_hub_validator_stubs()
