from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app import llm_client


def _client_error(payload: dict, *, headers: dict | None = None) -> genai_errors.ClientError:
    response = None
    if headers is not None:
        response = SimpleNamespace(headers=headers)
    return genai_errors.ClientError(429, payload, response)


def _retry_info_payload(
    retry_delay: str | None,
    *,
    nested: bool = True,
    quota_id: str | None = None,
    message: str | None = None,
) -> dict:
    details: list[dict] = []
    if quota_id is not None:
        details.append(
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": quota_id}],
            }
        )
    if retry_delay is not None:
        details.append(
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": retry_delay,
            }
        )
    error = {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "message": message
        or "You exceeded your current quota. Please retry in 53.016342224s.",
        "details": details,
    }
    return {"error": error} if nested else error


class TestParseRetryDelay:
    def test_nested_details_53s(self):
        err = _client_error(_retry_info_payload("53s", nested=True))
        assert llm_client.parse_retry_delay_seconds(err) == 53.0

    def test_flat_details_fractional(self):
        err = _client_error(_retry_info_payload("34.4s", nested=False))
        assert llm_client.parse_retry_delay_seconds(err) == 34.4

    def test_missing_retry_delay_returns_none_without_message_match(self):
        payload = {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "rate limit",
                "details": [],
            }
        }
        err = _client_error(payload)
        assert llm_client.parse_retry_delay_seconds(err) is None

    def test_retry_after_header_fallback(self):
        payload = {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "rate limit",
                "details": [],
            }
        }
        err = _client_error(payload, headers={"Retry-After": "12"})
        assert llm_client.parse_retry_delay_seconds(err) == 12.0

    def test_message_fallback(self):
        payload = {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Please retry in 7.5s.",
                "details": [],
            }
        }
        err = _client_error(payload)
        assert llm_client.parse_retry_delay_seconds(err) == 7.5


class TestDailyQuota:
    def test_per_day_quota_id(self):
        err = _client_error(
            _retry_info_payload(
                "1s",
                quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            )
        )
        assert llm_client.is_daily_quota_error(err) is True

    def test_per_minute_quota_id(self):
        err = _client_error(
            _retry_info_payload(
                "53s",
                quota_id="GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
            )
        )
        assert llm_client.is_daily_quota_error(err) is False


class TestNextWait:
    def test_first_wait_uses_server_delay(self):
        assert llm_client._next_wait(53.0, None) == 53.0

    def test_first_wait_falls_back_to_initial_delay(self, monkeypatch):
        monkeypatch.setenv("LLM_RETRY_INITIAL_DELAY", "1.0")
        assert llm_client._next_wait(None, None) == 1.0

    def test_later_wait_grows_and_respects_server_delay(self, monkeypatch):
        monkeypatch.setenv("LLM_RETRY_MAX_DELAY", "60")
        assert llm_client._next_wait(12.0, 53.0) == 60.0

    def test_cap(self, monkeypatch):
        monkeypatch.setenv("LLM_RETRY_MAX_DELAY", "60")
        assert llm_client._next_wait(90.0, None) == 60.0


class TestCallLlm:
    def test_retries_after_parsed_delay_then_succeeds(self, monkeypatch):
        sleeps: list[float] = []
        calls = {"n": 0}
        err = _client_error(_retry_info_payload("53s"))

        def fake_generate_content(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise err
            return SimpleNamespace(text='{"model": "Pixel"}')

        monkeypatch.setattr(llm_client, "_sleep", sleeps.append)
        monkeypatch.setattr(
            llm_client.client.models, "generate_content", fake_generate_content
        )

        result = llm_client.call_llm(
            [{"role": "user", "content": "hello"}], model="test-model"
        )

        assert result == '{"model": "Pixel"}'
        assert sleeps == [53.0]
        assert calls["n"] == 2

    def test_daily_quota_does_not_sleep(self, monkeypatch):
        sleeps: list[float] = []
        err = _client_error(
            _retry_info_payload(
                "1s",
                quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            )
        )

        def fake_generate_content(**kwargs):
            raise err

        monkeypatch.setattr(llm_client, "_sleep", sleeps.append)
        monkeypatch.setattr(
            llm_client.client.models, "generate_content", fake_generate_content
        )

        with pytest.raises(genai_errors.ClientError) as exc_info:
            llm_client.call_llm(
                [{"role": "user", "content": "hello"}], model="test-model"
            )

        assert exc_info.value is err
        assert sleeps == []

    def test_exhausted_retries_reraise_client_error(self, monkeypatch):
        sleeps: list[float] = []
        err = _client_error(_retry_info_payload("2s"))
        monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "3")

        def fake_generate_content(**kwargs):
            raise err

        monkeypatch.setattr(llm_client, "_sleep", sleeps.append)
        monkeypatch.setattr(
            llm_client.client.models, "generate_content", fake_generate_content
        )

        with pytest.raises(genai_errors.ClientError) as exc_info:
            llm_client.call_llm(
                [{"role": "user", "content": "hello"}], model="test-model"
            )

        assert exc_info.value is err
        assert sleeps == [2.0, 4.0]
