"""Fallback model: when it is used, when it is not, the budget, and caching."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai import errors as genai_errors

from app import guard as guard_module
from app import llm_client
from app.llm_client import FALLBACK_BUDGET, FALLBACK_MODEL, call_with_fallback
from benchmark import baseline_runner, guarded_runner, run_utils

PRIMARY = "gemini-test"  # set by conftest
GOOD = {"model": "Pixel 8", "storage": "256GB", "camera": "50MP", "battery": "5000mAh", "year": 2023}
MESSAGES = [{"role": "user", "content": "x"}]


def _server_error():
    return genai_errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})


def _daily_quota():
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}]
    return genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "daily", "details": details}})


def _per_minute_429():
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1s"}]
    return genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "per minute", "details": details}})


def _client(*effects):
    """Fake Gemini: each item is an exception to raise or a dict to return as JSON text."""
    models = MagicMock()
    models.generate_content.side_effect = [
        e if isinstance(e, BaseException) else SimpleNamespace(text=json.dumps(e)) for e in effects
    ]
    return models, SimpleNamespace(models=models)


def _models_called(models):
    return [c.kwargs["model"] for c in models.generate_content.call_args_list]


@pytest.mark.parametrize(
    "error, reason",
    [(_server_error(), "server_error"), (httpx.ReadTimeout("slow"), "timeout"), (_daily_quota(), "daily_quota")],
    ids=["503", "timeout", "daily-quota"],
)
def test_each_trigger_switches_to_fallback(error, reason):
    models, client = _client(error, GOOD)
    with patch.object(llm_client, "client", client):
        result = call_with_fallback(MESSAGES, PRIMARY)
    assert _models_called(models) == [PRIMARY, FALLBACK_MODEL]
    assert result["model_used"] == FALLBACK_MODEL and result["fallback_reason"] == reason
    assert llm_client.fallback_budget.used == 1


def test_per_minute_429_waits_and_retries_primary_not_fallback():
    models, client = _client(_per_minute_429(), GOOD)
    with patch.object(llm_client, "client", client), patch.object(llm_client, "_sleep"):
        result = call_with_fallback(MESSAGES, PRIMARY)
    assert _models_called(models) == [PRIMARY, PRIMARY]
    assert result["fallback_reason"] is None and llm_client.fallback_budget.used == 0


def test_other_errors_do_not_use_fallback():
    models, client = _client(genai_errors.ClientError(400, {"error": {"code": 400, "message": "bad request"}}))
    with patch.object(llm_client, "client", client), pytest.raises(genai_errors.ClientError):
        call_with_fallback(MESSAGES, PRIMARY)
    assert _models_called(models) == [PRIMARY]


def test_normal_call_never_touches_fallback():
    models, client = _client(GOOD)
    with patch.object(llm_client, "client", client):
        out = baseline_runner.call("Source: Pixel 8")
    assert _models_called(models) == [PRIMARY]
    assert out["model_used"] == PRIMARY and out["fallback_reason"] is None


def test_budget_stops_at_limit_then_row_is_llm_call_failed(tmp_path):
    n = FALLBACK_BUDGET + 2
    prompts = [{"id": f"p{i}", "prompt": f"prompt {i}", "expected_failure": "none", "answer": "refuse"} for i in range(n)]

    def fake_generate(*, model, **kwargs):
        if model == PRIMARY:
            raise _server_error()
        return SimpleNamespace(text=json.dumps(GOOD))

    models = MagicMock()
    models.generate_content.side_effect = fake_generate
    with patch.object(llm_client, "client", SimpleNamespace(models=models)):
        payload = run_utils.run_benchmark("baseline", baseline_runner.call, PRIMARY, tmp_path / "r.json", prompts, cache_dir=tmp_path)

    meta, rows = payload["meta"], payload["results"]
    assert meta["fallback_calls"] == FALLBACK_BUDGET  # never more than 20 fallback calls
    assert meta["fallback_used"] == FALLBACK_BUDGET
    assert meta["llm_call_failed"] == 2  # the rest: primary failed, no budget left
    assert all(r["error"].startswith("llm_call_failed") for r in rows[FALLBACK_BUDGET:])
    assert _models_called(models).count(FALLBACK_MODEL) == FALLBACK_BUDGET


def test_failed_reask_falls_back_once_to_fallback_model():
    bad = {**GOOD, "year": 1999}
    models, client = _client(bad, bad, GOOD)  # primary: answer + reask fail; fallback fixes it
    with patch.object(llm_client, "client", client):
        result = guard_module.guarded_call("Source: Pixel 8 from 2023")
    assert _models_called(models) == [PRIMARY, PRIMARY, FALLBACK_MODEL]
    assert result == {"output": GOOD, "model_used": FALLBACK_MODEL, "fallback_reason": "guard_failed"}


def test_failed_reask_without_budget_stays_guard_failed():
    llm_client.fallback_budget.used = FALLBACK_BUDGET
    bad = {**GOOD, "year": 1999}
    models, client = _client(bad, bad)
    with patch.object(llm_client, "client", client):
        out = guarded_runner.call("Source: x")
    assert _models_called(models) == [PRIMARY, PRIMARY]
    assert out["error"] == "guard_failed" and out["fallback_reason"] is None


def test_guarded_server_error_uses_fallback_inside_guard():
    models, client = _client(_server_error(), GOOD)
    with patch.object(llm_client, "client", client):
        result = guard_module.guarded_call("Source: Pixel 8")
    assert result["output"] == GOOD
    assert result["model_used"] == FALLBACK_MODEL and result["fallback_reason"] == "server_error"


def test_fallback_answers_are_cached_per_model(tmp_path):
    prompts = [{"id": "p1", "prompt": "prompt 1", "expected_failure": "none", "answer": "refuse"},
               {"id": "p2", "prompt": "prompt 2", "expected_failure": "none", "answer": "refuse"}]
    models, client = _client(_server_error(), GOOD, GOOD)  # p1 via fallback, p2 via primary
    args = ("baseline", baseline_runner.call, PRIMARY, tmp_path / "r.json", prompts)
    with patch.object(llm_client, "client", client):
        run_utils.run_benchmark(*args, cache_dir=tmp_path)

    primary_cache = json.loads(run_utils.cache_path("baseline", PRIMARY, tmp_path).read_text(encoding="utf-8"))
    fallback_cache = json.loads(run_utils.cache_path("baseline", FALLBACK_MODEL, tmp_path).read_text(encoding="utf-8"))
    k1, k2 = run_utils.cache_key("prompt 1"), run_utils.cache_key("prompt 2")
    assert set(primary_cache) == {k2} and set(fallback_cache) == {k1}
    assert fallback_cache[k1]["model_used"] == FALLBACK_MODEL

    # Rerun: both come from the caches, no calls, model_used kept.
    with patch.object(llm_client, "client", client):
        again = run_utils.run_benchmark(*args, cache_dir=tmp_path)
    assert models.generate_content.call_count == 3
    assert [r["model_used"] for r in again["results"]] == [FALLBACK_MODEL, PRIMARY]


def test_existing_primary_cache_still_hits():
    # The committed baseline cache (no model_used field) must keep working.
    cache = json.loads(run_utils.cache_path("baseline", "gemini-3.5-flash-lite").read_text(encoding="utf-8"))
    prompts = run_utils.load_prompts()
    assert sum(run_utils.cache_key(p["prompt"]) in cache for p in prompts) == 200
    row = run_utils.run_one(prompts[0], MagicMock(), "gemini-3.5-flash-lite", cache)
    assert row["cached"] and row["model_used"] == "gemini-3.5-flash-lite" and row["fallback_reason"] is None
