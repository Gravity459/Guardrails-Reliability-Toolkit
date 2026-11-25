"""Runners: what reaches (fake) Gemini, raw results saved, cache behaviour."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import llm_client
from app.prompts import SYSTEM_PROMPT
from benchmark import baseline_runner, guarded_runner, run_utils

ALL_FIELDS = '{"model":"Pixel 8","storage":"128GB","camera":"50MP dual","battery":"4575mAh","year":2023}'
PROMPTS = [
    {"id": "p1", "prompt": "Pixel 8, 128GB", "expected_failure": "none", "answer": "refuse"},
    {"id": "p2", "prompt": "Source: ", "expected_failure": "empty input", "answer": "refuse"},
]


def _fake_client(text=ALL_FIELDS):
    models = MagicMock()
    models.generate_content.return_value = SimpleNamespace(text=text)
    return models, SimpleNamespace(models=models)


def test_baseline_call_sends_rules_prompt_and_temperature_zero():
    models, client = _fake_client()
    with patch.object(llm_client, "client", client):
        out = baseline_runner.call("Extract the phone")

    kwargs = models.generate_content.call_args.kwargs
    assert kwargs["contents"] == "Extract the phone"  # regression: prompt is sent as a message
    assert kwargs["config"].system_instruction == SYSTEM_PROMPT
    assert kwargs["config"].temperature == 0
    assert out == {"raw_output": ALL_FIELDS, "error": None}


def test_guarded_call_saves_validated_json_with_temperature_zero():
    models, client = _fake_client()
    with patch.object(llm_client, "client", client):
        out = guarded_runner.call("Extract the phone")

    assert models.generate_content.call_args.kwargs["config"].temperature == 0
    assert out["error"] is None
    assert json.loads(out["raw_output"])["model"] == "Pixel 8"


def test_guarded_call_records_guard_failure():
    with patch.object(guarded_runner, "guarded_call_llm", return_value=None):
        assert guarded_runner.call("x") == {"raw_output": None, "error": "guard_failed"}


def test_run_benchmark_saves_raw_results_and_reuses_cache(tmp_path):
    call = MagicMock(return_value={"raw_output": ALL_FIELDS, "error": None})
    results_path = tmp_path / "results.json"

    first = run_utils.run_benchmark("baseline", call, "gemini-test", results_path, PROMPTS, cache_dir=tmp_path)
    second = run_utils.run_benchmark("baseline", call, "gemini-test", results_path, PROMPTS, cache_dir=tmp_path)

    assert call.call_count == 2  # only the first run called the model
    assert [r["cached"] for r in first["results"]] == [False, False]
    assert [r["cached"] for r in second["results"]] == [True, True]
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert saved["meta"]["temperature"] == 0
    assert set(saved["results"][0]) == {
        "id", "expected_failure", "model", "raw_output", "latency_ms", "error", "cached"
    }
    assert "passed" not in saved["results"][0]  # runners don't grade


def test_llm_call_failures_are_not_cached(tmp_path):
    call = MagicMock(side_effect=RuntimeError("429 quota"))
    payload = run_utils.run_benchmark("baseline", call, "m", tmp_path / "r.json", PROMPTS[:1], cache_dir=tmp_path)

    assert payload["results"][0]["error"].startswith("llm_call_failed")
    assert payload["meta"]["llm_call_failed"] == 1
    assert json.loads(run_utils.cache_path("baseline", "m", tmp_path).read_text()) == {}


def test_refresh_ignores_cache(tmp_path):
    call = MagicMock(return_value={"raw_output": ALL_FIELDS, "error": None})
    run_utils.run_benchmark("baseline", call, "m", tmp_path / "r.json", PROMPTS[:1], cache_dir=tmp_path)
    run_utils.run_benchmark("baseline", call, "m", tmp_path / "r.json", PROMPTS[:1], refresh=True, cache_dir=tmp_path)
    assert call.call_count == 2


def test_cache_key_changes_with_system_prompt():
    before = run_utils.cache_key("same prompt")
    with patch.object(run_utils, "SYSTEM_PROMPT", SYSTEM_PROMPT + " changed"):
        assert run_utils.cache_key("same prompt") != before
