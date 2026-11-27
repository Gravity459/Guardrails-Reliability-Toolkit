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
    assert out == {"raw_output": ALL_FIELDS, "error": None, "model_used": "gemini-test", "fallback_reason": None}


def test_guarded_call_saves_validated_json_with_temperature_zero():
    models, client = _fake_client()
    with patch.object(llm_client, "client", client):
        out = guarded_runner.call("Extract the phone")

    assert models.generate_content.call_args.kwargs["config"].temperature == 0
    assert out["error"] is None
    assert json.loads(out["raw_output"])["model"] == "Pixel 8"


def test_guarded_call_records_guard_failure():
    failed = {"output": None, "model_used": "gemini-3.8-flash", "fallback_reason": "guard_failed"}
    with patch.object(guarded_runner, "guarded_call", return_value=failed):
        assert guarded_runner.call("x") == {
            "raw_output": None, "error": "guard_failed", "model_used": "gemini-3.8-flash", "fallback_reason": "guard_failed"
        }


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
        "id", "expected_failure", "model", "raw_output", "latency_ms", "wait_ms", "error", "cached",
        "model_used", "fallback_reason",
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


def test_cache_file_name_uses_single_underscore(tmp_path):
    path = run_utils.cache_path("baseline", "gemini-3.5-flash-lite", tmp_path)
    assert path.name == "baseline_gemini-3.5-flash-lite.json"
    # Matches the cache files committed in the repo.
    assert (run_utils.CACHE_DIR / path.name).exists()


def test_cache_key_changes_with_system_prompt():
    before = run_utils.cache_key("same prompt")
    with patch.object(run_utils, "SYSTEM_PROMPT", SYSTEM_PROMPT + " changed"):
        assert run_utils.cache_key("same prompt") != before


# --- failed calls: guard blips, retry at the end, daily quota -------------------

def _daily_quota_error():
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}]
    return llm_client.genai_errors.ClientError(
        429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "daily", "details": details}}
    )


THREE = [{"id": f"p{i}", "prompt": f"prompt {i}", "expected_failure": "none", "answer": "refuse"} for i in (1, 2, 3)]
OK = {"raw_output": ALL_FIELDS, "error": None}


def test_network_blip_in_guarded_is_llm_call_failed_not_guard_failed(tmp_path):
    models = MagicMock()
    models.generate_content.side_effect = ConnectionError("connection reset")
    with patch.object(llm_client, "client", SimpleNamespace(models=models)):
        payload = run_utils.run_benchmark(
            "guarded", guarded_runner.call, "m", tmp_path / "r.json", PROMPTS[:1], cache_dir=tmp_path
        )
    row = payload["results"][0]
    assert row["error"].startswith("llm_call_failed")  # not "guard_failed"
    assert json.loads(run_utils.cache_path("guarded", "m", tmp_path).read_text()) == {}  # not cached


def test_failed_call_is_retried_at_the_end(tmp_path):
    order = []

    def call(prompt):
        order.append(prompt)
        if prompt == "prompt 1" and order.count(prompt) == 1:
            raise ConnectionError("blip")
        return OK

    payload = run_utils.run_benchmark("baseline", call, "m", tmp_path / "r.json", THREE, cache_dir=tmp_path)
    assert order == ["prompt 1", "prompt 2", "prompt 3", "prompt 1"]  # retry after the main pass
    assert payload["results"][0]["error"] is None and payload["results"][0]["retried"] is True
    assert payload["meta"]["retried"] == 1 and payload["meta"]["llm_call_failed"] == 0
    assert len(json.loads(run_utils.cache_path("baseline", "m", tmp_path).read_text())) == 3  # success cached


def test_failed_twice_stays_failed(tmp_path):
    call = MagicMock(side_effect=ConnectionError("down"))
    payload = run_utils.run_benchmark("baseline", call, "m", tmp_path / "r.json", THREE[:1], cache_dir=tmp_path)
    assert call.call_count == 2  # main pass + one retry, no more
    assert payload["meta"]["llm_call_failed"] == 1


def test_daily_quota_stops_run_and_rerun_resumes_from_cache(tmp_path, capsys):
    calls = []

    def call(prompt):
        calls.append(prompt)
        if prompt == "prompt 2":
            raise _daily_quota_error()
        return OK

    args = ("baseline", call, "m", tmp_path / "r.json", THREE)
    first = run_utils.run_benchmark(*args, cache_dir=tmp_path)
    assert calls == ["prompt 1", "prompt 2"]  # stopped at once: no prompt 3, no retry pass
    errors = [r["error"] for r in first["results"]]
    assert errors[0] is None and "daily_quota" in errors[1] and "not_run" in errors[2]
    assert first["meta"]["stopped_on_daily_quota"] is True and first["meta"]["not_run"] == 1
    assert "Daily quota reached at 2/3. Rerun tomorrow; cached answers will be reused." in capsys.readouterr().out

    calls.clear()
    second = run_utils.run_benchmark("baseline", lambda p: calls.append(p) or OK, "m", tmp_path / "r.json", THREE, cache_dir=tmp_path)
    assert calls == ["prompt 2", "prompt 3"]  # prompt 1 came from the cache
    assert second["meta"]["llm_call_failed"] == 0


def test_daily_quota_found_inside_wrapped_error():
    try:
        try:
            raise _daily_quota_error()
        except Exception as inner:
            raise RuntimeError("guardrails wrapper") from inner
    except RuntimeError as wrapped:
        assert run_utils._is_daily_quota(wrapped)
    assert not run_utils._is_daily_quota(ConnectionError("x"))


# --- #3: guard changes invalidate the guarded cache -----------------------------

def test_cache_salt_changes_key_but_no_salt_keeps_old_keys():
    assert run_utils.cache_key("p", salt="v1") != run_utils.cache_key("p", salt="v2")
    assert run_utils.cache_key("p", salt="") == run_utils.cache_key("p")  # baseline keys unchanged


def test_files_hash_tracks_content_not_line_endings(tmp_path):
    f = tmp_path / "guard.py"
    f.write_bytes(b"a = 1\r\nb = 2\r\n")
    crlf = run_utils.files_hash(f)
    f.write_bytes(b"a = 1\nb = 2\n")
    assert run_utils.files_hash(f) == crlf  # Windows vs CI checkout
    f.write_bytes(b"a = 1\nb = 3\n")
    assert run_utils.files_hash(f) != crlf  # real edit


def test_guarded_cache_misses_after_guard_change(tmp_path):
    call = MagicMock(return_value={"raw_output": ALL_FIELDS, "error": None})
    args = ("guarded", call, "m", tmp_path / "r.json", PROMPTS[:1])
    run_utils.run_benchmark(*args, cache_dir=tmp_path, cache_salt="guard-v1")
    run_utils.run_benchmark(*args, cache_dir=tmp_path, cache_salt="guard-v1")
    assert call.call_count == 1  # same guard: cache hit
    run_utils.run_benchmark(*args, cache_dir=tmp_path, cache_salt="guard-v2")
    assert call.call_count == 2  # guard changed: called again


def test_guarded_runner_salt_is_hash_of_guard_files():
    assert guarded_runner.GUARD_VERSION == run_utils.files_hash(*guarded_runner.GUARD_FILES)
    assert {p.name for p in guarded_runner.GUARD_FILES} == {"guard.py", "guarded_schema.py"}


# --- #4: rate-limit waits are not latency ---------------------------------------

def _quota_error():
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "53s"}]
    return llm_client.genai_errors.ClientError(
        429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota", "details": details}}
    )


def test_call_llm_adds_429_sleep_to_wait_counter():
    models = MagicMock()
    models.generate_content.side_effect = [_quota_error(), SimpleNamespace(text=ALL_FIELDS)]
    before = llm_client.rate_limit_wait_s
    with patch.object(llm_client, "client", SimpleNamespace(models=models)), patch.object(llm_client, "_sleep"):
        llm_client.call_llm([{"role": "user", "content": "x"}], model="m")
    assert llm_client.rate_limit_wait_s - before == 53.0


def test_wait_is_saved_as_wait_ms_not_latency(tmp_path):
    def call(prompt):
        llm_client.rate_limit_wait_s += 53.0  # what call_llm does after a 53s 429 sleep (no real sleep here)
        return {"raw_output": ALL_FIELDS, "error": None}

    first = run_utils.run_benchmark("baseline", call, "m", tmp_path / "r.json", PROMPTS[:1], cache_dir=tmp_path)
    row = first["results"][0]
    assert row["wait_ms"] == 53000.0
    assert row["latency_ms"] < 1000  # the 53s wait is not counted as model time
    assert first["meta"]["wait_ms_total"] == 53000.0

    again = run_utils.run_benchmark("baseline", call, "m", tmp_path / "r.json", PROMPTS[:1], cache_dir=tmp_path)
    assert again["results"][0]["wait_ms"] == 53000.0  # cached rows keep their original numbers
    assert again["meta"]["wait_ms_total"] == 0  # no new waiting in this run
