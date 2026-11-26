"""Shared benchmark loop: call the model per prompt, cache responses, save raw results.

Runners only record what the model returned. Grading happens in scorer.py.

Cache: benchmark/cache/<strategy>_<model>.json maps a hash of
(system prompt, prompt text, temperature[, cache_salt]) to the saved response,
so reruns don't call Gemini again. Changing the system prompt changes every key.
The guarded runner passes a hash of its guard code as cache_salt, so changing a
validator or the reask logic also changes every guarded key.
LLM call failures (e.g. quota) are never cached.

Failed calls: every prompt whose call failed gets one retry after the main
pass. If the daily quota runs out, the run stops at once, the rest are saved
as not_run, and a rerun skips everything already cached.

Latency: latency_ms is the time spent working, without rate-limit (429) waits.
Those waits are saved separately as wait_ms.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from google.genai import errors as genai_errors

from app import llm_client
from app.prompts import SYSTEM_PROMPT

BENCHMARK_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = BENCHMARK_DIR / "prompts.json"
CACHE_DIR = BENCHMARK_DIR / "cache"
TEMPERATURE = 0
LLM_CALL_FAILED = "llm_call_failed"
DAILY_QUOTA = "daily_quota"
NOT_RUN = "not_run"


def load_prompts(path: Path = PROMPTS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def cache_key(prompt: str, temperature: float = TEMPERATURE, salt: str = "") -> str:
    parts = [SYSTEM_PROMPT, prompt, temperature] + ([salt] if salt else [])
    payload = json.dumps(parts, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def files_hash(*paths: Path) -> str:
    """Short hash of source files (line endings ignored, so Windows and CI agree)."""
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def cache_path(strategy: str, model: str | None, cache_dir: Path = CACHE_DIR) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model or "unknown")
    return cache_dir / f"{strategy}_{safe_model}.json"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def _is_daily_quota(error: BaseException | None) -> bool:
    """True if the error (or one it was raised from, e.g. inside Guardrails) is a daily-quota 429."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, genai_errors.APIError) and llm_client.is_daily_quota_error(error):
            return True
        error = error.__cause__ or error.__context__
    return False


def _call_failed(result: dict) -> bool:
    return str(result["error"] or "").startswith(LLM_CALL_FAILED)


def _not_run(entry: dict, model: str | None) -> dict:
    """Row for a prompt skipped after the daily quota ran out."""
    return {
        "id": entry["id"],
        "expected_failure": entry["expected_failure"],
        "model": model,
        "raw_output": None,
        "latency_ms": None,
        "wait_ms": 0.0,
        "error": f"{LLM_CALL_FAILED}: {NOT_RUN} ({DAILY_QUOTA})",
        "cached": False,
    }


def run_one(
    entry: dict,
    call: Callable[[str], dict],
    model: str | None,
    cache: dict,
    refresh: bool = False,
    cache_salt: str = "",
) -> dict:
    """Run one prompt. `call(prompt)` returns {"raw_output": str|None, "error": str|None}."""
    key = cache_key(entry["prompt"], salt=cache_salt)
    hit = None if refresh else cache.get(key)
    if hit is not None:
        response, cached = hit, True
        latency_ms, wait_ms = hit["latency_ms"], hit.get("wait_ms", 0.0)
    else:
        start = time.perf_counter()
        waited_before = llm_client.rate_limit_wait_s
        try:
            response = call(entry["prompt"])
        except Exception as e:  # network, quota, SDK errors
            reason = f"{DAILY_QUOTA}: " if _is_daily_quota(e) else ""
            response = {"raw_output": None, "error": f"{LLM_CALL_FAILED}: {reason}{e}"}
        total_ms = (time.perf_counter() - start) * 1000
        wait_ms = round((llm_client.rate_limit_wait_s - waited_before) * 1000, 1)
        latency_ms = round(max(total_ms - wait_ms, 0.0), 1)
        cached = False
        if not str(response.get("error") or "").startswith(LLM_CALL_FAILED):
            cache[key] = {**response, "latency_ms": latency_ms, "wait_ms": wait_ms}

    return {
        "id": entry["id"],
        "expected_failure": entry["expected_failure"],
        "model": model,
        "raw_output": response.get("raw_output"),
        "latency_ms": latency_ms,
        "wait_ms": wait_ms,
        "error": response.get("error"),
        "cached": cached,
    }


def run_benchmark(
    strategy: str,
    call: Callable[[str], dict],
    model: str | None,
    results_path: Path,
    prompts: list[dict] | None = None,
    refresh: bool = False,
    cache_dir: Path = CACHE_DIR,
    cache_salt: str = "",
) -> dict:
    """Run all prompts, save raw results to `results_path`, return the payload."""
    prompts = prompts if prompts is not None else load_prompts()
    cpath = cache_path(strategy, model, cache_dir)
    cache = _read_json(cpath)

    total = len(prompts)
    results: list[dict | None] = [None] * total
    quota_hit = False

    def attempt(i: int, label: str) -> None:
        nonlocal quota_hit
        result = run_one(prompts[i], call, model, cache, refresh, cache_salt)
        tag = "cache" if result["cached"] else f"{result['latency_ms']:.0f}ms"
        status = "ERROR" if result["error"] else "ok"
        print(f"{label} {prompts[i]['id']} {status} ({tag})", flush=True)
        results[i] = result
        if not result["cached"]:
            _write_json(cpath, cache)  # save as we go, so a crash keeps progress
        quota_hit = DAILY_QUOTA in str(result["error"] or "")

    # Main pass.
    for i in range(total):
        attempt(i, f"[{i + 1}/{total}]")
        if quota_hit:
            break

    # Retry pass: each failed call gets one more try at the end.
    # (call_llm already waits out short per-minute 429s.)
    failed = [] if quota_hit else [i for i in range(total) if _call_failed(results[i])]
    if failed:
        print(f"Retrying {len(failed)} failed call(s)...", flush=True)
    for n, i in enumerate(failed, start=1):
        attempt(i, f"[retry {n}/{len(failed)}]")
        results[i]["retried"] = True
        if quota_hit:
            break

    if quota_hit:
        done = sum(r is not None for r in results)
        print(f"Daily quota reached at {done}/{total}. Rerun tomorrow; cached answers will be reused.")
    for i in range(total):
        if results[i] is None:
            results[i] = _not_run(prompts[i], model)

    payload = {
        "meta": {
            "strategy": strategy,
            "model": model,
            "temperature": TEMPERATURE,
            "system_prompt_sha": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16],
            "cache_salt": cache_salt or None,
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total": len(results),
            "cached": sum(r["cached"] for r in results),
            "llm_call_failed": sum(_call_failed(r) for r in results),
            "retried": sum(r.get("retried", False) for r in results),
            "not_run": sum(NOT_RUN in str(r["error"]) for r in results),
            "stopped_on_daily_quota": quota_hit,
            "wait_ms_total": round(sum(r["wait_ms"] for r in results if not r["cached"]), 1),
        },
        "results": results,
    }
    _write_json(results_path, payload)
    print(f"Saved {len(results)} results to {results_path}. Score them with: uv run python -m benchmark.report")
    return payload
