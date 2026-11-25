"""Shared benchmark loop: call the model per prompt, cache responses, save raw results.

Runners only record what the model returned. Grading happens in scorer.py.

Cache: benchmark/cache/<strategy>__<model>.json maps a hash of
(system prompt, prompt text, temperature) to the saved response, so reruns
don't call Gemini again. Changing the system prompt changes every key.
LLM call failures (e.g. quota) are never cached.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.prompts import SYSTEM_PROMPT

BENCHMARK_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = BENCHMARK_DIR / "prompts.json"
CACHE_DIR = BENCHMARK_DIR / "cache"
TEMPERATURE = 0
LLM_CALL_FAILED = "llm_call_failed"


def load_prompts(path: Path = PROMPTS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def cache_key(prompt: str, temperature: float = TEMPERATURE) -> str:
    payload = json.dumps([SYSTEM_PROMPT, prompt, temperature], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_path(strategy: str, model: str | None, cache_dir: Path = CACHE_DIR) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model or "unknown")
    return cache_dir / f"{strategy}__{safe_model}.json"


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


def run_one(
    entry: dict,
    call: Callable[[str], dict],
    model: str | None,
    cache: dict,
    refresh: bool = False,
) -> dict:
    """Run one prompt. `call(prompt)` returns {"raw_output": str|None, "error": str|None}."""
    key = cache_key(entry["prompt"])
    hit = None if refresh else cache.get(key)
    if hit is not None:
        response, latency_ms, cached = hit, hit["latency_ms"], True
    else:
        start = time.perf_counter()
        try:
            response = call(entry["prompt"])
        except Exception as e:  # network, quota, SDK errors
            response = {"raw_output": None, "error": f"{LLM_CALL_FAILED}: {e}"}
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        cached = False
        if not str(response.get("error") or "").startswith(LLM_CALL_FAILED):
            cache[key] = {**response, "latency_ms": latency_ms}

    return {
        "id": entry["id"],
        "expected_failure": entry["expected_failure"],
        "model": model,
        "raw_output": response.get("raw_output"),
        "latency_ms": latency_ms,
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
) -> dict:
    """Run all prompts, save raw results to `results_path`, return the payload."""
    prompts = prompts if prompts is not None else load_prompts()
    cpath = cache_path(strategy, model, cache_dir)
    cache = _read_json(cpath)

    results = []
    for i, entry in enumerate(prompts, start=1):
        result = run_one(entry, call, model, cache, refresh)
        tag = "cache" if result["cached"] else f"{result['latency_ms']:.0f}ms"
        status = "ERROR" if result["error"] else "ok"
        print(f"[{i}/{len(prompts)}] {entry['id']} {status} ({tag})", flush=True)
        results.append(result)
        if not result["cached"]:
            _write_json(cpath, cache)  # save as we go, so a crash keeps progress

    payload = {
        "meta": {
            "strategy": strategy,
            "model": model,
            "temperature": TEMPERATURE,
            "system_prompt_sha": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16],
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total": len(results),
            "cached": sum(r["cached"] for r in results),
            "llm_call_failed": sum(str(r["error"] or "").startswith(LLM_CALL_FAILED) for r in results),
        },
        "results": results,
    }
    _write_json(results_path, payload)
    print(f"Saved {len(results)} results to {results_path}. Score them with: uv run python -m benchmark.report")
    return payload
