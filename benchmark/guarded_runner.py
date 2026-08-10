from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import guarded_call_llm
from app.schemas.guarded_schema import GuardedPhone
from benchmark.baseline_runner import build_meta

PROMPTS_PATH = BENCHMARK_DIR / "prompts.json"
RESULTS_PATH = BENCHMARK_DIR / "results_guarded.json"
MODEL = os.getenv("LLM_PRIMARY_MODEL")


def run_prompt(entry: dict) -> dict:
    start = time.perf_counter()
    try:
        validated = guarded_call_llm(entry["prompt"])
        error = None
    except Exception as e:
        validated, error = None, str(e)
    latency_ms = (time.perf_counter() - start) * 1000

    ok = validated is not None
    return {
        "id": entry["id"],
        "expected_failure": entry["expected_failure"],
        "model": MODEL,
        "raw_output": (
            GuardedPhone.model_validate(validated).model_dump_json() if ok else None
        ),
        "latency_ms": latency_ms,
        "valid_json": ok,
        "json_error": None if ok else "guard_failed",
        "schema_valid": ok,
        "schema_error": None if ok else "guard_failed",
        "passed": ok,
        "error": error,
    }


def main() -> None:
    with PROMPTS_PATH.open(encoding="utf-8") as f:
        prompts = json.load(f)

    total = len(prompts)
    results: list[dict] = []

    for i, entry in enumerate(prompts, start=1):
        print(f"[{i}/{total}] {entry['id']} ...", flush=True)
        result = run_prompt(entry)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  -> {status}", flush=True)
        results.append(result)

    payload = {
        "meta": build_meta(results),
        "results": results,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    meta = payload["meta"]
    print(
        f"\nDone. {meta['passed']}/{meta['total']} passed "
        f"({meta['pass_rate']:.1%}). Wrote {RESULTS_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
