from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

from pydantic import ValidationError

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import call_llm
from app.schemas.baseline_schema import Phone

PROMPTS_PATH = BENCHMARK_DIR / "prompts.json"
RESULTS_PATH = BENCHMARK_DIR / "results_baseline.json"
MODEL = os.getenv("LLM_PRIMARY_MODEL")

_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)


def extract_json_text(raw: str) -> str:
    """Strip markdown fences like ```json ... ``` when present."""
    match = _FENCE_RE.match(raw.strip())
    if match:
        return match.group(1).strip()
    return raw.strip()


def is_valid_json(raw: str) -> tuple[bool, str | None]:
    try:
        json.loads(extract_json_text(raw))
        return True, None
    except json.JSONDecodeError as e:
        return False, f"not_valid_json: {e}"


def matches_schema(raw: str, schema_cls) -> tuple[bool, str | None]:
    try:
        parsed = json.loads(extract_json_text(raw))
        schema_cls.model_validate(parsed)  # pydantic v2
        return True, None
    except json.JSONDecodeError as e:
        return False, f"not_valid_json: {e}"
    except ValidationError as e:
        return False, f"schema_violation: {e.errors()}"


def run_prompt(entry: dict) -> dict:
    prompt_id = entry["id"]
    expected_failure = entry["expected_failure"]
    prompt = entry["prompt"]

    raw_output: str | None = None
    error: str | None = None
    latency_ms: float | None = None

    start = time.perf_counter()
    try:
        raw_output = call_llm(prompt, MODEL)
        latency_ms = (time.perf_counter() - start) * 1000
        print(f"Raw output: {raw_output}")
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        error = str(e)
        return {
            "id": prompt_id,
            "expected_failure": expected_failure,
            "model": MODEL,
            "raw_output": None,
            "latency_ms": latency_ms,
            "valid_json": False,
            "json_error": "llm_call_failed",
            "schema_valid": False,
            "schema_error": "llm_call_failed",
            "passed": False,
            "error": error,
        }

    valid_json, json_error = is_valid_json(raw_output)
    schema_valid, schema_error = matches_schema(raw_output, Phone)

    return {
        "id": prompt_id,
        "expected_failure": expected_failure,
        "model": MODEL,
        "raw_output": raw_output,
        "latency_ms": latency_ms,
        "valid_json": valid_json,
        "json_error": json_error,
        "schema_valid": schema_valid,
        "schema_error": schema_error,
        "passed": valid_json and schema_valid,
        "error": None,
    }


def build_meta(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    by_expected: dict[str, dict[str, int]] = defaultdict[str, dict[str, int]](
        lambda: {"total": 0, "passed": 0, "failed": 0}
    )
    for r in results:
        key = r["expected_failure"]
        by_expected[key]["total"] += 1
        if r["passed"]:
            by_expected[key]["passed"] += 1
        else:
            by_expected[key]["failed"] += 1

    return {
        "model": MODEL,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "by_expected_failure": dict[str, dict[str, int]](by_expected),
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
