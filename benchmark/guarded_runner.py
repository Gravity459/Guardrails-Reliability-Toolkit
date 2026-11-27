"""Guarded: same prompt and model, wrapped in Guardrails validation + re-asks.

Run: uv run python -m benchmark.guarded_runner [--refresh]
Saves the guard's validated output (as JSON text) to results_guarded.json;
score it with benchmark.report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.guard import guarded_call  # noqa: E402
from benchmark.run_utils import TEMPERATURE, files_hash, run_benchmark  # noqa: E402

RESULTS_PATH = BENCHMARK_DIR / "results_guarded.json"
MODEL = os.getenv("LLM_PRIMARY_MODEL")
# Part of every guarded cache key: editing the guard or its validators
# means old cached guarded answers are not reused.
GUARD_FILES = (ROOT / "app" / "guard.py", ROOT / "app" / "schemas" / "guarded_schema.py")
GUARD_VERSION = files_hash(*GUARD_FILES)


def call(prompt: str) -> dict:
    # API errors (e.g. quota) raise and are recorded as llm_call_failed by run_utils.
    result = guarded_call(prompt, temperature=TEMPERATURE)
    used = {"model_used": result["model_used"], "fallback_reason": result["fallback_reason"]}
    if result["output"] is None:
        return {"raw_output": None, "error": "guard_failed", **used}
    return {"raw_output": json.dumps(result["output"], ensure_ascii=False), "error": None, **used}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and call Gemini again")
    args = parser.parse_args(argv)
    run_benchmark("guarded", call, MODEL, RESULTS_PATH, refresh=args.refresh, cache_salt=GUARD_VERSION)


if __name__ == "__main__":
    main()
