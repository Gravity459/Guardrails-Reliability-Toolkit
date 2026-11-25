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

from app import guarded_call_llm  # noqa: E402
from benchmark.run_utils import TEMPERATURE, run_benchmark  # noqa: E402

RESULTS_PATH = BENCHMARK_DIR / "results_guarded.json"
MODEL = os.getenv("LLM_PRIMARY_MODEL")


def call(prompt: str) -> dict:
    # API errors (e.g. quota) raise and are recorded as llm_call_failed by run_utils.
    validated = guarded_call_llm(prompt, temperature=TEMPERATURE)
    if validated is None:
        return {"raw_output": None, "error": "guard_failed"}
    return {"raw_output": json.dumps(validated, ensure_ascii=False), "error": None}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and call Gemini again")
    args = parser.parse_args(argv)
    run_benchmark("guarded", call, MODEL, RESULTS_PATH, refresh=args.refresh)


if __name__ == "__main__":
    main()
