"""Baseline: plain Gemini call with the shared system prompt. No Guardrails.

Run: uv run python -m benchmark.baseline_runner [--refresh]
Saves raw outputs to results_baseline.json; score them with benchmark.report.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import call_llm  # noqa: E402
from app.prompts import build_messages  # noqa: E402
from benchmark.run_utils import TEMPERATURE, run_benchmark  # noqa: E402

RESULTS_PATH = BENCHMARK_DIR / "results_baseline.json"
MODEL = os.getenv("LLM_PRIMARY_MODEL")


def call(prompt: str) -> dict:
    raw = call_llm(build_messages(prompt), MODEL, temperature=TEMPERATURE)
    return {"raw_output": raw, "error": None}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and call Gemini again")
    args = parser.parse_args(argv)
    run_benchmark("baseline", call, MODEL, RESULTS_PATH, refresh=args.refresh)


if __name__ == "__main__":
    main()
