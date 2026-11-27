"""Score both runs against the answer key and compare them side by side.

Run: uv run python -m benchmark.report
Reads results_baseline.json + results_guarded.json, prints the table and
writes benchmark/report.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.run_utils import load_prompts  # noqa: E402
from benchmark.scorer import score_run  # noqa: E402

RUNS = {
    "baseline": BENCHMARK_DIR / "results_baseline.json",
    "guarded": BENCHMARK_DIR / "results_guarded.json",
}
REPORT_PATH = BENCHMARK_DIR / "report.md"

METRICS = [
    ("valid_json_pct", "Valid JSON %", True),
    ("schema_valid_pct", "Schema valid %", True),
    ("field_accuracy_pct", "Field accuracy %", True),
    ("exact_match_pct", "Exact match (all 5 fields) %", True),
    ("hallucination_pct", "Hallucination % (null fields filled, usable outputs only)", False),
    ("invalid_outputs", "Unusable outputs (count)", False),
    ("guard_failed", "  of which guard_failed", False),
    ("correct_refusal_pct", "Correct refusal %", True),
    ("latency_p50_ms", "Latency p50 (ms)", False),
    ("latency_p95_ms", "Latency p95 (ms)", False),
]


def _fmt(value) -> str:
    return "–" if value is None else f"{value:g}"


def _delta(base, guard, higher_is_better: bool) -> str:
    if base is None or guard is None:
        return "–"
    diff = round(guard - base, 1)
    if diff == 0:
        return "0"
    better = (diff > 0) == higher_is_better
    return f"{diff:+g} {'✅' if better else '❌'}"


def build_report(scores: dict[str, dict], metas: dict[str, dict]) -> str:
    base, guard = scores["baseline"], scores["guarded"]
    lines = ["# Benchmark report", ""]
    for name in RUNS:
        m = metas[name]
        lines.append(
            f"- **{name}**: model `{m.get('model')}`, temperature {m.get('temperature')}, "
            f"system prompt `{m.get('system_prompt_sha')}`, run {m.get('run_at')}, "
            f"{m.get('total')} prompts ({m.get('cached', 0)} from cache, "
            f"{m.get('llm_call_failed', 0)} LLM call failures, "
            f"fallback used: {m.get('fallback_used', 0)})"
        )
    for name, s in scores.items():
        o = s["overall"]
        lines.append(
            f"- **{name} scored: {o['scored']}/{o['prompts']}** "
            f"({o['llm_call_failed']} failed calls left out of every metric)"
        )
    lines += ["", "## Overall", "", "| Metric | Baseline | Guarded | Change |", "| --- | --- | --- | --- |"]
    for key, label, higher in METRICS:
        b, g = base["overall"][key], guard["overall"][key]
        lines.append(f"| {label} | {_fmt(b)} | {_fmt(g)} | {_delta(b, g, higher)} |")

    lines += ["", "## Field outcomes", "", "| Outcome | Baseline | Guarded |", "| --- | --- | --- |"]
    for outcome in base["overall"]["outcomes"]:
        lines.append(
            f"| {outcome} | {base['overall']['outcomes'][outcome]} | {guard['overall']['outcomes'][outcome]} |"
        )

    lines += [
        "",
        "## By failure type (exact match %)",
        "",
        "| Failure type | Prompts | Baseline | Guarded | Change |",
        "| --- | --- | --- | --- | --- |",
    ]
    for ftype, b_sub in base["by_expected_failure"].items():
        g_sub = guard["by_expected_failure"].get(ftype, {})
        b, g = b_sub["exact_match_pct"], g_sub.get("exact_match_pct")
        lines.append(f"| {ftype} | {b_sub['prompts']} | {_fmt(b)} | {_fmt(g)} | {_delta(b, g, True)} |")
    return "\n".join(lines) + "\n"


def main() -> str:
    prompts = load_prompts()
    scores, metas = {}, {}
    for name, path in RUNS.items():
        if not path.exists():
            sys.exit(f"{path.name} not found. Run: uv run python -m benchmark.{name}_runner")
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        metas[name] = payload.get("meta", {})
        scores[name] = score_run(payload["results"], prompts)
    report = build_report(scores, metas)
    REPORT_PATH.write_text(report, encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    print(report)
    print(f"Wrote {REPORT_PATH}")
    return report


if __name__ == "__main__":
    main()
