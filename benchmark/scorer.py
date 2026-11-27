"""Score model outputs against the hand-written answer key in prompts.json.

Matching rules:
- model, camera: exact text. Only outer whitespace is trimmed; case-sensitive.
- storage: "256 GB" / "256gb" -> "256GB", "1 tb" -> "1TB", then exact.
- battery: "5,000 mAh" -> "5000mAh", then exact.
- year: integer compare.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from statistics import quantiles

from pydantic import ValidationError

from app.schemas.baseline_schema import Phone

FIELDS = ("model", "storage", "camera", "battery", "year")
REFUSE = "refuse"

# Per-field outcomes.
CORRECT = "correct"  # answer has a value, output matches it
CORRECT_NULL = "correct_null"  # answer is null, output is null
WRONG = "wrong"  # both have values, they differ
HALLUCINATED = "hallucinated"  # answer is null, output has a value
MISSED = "missed"  # answer has a value, output is null
INVALID = "invalid"  # output was not valid JSON / schema, nothing to compare
GOOD = {CORRECT, CORRECT_NULL}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)
_STORAGE_RE = re.compile(r"^(\d+)\s*(gb|tb)$", re.IGNORECASE)
_BATTERY_RE = re.compile(r"^(\d+)\s*mah$", re.IGNORECASE)


def parse_output(raw: str | None) -> dict:
    """Parse raw model text into the 5 fields.

    Returns {"valid_json", "schema_valid", "data", "error"}. `data` is a dict
    with all 5 fields (missing keys -> None) only when the schema is valid.
    """
    result = {"valid_json": False, "schema_valid": False, "data": None, "error": None}
    if raw is None:
        result["error"] = "no_output"
        return result

    text = raw.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        result["error"] = f"not_valid_json: {e}"
        return result
    result["valid_json"] = True

    if not isinstance(parsed, dict):
        result["error"] = f"schema_violation: expected an object, got {type(parsed).__name__}"
        return result
    extra = sorted(set(parsed) - set(FIELDS))
    if extra:
        result["error"] = f"schema_violation: extra keys {extra}"
        return result
    try:
        # strict: "2023" is not an int, 512 is not a str.
        phone = Phone.model_validate(parsed, strict=True)
    except ValidationError as e:
        result["error"] = f"schema_violation: {e.errors(include_url=False)}"
        return result

    result["schema_valid"] = True
    result["data"] = phone.model_dump()
    return result


def normalize(field: str, value):
    """Normalise one field value for comparison (None stays None)."""
    if value is None:
        return None
    if field == "year":
        return value
    text = str(value).strip()
    if field == "storage":
        m = _STORAGE_RE.match(text.replace(" ", ""))
        return f"{m.group(1)}{m.group(2).upper()}" if m else text
    if field == "battery":
        m = _BATTERY_RE.match(text.replace(",", "").replace(" ", ""))
        return f"{m.group(1)}mAh" if m else text
    return text  # model, camera: exact text


def score_prompt(output: dict | None, answer) -> dict:
    """Compare parsed output (or None if invalid) with the answer.

    Returns {"fields": {field: outcome}, "exact_match": bool, "refusal_correct": bool|None}.
    A "refuse" answer means every field should be null.
    """
    refuse = answer == REFUSE
    expected = dict.fromkeys(FIELDS) if refuse else answer

    fields = {}
    for field in FIELDS:
        want = normalize(field, expected[field])
        if output is None:
            fields[field] = INVALID
            continue
        got = normalize(field, output.get(field))
        if want is None:
            fields[field] = CORRECT_NULL if got is None else HALLUCINATED
        elif got is None:
            fields[field] = MISSED
        else:
            fields[field] = CORRECT if got == want else WRONG

    exact = all(o in GOOD for o in fields.values())
    return {
        "fields": fields,
        "exact_match": exact,
        "refusal_correct": exact if refuse else None,
    }


def _pct(part: int, whole: int) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def _percentile(values: list[float], q: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(quantiles(values, n=100, method="inclusive")[q - 1], 1)


def _call_failed(row: dict) -> bool:
    return str(row.get("run_error") or "").startswith("llm_call_failed")


def score_run(results: list[dict], prompts: list[dict]) -> dict:
    """Score one run (list of saved results) against the answer key."""
    answers = {p["id"]: p["answer"] for p in prompts}
    rows = []
    for r in results:
        parsed = parse_output(r.get("raw_output"))
        scored = score_prompt(parsed["data"], answers[r["id"]])
        # parse_output's "error" would hide the runner's error (guard_failed,
        # llm_call_failed), so keep that one as run_error.
        rows.append({**r, "run_error": r.get("error"), **parsed, **scored})

    def summarize(all_rows: list[dict]) -> dict:
        # Failed calls (quota, network, not run) say nothing about the model or
        # the guard, so they are left out of every quality metric. guard_failed
        # is a real guard outcome and stays in.
        subset = [row for row in all_rows if not _call_failed(row)]
        n = len(subset)
        outcomes = [o for row in subset for o in row["fields"].values()]
        # Fields whose answer is null, in outputs that could be parsed.
        null_answer_fields = [o for o in outcomes if o in {CORRECT_NULL, HALLUCINATED}]
        refusals = [row for row in subset if row["refusal_correct"] is not None]
        latencies = [row["latency_ms"] for row in subset if row.get("latency_ms") is not None]
        return {
            "prompts": len(all_rows),
            "scored": n,
            "valid_json_pct": _pct(sum(row["valid_json"] for row in subset), n),
            "schema_valid_pct": _pct(sum(row["schema_valid"] for row in subset), n),
            "field_accuracy_pct": _pct(sum(o in GOOD for o in outcomes), len(outcomes)),
            "hallucination_pct": _pct(
                sum(o == HALLUCINATED for o in null_answer_fields), len(null_answer_fields)
            ),
            "correct_refusal_pct": _pct(sum(row["refusal_correct"] for row in refusals), len(refusals)),
            "exact_match_pct": _pct(sum(row["exact_match"] for row in subset), n),
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
            # Rows with no usable output: they are left out of hallucination_pct,
            # so show how many there were and why.
            "invalid_outputs": sum(row["data"] is None for row in subset),
            "guard_failed": sum(row["run_error"] == "guard_failed" for row in subset),
            "llm_call_failed": len(all_rows) - n,  # not scored
            "outcomes": {o: outcomes.count(o) for o in (CORRECT, CORRECT_NULL, WRONG, HALLUCINATED, MISSED, INVALID)},
        }

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["expected_failure"]].append(row)

    return {
        "overall": summarize(rows),
        "by_expected_failure": {t: summarize(sub) for t, sub in by_type.items()},
        "rows": rows,
    }
