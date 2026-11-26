"""Scorer: parsing, normalisation, exact matching, outcomes, metrics, report."""

import pytest

from benchmark import report
from benchmark.scorer import (
    CORRECT,
    CORRECT_NULL,
    HALLUCINATED,
    INVALID,
    MISSED,
    WRONG,
    normalize,
    parse_output,
    score_prompt,
    score_run,
)

ANSWER = {"model": "Google Pixel 8 Pro", "storage": "256GB", "camera": "50MP dual", "battery": "5050mAh", "year": 2023}
GOOD_JSON = '{"model":"Google Pixel 8 Pro","storage":"256GB","camera":"50MP dual","battery":"5050mAh","year":2023}'


# --- parse_output -------------------------------------------------------------

def test_parse_plain_and_fenced_json():
    assert parse_output(GOOD_JSON)["data"] == ANSWER
    fenced = parse_output(f"```json\n{GOOD_JSON}\n```")
    assert fenced["schema_valid"] and fenced["data"] == ANSWER


def test_parse_missing_keys_become_null():
    data = parse_output('{"camera": "50MP"}')["data"]
    assert data == {"model": None, "storage": None, "camera": "50MP", "battery": None, "year": None}


@pytest.mark.parametrize(
    "raw, valid_json",
    [
        (None, False),
        ("Here is a nice paragraph about the phone.", False),
        ('["Pixel", "256GB"]', True),  # array, not object
        ('{"model": "Pixel", "price": 999}', True),  # extra key
        ('{"year": "2023"}', True),  # year as string
        ('{"year": 2024.0}', True),  # year as float
        ('{"storage": 512}', True),  # storage as number
    ],
    ids=["none", "prose", "array", "extra-key", "year-str", "year-float", "storage-int"],
)
def test_parse_rejects_bad_outputs(raw, valid_json):
    result = parse_output(raw)
    assert result["valid_json"] is valid_json
    assert result["schema_valid"] is False
    assert result["data"] is None
    assert result["error"]


# --- normalisation + exact matching -------------------------------------------

@pytest.mark.parametrize(
    "field, raw, expected",
    [
        ("storage", "256 GB", "256GB"),
        ("storage", "256gb", "256GB"),
        ("storage", "1 tb", "1TB"),
        ("battery", "5,000 mAh", "5000mAh"),
        ("battery", "5000MAH", "5000mAh"),
        ("model", "  Google Pixel 8 Pro ", "Google Pixel 8 Pro"),  # outer spaces only
        ("camera", "50MP Dual", "50MP Dual"),  # case kept
        ("year", 2023, 2023),
    ],
)
def test_normalize(field, raw, expected):
    assert normalize(field, raw) == expected


@pytest.mark.parametrize(
    "field, output_value",
    [
        ("model", "Pixel 8 Pro"),  # brand missing -> wrong (exact only)
        ("model", "google pixel 8 pro"),  # case differs -> wrong
        ("camera", "50MP"),  # qualifier missing -> wrong
        ("camera", "50MP dual camera"),  # extra word -> wrong
        ("camera", "50MP Dual"),  # case differs -> wrong
        ("year", 2022),
    ],
)
def test_exact_match_is_strict_for_model_camera(field, output_value):
    output = {**ANSWER, field: output_value}
    assert score_prompt(output, ANSWER)["fields"][field] == WRONG


def test_storage_and_battery_formatting_still_correct():
    output = {**ANSWER, "storage": "256 GB", "battery": "5,050 mAh"}
    assert score_prompt(output, ANSWER)["exact_match"] is True


# --- outcomes ------------------------------------------------------------------

def test_each_outcome_type():
    answer = {"model": "Pixel 7a", "storage": None, "camera": "64MP dual", "battery": None, "year": 2023}
    output = {"model": "Pixel 7a", "storage": None, "camera": "48MP", "battery": "4385mAh", "year": None}
    fields = score_prompt(output, answer)["fields"]
    assert fields == {
        "model": CORRECT,
        "storage": CORRECT_NULL,
        "camera": WRONG,
        "battery": HALLUCINATED,
        "year": MISSED,
    }


def test_invalid_output_marks_every_field_invalid():
    scored = score_prompt(None, ANSWER)
    assert set(scored["fields"].values()) == {INVALID}
    assert scored["exact_match"] is False


def test_refusal_correct_only_when_all_null():
    all_null = dict.fromkeys(ANSWER)
    assert score_prompt(all_null, "refuse")["refusal_correct"] is True
    assert score_prompt({**all_null, "model": "Pixel"}, "refuse")["refusal_correct"] is False
    assert score_prompt(None, "refuse")["refusal_correct"] is False
    assert score_prompt(all_null, ANSWER)["refusal_correct"] is None  # not a refuse prompt


# --- run metrics + report -------------------------------------------------------

PROMPTS = [
    {"id": "a", "expected_failure": "none", "answer": ANSWER},
    {"id": "b", "expected_failure": "missing year", "answer": {**ANSWER, "year": None}},
    {"id": "c", "expected_failure": "empty input", "answer": "refuse"},
]


def _results(raws):
    return [
        {"id": p["id"], "expected_failure": p["expected_failure"], "raw_output": raw, "latency_ms": ms}
        for p, raw, ms in zip(PROMPTS, raws, [100.0, 200.0, 300.0])
    ]


def test_score_run_metrics():
    raws = [
        GOOD_JSON,  # all correct
        GOOD_JSON,  # year hallucinated (answer null)
        "not json",  # invalid, refusal failed
    ]
    overall = score_run(_results(raws), PROMPTS)["overall"]
    assert overall["valid_json_pct"] == pytest.approx(66.7)
    assert overall["exact_match_pct"] == pytest.approx(33.3)
    assert overall["field_accuracy_pct"] == 60.0  # 9 good of 15
    assert overall["hallucination_pct"] == 100.0  # 1 of 1 parsed null-answer field
    assert overall["correct_refusal_pct"] == 0.0
    assert overall["latency_p50_ms"] == 200.0


def test_by_failure_type_breakdown():
    by_type = score_run(_results([GOOD_JSON, GOOD_JSON, "{}"]), PROMPTS)["by_expected_failure"]
    assert by_type["none"]["exact_match_pct"] == 100.0
    assert by_type["missing year"]["exact_match_pct"] == 0.0
    assert by_type["empty input"]["correct_refusal_pct"] == 100.0


def test_failed_calls_not_scored_but_guard_failed_is():
    # #1: a failed call says nothing about the model -> left out of every metric.
    # #5: guard_failed is a real outcome -> scored, and counted next to hallucination %.
    results = _results([GOOD_JSON, None, None])
    results[1]["error"] = "guard_failed"
    results[2]["error"] = "llm_call_failed: 429 quota"
    overall = score_run(results, PROMPTS)["overall"]
    assert (overall["prompts"], overall["scored"], overall["llm_call_failed"]) == (3, 2, 1)
    assert overall["invalid_outputs"] == 1 and overall["guard_failed"] == 1
    assert overall["outcomes"]["invalid"] == 5  # only the guard_failed row's fields
    assert overall["exact_match_pct"] == 50.0  # 1 of 2 scored, not 1 of 3
    assert overall["valid_json_pct"] == 50.0
    assert overall["correct_refusal_pct"] is None  # the only refuse prompt failed to call
    assert overall["latency_p50_ms"] == 150.0  # failed call's latency left out


def test_build_report_side_by_side():
    base = score_run(_results([GOOD_JSON, GOOD_JSON, "not json"]), PROMPTS)
    guard = score_run(_results([GOOD_JSON, GOOD_JSON.replace("2023", "null"), "{}"]), PROMPTS)
    meta = {"model": "gemini-test", "temperature": 0, "total": 3}
    md = report.build_report({"baseline": base, "guarded": guard}, {"baseline": meta, "guarded": meta})

    assert "| Exact match (all 5 fields) % | 33.3 | 100 | +66.7 ✅ |" in md
    assert "| missing year | 1 | 0 | 100 | +100 ✅ |" in md
    assert "| Unusable outputs (count) | 1 | 0 | -1 ✅ |" in md  # shown next to hallucination %
    assert "- **baseline scored: 3/3** (0 failed calls left out of every metric)" in md
