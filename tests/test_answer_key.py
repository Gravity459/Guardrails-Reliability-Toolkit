"""Checks the hand-written answer key in benchmark/prompts.json."""

import json
from pathlib import Path

import pytest

PROMPTS = json.loads(
    (Path(__file__).resolve().parents[1] / "benchmark" / "prompts.json").read_text(
        encoding="utf-8"
    )
)
FIELDS = {"model", "storage", "camera", "battery", "year"}
REFUSE_TYPES = {"empty input", "non-phone content"}


def _ids(entries):
    return [e["id"] for e in entries]


def test_has_200_prompts_with_unique_ids():
    ids = _ids(PROMPTS)
    assert len(ids) == 200
    assert len(set(ids)) == 200


@pytest.mark.parametrize("entry", PROMPTS, ids=_ids(PROMPTS))
def test_answer_shape(entry):
    assert "answer" in entry
    answer = entry["answer"]

    if answer == "refuse":
        assert entry["expected_failure"] in REFUSE_TYPES
        return

    assert isinstance(answer, dict)
    assert set(answer) == FIELDS
    assert answer["year"] is None or type(answer["year"]) is int
    for field in FIELDS - {"year"}:
        value = answer[field]
        assert value is None or (isinstance(value, str) and value.strip())


@pytest.mark.parametrize(
    "entry",
    [e for e in PROMPTS if e["expected_failure"] in REFUSE_TYPES],
    ids=_ids([e for e in PROMPTS if e["expected_failure"] in REFUSE_TYPES]),
)
def test_empty_and_non_phone_prompts_are_refused(entry):
    assert entry["answer"] == "refuse"


# The failure type says which field the prompt makes unknowable:
# the answer must leave that field null instead of guessing.
NULL_FIELD_BY_TYPE = {
    "missing model": "model",
    "missing storage": "storage",
    "missing camera": "camera",
    "missing battery": "battery",
    "missing year": "year",
    "contradictory model": "model",
    "contradictory storage": "storage",
    "contradictory year": "year",
    "ambiguous year format": "year",
}
TYPED = [e for e in PROMPTS if e["expected_failure"] in NULL_FIELD_BY_TYPE]


@pytest.mark.parametrize("entry", TYPED, ids=_ids(TYPED))
def test_failure_type_field_is_null(entry):
    field = NULL_FIELD_BY_TYPE[entry["expected_failure"]]
    assert entry["answer"][field] is None


@pytest.mark.parametrize(
    "entry",
    [e for e in PROMPTS if e["expected_failure"] == "none"],
    ids=_ids([e for e in PROMPTS if e["expected_failure"] == "none"]),
)
def test_normal_prompts_have_every_field(entry):
    assert None not in entry["answer"].values()
