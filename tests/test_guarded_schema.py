"""Optional fields: nulls pass the guard, but real bad values still fail."""

import json

import pytest

from app.guard import guard
from app.schemas.baseline_schema import Phone

FULL = {"model": "Pixel 8", "storage": "256GB", "camera": "50MP", "battery": "5000mAh", "year": 2023}


def test_baseline_schema_accepts_nulls():
    phone = Phone.model_validate({"model": "Pixel 8", "storage": None, "year": None})
    assert phone.storage is None
    assert phone.camera is None


def test_guard_passes_null_fields():
    result = guard.validate(json.dumps({**FULL, "storage": None, "year": None, "battery": None}))
    assert result.validation_passed


@pytest.mark.parametrize(
    "bad",
    [{"storage": "999GB"}, {"model": "P"}, {"year": 1999}],
    ids=["bad-storage", "short-model", "year-out-of-range"],
)
def test_guard_still_rejects_bad_values(bad):
    with pytest.raises(Exception):
        guard.validate(json.dumps({**FULL, **bad}))
