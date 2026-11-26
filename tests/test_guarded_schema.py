"""Guarded schema: nulls pass, bad values are re-asked once, then rejected."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import guard as guard_module
from app import llm_client
from app.guard import guard
from app.prompts import SYSTEM_PROMPT
from app.schemas.baseline_schema import Phone

FULL = {"model": "Pixel 8", "storage": "256GB", "camera": "50MP", "battery": "5000mAh", "year": 2023}


def _fake_client(*answers):
    models = MagicMock()
    models.generate_content.side_effect = [SimpleNamespace(text=json.dumps(a)) for a in answers]
    return models, SimpleNamespace(models=models)


def test_baseline_schema_accepts_nulls():
    phone = Phone.model_validate({"model": "Pixel 8", "storage": None, "year": None})
    assert phone.storage is None
    assert phone.camera is None


def test_guard_passes_null_fields():
    result = guard.validate(json.dumps({**FULL, "storage": None, "year": None, "battery": None}))
    assert result.validation_passed


def test_reask_sends_error_back_and_accepts_fixed_answer():
    models, client = _fake_client({**FULL, "year": 1999}, FULL)
    with patch.object(llm_client, "client", client):
        result = guard_module.guarded_call_llm("Source: Pixel 8 from 2023")

    assert result == FULL
    assert models.generate_content.call_count == 2
    reask = models.generate_content.call_args_list[1].kwargs
    assert "Value 1999 is less than 2000." in reask["contents"]  # validator error goes back
    assert "Source: Pixel 8 from 2023" in reask["contents"]  # so does the source text
    assert reask["config"].system_instruction == SYSTEM_PROMPT  # and our rules, not Guardrails'


@pytest.mark.parametrize(
    "bad",
    [{"storage": "999GB"}, {"model": "P"}, {"year": 1999}],
    ids=["bad-storage", "short-model", "year-out-of-range"],
)
def test_still_bad_after_one_reask_is_rejected(bad):
    # Guardrails would swap in a made-up "fix" (e.g. year 2000); we reject instead.
    models, client = _fake_client({**FULL, **bad}, {**FULL, **bad})
    with patch.object(llm_client, "client", client):
        assert guard_module.guarded_call_llm("Source: x") is None
    assert models.generate_content.call_count == 2  # num_reasks = 1


def test_baseline_runner_never_loads_the_guard():
    code = "import sys, benchmark.baseline_runner; print('app.guard' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
