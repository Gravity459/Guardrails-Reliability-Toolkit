from unittest.mock import patch

from app.prompts import SYSTEM_PROMPT, build_messages
from benchmark import baseline_runner

VALID_JSON = (
    '{"model":"Pixel 8 Pro","storage":"256GB","camera":"50MP",'
    '"battery":"5050mAh","year":2023}'
)


def test_run_prompt_sends_message_list_and_passes():
    entry = {"id": "normal-001", "prompt": "Extract the phone", "expected_failure": "none"}

    with patch.object(baseline_runner, "call_llm", return_value=VALID_JSON) as fake:
        result = baseline_runner.run_prompt(entry)

    messages = fake.call_args.args[0]
    assert messages == build_messages("Extract the phone")
    assert result["passed"] is True
    assert result["error"] is None


def test_run_prompt_with_real_call_llm_reaches_gemini():
    """Regression: a plain-string prompt crashed inside call_llm before any request."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from app import llm_client

    models = MagicMock()
    models.generate_content.return_value = SimpleNamespace(text=VALID_JSON)
    entry = {"id": "normal-001", "prompt": "Extract the phone", "expected_failure": "none"}

    with patch.object(llm_client, "client", SimpleNamespace(models=models)):
        result = baseline_runner.run_prompt(entry)

    assert result["error"] is None
    assert result["passed"] is True
    kwargs = models.generate_content.call_args.kwargs
    assert kwargs["contents"] == "Extract the phone"
    assert kwargs["config"].system_instruction == SYSTEM_PROMPT
