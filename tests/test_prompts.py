"""The shared rules prompt reaches Gemini on both paths."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import guard as guard_module
from app import llm_client
from app.prompts import SYSTEM_PROMPT, build_messages

ALL_NULL = '{"model":null,"storage":null,"camera":null,"battery":null,"year":null}'


def _fake_client():
    models = MagicMock()
    models.generate_content.return_value = SimpleNamespace(text=ALL_NULL)
    return models, SimpleNamespace(models=models)


def test_build_messages_puts_rules_first():
    assert build_messages("hi") == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "hi"},
    ]


def test_call_llm_sends_system_as_system_instruction():
    models, client = _fake_client()
    with patch.object(llm_client, "client", client):
        llm_client.call_llm(build_messages("the prompt"), model="gemini-test")

    kwargs = models.generate_content.call_args.kwargs
    assert kwargs["contents"] == "the prompt"  # rules are not mixed into the prompt
    assert kwargs["config"].system_instruction == SYSTEM_PROMPT


def test_call_llm_without_system_sends_no_config():
    models, client = _fake_client()
    with patch.object(llm_client, "client", client):
        llm_client.call_llm([{"role": "user", "content": "x"}], model="gemini-test")

    assert models.generate_content.call_args.kwargs["config"] is None


def test_guarded_path_sends_same_rules_and_accepts_all_null_refusal():
    models, client = _fake_client()
    with patch.object(llm_client, "client", client):
        result = guard_module.guarded_call_llm("Source: ")

    kwargs = models.generate_content.call_args.kwargs
    assert kwargs["config"].system_instruction == SYSTEM_PROMPT
    assert result is not None  # all-null refusal passes the guard
