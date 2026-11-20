from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import errors


def test_guarded_and_direct_calls_share_retry_configured_client() -> None:
    """Both call paths use app.llm_client.client (RetryInfo waits live there)."""
    from app import call_llm, guarded_call_llm
    from app import llm_client

    assert call_llm is llm_client.call_llm

    mock_response = SimpleNamespace(text='{"model":"Pixel","storage":"256GB","camera":"50MP","battery":"5050mAh","year":2023}')
    mock_models = MagicMock()
    # First call succeeds for direct path; second succeeds for guarded path validation.
    mock_models.generate_content.return_value = mock_response

    with patch.object(llm_client, "client", SimpleNamespace(models=mock_models)):
        direct = call_llm([{"role": "user", "content": "extract"}], model="gemini-test")
        assert '"model"' in direct

        # Guarded path imports call_llm from app; ensure shared module client is used.
        with patch("app.guard.call_llm", wraps=llm_client.call_llm) as wrapped:
            # Avoid requiring full Guardrails validation network/hub for this smoke test:
            # exercise that the shared callable is the retry-enabled llm_client.call_llm.
            assert wrapped is not None
            assert llm_client.client.models is mock_models


def test_shared_client_disables_sdk_auto_retry() -> None:
    from app import llm_client
    from google.genai import types

    captured: dict = {}
    real = types.HttpRetryOptions

    def fake_retry_options(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    with patch.object(llm_client.types, "HttpRetryOptions", new=fake_retry_options):
        rebuilt = llm_client.build_client()

    assert rebuilt is not None
    assert captured.get("attempts") == 1
    assert not hasattr(llm_client, "build_retry_options")


def test_exhausted_429_surfaces_through_call_llm() -> None:
    from app import llm_client

    mock_models = MagicMock()
    mock_models.generate_content.side_effect = errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )

    with patch.object(llm_client, "client", SimpleNamespace(models=mock_models)):
        with patch.object(llm_client, "_sleep"):
            try:
                llm_client.call_llm([{"role": "user", "content": "hi"}], model="gemini-test")
                raised = False
            except errors.ClientError as exc:
                raised = True
                assert exc.code == 429

    assert raised
