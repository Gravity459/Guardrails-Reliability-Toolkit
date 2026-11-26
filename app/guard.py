from guardrails import Guard
from guardrails.errors import ValidationError

from app import call_llm
from app.prompts import SYSTEM_PROMPT, build_messages
from app.schemas.guarded_schema import GuardedPhone
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


guard = Guard.for_pydantic(output_class=GuardedPhone)
# Turn off Guardrails' usage telemetry (traces sent to Guardrails' servers).
# Every guard call reads this flag, whatever ~/.guardrailsrc says.
guard.configure(allow_metrics_collection=False)


def _make_llm_api(prompt: str):
    """Build the function Guardrails calls (it passes `messages` keyword-only).

    The first call gets our own messages. On a reask, Guardrails sends its own
    system text plus a message with the bad JSON, the errors and the schema,
    but not the source text. So we resend our system prompt and put the
    original request in front of Guardrails' reask message.
    (Guardrails' `reask_messages` option can't do this: it doesn't fill in
    ${previous_response} / ${error_messages}.)
    """

    def llm_api(*, messages: list[dict], **kwargs) -> str:
        is_reask = messages[0]["content"] != SYSTEM_PROMPT
        if is_reask:
            reask = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
            messages = build_messages(f"Original request:\n{prompt}\n\n{reask}")
        return call_llm(messages, **kwargs)

    return llm_api


def guarded_call_llm(prompt: str, temperature: float = 0) -> GuardedPhone:
    try:
        result = guard(
            llm_api=_make_llm_api(prompt),
            messages=build_messages(prompt),
            model=os.getenv("LLM_PRIMARY_MODEL"),
            temperature=temperature,
        )
        logger.debug("guard result: %s", result)
        # After the last reask, Guardrails swaps a still-bad value for the
        # validator's "fix" (e.g. year 1999 -> 2000) and reports passed.
        # That value is made up, so count it as a guard failure.
        still_failing = any(
            s.validator_status == "fail" for s in (result.validation_summaries or [])
        )
        if not result.validation_passed or still_failing:
            return None
        return result.validated_output
    except ValidationError as e:
        # A real guard outcome: the answer failed validation.
        print(f"Guard failed: {e}")
        return None
    # Anything else (API, quota, network, timeout) is not the guard's verdict:
    # it propagates, and the runner records it as llm_call_failed (never cached).


if __name__ == "__main__":
    result = guarded_call_llm(
        "Extract the phone's model, storage, camera, battery, and year as JSON. Source: The Google Pixel 8 Pro ships with 256GB of storage, a 50MP dual camera system, a 5050mAh battery, and was released in 2023."
    )
    print(result)
