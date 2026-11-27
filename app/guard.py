from guardrails import Guard
from guardrails.errors import ValidationError

from app import call_llm, llm_client
from app.llm_client import FALLBACK_MODEL, FallbackBudgetExhausted, call_with_fallback
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


def _make_llm_api(prompt: str, info: dict, fallback_only: bool = False):
    """Build the function Guardrails calls (it passes `messages` keyword-only).

    The first call gets our own messages. On a reask, Guardrails sends its own
    system text plus a message with the bad JSON, the errors and the schema,
    but not the source text. So we resend our system prompt and put the
    original request in front of Guardrails' reask message.
    (Guardrails' `reask_messages` option can't do this: it doesn't fill in
    ${previous_response} / ${error_messages}.)

    Normal mode: primary model, switching to the fallback model on server
    errors / timeouts / daily quota. fallback_only: every call goes to the
    fallback model (used after a failed reask). `info` records which model
    answered and why the fallback was used.
    """

    def llm_api(*, messages: list[dict], **kwargs) -> str:
        is_reask = messages[0]["content"] != SYSTEM_PROMPT
        if is_reask:
            reask = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
            messages = build_messages(f"Original request:\n{prompt}\n\n{reask}")
        temperature = kwargs.get("temperature")
        if fallback_only:
            if not llm_client.fallback_budget.take():
                raise FallbackBudgetExhausted("fallback budget used up")
            return call_llm(messages, FALLBACK_MODEL, temperature=temperature)
        result = call_with_fallback(messages, kwargs.get("model"), temperature=temperature)
        if result["fallback_reason"]:
            info.update(model_used=result["model_used"], fallback_reason=result["fallback_reason"])
        return result["text"]

    return llm_api


def guarded_call(prompt: str, temperature: float = 0) -> dict:
    """Run the guard; if it still fails after the reask, run it once more on the fallback model.

    Returns {"output": dict | None, "model_used", "fallback_reason"}.
    output None = guard_failed.
    """
    info = {"model_used": os.getenv("LLM_PRIMARY_MODEL"), "fallback_reason": None}
    output = _run_guard(prompt, temperature, _make_llm_api(prompt, info))
    if output is None and llm_client.fallback_budget.remaining() > 0:
        print(f"Guard failed; trying {FALLBACK_MODEL}", flush=True)
        info = {"model_used": FALLBACK_MODEL, "fallback_reason": "guard_failed"}
        output = _run_guard(prompt, temperature, _make_llm_api(prompt, info, fallback_only=True))
    return {"output": output, **info}


def guarded_call_llm(prompt: str, temperature: float = 0) -> GuardedPhone:
    return guarded_call(prompt, temperature)["output"]


def _run_guard(prompt: str, temperature: float, llm_api) -> dict | None:
    try:
        result = guard(
            llm_api=llm_api,
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
