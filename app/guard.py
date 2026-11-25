from guardrails import Guard
from google.genai import errors as genai_errors

from app import call_llm
from app.prompts import build_messages
from app.schemas.guarded_schema import GuardedPhone
import os
from dotenv import load_dotenv

load_dotenv()


guard = Guard.for_pydantic(output_class=GuardedPhone)


def _llm_api(*, messages: list[dict], **kwargs) -> str:
    """Guardrails expects `messages` as a keyword-only argument."""
    return call_llm(messages, **kwargs)


def guarded_call_llm(prompt: str, temperature: float = 0) -> GuardedPhone:
    try:
        result = guard(
            llm_api=_llm_api,
            messages=build_messages(prompt),
            model=os.getenv("LLM_PRIMARY_MODEL"),
            temperature=temperature,
        )
        print(result)
        return result.validated_output
    except Exception as e:
        print(f"Error: {e}")
        if isinstance(e, genai_errors.APIError) or isinstance(
            e.__cause__, genai_errors.APIError
        ):
            raise
        return None


if __name__ == "__main__":
    result = guarded_call_llm(
        "Extract the phone's model, storage, camera, battery, and year as JSON. Source: The Google Pixel 8 Pro ships with 256GB of storage, a 50MP dual camera system, a 5050mAh battery, and was released in 2023."
    )
    print(result)
