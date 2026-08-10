from guardrails import Guard

from app import call_llm
from app.schemas.guarded_schema import Phone
import os
from dotenv import load_dotenv

load_dotenv()


guard = Guard.for_pydantic(output_class=Phone)


def guarded_call_llm(prompt: str) -> Phone:
    try:
        result = guard(
            llm_api=call_llm,
            messages=[{"role": "user", "content": prompt}],
            model=os.getenv("LLM_PRIMARY_MODEL"),
        )
        print(result)
        return result.validated_output
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    result = guarded_call_llm(
        "Extract the phone's model, storage, camera, battery, and year as JSON. Source: The Google Pixel 8 Pro ships with 256GB of storage, a 50MP dual camera system, a 5050mAh battery, and was released in 2023."
    )
    print(result)
