from google import genai
import os

from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("LLM_API_KEY"))


def call_llm(prompt: str, model: str) -> str:
    interaction = client.interactions.create(model=model, input=prompt)
    return interaction.output_text
