from fastapi import FastAPI
from app.llm_client import call_llm

app = FastAPI()

PRIMARY_LLM_MODEL = "gemini-3.5-flash-lite"
FALLBACK_LLM_MODEL = "gemini-3.6-flash"


@app.get("/")
def root() -> str:
    return "Hello World"


@app.get("/chat")
def chat(prompt: str) -> str:
    return call_llm(prompt, PRIMARY_LLM_MODEL)
