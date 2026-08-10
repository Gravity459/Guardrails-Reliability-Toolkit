from fastapi import FastAPI
from app.llm_client import call_llm
from dotenv import load_dotenv
import os

load_dotenv()


app = FastAPI()


@app.get("/")
def root() -> str:
    return "Hello World"


@app.get("/chat")
def chat(prompt: str) -> str:
    return call_llm(
        [{"role": "user", "content": prompt}], os.getenv("LLM_PRIMARY_MODEL")
    )
