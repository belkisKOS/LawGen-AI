import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from rag.pipeline import LawGenPipeline


app = FastAPI(
    title="LawGen AI API",
    description="Legal RAG API for Algerian legal documents.",
    version="0.1.0",
)


bot = LawGenPipeline()


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    source_file: str
    title: str
    page_start: Optional[int | str] = ""
    page_end: Optional[int | str] = ""
    chunk_type: str
    score: float | int
    text_preview: str


class AskResponse(BaseModel):
    question: str
    reformulated_question: str
    answer: str
    sources: List[Source]


@app.get("/")
def root():
    return {
        "name": "LawGen AI API",
        "status": "running",
        "endpoint": "/ask",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return bot.ask_with_sources(request.question)
