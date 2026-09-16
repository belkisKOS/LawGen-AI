import sys
import shutil
import tempfile
from pathlib import Path
from pathlib import Path as PathLib
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from rag.pipeline import LawGenPipeline
from api.document_validation import check_basic_rules, classify_with_llm
from api.document_ingest import ingest_document
from ingestion.extract_generic import extract_text_generic


app = FastAPI(
    title="LawGen AI API",
    description="Legal RAG API for Algerian legal documents.",
    version="0.1.0",
)


bot = LawGenPipeline()


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: List[Message] = []


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
    history = [
        {
            "role": msg.role,
            "content": msg.content
        }
        for msg in request.history
    ]

    return bot.ask_with_sources(
        request.question,
        history=history
    )


@app.post("/documents/validate")
async def validate_document(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=PathLib(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = PathLib(tmp.name)

    try:
        text = extract_text_generic(tmp_path)
        ok, reason, already_present = check_basic_rules(tmp_path, text)

        if not ok:
            return {"valid": False, "reason": reason, "already_present": False}

        if already_present:
            return {"valid": True, "reason": "OK", "already_present": True}

        is_legal, llm_reason = classify_with_llm(text, bot.llm.generate)
        if not is_legal:
            return {
                "valid": False,
                "reason": f"Not recognized as an Algerian legal document: {llm_reason}",
                "already_present": False,
            }

        return {"valid": True, "reason": "OK", "already_present": False}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/documents/ingest")
async def ingest_document_route(file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp()
    original_name = file.filename or "uploaded_file"
    tmp_path = PathLib(tmp_dir) / original_name

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunk_ids = ingest_document(tmp_path, bot.retriever)
        return {"chunk_ids": chunk_ids}
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass