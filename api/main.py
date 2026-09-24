import json
import sys
import shutil
import tempfile
from pathlib import Path
from pathlib import Path as PathLib
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import get_logger

logger = get_logger(__name__)

from rag.pipeline import LawGenPipeline, PipelineError
from api.document_validation import check_basic_rules, classify_with_llm, DocumentValidationError
from api.document_ingest import ingest_document, DocumentIngestError
from ingestion.extract_generic import extract_text_generic
from api.image_ask import answer_from_images, ImageAskError
from api.document_ask import answer_from_documents, DocumentAskError


app = FastAPI(
    title="LawGen AI API",
    description="Legal RAG API for Algerian legal documents.",
    version="0.1.0",
)


logger.info("Starting LawGen AI API - initializing pipeline...")
try:
    bot = LawGenPipeline()
    if bot.ready:
        logger.info("Pipeline initialized successfully.")
    else:
        logger.error("Pipeline initialized in a NOT READY state: %s", bot.init_error)
except Exception as error:
    logger.exception("Unexpected error while constructing LawGenPipeline.")
    bot = None


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {exc}"},
    )


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
    checks = {}

    pipeline_ok = bot is not None and bot.ready
    checks["pipeline"] = {
        "ok": pipeline_ok,
        "detail": "ok" if pipeline_ok else (
            (bot.init_error if bot else "Pipeline object failed to construct")
        ),
    }

    if pipeline_ok:
        vector_ok, vector_detail = bot.retriever.check_connection()
        checks["vector_store"] = {"ok": vector_ok, "detail": vector_detail}

        llm_ok, llm_detail = bot.llm.check_connection()
        checks["llm"] = {"ok": llm_ok, "detail": llm_detail}
    else:
        checks["vector_store"] = {"ok": False, "detail": "skipped - pipeline not ready"}
        checks["llm"] = {"ok": False, "detail": "skipped - pipeline not ready"}

    chunk_files_ok, chunk_files_detail = _check_chunk_files()
    checks["chunk_files"] = {"ok": chunk_files_ok, "detail": chunk_files_detail}

    overall_ok = all(check["ok"] for check in checks.values())

    body = {
        "status": "ok" if overall_ok else "degraded",
        "checks": checks,
    }

    if not overall_ok:
        logger.warning("Health check reported degraded status: %s", checks)

    return JSONResponse(status_code=200 if overall_ok else 503, content=body)


def _check_chunk_files() -> tuple[bool, str]:
    curated = PathLib("data/final_chunks_v3/final_chunks.json")
    if not curated.exists():
        return False, f"Missing curated chunk file: {curated}"

    try:
        import json
        json.loads(curated.read_text(encoding="utf-8"))
    except Exception as error:
        return False, f"Curated chunk file exists but failed to parse: {error}"

    return True, "ok"


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if bot is None or not bot.ready:
        detail = bot.init_error if bot else "Pipeline failed to initialize."
        logger.error("Rejecting /ask because the pipeline isn't ready: %s", detail)
        return JSONResponse(
            status_code=503,
            content={"detail": f"AI service is not ready: {detail}"},
        )

    history = [
        {"role": msg.role, "content": msg.content}
        for msg in request.history
    ]

    try:
        return bot.ask_with_sources(request.question, history=history)
    except PipelineError as error:
        logger.error("PipelineError on /ask: %s", error)
        return JSONResponse(status_code=502, content={"detail": str(error)})
    except Exception as error:
        logger.exception("Unexpected error on /ask for question: %s", request.question)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected error while answering: {error}"},
        )


@app.post("/documents/validate")
async def validate_document(file: UploadFile = File(...)):
    if bot is None or not bot.ready:
        detail = bot.init_error if bot else "Pipeline failed to initialize."
        return JSONResponse(
            status_code=503,
            content={"detail": f"AI service is not ready: {detail}"},
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=PathLib(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = PathLib(tmp.name)
    except Exception as error:
        logger.exception("Failed to save uploaded file for validation.")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Could not read the uploaded file: {error}"},
        )

    try:
        try:
            text = extract_text_generic(tmp_path)
        except Exception as error:
            logger.exception("Text extraction failed during validation for %s", file.filename)
            return JSONResponse(
                status_code=400,
                content={"detail": f"Could not extract text from file: {error}"},
            )

        ok, reason, already_present = check_basic_rules(tmp_path, text)

        if not ok:
            return {"valid": False, "reason": reason, "already_present": False}

        if already_present:
            return {"valid": True, "reason": "OK", "already_present": True}

        try:
            is_legal, llm_reason = classify_with_llm(text, bot.llm.generate)
        except DocumentValidationError as error:
            logger.error("Validation LLM check failed: %s", error)
            return JSONResponse(
                status_code=502,
                content={"detail": str(error)},
            )

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
    if bot is None or not bot.ready:
        detail = bot.init_error if bot else "Pipeline failed to initialize."
        return JSONResponse(
            status_code=503,
            content={"detail": f"AI service is not ready: {detail}"},
        )

    tmp_dir = tempfile.mkdtemp()
    original_name = file.filename or "uploaded_file"
    tmp_path = PathLib(tmp_dir) / original_name

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as error:
        logger.exception("Failed to save uploaded file for ingestion.")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Could not read the uploaded file: {error}"},
        )

    try:
        chunk_ids = ingest_document(tmp_path, bot.retriever)
        return {"chunk_ids": chunk_ids}
    except DocumentIngestError as error:
        logger.error("Ingestion failed for %s: %s", original_name, error)
        return JSONResponse(status_code=502, content={"detail": str(error)})
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


@app.post("/ask-image")
async def ask_image(
    files: List[UploadFile] = File(...),
    question: str = Form(""),
    history: str = Form("[]"),
):
    if bot is None or not bot.ready:
        detail = bot.init_error if bot else "Pipeline failed to initialize."
        return JSONResponse(status_code=503, content={"detail": f"AI service is not ready: {detail}"})

    try:
        history_list = json.loads(history)
    except Exception:
        history_list = []

    images = []
    for f in files:
        try:
            content = await f.read()
        except Exception as error:
            return JSONResponse(status_code=500, content={"detail": f"Could not read {f.filename}: {error}"})
        images.append((content, f.filename or "photo.jpg"))

    try:
        return answer_from_images(images, question, bot, history=history_list)
    except ImageAskError as error:
        return JSONResponse(status_code=400, content={"detail": str(error)})
    except Exception as error:
        logger.exception("Unexpected error on /ask-image")
        return JSONResponse(status_code=500, content={"detail": f"Unexpected error: {error}"})


@app.post("/ask-file")
async def ask_file(
    files: List[UploadFile] = File(...),
    question: str = Form(""),
    history: str = Form("[]"),
):
    if bot is None or not bot.ready:
        detail = bot.init_error if bot else "Pipeline failed to initialize."
        return JSONResponse(status_code=503, content={"detail": f"AI service is not ready: {detail}"})

    try:
        history_list = json.loads(history)
    except Exception:
        history_list = []

    documents = []
    for f in files:
        try:
            content = await f.read()
        except Exception as error:
            return JSONResponse(status_code=500, content={"detail": f"Could not read {f.filename}: {error}"})
        documents.append((content, f.filename or "document"))

    try:
        return answer_from_documents(documents, question, bot, history=history_list)
    except DocumentAskError as error:
        return JSONResponse(status_code=400, content={"detail": str(error)})
    except Exception as error:
        logger.exception("Unexpected error on /ask-file")
        return JSONResponse(status_code=500, content={"detail": f"Unexpected error: {error}"})