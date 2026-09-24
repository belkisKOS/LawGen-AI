import tempfile
from pathlib import Path
from typing import List, Tuple

from core.logger import get_logger
from ingestion.extract_generic import extract_text_generic
from api.document_validation import (
    classify_with_llm,
    ALLOWED_EXTENSIONS,
    MAX_SIZE_BYTES,
    MIN_WORDS,
)
from rag.context_builder import clean_text

logger = get_logger(__name__)


class DocumentAskError(Exception):
    """Raised when a file cannot be processed (unexpected failure, not a normal rejection)."""


def answer_from_documents(files: List[Tuple[bytes, str]], question: str, pipeline, history=None):
    """
    files: list of (raw_bytes, filename) tuples, in the order the user attached them.

    Validates EVERY file (extension, size, extractable text, legal-content check).
    If ANY single file fails, the WHOLE batch is rejected and the pipeline is never
    called. Each file is written to a temp path only long enough to extract its
    text, then deleted — nothing is kept on disk and nothing is ingested into Chroma.
    """
    history = history or []

    if not files:
        raise DocumentAskError("No files provided.")

    extracted_texts = []

    for index, (file_bytes, filename) in enumerate(files, start=1):
        suffix = Path(filename).suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            return {
                "accepted": False,
                "reason": f"صيغة الملف رقم {index} غير مدعومة ({suffix}). الصيغ المدعومة: PDF, DOCX, TXT.",
                "answer": None,
                "sources": [],
            }

        if len(file_bytes) > MAX_SIZE_BYTES:
            return {
                "accepted": False,
                "reason": f"الملف رقم {index} يتجاوز الحجم المسموح به (10 ميغابايت).",
                "answer": None,
                "sources": [],
            }

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)
            text = extract_text_generic(tmp_path).strip()
        except Exception:
            logger.exception("Text extraction failed for file %s (%s)", index, filename)
            return {
                "accepted": False,
                "reason": f"تعذرت قراءة محتوى الملف رقم {index}. تأكد أنه غير تالف أو صورة ممسوحة بدون طبقة نصية.",
                "answer": None,
                "sources": [],
            }
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        word_count = len(text.split())
        if word_count < MIN_WORDS:
            return {
                "accepted": False,
                "reason": f"الملف رقم {index} قصير جدا أو لا يحتوي على نص كاف ({word_count} كلمة).",
                "answer": None,
                "sources": [],
            }

        try:
            is_legal, llm_reason = classify_with_llm(text, pipeline.llm.generate)
        except Exception as error:
            logger.exception("Legal classification failed for file %s", index)
            raise DocumentAskError(f"Could not verify file {index}: {error}") from error

        if not is_legal:
            return {
                "accepted": False,
                "reason": f"الملف رقم {index} لا يبدو وثيقة قانونية جزائرية: {llm_reason}",
                "answer": None,
                "sources": [],
            }

        extracted_texts.append(text)

    combined_context = "\n\n---\n\n".join(
        clean_text(t, max_chars=1500) for t in extracted_texts
    )
    user_question = (question or "").strip()

    if user_question:
        retrieval_question = (
            f"{user_question}\n\nنصوص مستخرجة من ملفات أرفقها المستخدم:\n{combined_context}"
        )
    else:
        retrieval_question = (
            "اشرح لي هذه الوثائق وأجب بالاعتماد على القانون الجزائري:\n"
            f"{combined_context}"
        )

    result = pipeline.ask_with_sources(retrieval_question, history=history)

    return {
        "accepted": True,
        "reason": "OK",
        "answer": result["answer"],
        "sources": result["sources"],
        "extracted_text": combined_context,
    }