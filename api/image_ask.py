import mimetypes
from typing import List, Tuple

from core.logger import get_logger
from ingestion.deepseek_ocr import run_ocr_on_image
from api.document_validation import classify_with_llm
from rag.context_builder import clean_text

logger = get_logger(__name__)

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE_BYTES = 8 * 1024 * 1024  # 8 MB
MIN_OCR_CHARS = 15


class ImageAskError(Exception):
    """Raised when an image cannot be processed (unexpected failure, not a normal rejection)."""


def guess_mime_type(filename: str, fallback: str = "image/jpeg") -> str:
    mime_type, _ = mimetypes.guess_type(filename or "")
    return mime_type or fallback


def answer_from_images(images: List[Tuple[bytes, str]], question: str, pipeline, history=None):
    """
    images: list of (raw_bytes, filename) tuples, in the order the user attached them.

    Validates EVERY image. If any single one fails (too big, wrong type, unreadable,
    not a legal document), the WHOLE batch is rejected and the pipeline is never
    called — no wasted LLM call. Nothing here is written to disk or ingested into
    Chroma; this is a per-request, throwaway flow, same spirit as /documents/validate
    but for photos instead of files.
    """
    history = history or []

    if not images:
        raise ImageAskError("No images provided.")

    extracted_texts = []

    for index, (image_bytes, filename) in enumerate(images, start=1):
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            return {
                "accepted": False,
                "reason": f"الصورة رقم {index} تتجاوز الحجم المسموح به (8 ميغابايت).",
                "answer": None,
                "sources": [],
            }

        mime_type = guess_mime_type(filename)
        if mime_type not in ALLOWED_MIME_TYPES:
            return {
                "accepted": False,
                "reason": f"الصورة رقم {index} بصيغة غير مدعومة ({mime_type}).",
                "answer": None,
                "sources": [],
            }

        try:
            text = run_ocr_on_image(image_bytes, mime_type=mime_type).strip()
        except Exception:
            logger.exception("OCR failed for image %s (%s)", index, filename)
            return {
                "accepted": False,
                "reason": f"تعذرت قراءة الصورة رقم {index}. حاول التقاط صورة أوضح.",
                "answer": None,
                "sources": [],
            }

        if len(text) < MIN_OCR_CHARS:
            return {
                "accepted": False,
                "reason": f"لم أتمكن من قراءة نص واضح من الصورة رقم {index}.",
                "answer": None,
                "sources": [],
            }

        try:
            is_legal, llm_reason = classify_with_llm(text, pipeline.llm.generate)
        except Exception as error:
            logger.exception("Legal classification failed for image %s", index)
            raise ImageAskError(f"Could not verify image {index}: {error}") from error

        if not is_legal:
            return {
                "accepted": False,
                "reason": f"الصورة رقم {index} لا تبدو وثيقة قانونية جزائرية: {llm_reason}",
                "answer": None,
                "sources": [],
            }

        extracted_texts.append(text)

    # Every image passed validation — build one combined context, answer once.
    combined_context = "\n\n---\n\n".join(
        clean_text(t, max_chars=1200) for t in extracted_texts
    )
    user_question = (question or "").strip()

    if user_question:
        retrieval_question = (
            f"{user_question}\n\nنصوص مستخرجة من صور أرفقها المستخدم:\n{combined_context}"
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