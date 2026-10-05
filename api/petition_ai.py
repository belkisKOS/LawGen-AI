import json
import re
import tempfile
from pathlib import Path
from typing import List, Tuple

from core.logger import get_logger
from ingestion.deepseek_ocr import (
    run_ocr_on_image,
    render_pdf_page_to_png,
    _call_ocr_api,
)
from ingestion.extract_generic import extract_text_generic
from rag.context_builder import build_context

logger = get_logger(__name__)

IMAGE_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
DOC_EXT = {".pdf", ".docx", ".txt"}
MAX_BYTES = 10 * 1024 * 1024
MAX_FILES = 8
MAX_PDF_OCR_PAGES = 6
MIN_CHARS = 15
MAX_TEXT_FOR_LLM = 14000  # chars sent to the LLM

DOC_TYPES = {
    "reply_memo": "مذكرة جوابية",
    "complaint": "عريضة افتتاح دعوى",
    "appeal": "عريضة استئناف",
    "legal_summary": "ملخص قانوني",
}

CONTEXT_KEYS = [
    "case_type", "plaintiff", "defendant", "stage",
    "court", "facts", "requests", "key_dates", "amounts",
]


def _reject(reason: str) -> dict:
    return {"accepted": False, "reason": reason}


def _ocr_pdf(pdf_path: Path) -> str:
    import fitz
    with fitz.open(pdf_path) as doc:
        pages = min(len(doc), MAX_PDF_OCR_PAGES)
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in range(1, pages + 1):
            data_url = render_pdf_page_to_png(pdf_path, n, Path(tmp) / f"p{n}.png")
            parts.append(_call_ocr_api(data_url))
    return "\n\n".join(parts).strip()


def _extract_one(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_MIME:
        return run_ocr_on_image(data, mime_type=IMAGE_MIME[suffix]).strip()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            t.write(data)
            tmp_path = Path(t.name)
        text = ""
        try:
            text = (extract_text_generic(tmp_path) or "").strip()
        except Exception:
            logger.exception("Text extraction failed for %s", filename)
        # scanned PDF with no text layer -> OCR it
        if suffix == ".pdf" and len(text) < MIN_CHARS * 3:
            text = _ocr_pdf(tmp_path)
        return text
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


def _parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError("no JSON object in LLM output")
    return json.loads(match.group(0))


UNDERSTAND_PROMPT = """أنت مساعد قانوني جزائري. اقرأ نصوص المستندات التالية (مستخرجة بالـ OCR وقد تحتوي أخطاء) واستخرج المعلومات.

قواعد صارمة:
- أعد كائن JSON فقط، بلا شرح وبلا markdown.
- لا تخترع أي معلومة. إذا لم تجدها اكتب "" (نص فارغ).
- استعمل المصطلحات الجزائرية فقط: المحكمة، القسم المدني/التجاري/العقاري/الاجتماعي/الجزائي، المجلس القضائي، المحكمة الإدارية، مجلس الدولة، المحكمة العليا. لا تذكر محاكم أي دولة أخرى.
- لغة القيم = لغة المستند.
- "stage" يجب أن تكون إحدى: "ابتدائي", "استئناف", "طعن بالنقض", "تنفيذ" أو "".

المفاتيح بالضبط:
{{"case_type": "", "plaintiff": "", "defendant": "", "stage": "", "court": "", "facts": "ملخص الوقائع في فقرة واضحة", "requests": "الطلبات", "key_dates": "", "amounts": ""}}

المستندات:
\"\"\"
{text}
\"\"\"
"""


def understand_documents(files: List[Tuple[bytes, str]], pipeline) -> dict:
    if not files:
        return _reject("لم يتم إرسال أي ملف.")
    if len(files) > MAX_FILES:
        return _reject(f"الحد الأقصى {MAX_FILES} ملفات.")

    texts = []
    for i, (data, name) in enumerate(files, start=1):
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_MIME and suffix not in DOC_EXT:
            return _reject(f"صيغة الملف رقم {i} غير مدعومة ({suffix}).")
        if len(data) > MAX_BYTES:
            return _reject(f"الملف رقم {i} يتجاوز 10 ميغابايت.")
        try:
            text = _extract_one(data, name)
        except Exception:
            logger.exception("Extraction failed for file %s (%s)", i, name)
            return _reject(f"تعذرت قراءة الملف رقم {i}. جرّب صورة أوضح.")
        if len(text) < MIN_CHARS:
            return _reject(f"لم أتمكن من قراءة نص واضح من الملف رقم {i}.")
        texts.append(text)

    combined = "\n\n=== مستند جديد ===\n\n".join(texts)[:MAX_TEXT_FOR_LLM]

    raw = pipeline.llm.generate(UNDERSTAND_PROMPT.format(text=combined))
    try:
        data = _parse_json(raw)
    except Exception:
        logger.exception("Understand: bad JSON from LLM: %s", raw[:500])
        return _reject("تعذر تحليل المستند. حاول مرة أخرى.")

    context = {k: str(data.get(k, "") or "") for k in CONTEXT_KEYS}
    return {"accepted": True, "context": context, "extracted_text": combined}


GENERATE_PROMPT = """أنت محامٍ جزائري محترف. اكتب {doc_name} كاملة وجاهزة للتقديم أمام القضاء الجزائري.

المعطيات المؤكدة من المستخدم:
- نوع القضية: {case_type}
- المدعي / الطالب: {plaintiff}
- المدعى عليه: {defendant}
- المرحلة: {stage}
- الجهة القضائية: {court}
- الوقائع: {facts}
- الطلبات: {requests}
- تواريخ: {key_dates}
- مبالغ: {amounts}

نص المستندات الأصلية (للرجوع إليه):
\"\"\"
{text}
\"\"\"

توجيهات إضافية من المستخدم: {instructions}

النصوص القانونية الجزائرية المسترجعة (استند إليها فقط عند الاستشهاد بالمواد):
{law_context}

قواعد:
- اكتب بالعربية القانونية الرسمية وبالهيكل الإجرائي الجزائري للمستند المطلوب (الديباجة، الوقائع، الدفوع/الأسباب القانونية، الطلبات، الخاتمة).
- لا تستشهد بأي مادة قانونية غير موجودة في النصوص المسترجعة أعلاه. لا تخترع أرقام مواد.
- لا تخترع أسماء أو تواريخ أو مبالغ غير مذكورة. إذا نقصت معلومة اكتب [.....].
- لا تذكر محاكم أو قوانين دولة أخرى.
- الصيغة المطلوبة للإخراج:
السطر الأول: العنوان فقط بعد "TITLE:" 
ثم سطر يحتوي فقط على ---
ثم نص المستند كاملاً بنص عادي (بدون markdown وبدون **).
"""


def generate_document(payload: dict, pipeline) -> dict:
    doc_type = payload.get("doc_type", "")
    if doc_type not in DOC_TYPES:
        return _reject("نوع المستند غير مدعوم.")
    ctx = payload.get("context") or {}
    get = lambda k: (ctx.get(k) or "—")

    # retrieve REAL Algerian law for citations
    query = f"{DOC_TYPES[doc_type]} {get('case_type')} {get('facts')}"[:600]
    try:
        results = pipeline.retriever.hybrid_search(query, k=5)
        law_context = build_context(results, max_sources=5)
        sources = pipeline.format_sources(pipeline.select_main_sources(results))
    except Exception:
        logger.exception("Retrieval failed during petition generation")
        law_context, sources = "لا توجد نصوص مسترجعة.", []

    prompt = GENERATE_PROMPT.format(
        doc_name=DOC_TYPES[doc_type],
        case_type=get("case_type"), plaintiff=get("plaintiff"),
        defendant=get("defendant"), stage=get("stage"), court=get("court"),
        facts=get("facts"), requests=get("requests"),
        key_dates=get("key_dates"), amounts=get("amounts"),
        text=(payload.get("extracted_text") or "")[:MAX_TEXT_FOR_LLM],
        instructions=(payload.get("instructions") or "لا يوجد"),
        law_context=law_context,
    )
    raw = (pipeline.llm.generate(prompt) or "").strip()

    title, body = DOC_TYPES[doc_type], raw
    if "---" in raw:
        head, _, rest = raw.partition("---")
        title = head.replace("TITLE:", "").strip() or title
        body = rest.strip()
    return {"accepted": True, "title": title[:200], "content": body, "sources": sources}