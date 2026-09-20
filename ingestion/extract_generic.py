from pathlib import Path
import fitz  # PyMuPDF
from docx import Document as DocxDocument


def extract_text_generic(file_path: Path) -> str:
    """
    Extract raw text from an uploaded file: PDF, DOCX, or TXT.
    Unlike extract_pdf.py, this returns text directly (no page markers,
    no OCR fallback) — good enough for validation + generic chunking.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(file_path)
        pages = [page.get_text("text").strip() for page in doc]
        return "\n\n".join(p for p in pages if p)

    if suffix == ".docx":
        doc = DocxDocument(str(file_path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {suffix}")