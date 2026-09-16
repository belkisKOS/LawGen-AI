import hashlib
from typing import List, Dict


def chunk_generic(text: str, source_filename: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
    """
    Simple sliding-window paragraph chunker for arbitrary uploaded documents.
    Unlike chunk_documents_v3.py (article-numbered, structured government PDFs),
    this makes no assumptions about document structure.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    buffer = ""
    chunk_index = 0

    for para in paragraphs:
        if len(buffer) + len(para) > chunk_size and buffer:
            chunks.append(_make_chunk(buffer, source_filename, chunk_index))
            # keep overlap tail for context continuity
            buffer = buffer[-overlap:] + "\n\n" + para
            chunk_index += 1
        else:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para

    if buffer.strip():
        chunks.append(_make_chunk(buffer, source_filename, chunk_index))

    return chunks


def _make_chunk(text: str, source_filename: str, index: int) -> Dict:
    chunk_id = hashlib.sha256(f"{source_filename}-{index}-{text[:50]}".encode()).hexdigest()[:16]
    return {
        "id": f"upload_{chunk_id}",
        "text": text,
        "source_file": source_filename,
        "document_type": "user_upload",
        "chunk_type": "user_upload",
        "title": f"{source_filename} (part {index + 1})",
        "page_start": "",
        "page_end": "",
        "retrieval_label": f"{source_filename} - {index + 1}",
    }