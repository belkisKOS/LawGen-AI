import json
from pathlib import Path
from typing import List, Dict

from ingestion.extract_generic import extract_text_generic
from ingestion.chunk_generic import chunk_generic
from api.document_validation import save_hash, compute_hash

UPLOADED_CHUNKS_FILE = Path("data/uploaded_chunks.json")


def ingest_document(file_path: Path, retriever) -> List[str]:
    """
    retriever: the running LawGenRetriever instance (from the pipeline),
    so we add to the SAME vectorstore connection already open in memory
    and can call reload_chunks() on it directly afterward.
    """
    text = extract_text_generic(file_path)
    chunks = chunk_generic(text, source_filename=file_path.name)

    if not chunks:
        return []

    # Add to Chroma via the existing vectorstore instance
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=c["text"],
            metadata={k: v for k, v in c.items() if k != "text" and v is not None},
        )
        for c in chunks
    ]
    retriever.vectorstore.add_documents(docs)
    retriever.vectorstore.persist()

    # Append to the uploaded chunks manifest (never touch the curated file)
    existing = []
    if UPLOADED_CHUNKS_FILE.exists():
        existing = json.loads(UPLOADED_CHUNKS_FILE.read_text(encoding="utf-8"))
    existing.extend(chunks)
    UPLOADED_CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    UPLOADED_CHUNKS_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save the hash so future duplicate uploads are caught
    save_hash(compute_hash(text))

    # Reload in-memory chunks so exact-article lexical search sees new content immediately
    retriever.reload_chunks()

    return [c["id"] for c in chunks]