import json
from pathlib import Path
from typing import List, Dict

from ingestion.extract_generic import extract_text_generic
from ingestion.chunk_generic import chunk_generic
from api.document_validation import save_hash, compute_hash

from core.logger import get_logger

logger = get_logger(__name__)

UPLOADED_CHUNKS_FILE = Path("data/uploaded_chunks.json")


class DocumentIngestError(Exception):
    """Raised when a document fails to be chunked/embedded/added to the index."""


def ingest_document(file_path: Path, retriever) -> List[str]:
    """
    retriever: the running LawGenRetriever instance (from the pipeline),
    so we add to the SAME vectorstore connection already open in memory
    and can call reload_chunks() on it directly afterward.
    """
    try:
        text = extract_text_generic(file_path)
    except Exception as error:
        logger.exception("Text extraction failed for %s", file_path.name)
        raise DocumentIngestError(f"Could not extract text from file: {error}") from error

    try:
        chunks = chunk_generic(text, source_filename=file_path.name)
    except Exception as error:
        logger.exception("Chunking failed for %s", file_path.name)
        raise DocumentIngestError(f"Could not chunk document: {error}") from error

    if not chunks:
        logger.warning("No chunks produced for %s - nothing to ingest.", file_path.name)
        return []

    try:
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
    except Exception as error:
        logger.exception("Failed to add %s to Chroma.", file_path.name)
        raise DocumentIngestError(f"Failed to add document to the vector store: {error}") from error

    try:
        existing = []
        if UPLOADED_CHUNKS_FILE.exists():
            existing = json.loads(UPLOADED_CHUNKS_FILE.read_text(encoding="utf-8"))
        existing.extend(chunks)
        UPLOADED_CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        UPLOADED_CHUNKS_FILE.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as error:
        logger.exception(
            "Chunks were added to Chroma but failed to save to %s. "
            "The manifest file is now out of sync.",
            UPLOADED_CHUNKS_FILE,
        )
        raise DocumentIngestError(
            f"Document was embedded, but saving the manifest failed: {error}"
        ) from error

    save_hash(compute_hash(text))

    try:
        retriever.reload_chunks()
    except Exception as error:
        logger.exception("Ingestion succeeded but reload_chunks() failed.")
        raise DocumentIngestError(
            f"Document ingested, but the retriever failed to reload "
            f"(a service restart will be needed for it to be searchable): {error}"
        ) from error

    logger.info("Ingested %s (%d chunks).", file_path.name, len(chunks))
    return [c["id"] for c in chunks]