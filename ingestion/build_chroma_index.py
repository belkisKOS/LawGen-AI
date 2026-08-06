from pathlib import Path
import json
from typing import List, Dict

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

FINAL_CHUNKS_FILE = Path("data/final_chunks_v3/final_chunks.json")
CHROMA_DIR = Path("data/chroma_db")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks() -> List[Dict]:
    if not FINAL_CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing final chunks file: {FINAL_CHUNKS_FILE}")

    chunks = json.loads(FINAL_CHUNKS_FILE.read_text(encoding="utf-8"))

    if not chunks:
        raise ValueError("Final chunks file is empty.")

    return chunks


def chunks_to_documents(chunks: List[Dict]) -> List[Document]:
    documents = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        if not text:
            continue

        metadata = {
            "id": chunk.get("id"),
            "source_file": chunk.get("source_file"),
            "document_type": chunk.get("document_type"),
            "chunk_type": chunk.get("chunk_type"),
            "title": chunk.get("title"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "subchunk_index": chunk.get("subchunk_index"),
            "subchunk_count": chunk.get("subchunk_count"),
            "retrieval_label": chunk.get("retrieval_label"),
        }

        metadata = {key: value for key, value in metadata.items() if value is not None}

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


def build_index() -> None:
    chunks = load_chunks()
    documents = chunks_to_documents(chunks)

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Documents to index: {len(documents)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="lawgen_v3",
    )

    vectorstore.persist()

    print("\n✅ Chroma index created successfully.")
    print(f"Location: {CHROMA_DIR}")
    print("Collection: lawgen_v3")


if __name__ == "__main__":
    build_index()
