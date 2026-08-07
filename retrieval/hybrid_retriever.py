from pathlib import Path
import json
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


FINAL_CHUNKS_FILE = Path("data/final_chunks_v3/final_chunks.json")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "lawgen_v3"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("ـ", "")
    text = text.replace("-", " ")
    text = text.replace("–", " ")
    text = text.replace("—", " ")

    arabic_to_latin_digits = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    text = text.translate(arabic_to_latin_digits)

    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+", normalize(text))


class HybridRetriever:
    def __init__(self) -> None:
        if not FINAL_CHUNKS_FILE.exists():
            raise FileNotFoundError(f"Missing final chunks file: {FINAL_CHUNKS_FILE}")

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(f"Missing Chroma database: {CHROMA_DIR}")

        self.chunks = json.loads(FINAL_CHUNKS_FILE.read_text(encoding="utf-8"))

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
        )

    def lexical_score(self, query: str, chunk: Dict) -> float:
        q = normalize(query)
        title = normalize(chunk.get("title", ""))
        body = normalize(chunk.get("text", ""))
        source = chunk.get("source_file", "")
        chunk_type = chunk.get("chunk_type", "")

        score = 0.0

        query_terms = [term for term in q.split() if len(term) >= 3]
        for term in query_terms:
            if term in title:
                score += 5
            if term in body:
                score += 1

        q_numbers = extract_numbers(query)
        title_numbers = extract_numbers(title)
        body_numbers = extract_numbers(body)

        for number in q_numbers:
            if number in title_numbers:
                score += 20
            elif number in body_numbers:
                score += 5

        if len(q_numbers) >= 2:
            all_in_title = all(number in title_numbers for number in q_numbers)
            all_in_body = all(number in body_numbers for number in q_numbers)

            if all_in_title:
                score += 120
            elif all_in_body:
                score += 50
            else:
                score -= 25

        if "دستور" in q and source == "Constitution_extracted.txt":
            score += 15

        if "قانون المالية" in q and source == "LoiFinance2026_extracted.txt":
            score += 15

        if "مرسوم" in q and source == "JournalOff_extracted.txt":
            score += 15

        if "دين الدولة" in q and "دين الدولة" in body:
            score += 70

        if "اللغة الوطنية والرسمية" in q and "اللغة الوطنية والرسمية" in body:
            score += 70

        if "ايرادات" in q or "الإيرادات" in q or "الايرادات" in q:
            if "تقدر الإيرادات" in body or "تقدر الايرادات" in body:
                score += 80
            elif "الإيرادات" in body or "الايرادات" in body:
                score += 20

        if chunk_type in {"article", "legal_act"}:
            score += 8

        if chunk_type in {"document_info", "preamble"}:
            score -= 25

        return score

    def search(self, query: str, k: int = 5, vector_k: int = 20) -> List[Tuple[Dict, float]]:
        vector_results = self.vectorstore.similarity_search_with_score(query, k=vector_k)

        scores = defaultdict(float)
        chunk_by_id = {}

        for rank, (doc, distance) in enumerate(vector_results, start=1):
            chunk_id = doc.metadata.get("id")
            if not chunk_id:
                continue

            chunk_by_id[chunk_id] = {
                "id": chunk_id,
                "source_file": doc.metadata.get("source_file"),
                "chunk_type": doc.metadata.get("chunk_type"),
                "title": doc.metadata.get("title"),
                "page_start": doc.metadata.get("page_start"),
                "page_end": doc.metadata.get("page_end"),
                "text": doc.page_content,
            }

            scores[chunk_id] += 30 / rank

        for chunk in self.chunks:
            chunk_id = chunk["id"]
            lexical = self.lexical_score(query, chunk)

            if lexical > 0:
                scores[chunk_id] += lexical
                chunk_by_id[chunk_id] = chunk

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        return [(chunk_by_id[chunk_id], score) for chunk_id, score in ranked[:k]]
