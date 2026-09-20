import json
import re
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from core.logger import get_logger

logger = get_logger(__name__)


class RetrieverInitError(Exception):
    """Raised when the retriever (Chroma / embeddings / chunk files) fails to load."""


class LawGenRetriever:

    def __init__(self):
        try:
            logger.info("Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as error:
            logger.exception("Failed to load the embedding model.")
            raise RetrieverInitError(f"Embedding model failed to load: {error}") from error

        try:
            logger.info("Connecting to Chroma at data/chroma_db...")
            self.vectorstore = Chroma(
                collection_name="lawgen_v3",
                persist_directory="data/chroma_db",
                embedding_function=self.embeddings,
            )
        except Exception as error:
            logger.exception("Failed to connect to Chroma.")
            raise RetrieverInitError(f"Chroma connection failed: {error}") from error

        try:
            self.reload_chunks()
        except Exception as error:
            logger.exception("Failed to load chunk manifest files.")
            raise RetrieverInitError(f"Chunk manifest loading failed: {error}") from error

        logger.info("LawGenRetriever ready (%d chunks loaded).", len(self.chunks))

    def reload_chunks(self):
        """Re-read curated + uploaded chunk files into memory. Call after ingest/delete."""
        curated_path = Path("data/final_chunks_v3/final_chunks.json")
        if not curated_path.exists():
            raise FileNotFoundError(f"Missing curated chunk file: {curated_path}")

        curated = json.loads(curated_path.read_text(encoding="utf-8"))

        uploaded_path = Path("data/uploaded_chunks.json")
        uploaded = []
        if uploaded_path.exists():
            uploaded = json.loads(uploaded_path.read_text(encoding="utf-8"))

        self.chunks = curated + uploaded

    def check_connection(self) -> tuple[bool, str]:
        """Cheap check for /health: confirms Chroma is actually reachable and has data."""
        try:
            count = self.vectorstore._collection.count()
            if count == 0:
                return False, "Chroma collection is reachable but empty (0 vectors)."
            return True, f"ok ({count} vectors, {len(self.chunks)} chunks loaded)"
        except Exception as error:
            logger.warning("Chroma connectivity check failed: %s", error)
            return False, str(error)

    def normalize_digits(self, text):
        text = text or ""

        return text.translate(
            str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        )


    def normalize(self, text):
        text = self.normalize_digits(text)
        text = text.lower()
        text = text.replace("ـ", "")
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    def extract_article_numbers(self, text):
        text = self.normalize(text)

        numbers = re.findall(
            r"المادة\s+([0-9]+)",
            text
        )

        return set(numbers)


    def to_result(self, chunk, score=0):
        return {
            "id": chunk.get("id", ""),
            "text": self.normalize_digits(chunk.get("text", "")),
            "score": score,
            "source_file": chunk.get("source_file", ""),
            "document_type": chunk.get("document_type", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "title": self.normalize_digits(chunk.get("title", "")),
            "page_start": chunk.get("page_start", ""),
            "page_end": chunk.get("page_end", ""),
            "retrieval_label": self.normalize_digits(
                chunk.get("retrieval_label", "")
            ),
        }


    def lexical_score(self, question, chunk):
        question_n = self.normalize(question)
        text_n = self.normalize(chunk.get("text", ""))
        title_n = self.normalize(chunk.get("title", ""))

        score = 0

        # Strong concept boosts for constitutional identity and trade terms
        if "تمازيغت" in question_n:
            if "تمازيغت" in text_n or "تمازيغت" in title_n:
                score += 250
            if "المادة 4" in title_n:
                score += 300

        if "التجارة الخارجية" in question_n:
            if "التجارة الخارجية" in text_n or "التجارة الخارجية" in title_n:
                score += 250
            if "المادة 32" in title_n:
                score += 300

        if "الضمان الاجتماعي" in question_n or "هيئات الضمان الاجتماعي" in question_n:
            if "هيئات الضمان الاجتماعي" in text_n:
                score += 400
            if "القطاعات الصحية" in text_n or "المؤسسات الاستشفائية" in text_n:
                score += 150
            if chunk.get("source_file") == "LoiFinance2026_extracted.txt":
                score += 150
            if "المادة 4" in title_n:
                score += 300

        if (
            "الإيرادات" in question_n
            or "ايرادات" in question_n
            or "الحواصل" in question_n
            or "المداخيل" in question_n
        ):
            if chunk.get("source_file") == "LoiFinance2026_extracted.txt":
                score += 200
            if "الإيرادات والحواصل والمداخيل" in text_n:
                score += 500
            if "تقدر الإيرادات" in text_n or "تقدر الايرادات" in text_n:
                score += 400
            if "المادة 2" in title_n:
                score += 300


        question_articles = self.extract_article_numbers(question)
        title_articles = self.extract_article_numbers(chunk.get("title", ""))
        text_articles = self.extract_article_numbers(chunk.get("text", "")[:120])

        # Exact article-number matching
        for article_number in question_articles:
            if article_number in title_articles:
                score += 500

            elif article_number in text_articles:
                score += 450

        # Penalize article chunks with other article numbers
        if question_articles and title_articles:
            if question_articles.isdisjoint(title_articles):
                score -= 80

        exact_phrases = [
            "الإسلام دين الدولة",
            "دين الدولة",
            "اللغة العربية هي اللغة الوطنية والرسمية",
            "اللغة الوطنية والرسمية",
            "تمازيغت هي كذلك لغة وطنية ورسمية",
            "تمازيغت كلغة وطنية ورسمية",
            "قانون المالية لسنة 2026",
            "الإيرادات والحواصل والمداخيل",
            "مرسوم رئاسي رقم",
            "26 - 239",
            "26-239",
        ]

        for phrase in exact_phrases:
            phrase_n = self.normalize(phrase)

            if phrase_n in question_n and phrase_n in text_n:
                score += 120

            if phrase_n in question_n and phrase_n in title_n:
                score += 150

        important_words = [
            "الإسلام",
            "دين",
            "الدولة",
            "اللغة",
            "الوطنية",
            "الرسمية",
            "الإيرادات",
            "المالية",
            "مرسوم",
            "رئاسي",
        ]

        for word in important_words:
            word_n = self.normalize(word)

            if word_n in question_n and word_n in text_n:
                score += 5

            if word_n in question_n and word_n in title_n:
                score += 10

        decree_match = re.search(
            r"(\d+)\s*-\s*(\d+)",
            self.normalize(question)
        )

        if decree_match:
            number_a = decree_match.group(1)
            number_b = decree_match.group(2)

            compact = f"{number_a}-{number_b}"
            spaced = f"{number_a} - {number_b}"

            text_full = text_n + " " + title_n

            if compact in text_full or spaced in text_full:
                score += 300

        return score


    def semantic_search(self, question, k=10):
        results = self.vectorstore.similarity_search_with_score(
            question,
            k=k,
        )

        output = []

        for doc, distance in results:
            metadata = doc.metadata or {}

            chunk = {
                "id": metadata.get("id", ""),
                "text": doc.page_content,
                "source_file": metadata.get("source_file", ""),
                "document_type": metadata.get("document_type", ""),
                "chunk_type": metadata.get("chunk_type", ""),
                "title": metadata.get("title", ""),
                "page_start": metadata.get("page_start", ""),
                "page_end": metadata.get("page_end", ""),
                "retrieval_label": metadata.get("retrieval_label", ""),
            }

            result = self.to_result(
                chunk,
                score=float(distance),
            )

            result["semantic_score"] = max(
                0,
                100 - float(distance) * 100,
            )

            output.append(result)

        return output


    def lexical_search(self, question, min_score=20):
        output = []

        for chunk in self.chunks:
            score = self.lexical_score(
                question,
                chunk,
            )

            if score >= min_score:
                result = self.to_result(
                    chunk,
                    score=score,
                )

                result["lexical_score"] = score

                output.append(result)

        return output


    def hybrid_search(self, question, k=10):
        question_n = self.normalize(question)
        requested_articles = self.extract_article_numbers(question)

        source_filter = None

        if "قانون المالية" in question_n or "الميزانية" in question_n:
            source_filter = "LoiFinance2026_extracted.txt"

        elif "مرسوم" in question_n or "الجريدة الرسمية" in question_n:
            source_filter = "JournalOff_extracted.txt"

        elif "دستور" in question_n:
            source_filter = "Constitution_extracted.txt"

        # If the question explicitly asks for article numbers,
        # use deterministic article retrieval first.
        if requested_articles:
            article_results = []

            for chunk in self.chunks:
                source_file = chunk.get("source_file", "")
                title_articles = self.extract_article_numbers(
                    chunk.get("title", "")
                )

                if source_filter and source_file != source_filter:
                    continue

                for article_number in requested_articles:
                    if article_number in title_articles:
                        result = self.to_result(
                            chunk,
                            score=1000
                        )

                        result["lexical_score"] = 1000
                        result["hybrid_score"] = 1000
                        article_results.append(result)

            # Keep only one best chunk per requested article number
            unique_results = []
            seen_articles = set()

            for article_number in requested_articles:
                for result in article_results:
                    result_articles = self.extract_article_numbers(
                        result.get("title", "")
                    )

                    if article_number in result_articles and article_number not in seen_articles:
                        unique_results.append(result)
                        seen_articles.add(article_number)
                        break

            if unique_results:
                return unique_results[:k]

        # Normal hybrid retrieval for general questions
        semantic_results = self.semantic_search(
            question,
            k=k,
        )

        lexical_results = self.lexical_search(
            question,
            min_score=20,
        )

        merged = {}

        for result in semantic_results + lexical_results:
            key = result.get("id") or result.get("text")[:80]

            if key not in merged:
                merged[key] = result

            else:
                existing = merged[key]

                existing["semantic_score"] = max(
                    existing.get("semantic_score", 0),
                    result.get("semantic_score", 0),
                )

                existing["lexical_score"] = max(
                    existing.get("lexical_score", 0),
                    result.get("lexical_score", 0),
                )

                existing["hybrid_score"] = (
                    existing.get("semantic_score", 0)
                    + existing.get("lexical_score", 0)
                )

        final_results = []

        for result in merged.values():
            result["hybrid_score"] = (
                result.get("semantic_score", 0)
                + result.get("lexical_score", 0)
            )

            final_results.append(result)

        final_results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        return final_results[:k]