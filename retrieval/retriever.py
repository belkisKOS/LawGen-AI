import json
import re
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class LawGenRetriever:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = Chroma(
            collection_name="lawgen_v3",
            persist_directory="data/chroma_db",
            embedding_function=self.embeddings,
        )

        self.chunks = json.loads(
            Path("data/final_chunks_v3/final_chunks.json").read_text(
                encoding="utf-8"
            )
        )


    def normalize(self, text):

        text = text or ""
        text = text.lower()
        text = text.replace("ـ", "")
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    def to_result(self, chunk, score=0):

        return {
            "id": chunk.get("id", ""),
            "text": chunk.get("text", ""),
            "score": score,
            "source_file": chunk.get("source_file", ""),
            "document_type": chunk.get("document_type", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "title": chunk.get("title", ""),
            "page_start": chunk.get("page_start", ""),
            "page_end": chunk.get("page_end", ""),
            "retrieval_label": chunk.get("retrieval_label", ""),
        }


    def lexical_score(self, question, chunk):

        question_n = self.normalize(question)
        text_n = self.normalize(chunk.get("text", ""))
        title_n = self.normalize(chunk.get("title", ""))

        score = 0

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

        # Exact decree number handling
        decree_match = re.search(r"(\d+)\s*-\s*(\d+)", question)

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

            result["semantic_score"] = max(0, 100 - float(distance) * 100)

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
                merged[key].update(
                    {
                        field: result.get(field) or merged[key].get(field)
                        for field in result
                    }
                )

                merged[key]["semantic_score"] = max(
                    merged[key].get("semantic_score", 0),
                    result.get("semantic_score", 0),
                )

                merged[key]["lexical_score"] = max(
                    merged[key].get("lexical_score", 0),
                    result.get("lexical_score", 0),
                )

        final_results = []

        for result in merged.values():

            semantic_score = result.get("semantic_score", 0)
            lexical_score = result.get("lexical_score", 0)

            result["hybrid_score"] = lexical_score + semantic_score

            final_results.append(result)

        final_results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        return final_results[:k]
