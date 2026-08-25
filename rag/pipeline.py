from retrieval.retriever import LawGenRetriever
from rag.context_builder import build_context, build_legal_prompt
from rag.query_reformulator import reformulate_legal_query
from llm.gemma_client import GemmaClient


class LawGenPipeline:

    def __init__(self):
        print("Loading retriever...")
        self.retriever = LawGenRetriever()

        print("Loading Gemma...")
        self.llm = GemmaClient()


    def format_sources(self, results):
        sources = []

        for result in results:
            sources.append(
                {
                    "source_file": result.get("source_file", ""),
                    "title": result.get("title", ""),
                    "page_start": result.get("page_start", ""),
                    "page_end": result.get("page_end", ""),
                    "chunk_type": result.get("chunk_type", ""),
                    "score": result.get("hybrid_score", result.get("score", 0)),
                    "text_preview": result.get("text", "")[:500],
                }
            )

        return sources


    def ask_with_sources(self, question):
        retrieval_question = reformulate_legal_query(
            question,
            llm_client=self.llm,
        )

        results = self.retriever.hybrid_search(
            retrieval_question,
            k=5
        )

        context = build_context(
            results,
            max_sources=5
        )

        prompt = build_legal_prompt(
            question,
            context
        )

        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "reformulated_question": retrieval_question,
            "answer": answer,
            "sources": self.format_sources
                (self.select_main_sources(results)),
        }


    def ask(self, question):
        response = self.ask_with_sources(question)
        return response["answer"]

    def select_main_sources(self, results, max_sources=3):

        if not results:
            return []

        top_score = results[0].get(
            "hybrid_score",
            results[0].get("score", 0)
        )

        if top_score == 0:
            return results[:max_sources]

        selected = []

        for item in results:
            score = item.get(
                "hybrid_score",
                item.get("score", 0)
            )

            # keep only sources close to best match
            if score >= top_score * 0.5:
                selected.append(item)

            if len(selected) >= max_sources:
                break

        return selected
    