from retrieval.retriever import LawGenRetriever
from rag.context_builder import build_context, build_legal_prompt
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
        results = self.retriever.hybrid_search(
            question,
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
            "answer": answer,
            "sources": self.format_sources(results),
        }


    def ask(self, question):
        response = self.ask_with_sources(question)
        return response["answer"]
