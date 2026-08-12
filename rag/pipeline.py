from retrieval.retriever import LawGenRetriever
from rag.context_builder import build_context, build_legal_prompt
from llm.gemma_client import GemmaClient


class LawGenPipeline:

    def __init__(self):
        print("Loading retriever...")
        self.retriever = LawGenRetriever()

        print("Loading Gemma...")
        self.llm = GemmaClient()


    def ask(self, question):
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

        return answer
