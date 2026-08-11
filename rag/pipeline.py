from retrieval.retriever import LawGenRetriever
from reranking.context_enhancer import enhance_context
from rag.context_builder import build_context, build_legal_prompt
from llm.gemma_client import GemmaClient


class LawGenPipeline:


    def __init__(self):

        print("Loading retriever...")
        self.retriever = LawGenRetriever()

        print("Loading Gemma...")
        self.llm = GemmaClient()



    def ask(self, question):

                # 1. Retrieve legal chunks
        results = self.retriever.hybrid_search(
            question,
            k=10
        )


   


        # 2. Improve retrieved context
        enhanced = enhance_context(
            question,
            results
        )


        # 3. Build legal context
        context = build_context(
            enhanced
        )


        # 4. Build prompt
        prompt = build_legal_prompt(
            question,
            context
        )


        # 5. Generate answer
        answer = self.llm.generate(
            prompt
        )


        return answer
