from retrieval.retriever import LawGenRetriever, RetrieverInitError
from rag.context_builder import build_context, build_legal_prompt
from rag.query_reformulator import reformulate_legal_query
from llm.gemma_client import GemmaClient, GemmaClientError
from core.logger import get_logger

logger = get_logger(__name__)


class PipelineError(Exception):
    """Raised when answering a question fails for a known, identifiable reason."""


class LawGenPipeline:

    def __init__(self):
        self.ready = False
        self.init_error = None
        self.retriever = None
        self.llm = None

        try:
            logger.info("Loading retriever...")
            self.retriever = LawGenRetriever()
        except RetrieverInitError as error:
            self.init_error = f"Retriever failed to load: {error}"
            logger.error(self.init_error)
            return
        except Exception as error:
            self.init_error = f"Unexpected error loading retriever: {error}"
            logger.exception(self.init_error)
            return

        try:
            logger.info("Loading Gemma...")
            self.llm = GemmaClient()
        except GemmaClientError as error:
            self.init_error = f"LLM client failed to load: {error}"
            logger.error(self.init_error)
            return
        except Exception as error:
            self.init_error = f"Unexpected error loading LLM client: {error}"
            logger.exception(self.init_error)
            return

        self.ready = True
        logger.info("LawGenPipeline ready.")


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


    def ask_with_sources(self, question, history=None):
        if not self.ready:
            raise PipelineError(
                f"Pipeline is not ready: {self.init_error or 'unknown initialization error'}"
            )

        try:
            retrieval_question = reformulate_legal_query(
                question,
                llm_client=self.llm,
                history=history,
            )
        except Exception as error:
            logger.exception("Query reformulation failed for question: %s", question)
            raise PipelineError(f"Query reformulation failed: {error}") from error

        try:
            results = self.retriever.hybrid_search(
                retrieval_question,
                k=5
            )
        except Exception as error:
            logger.exception("Retrieval failed for question: %s", retrieval_question)
            raise PipelineError(f"Document retrieval failed: {error}") from error

        context = build_context(
            results,
            max_sources=5
        )

        prompt = build_legal_prompt(
            question,
            context,
            history=history
        )

        try:
            answer = self.llm.generate(prompt)
        except GemmaClientError as error:
            logger.exception("LLM generation failed for question: %s", question)
            raise PipelineError(f"Answer generation failed: {error}") from error

        return {
            "question": question,
            "reformulated_question": retrieval_question,
            "answer": answer,
            "sources": self.format_sources(self.select_main_sources(results)),
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