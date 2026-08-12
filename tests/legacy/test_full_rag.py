import json
from pathlib import Path
from dotenv import load_dotenv

from reranking.context_enhancer import enhance_context
from rag.prompt_builder import build_prompt
from llm.gemma_client import GemmaClient


load_dotenv()


question = "ما هو دين الدولة حسب الدستور الجزائري؟"


# Load chunks
from retrieval.retriever import LawGenRetriever


retriever = LawGenRetriever()


retrieved = retriever.hybrid_search(
    question,
    k=10
)


# Context enhancement

enhanced = enhance_context(
    question,
    retrieved
)



# Build Gemma prompt

prompt = build_prompt(
    question,
    enhanced
)



# Call Gemma

gemma = GemmaClient()

answer = gemma.generate(
    prompt
)


print("=" * 80)
print("FINAL ANSWER")
print("=" * 80)

print(answer)
