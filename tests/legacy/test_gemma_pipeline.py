import json
from pathlib import Path

from reranking.context_enhancer import enhance_context
from rag.prompt_builder import build_prompt
from llm.gemma_client import GemmaClient


question = "ما هو دين الدولة حسب الدستور الجزائري?"


chunks = json.loads(
    Path(
        "data/final_chunks_v3/final_chunks.json"
    ).read_text(
        encoding="utf-8"
    )
)


# temporary retrieval simulation
retrieved = []

for chunk in chunks:

    if "دين الدولة" in chunk.get("text",""):

        chunk["score"] = 50

        retrieved.append(chunk)



enhanced = enhance_context(
    question,
    retrieved
)


prompt = build_prompt(
    question,
    enhanced
)


print("="*80)
print("PROMPT SENT TO GEMMA")
print("="*80)

print(prompt)


# Uncomment after API is confirmed
#
# gemma = GemmaClient()
# answer = gemma.generate(prompt)
#
# print(answer)
