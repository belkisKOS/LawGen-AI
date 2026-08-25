import json
from pathlib import Path

from reranking.context_enhancer import (
    enhance_context,
    print_context
)


# Test question
question = "ما هو دين الدولة حسب الدستور الجزائري؟"


# Load final chunks
path = Path(
    "data/final_chunks_v3/final_chunks.json"
)

chunks = json.loads(
    path.read_text(encoding="utf-8")
)


# Simple retrieval simulation
# We take chunks containing relevant words
retrieved_chunks = []

for chunk in chunks:

    text = chunk.get("text", "")

    if (
        "دين الدولة" in text
        or "الإسلام" in text
    ):
        chunk["score"] = 50
        retrieved_chunks.append(chunk)


# Enhance context
enhanced = enhance_context(
    question,
    retrieved_chunks
)


# Show result
print_context(enhanced)
