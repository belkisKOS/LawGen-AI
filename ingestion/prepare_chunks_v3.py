from pathlib import Path
import json
import re
from typing import List, Dict

CHUNKS_V3_DIR = Path("data/chunks_v3")
FINAL_V3_DIR = Path("data/final_chunks_v3")
FINAL_OUTPUT = FINAL_V3_DIR / "final_chunks.json"

MAX_CHARS = 3000
OVERLAP_CHARS = 250


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_text(text: str, max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS) -> List[str]:
    text = clean_text(text)

    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)

    parts = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) > max_chars:
            if current.strip():
                parts.append(current.strip())
                current = ""

            start = 0
            while start < len(paragraph):
                end = start + max_chars
                piece = paragraph[start:end].strip()
                if piece:
                    parts.append(piece)

                start = end - overlap_chars
                if start >= len(paragraph):
                    break

            continue

        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current.strip():
                parts.append(current.strip())

            overlap = parts[-1][-overlap_chars:].strip() if parts else ""
            current = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph

    if current.strip():
        parts.append(current.strip())

    return parts


def prepare_final_chunks_v3() -> None:
    FINAL_V3_DIR.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(CHUNKS_V3_DIR.glob("*_chunks_v3.json"))

    if not chunk_files:
        print(f"❌ No V3 chunk files found in {CHUNKS_V3_DIR}")
        return

    final_chunks: List[Dict] = []
    global_index = 0

    print(f"Found {len(chunk_files)} V3 chunk file(s).")

    for chunk_file in chunk_files:
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))

        print(f"\nProcessing: {chunk_file.name}")
        print(f"Input chunks: {len(chunks)}")

        produced = 0

        for chunk in chunks:
            text = clean_text(chunk.get("text", ""))

            if not text:
                continue

            subchunks = split_text(text)

            for sub_index, sub_text in enumerate(subchunks, start=1):
                global_index += 1
                produced += 1

                final_chunks.append({
                    "id": f"final_v3_chunk_{global_index}",
                    "source_file": chunk.get("source_file"),
                    "document_type": chunk.get("document_type"),
                    "chunk_type": chunk.get("chunk_type"),
                    "parent_chunk_id": chunk.get("chunk_id"),
                    "title": chunk.get("title"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "subchunk_index": sub_index,
                    "subchunk_count": len(subchunks),
                    "text": sub_text,
                    "char_count": len(sub_text),
                    "retrieval_label": (
                        f"{chunk.get('source_file')} | "
                        f"{chunk.get('title')} | "
                        f"pages {chunk.get('page_start')}-{chunk.get('page_end')}"
                    ),
                })

        print(f"Final chunks produced: {produced}")

    FINAL_OUTPUT.write_text(
        json.dumps(final_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n✅ Final V3 chunk preparation completed.")
    print(f"Total final chunks: {len(final_chunks)}")
    print(f"Output: {FINAL_OUTPUT}")

    if final_chunks:
        sizes = [chunk["char_count"] for chunk in final_chunks]
        print("\nSize stats:")
        print(f"Smallest: {min(sizes)} chars")
        print(f"Largest: {max(sizes)} chars")
        print(f"Average: {sum(sizes) / len(sizes):.0f} chars")


if __name__ == "__main__":
    prepare_final_chunks_v3()
