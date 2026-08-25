import json
from pathlib import Path


CHUNKS_PATH = Path("data/final_chunks_v3/final_chunks.json")


def fix_constitution_article_223(chunks):
    fixed = 0

    for chunk in chunks:
        text = chunk.get("text", "")

        if (
            chunk.get("source_file") == "Constitution_extracted.txt"
            and "لا يمكن أي تعديل دستوري أن يمس" in text
            and "الإسلام باعتباره دين الدولة" in text
        ):
            chunk["title"] = "المادة 223"
            chunk["text"] = text.replace("المادة ٣2", "المادة 223", 1)
            chunk["retrieval_label"] = (
                f"{chunk.get('source_file')} | المادة 223 | pages "
                f"{chunk.get('page_start')}-{chunk.get('page_end')}"
            )
            fixed += 1

    return fixed


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PATH}")

    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    fixed = fix_constitution_article_223(chunks)

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Metadata fixes applied: {fixed}")


if __name__ == "__main__":
    main()
