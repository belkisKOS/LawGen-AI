from retrieval.hybrid_retriever import HybridRetriever


def print_result(index, chunk, score):
    text = chunk.get("text", "").replace("\n", " ")
    print(f"\n[{index}] Score: {round(score, 2)}")
    print(f"Source: {chunk.get('source_file')}")
    print(f"Type: {chunk.get('chunk_type')}")
    print(f"Title: {chunk.get('title')}")
    print(f"Pages: {chunk.get('page_start')} - {chunk.get('page_end')}")
    print(f"Text: {text[:700]}")


def main():
    print("Loading LawGen hybrid retriever...")
    retriever = HybridRetriever()
    print("✅ Ready. Ask a legal question. Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        if not question:
            continue

        results = retriever.search(question, k=5)

        print("\nTop retrieved legal chunks:")
        print("=" * 100)

        for i, (chunk, score) in enumerate(results, start=1):
            print_result(i, chunk, score)

        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
