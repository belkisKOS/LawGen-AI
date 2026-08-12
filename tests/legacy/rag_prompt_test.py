from retrieval.hybrid_retriever import HybridRetriever
from rag.context_builder import build_context, build_legal_prompt


def main():
    print("Loading LawGen retriever...")
    retriever = HybridRetriever()
    print("✅ Ready. Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        results = retriever.search(question, k=5)
        context = build_context(results)
        prompt = build_legal_prompt(question, context)

        print("\n" + "=" * 100)
        print("PROMPT THAT WILL BE SENT TO THE LLM")
        print("=" * 100)
        print(prompt)
        print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
