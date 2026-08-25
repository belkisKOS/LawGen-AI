from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = Path("data/chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Missing Chroma database: {CHROMA_DIR}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="lawgen_v3",
    )

    questions = [
        "ما هو دين الدولة حسب الدستور الجزائري؟",
        "ما هي اللغة الوطنية والرسمية في الجزائر حسب الدستور؟",
        "ما قيمة الإيرادات المقدرة في قانون المالية لسنة 2026؟",
        "ما هو المرسوم الرئاسي رقم 26-239؟",
    ]

    for question in questions:
        print("\n" + "=" * 100)
        print("QUESTION:", question)
        print("=" * 100)

        results = vectorstore.similarity_search_with_score(question, k=4)

        for i, (doc, score) in enumerate(results, start=1):
            meta = doc.metadata
            text = doc.page_content.replace("\n", " ")

            print(f"\nRESULT {i}")
            print("-" * 80)
            print("Score:", score)
            print("Source:", meta.get("source_file"))
            print("Type:", meta.get("chunk_type"))
            print("Title:", meta.get("title"))
            print("Pages:", meta.get("page_start"), "-", meta.get("page_end"))
            print("Text:", text[:600])


if __name__ == "__main__":
    main()
