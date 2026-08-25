import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from rag.pipeline import LawGenPipeline


QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"


def normalize_digits(text):
    text = text or ""
    return text.translate(
        str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    )


def normalize_ref(text):
    text = normalize_digits(text)
    text = text.lower()
    text = text.replace("ـ", "")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_article_numbers(text):
    text = normalize_ref(text)
    return set(
        re.findall(r"المادة\s*([0-9]+)", text)
    )


def reference_found(expected_reference, expected_document, results):
    expected_numbers = extract_article_numbers(expected_reference)
    expected_ref = normalize_ref(expected_reference)

    for result in results:
        source_file = result.get("source_file", "")

        if expected_document and source_file != expected_document:
            continue

        haystack = " ".join(
            [
                result.get("title", ""),
                result.get("retrieval_label", ""),
                result.get("text", "")[:500],
            ]
        )

        haystack_ref = normalize_ref(haystack)
        haystack_numbers = extract_article_numbers(haystack)

        if expected_numbers:
            if expected_numbers.intersection(haystack_numbers):
                return True

        elif expected_ref in haystack_ref:
            return True

    return False


def main():
    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    bot = LawGenPipeline()

    passed = 0
    total = len(questions)

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_document = item.get("expected_document", "")
        expected_references = item.get("expected_articles", [])

        print("=" * 80)
        print(f"TEST {index}/{total}")
        print("QUESTION:", question)

        results = bot.retriever.hybrid_search(
            question,
            k=5
        )

        retrieved = [
            f"{result.get('title', '')} [{result.get('source_file', '')}]"
            for result in results
        ]

        print("RETRIEVED:", retrieved)

        retrieval_ok = all(
            reference_found(
                expected,
                expected_document,
                results,
            )
            for expected in expected_references
        )

        answer = bot.ask(question)

        print("ANSWER:")
        print(answer)

        if retrieval_ok:
            passed += 1
            print("RESULT: PASS ✅")
        else:
            print("RESULT: FAIL ❌")
            print("EXPECTED DOCUMENT:", expected_document)
            print("EXPECTED REFERENCES:", expected_references)

    print("=" * 80)
    print(f"EVALUATION SCORE: {passed}/{total}")


if __name__ == "__main__":
    main()
