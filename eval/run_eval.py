import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from rag.pipeline import LawGenPipeline


QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"


def main():
    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    bot = LawGenPipeline()

    passed = 0
    total = len(questions)

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_articles = item.get("expected_articles", [])

        print("=" * 80)
        print(f"TEST {index}/{total}")
        print("QUESTION:", question)

        results = bot.retriever.hybrid_search(
            question,
            k=5
        )

        retrieved_titles = [
            result.get("title", "")
            for result in results
        ]

        print("RETRIEVED:", retrieved_titles)

        retrieval_ok = all(
            expected in retrieved_titles
            for expected in expected_articles
        )

        answer = bot.ask(question)

        print("ANSWER:")
        print(answer)

        if retrieval_ok:
            passed += 1
            print("RESULT: PASS ✅")
        else:
            print("RESULT: FAIL ❌")
            print("EXPECTED:", expected_articles)

    print("=" * 80)
    print(f"EVALUATION SCORE: {passed}/{total}")


if __name__ == "__main__":
    main()
