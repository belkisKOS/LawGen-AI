import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from rag.pipeline import LawGenPipeline
from rag.query_reformulator import reformulate_legal_query


test_questions = [
    "واش يقول القانون على دين الدولة؟",
    "هل يقدرو يبدلو دين الدولة في الدستور؟",
    "شنو وضع تمازيغت في الدستور؟",
    "قداش الإيرادات في قانون المالية 2026؟",
    "وش موضوع المرسوم 26-239؟",
]


def main():
    bot = LawGenPipeline()

    for question in test_questions:
        print("=" * 80)
        print("ORIGINAL QUESTION:")
        print(question)

        reformulated = reformulate_legal_query(question)
        print("\nREFORMULATED QUESTION:")
        print(reformulated)

        response = bot.ask_with_sources(question)

        print("\nANSWER:")
        print(response["answer"])

        print("\nTOP SOURCES:")
        for source in response["sources"][:3]:
            print(
                "-",
                source["source_file"],
                "|",
                source["title"],
                "| pages:",
                source["page_start"],
                "-",
                source["page_end"],
            )


if __name__ == "__main__":
    main()
