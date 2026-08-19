def reformulate_legal_query(question: str) -> str:
    """
    Reformulate a user's question into a clearer legal retrieval query.

    This function is used only for retrieval.
    The final answer should still be based on the original user question.

    Version 1 is rule-based and safe:
    - no extra API call
    - no hallucination risk
    - deterministic behavior
    """

    original_question = question.strip()
    normalized = original_question.lower()

    legal_hints = []

    # Constitution-related concepts
    if "دين الدولة" in normalized or "الإسلام" in normalized:
        legal_hints.append(
            "النص الدستوري المتعلق بالإسلام باعتباره دين الدولة في الدستور الجزائري"
        )

    if "تمازيغت" in normalized or "الأمازيغية" in normalized:
        legal_hints.append(
            "الوضع الدستوري لتمازيغت كلغة وطنية ورسمية في الدستور الجزائري"
        )

    if "اللغة" in normalized and ("الرسمية" in normalized or "الوطنية" in normalized):
        legal_hints.append(
            "اللغة الوطنية والرسمية في الدستور الجزائري"
        )

    if "الجنسية" in normalized:
        legal_hints.append(
            "الأحكام الدستورية المتعلقة بالجنسية الجزائرية"
        )

    if "الحرب" in normalized:
        legal_hints.append(
            "موقف الدستور الجزائري من اللجوء إلى الحرب وتسوية الخلافات الدولية"
        )

    if "التجارة الخارجية" in normalized:
        legal_hints.append(
            "تنظيم التجارة الخارجية في الدستور الجزائري"
        )

    if "التعاون الدولي" in normalized or "العلاقات الودية" in normalized:
        legal_hints.append(
            "الأحكام الدستورية المتعلقة بالتعاون الدولي والعلاقات الودية بين الدول"
        )

    # Finance Law concepts
    if "قانون المالية" in normalized or "الميزانية" in normalized:
        legal_hints.append(
            "أحكام قانون المالية لسنة 2026"
        )

    if (
        "الإيرادات" in normalized
        or "ايرادات" in normalized
        or "الحواصل" in normalized
        or "المداخيل" in normalized
        or ("قداش" in normalized and "قانون المالية" in normalized)
    ):
        legal_hints.append(
            "قيمة الإيرادات والحواصل والمداخيل المقدرة في المادة 2 من قانون المالية لسنة 2026"
        )

    if "الضمان الاجتماعي" in normalized or "هيئات الضمان الاجتماعي" in normalized:
        legal_hints.append(
            "مساهمة هيئات الضمان الاجتماعي في قانون المالية لسنة 2026"
        )

    if "رخص الالتزام" in normalized:
        legal_hints.append(
            "رخص الالتزام في قانون المالية لسنة 2026"
        )

    # Official Journal / decree concepts
    if "مرسوم" in normalized or "المرسوم" in normalized:
        legal_hints.append(
            "موضوع المرسوم في الجريدة الرسمية الجزائرية"
        )

    if "الجريدة الرسمية" in normalized:
        legal_hints.append(
            "النصوص المنشورة في الجريدة الرسمية الجزائرية"
        )

    # If no legal hint was detected, return the original question.
    if not legal_hints:
        return original_question

    # Add legal hints without replacing the user's original wording.
    # This improves retrieval while keeping the query understandable.
    unique_hints = []
    for hint in legal_hints:
        if hint not in unique_hints:
            unique_hints.append(hint)

    return original_question + "\n\nصياغة قانونية للبحث:\n" + "\n".join(
        f"- {hint}" for hint in unique_hints
    )


if __name__ == "__main__":
    test_questions = [
        "واش يقول القانون على دين الدولة؟",
        "ما هو وضع تمازيغت في الدستور الجزائري؟",
        "ما قيمة الإيرادات في قانون المالية 2026؟",
        "ما موضوع المرسوم الرئاسي رقم 26-239؟",
    ]

    for question in test_questions:
        print("=" * 80)
        print("Original:")
        print(question)
        print("\nReformulated:")
        print(reformulate_legal_query(question))
