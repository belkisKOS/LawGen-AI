def rule_based_reformulation(question: str) -> str:
    """
    Deterministic legal hints.

    This is a safety layer. It helps retrieval with known legal concepts.
    It does not use an LLM.
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

    if not legal_hints:
        return ""

    unique_hints = []

    for hint in legal_hints:
        if hint not in unique_hints:
            unique_hints.append(hint)

    return "\n".join(
        f"- {hint}" for hint in unique_hints
    )


def build_ai_reformulation_prompt(
    question: str,
    rule_hints: str,
    history=None
) -> str:
    """
    Build a strict prompt for AI legal query reformulation.

    The model must not answer the legal question.
    It only rewrites the question into a better retrieval query.
    """
    conversation = ""

    if history:
        conversation = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in history
            ]
        )

    return f"""
أنت محرك لإعادة صياغة الأسئلة القانونية داخل نظام بحث قانوني جزائري.

مهمتك:
- لا تجب عن السؤال.
- لا تقدم رأيا قانونيا.
- أعد صياغة السؤال فقط إلى سؤال قانوني واضح ودقيق يساعد على البحث في النصوص القانونية الجزائرية.
- حافظ على كل أرقام المواد، أرقام المراسيم، أرقام القوانين، السنوات، التواريخ، والمبالغ كما هي.
- إذا كان السؤال باللهجة أو بالفرنسية، حوّله إلى عربية قانونية واضحة.
- إذا كان السؤال يتحدث عن الدستور، اذكر "الدستور الجزائري".
- إذا كان السؤال يتحدث عن قانون المالية، اذكر "قانون المالية لسنة 2026".
- إذا كان السؤال يتحدث عن مرسوم، اذكر "الجريدة الرسمية الجزائرية" و"المرسوم".
- أرجع سطرا واحدا فقط.
- لا تضع شرحا.
- لا تضع مقدمة.
- لا تضع نقاطا.

المحادثة السابقة:
{conversation if conversation else "لا توجد محادثة سابقة"}

السؤال الحالي:
{question}

مؤشرات قانونية مساعدة:
{rule_hints if rule_hints else "لا توجد مؤشرات محددة"}

الصياغة القانونية للبحث:
""".strip()


def clean_ai_reformulation(text: str) -> str:
    """
    Clean the AI output so we keep only the reformulated query.
    """

    text = (text or "").strip()

    prefixes = [
        "الصياغة القانونية للبحث:",
        "السؤال القانوني:",
        "إعادة الصياغة:",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Keep only first non-empty line if model returns multiple lines.
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    return lines[0]


def ai_reformulation(
    question: str,
    llm_client,
    rule_hints: str,
    history=None
) -> str:
    """
    Use the LLM to reformulate the question.

    If the LLM fails, return an empty string.
    """

    if llm_client is None:
        return ""

    prompt = build_ai_reformulation_prompt(
         question=question,
         rule_hints=rule_hints,
         history=history,
    )

    try:
        output = llm_client.generate(prompt)
        return clean_ai_reformulation(output)

    except Exception as error:
        print("Query reformulation failed:", error)
        return ""


def reformulate_legal_query(
    question: str,
    llm_client=None,
    history=None
) -> str:
    """
    Final reformulation function.

    It combines:
    - original user question
    - AI reformulation if available
    - rule-based legal hints

    This combined query is used for retrieval only.
    """

    original_question = question.strip()

    rule_hints = rule_based_reformulation(original_question)

    ai_query = ai_reformulation(
        question=original_question,
        llm_client=llm_client,
        rule_hints=rule_hints,
        history=history,
    )

    parts = [original_question]

    if ai_query and ai_query not in parts:
        parts.append("صياغة قانونية بالذكاء الاصطناعي:")
        parts.append(ai_query)

    if rule_hints:
        parts.append("مؤشرات قانونية:")
        parts.append(rule_hints)

    return "\n\n".join(parts)


if __name__ == "__main__":
    test_questions = [
        "واش يقول القانون على دين الدولة؟",
        "هل يقدرو يبدلو دين الدولة في الدستور؟",
        "شنو وضع تمازيغت في الدستور؟",
        "قداش الإيرادات في قانون المالية 2026؟",
        "c'est quoi le décret 26-239 ?",
    ]

    for question in test_questions:
        print("=" * 80)
        print("Original:")
        print(question)
        print("\nRule-based only:")
        print(reformulate_legal_query(question))
