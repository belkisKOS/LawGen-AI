from typing import Dict, List


def clean_text(text: str, max_chars: int = 1800) -> str:
    text = (text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n...[تم اختصار النص]"



def get_score(result: Dict) -> float:
    return float(
        result.get("hybrid_score")
        or result.get("enhancement_score")
        or result.get("lexical_score")
        or result.get("semantic_score")
        or result.get("score")
        or 0
    )



def filter_results_by_score(
    results: List[Dict],
    max_sources: int = 3,
    min_ratio: float = 0.50,
) -> List[Dict]:

    if not results:
        return []

    top_score = get_score(results[0])

    selected = []

    for result in results:

        if len(selected) >= max_sources:
            break

        score = get_score(result)

        if score == top_score or score >= top_score * min_ratio:
            selected.append(result)

    return selected



def format_source(
    index: int,
    result: Dict,
    max_source_chars: int = 1800,
) -> str:

    title = result.get("title", "بدون عنوان")
    source_file = result.get("source_file", "مصدر غير معروف")
    chunk_type = result.get("chunk_type", "unknown")
    page_start = result.get("page_start", "?")
    page_end = result.get("page_end", "?")
    score = get_score(result)

    text = clean_text(
        result.get("text", ""),
        max_chars=max_source_chars,
    )

    return f"""
[Source {index}]
Score: {round(score, 2)}
Source file: {source_file}
Type: {chunk_type}
Title: {title}
Pages: {page_start}-{page_end}
Text:
{text}
""".strip()



def build_context(
    results: List[Dict],
    max_sources: int = 3,
    max_chars: int = 5000,
    max_source_chars: int = 1800,
) -> str:

    filtered_results = filter_results_by_score(
        results,
        max_sources=max_sources,
        min_ratio=0.50,
    )

    parts = []
    total_chars = 0

    for index, result in enumerate(filtered_results, start=1):

        source_text = format_source(
            index,
            result,
            max_source_chars=max_source_chars,
        )

        if total_chars + len(source_text) > max_chars:
            break

        parts.append(source_text)
        total_chars += len(source_text)

    return "\n\n---\n\n".join(parts)



def build_legal_prompt(question: str, context: str) -> str:

    return f"""
أنت مساعد قانوني ذكي متخصص في القانون الجزائري.

مهمتك:
- أجب فقط اعتمادا على المصادر القانونية المقدمة في السياق.
- لا تضف معلومات من خارج السياق.
- إذا لم يكن السياق كافيا، قل: لا أتوفر على معلومات كافية من المصادر المتاحة للإجابة بثقة.
- اذكر المرجع القانوني بوضوح: اسم الوثيقة، المادة أو عنوان النص، والصفحة إن وجدت.
- اجعل الجواب واضحا ومختصرا.
- لا تقدم استشارة قانونية نهائية، بل مساعدة معلوماتية مبنية على النصوص المتاحة.

السؤال:
{question}

السياق القانوني المسترجع:
{context}

اكتب الجواب الآن وفق الشكل التالي:

الإجابة:
...

المصدر:
- الوثيقة:
- المادة أو النص:
- الصفحة:
""".strip()
