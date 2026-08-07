from typing import Dict, List, Tuple


def clean_text(text: str, max_chars: int = 1800) -> str:
    text = (text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n...[تم اختصار النص]"


def filter_results_by_score(
    results: List[Tuple[Dict, float]],
    max_sources: int = 3,
    min_ratio: float = 0.50,
) -> List[Tuple[Dict, float]]:
    """
    Keep the strongest sources only.

    Example:
    Top score = 196
    Source 2 score = 55
    55 / 196 = 0.28, so we remove it.

    This avoids confusing the LLM with similar but wrong legal references.
    """
    if not results:
        return []

    top_score = results[0][1]

    selected = []
    for chunk, score in results:
        if len(selected) >= max_sources:
            break

        if score == top_score or score >= top_score * min_ratio:
            selected.append((chunk, score))

    return selected


def format_source(index: int, chunk: Dict, score: float, max_source_chars: int = 1800) -> str:
    title = chunk.get("title", "بدون عنوان")
    source_file = chunk.get("source_file", "مصدر غير معروف")
    chunk_type = chunk.get("chunk_type", "unknown")
    page_start = chunk.get("page_start", "?")
    page_end = chunk.get("page_end", "?")
    text = clean_text(chunk.get("text", ""), max_chars=max_source_chars)

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
    results: List[Tuple[Dict, float]],
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

    for index, (chunk, score) in enumerate(filtered_results, start=1):
        source_text = format_source(index, chunk, score, max_source_chars=max_source_chars)

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

اكتب الجواب الآن:
""".strip()
