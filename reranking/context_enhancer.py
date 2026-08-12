import re


def normalize_text(text):
    text = text.lower()
    text = text.replace("ـ", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def lexical_score(question, chunk):

    question = normalize_text(question)

    title = normalize_text(
        chunk.get("title", "")
    )

    text = normalize_text(
        chunk.get("text", "")
    )

    score = 0


    # Exact legal concepts

    if "دين الدولة" in question:

        if "دين الدولة" in text:
            score += 100
        else:
            score -= 50


    if "اللغة الوطنية والرسمية" in question:

        if "اللغة الوطنية والرسمية" in text:
            score += 100
        else:
            score -= 30


    # General matching

    for word in question.split():

        if len(word) > 2:

            if word in title:
                score += 10

            if word in text:
                score += 3


    return score



def enhance_context(question, retrieved_chunks):

    enhanced = []


    for chunk in retrieved_chunks:

        semantic_score = chunk.get(
            "score",
            0
        )


        keyword_score = lexical_score(
            question,
            chunk
        )


        final_score = (
            semantic_score
            +
            keyword_score
        )


        new_chunk = chunk.copy()

        new_chunk["enhancement_score"] = final_score

        enhanced.append(new_chunk)



    enhanced.sort(
        key=lambda x: x["enhancement_score"],
        reverse=True
    )


    if not enhanced:
        return []


    best_score = enhanced[0]["enhancement_score"]


    selected = []

    for chunk in enhanced:

        if chunk["enhancement_score"] >= best_score * 0.55:
            selected.append(chunk)


    return selected



def print_context(chunks):

    print("\n" + "=" * 80)
    print("ENHANCED CONTEXT")
    print("=" * 80)


    for i, chunk in enumerate(chunks, 1):

        print(f"\n[{i}]")

        print(
            "Source:",
            chunk.get("source_file")
        )

        print(
            "Reference:",
            f"{chunk.get('source_file')} | "
            f"{chunk.get('title')} | "
            f"Pages {chunk.get('page_start')}-{chunk.get('page_end')}"
        )

        print(
            "Score:",
            chunk.get("enhancement_score")
        )

        print(
            "Text:"
        )

        print(
            chunk.get("text", "")[:700]
        )
