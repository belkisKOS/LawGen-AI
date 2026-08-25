from pathlib import Path
import json
import re
import unicodedata
from typing import List, Dict, Optional, Tuple

PROCESSED_DIR = Path("data/processed")
CHUNKS_V3_DIR = Path("data/chunks_v3")

MAX_CHARS = 3500

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

ORDINALS = {
    "الأولى": 1,
    "الاولى": 1,
    "الثانية": 2,
    "الثالثة": 3,
    "الرابعة": 4,
    "الخامسة": 5,
    "السادسة": 6,
    "السابعة": 7,
    "الثامنة": 8,
    "التاسعة": 9,
    "العاشرة": 10,
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ـ", "")
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace("：", ":")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = "\n".join(remove_noise_lines(text.splitlines()))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def remove_noise_lines(lines: List[str]) -> List[str]:
    cleaned = []

    for line in lines:
        n = normalize_text(line)
        c = compact_text(n)

        if not n:
            continue

        if "الجريدة الرسمية" in n:
            continue

        if "الجريدةالرسمية" in c:
            continue

        if "للجمهوريةالجزائرية" in c:
            continue

        if re.fullmatch(r"[0-9٠-٩]{1,3}", c):
            continue

        if "رجبعام" in c or "صفرعام" in c or "جمادى" in c:
            continue

        if "ديسمبرسنة" in c or "يوليوسنة" in c:
            continue

        cleaned.append(n)

    return cleaned


def extract_pages(text: str) -> List[Dict]:
    parts = re.split(r"\n\s*--- PAGE\s+(\d+)\s+---\s*\n", text)
    pages = []

    if parts[0].strip():
        pages.append({"page": None, "text": clean_text(parts[0])})

    for i in range(1, len(parts), 2):
        page_number = int(parts[i])
        page_text = parts[i + 1] if i + 1 < len(parts) else ""
        pages.append({"page": page_number, "text": clean_text(page_text)})

    return pages


def to_int_number(value: str) -> Optional[int]:
    value = normalize_text(value).strip()

    if value in ORDINALS:
        return ORDINALS[value]

    digits = re.sub(r"[^0-9٠-٩]", "", value)
    if not digits:
        return None

    try:
        return int(digits.translate(ARABIC_DIGITS))
    except ValueError:
        return None


def page_range(lines_with_pages: List[Tuple[Optional[int], str]]) -> Tuple[Optional[int], Optional[int]]:
    pages = [p for p, _ in lines_with_pages if p is not None]
    if not pages:
        return None, None
    return min(pages), max(pages)


def make_chunk(
    chunk_id: str,
    source_file: str,
    document_type: str,
    chunk_type: str,
    title: str,
    lines_with_pages: List[Tuple[Optional[int], str]],
) -> Optional[Dict]:
    text = clean_text("\n".join(line for _, line in lines_with_pages))
    if not text:
        return None

    start, end = page_range(lines_with_pages)

    return {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "document_type": document_type,
        "chunk_type": chunk_type,
        "title": title,
        "page_start": start,
        "page_end": end,
        "text": text,
        "char_count": len(text),
    }


def same_line_article_candidate(line: str) -> Optional[Tuple[str, Optional[int]]]:
    n = normalize_text(line)

    if "من قانون" in n or "من القانون" in n:
        return None

    match = re.match(
        r"^(المادة|مادة)\s*([0-9٠-٩]{1,4}|الأولى|الاولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة)\b",
        n,
    )

    if not match:
        return None

    number = to_int_number(match.group(2))

    if ":" in n or len(n) <= 100:
        return n, number

    return None


def split_line_article_candidate(lines: List[str], index: int) -> Optional[Tuple[int, str, Optional[int]]]:
    current = compact_text(lines[index])

    if current not in {"المادة", "مادة"}:
        return None

    if index + 1 >= len(lines):
        return None

    next_line = normalize_text(lines[index + 1])
    number = to_int_number(next_line)

    if number is None:
        return None

    extra = 1

    if index + 2 < len(lines):
        third = normalize_text(lines[index + 2])
        if third == ":" or third.startswith(":"):
            extra = 2

    title = " ".join(normalize_text(lines[index + offset]) for offset in range(extra + 1))
    title = re.sub(r"\s+", " ", title).strip()

    return extra, title, number


def reversed_finance_article_candidate(line: str) -> Optional[Tuple[str, Optional[int]]]:
    n = normalize_text(line)

    match = re.search(r"[:،]\s*([0-9٠-٩]{1,4})\s*ةدالما\b", n)
    if not match:
        match = re.search(r"^\s*([0-9٠-٩]{1,4})\s*ةدالما\s*$", n)

    if not match:
        return None

    number = to_int_number(match.group(1))
    if number is None:
        return None

    title = f"المادة {match.group(1)}"
    return title, number


def is_finance_table_page(page_text: str) -> bool:
    n = normalize_text(page_text)
    c = compact_text(n)

    table_words = [
        "الجدول",
        "جدول",
        "لودجلا",
        "الإيرادات",
        "الايرادات",
        "الاعتمادات",
        "النصوصالتشريعيةوالتنظيمية",
        "الهيئاتالمستفيدة",
        "الوحدة:دج",
        "الوحدة:بالدينار",
    ]

    return any(word in n or word in c for word in table_words)


def split_long_text(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    text = clean_text(text)

    if len(text) <= max_chars:
        return [text]

    pieces = []
    start = 0
    overlap = 250

    while start < len(text):
        end = start + max_chars
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        start = end - overlap
        if start >= len(text):
            break

    return pieces


def make_page_chunks(
    pages: List[Dict],
    source_file: str,
    document_type: str,
    chunk_type: str,
    skip_pages_before: int = 0,
) -> List[Dict]:
    chunks = []
    index = 0

    for page_obj in pages:
        page = page_obj["page"]

        if page is not None and page < skip_pages_before:
            continue

        text = clean_text(page_obj["text"])
        if not text:
            continue

        for part_index, part in enumerate(split_long_text(text), start=1):
            index += 1
            chunks.append({
                "chunk_id": f"{Path(source_file).stem}_v3_chunk_{index}",
                "source_file": source_file,
                "document_type": document_type,
                "chunk_type": chunk_type,
                "title": f"page_{page}" if page is not None else "unknown_page",
                "page_start": page,
                "page_end": page,
                "text": part,
                "char_count": len(part),
                "page_part": part_index,
            })

    return chunks


def chunk_constitution(source_file: str, text: str) -> List[Dict]:
    pages = extract_pages(text)
    lines_with_pages = []

    for page_obj in pages:
        page = page_obj["page"]
        for line in page_obj["text"].splitlines():
            lines_with_pages.append((page, line))

    chunks = []
    current_title = "preamble_or_unknown"
    current_type = "preamble"
    current_lines = []
    index = 0

    i = 0
    lines = [line for _, line in lines_with_pages]

    while i < len(lines_with_pages):
        page, line = lines_with_pages[i]
        n = normalize_text(line)

        if not n:
            i += 1
            continue

        candidate = same_line_article_candidate(n)
        extra_candidate = None if candidate else split_line_article_candidate(lines, i)

        is_heading = candidate is not None or extra_candidate is not None

        if is_heading:
            if current_lines:
                index += 1
                chunk = make_chunk(
                    f"{Path(source_file).stem}_v3_chunk_{index}",
                    source_file,
                    "constitution",
                    current_type,
                    current_title,
                    current_lines,
                )
                if chunk:
                    chunks.append(chunk)

            if candidate:
                title, _ = candidate
                current_lines = [(page, n)]
                i += 1
            else:
                extra, title, _ = extra_candidate
                current_lines = []
                for offset in range(extra + 1):
                    p, l = lines_with_pages[i + offset]
                    current_lines.append((p, normalize_text(l)))
                i += extra + 1

            current_title = title
            current_type = "article"
            continue

        current_lines.append((page, n))
        i += 1

    if current_lines:
        index += 1
        chunk = make_chunk(
            f"{Path(source_file).stem}_v3_chunk_{index}",
            source_file,
            "constitution",
            current_type,
            current_title,
            current_lines,
        )
        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_finance_law(source_file: str, text: str) -> List[Dict]:
    pages = extract_pages(text)

    chunks = []
    current_title = "preamble_or_unknown"
    current_type = "preamble"
    current_lines = []
    last_article_number = 0
    index = 0

    for page_obj in pages:
        page = page_obj["page"]
        page_text = page_obj["text"]

        if page is None:
            continue

        if page == 1:
            index += 1
            for part in split_long_text(page_text):
                chunks.append({
                    "chunk_id": f"{Path(source_file).stem}_v3_chunk_{index}",
                    "source_file": source_file,
                    "document_type": "finance_law",
                    "chunk_type": "document_info",
                    "title": "document_info",
                    "page_start": page,
                    "page_end": page,
                    "text": part,
                    "char_count": len(part),
                })
            continue

        if is_finance_table_page(page_text):
            if current_lines:
                index += 1
                chunk = make_chunk(
                    f"{Path(source_file).stem}_v3_chunk_{index}",
                    source_file,
                    "finance_law",
                    current_type,
                    current_title,
                    current_lines,
                )
                if chunk:
                    chunks.append(chunk)
                current_lines = []
                current_title = "preamble_or_unknown"
                current_type = "preamble"

            for part_index, part in enumerate(split_long_text(page_text), start=1):
                index += 1
                chunks.append({
                    "chunk_id": f"{Path(source_file).stem}_v3_chunk_{index}",
                    "source_file": source_file,
                    "document_type": "finance_law",
                    "chunk_type": "table_or_page",
                    "title": f"table_or_page_{page}",
                    "page_start": page,
                    "page_end": page,
                    "text": part,
                    "char_count": len(part),
                    "page_part": part_index,
                })

            continue

        page_lines = [(page, line) for line in page_text.splitlines()]
        lines_only = [line for _, line in page_lines]

        i = 0
        while i < len(page_lines):
            p, line = page_lines[i]
            n = normalize_text(line)

            if not n:
                i += 1
                continue

            heading_title = None
            heading_number = None
            heading_extra = 0

            candidate = same_line_article_candidate(n)
            split_candidate = None if candidate else split_line_article_candidate(lines_only, i)
            reversed_candidate = None if candidate or split_candidate else reversed_finance_article_candidate(n)

            if candidate:
                heading_title, heading_number = candidate
            elif split_candidate:
                heading_extra, heading_title, heading_number = split_candidate
            elif reversed_candidate:
                heading_title, heading_number = reversed_candidate

            valid_heading = False

            if heading_number is not None:
                if 1 <= heading_number <= 250 and heading_number > last_article_number:
                    valid_heading = True

            if valid_heading:
                if current_lines:
                    index += 1
                    chunk = make_chunk(
                        f"{Path(source_file).stem}_v3_chunk_{index}",
                        source_file,
                        "finance_law",
                        current_type,
                        current_title,
                        current_lines,
                    )
                    if chunk:
                        chunks.append(chunk)

                current_title = heading_title
                current_type = "article"
                current_lines = []

                for offset in range(heading_extra + 1):
                    if i + offset < len(page_lines):
                        pp, ll = page_lines[i + offset]
                        current_lines.append((pp, normalize_text(ll)))

                last_article_number = heading_number
                i += heading_extra + 1
                continue

            current_lines.append((p, n))
            i += 1

    if current_lines:
        index += 1
        chunk = make_chunk(
            f"{Path(source_file).stem}_v3_chunk_{index}",
            source_file,
            "finance_law",
            current_type,
            current_title,
            current_lines,
        )
        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_journal_officiel(source_file: str, text: str) -> List[Dict]:
    pages = extract_pages(text)

    return make_page_chunks(
        pages=pages,
        source_file=source_file,
        document_type="journal_officiel",
        chunk_type="legal_page",
        skip_pages_before=4,
    )


def choose_chunker(txt_file: Path):
    name = txt_file.name.lower()

    if "constitution" in name:
        return chunk_constitution

    if "loifinance" in name or "finance" in name:
        return chunk_finance_law

    if "journaloff" in name or "journal" in name:
        return chunk_journal_officiel

    return chunk_journal_officiel


def chunk_all_documents_v3() -> None:
    CHUNKS_V3_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(PROCESSED_DIR.glob("*_extracted.txt"))

    if not files:
        print(f"❌ No extracted files found in {PROCESSED_DIR}")
        return

    print(f"Found {len(files)} extracted file(s).")

    for txt_file in files:
        text = txt_file.read_text(encoding="utf-8")
        chunker = choose_chunker(txt_file)

        print(f"\nProcessing: {txt_file.name}")
        print(f"Chunker: {chunker.__name__}")

        chunks = chunker(txt_file.name, text)

        output_file = CHUNKS_V3_DIR / f"{txt_file.stem}_chunks_v3.json"
        output_file.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        types = sorted(set(c["chunk_type"] for c in chunks))
        print(f"Chunks created: {len(chunks)}")
        for t in types:
            print(f"{t}: {sum(1 for c in chunks if c['chunk_type'] == t)}")
        print(f"Output: {output_file}")

    print("\n✅ V3 chunking completed.")


if __name__ == "__main__":
    chunk_all_documents_v3()
