from pathlib import Path
import json
import re
import unicodedata
from typing import List, Dict, Optional, Tuple

INPUT_FILE = Path("data/processed/LoiFinance2026_extracted.txt")
OUTPUT_DIR = Path("data/chunks_v3")
OUTPUT_FILE = OUTPUT_DIR / "LoiFinance2026_extracted_chunks_v3.json"

MAX_TABLE_CHARS = 3500

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
    text = text.replace("﹕", ":")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def remove_header_lines(lines: List[str]) -> List[str]:
    cleaned = []

    for line in lines:
        n = normalize_text(line)
        c = compact_text(n)

        if not n:
            continue

        if "الجريدة الرسمية" in n or "الجريدةالرسمية" in c:
            continue

        if "للجمهوريةالجزائرية" in c:
            continue

        if "رجبعام" in c or "صفرعام" in c or "جمادى" in c:
            continue

        if "ديسمبرسنة" in c or "يوليوسنة" in c:
            continue

        cleaned.append(n)

    return cleaned


def clean_output_text(text: str) -> str:
    lines = remove_header_lines(text.splitlines())
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_pages(text: str) -> List[Dict]:
    parts = re.split(r"\n\s*--- PAGE\s+(\d+)\s+---\s*\n", text)
    pages = []

    for i in range(1, len(parts), 2):
        page_number = int(parts[i])
        page_text = parts[i + 1] if i + 1 < len(parts) else ""
        pages.append({
            "page": page_number,
            "text": unicodedata.normalize("NFKC", page_text),
        })

    return pages


def digits_to_int(value: str) -> Optional[int]:
    digits = re.sub(r"[^0-9٠-٩]", "", value)
    if not digits:
        return None

    try:
        return int(digits.translate(ARABIC_DIGITS))
    except ValueError:
        return None


def digits_to_int_reversed(value: str) -> Optional[int]:
    digits = re.sub(r"[^0-9٠-٩]", "", value)
    if not digits:
        return None

    try:
        return int(digits.translate(ARABIC_DIGITS)[::-1])
    except ValueError:
        return None


def number_variants(value: str, allow_reversed: bool = True) -> List[int]:
    value = normalize_text(value)

    if value in ORDINALS:
        return [ORDINALS[value]]

    variants = []

    normal = digits_to_int(value)
    if normal is not None:
        variants.append(normal)

    if allow_reversed:
        reversed_number = digits_to_int_reversed(value)
        if reversed_number is not None and reversed_number not in variants:
            variants.append(reversed_number)

    return variants


def is_reasonable_next_article(number: int, last_article_number: int) -> bool:
    if number <= last_article_number:
        return False

    if last_article_number == 0:
        return number <= 5

    return number <= last_article_number + 12


def same_line_candidate(line: str) -> Optional[Dict]:
    n = normalize_text(line)

    match = re.match(
        r"^(المادة|مادة)\s*([0-9٠-٩]{1,4}|الأولى|الاولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة)\b",
        n,
    )

    if not match:
        return None

    if "من قانون" in n or "من القانون" in n:
        return None

    number_text = match.group(2)
    # Same-line headings like "المادة 51" must not be reversed.
    # Otherwise table references become fake article 15.
    variants = number_variants(number_text, allow_reversed=False)

    if not variants:
        return None

    if ":" not in n and len(n) > 130:
        return None

    return {
        "title": n,
        "number_variants": variants,
        "extra_lines": 0,
    }


def split_line_candidate(lines: List[str], index: int) -> Optional[Dict]:
    raw_current = normalize_text(lines[index])
    current = compact_text(raw_current)

    # Ignore quoted inner articles from amended laws/codes:
    # "المادة
    # ٣١١
    # These are not top-level articles of LoiFinance2026.
    if raw_current.lstrip().startswith(('"', "“", "”")):
        return None

    if current not in {"المادة", "مادة"}:
        return None

    if index + 1 >= len(lines):
        return None

    number_line = normalize_text(lines[index + 1])
    variants = number_variants(number_line, allow_reversed=True)

    if not variants:
        return None

    extra_lines = 1

    if index + 2 < len(lines):
        third = normalize_text(lines[index + 2])
        if third == ":" or third.startswith(":"):
            extra_lines = 2

    title_parts = []
    for offset in range(extra_lines + 1):
        title_parts.append(normalize_text(lines[index + offset]))

    title = " ".join(title_parts)
    title = re.sub(r"\s+", " ", title).strip()

    return {
        "title": title,
        "number_variants": variants,
        "extra_lines": extra_lines,
    }


def reversed_line_candidate(line: str) -> Optional[Dict]:
    n = normalize_text(line)

    # Ignore quoted amended articles.
    if n.lstrip().startswith(('"', "“", "”")):
        return None

    matches = list(re.finditer(r"([0-9٠-٩]{1,4})\s*ةدالما", n))

    if not matches:
        return None

    candidates = []

    for match in matches:
        raw_number = match.group(1)
        reversed_number = digits_to_int_reversed(raw_number)
        if reversed_number is not None:
            candidates.append(reversed_number)

    if not candidates:
        return None

    return {
        "title": n,
        "number_variants": candidates,
        "extra_lines": 0,
    }


def candidate_at(lines: List[str], index: int) -> Optional[Dict]:
    line = normalize_text(lines[index])

    candidate = same_line_candidate(line)
    if candidate:
        return candidate

    candidate = split_line_candidate(lines, index)
    if candidate:
        return candidate

    candidate = reversed_line_candidate(line)
    if candidate:
        return candidate

    return None


def select_valid_candidate(candidate: Dict, last_article_number: int) -> Optional[Tuple[str, int, int]]:
    for number in candidate["number_variants"]:
        if is_reasonable_next_article(number, last_article_number):
            return candidate["title"], number, candidate["extra_lines"]

    return None


def page_has_valid_article_heading(lines: List[str], last_article_number: int) -> bool:
    for index in range(len(lines)):
        candidate = candidate_at(lines, index)
        if not candidate:
            continue

        valid = select_valid_candidate(candidate, last_article_number)
        if valid:
            return True

    return False


def is_table_page(page_text: str, page_number: int, last_article_number: int) -> bool:
    if page_number <= 2:
        return False

    n = normalize_text(page_text)
    c = compact_text(n)

    table_markers = [
        "جداولقانونالمالية",
        "الجدول\"أ\"",
        "الجدول”أ“",
        "الجدول\"ب\"",
        "الجدول”ب“",
        "الجدول”و“",
        "الرسومشبهالجبائية",
        "النصوصالتشريعيةوالتنظيمية",
        "الهيئاتالمستفيدة",
        "الوحدة:بالدينار",
        "الوحدة:دج",
        "الاعتماداتالمفتوحةللسنة",
    ]

    if not any(marker in c for marker in table_markers):
        return False

    lines = page_text.splitlines()

    if page_has_valid_article_heading(lines, last_article_number):
        return False

    return True


def split_long_text(text: str, max_chars: int = MAX_TABLE_CHARS) -> List[str]:
    text = clean_output_text(text)

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


def page_range(lines_with_pages: List[Tuple[int, str]]) -> Tuple[Optional[int], Optional[int]]:
    pages = [page for page, _ in lines_with_pages]

    if not pages:
        return None, None

    return min(pages), max(pages)


def make_article_chunk(
    chunk_id: str,
    title: str,
    lines_with_pages: List[Tuple[int, str]],
    chunk_type: str = "article",
) -> Optional[Dict]:
    text = clean_output_text("\n".join(line for _, line in lines_with_pages))

    if not text:
        return None

    start, end = page_range(lines_with_pages)

    return {
        "chunk_id": chunk_id,
        "source_file": "LoiFinance2026_extracted.txt",
        "document_type": "finance_law",
        "chunk_type": chunk_type,
        "title": title,
        "page_start": start,
        "page_end": end,
        "text": text,
        "char_count": len(text),
    }


def chunk_finance_law_v31() -> List[Dict]:
    raw_text = INPUT_FILE.read_text(encoding="utf-8")
    pages = extract_pages(raw_text)

    chunks = []
    current_lines: List[Tuple[int, str]] = []
    current_title = "preamble_or_unknown"
    current_type = "preamble"

    last_article_number = 0
    chunk_index = 0

    for page_obj in pages:
        page = page_obj["page"]
        page_text = page_obj["text"]

        if page == 1:
            for part_index, part in enumerate(split_long_text(page_text), start=1):
                chunk_index += 1
                chunks.append({
                    "chunk_id": f"LoiFinance2026_extracted_v31_chunk_{chunk_index}",
                    "source_file": "LoiFinance2026_extracted.txt",
                    "document_type": "finance_law",
                    "chunk_type": "document_info",
                    "title": "document_info",
                    "page_start": page,
                    "page_end": page,
                    "text": part,
                    "char_count": len(part),
                    "page_part": part_index,
                })
            continue

        if is_table_page(page_text, page, last_article_number):
            if current_lines:
                chunk_index += 1
                chunk = make_article_chunk(
                    f"LoiFinance2026_extracted_v31_chunk_{chunk_index}",
                    current_title,
                    current_lines,
                    current_type,
                )
                if chunk:
                    chunks.append(chunk)

                current_lines = []
                current_title = "preamble_or_unknown"
                current_type = "preamble"

            for part_index, part in enumerate(split_long_text(page_text), start=1):
                chunk_index += 1
                chunks.append({
                    "chunk_id": f"LoiFinance2026_extracted_v31_chunk_{chunk_index}",
                    "source_file": "LoiFinance2026_extracted.txt",
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

        lines = page_text.splitlines()
        i = 0

        while i < len(lines):
            line = normalize_text(lines[i])

            if not line:
                i += 1
                continue

            candidate = candidate_at(lines, i)
            valid = select_valid_candidate(candidate, last_article_number) if candidate else None

            if valid:
                title, article_number, extra_lines = valid

                if current_lines:
                    chunk_index += 1
                    chunk = make_article_chunk(
                        f"LoiFinance2026_extracted_v31_chunk_{chunk_index}",
                        current_title,
                        current_lines,
                        current_type,
                    )
                    if chunk:
                        chunks.append(chunk)

                current_title = f"المادة {article_number}"
                current_type = "article"
                current_lines = []

                for offset in range(extra_lines + 1):
                    if i + offset < len(lines):
                        current_lines.append((page, normalize_text(lines[i + offset])))

                last_article_number = article_number
                i += extra_lines + 1
                continue

            current_lines.append((page, line))
            i += 1

    if current_lines:
        chunk_index += 1
        chunk = make_article_chunk(
            f"LoiFinance2026_extracted_v31_chunk_{chunk_index}",
            current_title,
            current_lines,
            current_type,
        )
        if chunk:
            chunks.append(chunk)

    return chunks


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"❌ Missing input file: {INPUT_FILE}")
        return

    chunks = chunk_finance_law_v31()

    OUTPUT_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ Finance Law V3.1 chunking completed.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total chunks: {len(chunks)}")

    types = sorted(set(chunk["chunk_type"] for chunk in chunks))
    for chunk_type in types:
        print(f"{chunk_type}: {sum(1 for chunk in chunks if chunk['chunk_type'] == chunk_type)}")


if __name__ == "__main__":
    main()
