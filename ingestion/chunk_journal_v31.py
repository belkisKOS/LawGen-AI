from pathlib import Path
import json
import re
import unicodedata
from typing import List, Dict, Optional, Tuple

INPUT_FILE = Path("data/processed/JournalOff_extracted.txt")
OUTPUT_DIR = Path("data/chunks_v3")
OUTPUT_FILE = OUTPUT_DIR / "JournalOff_extracted_chunks_v3.json"

MAX_CHARS = 3500

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")


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


def remove_noise_lines(lines: List[str]) -> List[str]:
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

        if "صفرعام" in c or "رجبعام" in c or "جمادى" in c:
            continue

        if "يوليوسنة" in c or "ديسمبرسنة" in c:
            continue

        # Keep standalone numbers.
        # In the Journal Officiel, legal act identifiers can be split across lines:
        # مرسوم رئاسي رقم 26
        # -
        # 239
        # Removing standalone numbers would damage the legal reference.
        cleaned.append(n)

    return cleaned


def clean_output_text(text: str) -> str:
    lines = remove_noise_lines(text.splitlines())
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


def is_index_page(page_text: str) -> bool:
    c = compact_text(page_text)

    index_markers = [
        "فهرس",
        "فهرس(تابع)",
        "الاشتراك",
        "ثمنالنسخة",
    ]

    return any(marker in c for marker in index_markers)


def is_section_heading(line: str) -> bool:
    c = compact_text(line)

    section_markers = [
        "مراسيمتنظيمية",
        "مراسيمفردية",
        "وزارةالتعليمالعاليوالبحثالعلمي",
        "وزارةالشؤونالدينيةوالأوقاف",
        "وزارةالبيئةوجودةالحياة",
    ]

    return any(marker in c for marker in section_markers)


def is_legal_act_start(line: str) -> bool:
    n = normalize_text(line)
    c = compact_text(n)

    patterns = [
        "مرسومرئاسيرقم",
        "مرسومترئاسيرقم",
        "مرسومرئاسيمؤرخ",
        "مرسومرئاسيمؤرخ",
        "مرسومتنفيذيرقم",
        "مرسومتنفيذيمؤرخ",
        "قراروزاريمشترك",
        "قرارمؤرخ",
        "قراررقم",
    ]

    if any(pattern in c for pattern in patterns):
        return True

    # Some PDF lines may have extra spacing between words.
    regex_patterns = [
        r"^مرسوم\s+رئاسي\s+رقم",
        r"^مرسوم\s+رئاسي\s+مؤرخ",
        r"^مرسوم\s+تنفيذي\s+رقم",
        r"^مرسوم\s+تنفيذي\s+مؤرخ",
        r"^قرار\s+وزاري\s+مشترك",
        r"^قرار\s+مؤرخ",
        r"^قرار\s+رقم",
    ]

    return any(re.search(pattern, n) for pattern in regex_patterns)


def split_long_text(text: str, max_chars: int = MAX_CHARS) -> List[str]:
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


def make_chunk(
    chunk_id: str,
    chunk_type: str,
    title: str,
    lines_with_pages: List[Tuple[int, str]],
) -> Optional[Dict]:
    text = clean_output_text("\n".join(line for _, line in lines_with_pages))

    if not text:
        return None

    page_start, page_end = page_range(lines_with_pages)

    return {
        "chunk_id": chunk_id,
        "source_file": "JournalOff_extracted.txt",
        "document_type": "journal_officiel",
        "chunk_type": chunk_type,
        "title": title,
        "page_start": page_start,
        "page_end": page_end,
        "text": text,
        "char_count": len(text),
    }



def build_better_title(text: str, fallback: str) -> str:
    lines = [normalize_text(line) for line in text.splitlines() if normalize_text(line)]

    for idx, line in enumerate(lines):
        if is_legal_act_start(line):
            title_lines = [line]

            for extra in lines[idx + 1: idx + 8]:
                if is_section_heading(extra) or is_legal_act_start(extra):
                    break

                title_lines.append(extra)
                joined = " ".join(title_lines)

                if "،" in extra or "." in extra or "يتضمن" in joined or "يحدد" in joined:
                    break

            title = " ".join(title_lines)
            title = re.sub(r"\s+", " ", title).strip()
            return title[:220]

    return fallback[:220]


def chunk_journal_v31() -> List[Dict]:
    raw_text = INPUT_FILE.read_text(encoding="utf-8")
    pages = extract_pages(raw_text)

    chunks = []
    current_lines: List[Tuple[int, str]] = []
    current_title = "journal_section_or_unknown"
    current_type = "legal_act"
    chunk_index = 0
    current_section = None

    for page_obj in pages:
        page = page_obj["page"]
        page_text = page_obj["text"]

        # Skip cover and index pages.
        if page < 4:
            continue

        if is_index_page(page_text):
            continue

        lines = page_text.splitlines()

        for raw_line in lines:
            line = normalize_text(raw_line)

            if not line:
                continue

            if is_section_heading(line):
                current_section = line

                # Keep section headings attached to next legal act, not as separate chunk.
                if not current_lines:
                    current_lines.append((page, line))
                else:
                    current_lines.append((page, line))

                continue

            if is_legal_act_start(line):
                if current_lines:
                    chunk_index += 1
                    chunk = make_chunk(
                        f"JournalOff_extracted_v31_chunk_{chunk_index}",
                        current_type,
                        current_title,
                        current_lines,
                    )
                    if chunk:
                        chunks.append(chunk)

                current_title = line
                current_type = "legal_act"
                current_lines = []

                if current_section:
                    current_lines.append((page, current_section))

                current_lines.append((page, line))
                continue

            current_lines.append((page, line))

    if current_lines:
        chunk_index += 1
        chunk = make_chunk(
            f"JournalOff_extracted_v31_chunk_{chunk_index}",
            current_type,
            current_title,
            current_lines,
        )
        if chunk:
            chunks.append(chunk)

    # Remove tiny section-only chunks.
    chunks = [
        chunk for chunk in chunks
        if not (
            chunk.get("title") == "journal_section_or_unknown"
            and chunk.get("char_count", 0) < 100
        )
    ]

    # Split very long legal acts while keeping title.
    final_chunks = []
    split_index = 0

    for chunk in chunks:
        improved_title = build_better_title(chunk["text"], chunk["title"])
        parts = split_long_text(chunk["text"])

        for part_index, part in enumerate(parts, start=1):
            split_index += 1
            new_chunk = dict(chunk)
            new_chunk["chunk_id"] = f"JournalOff_extracted_v31_chunk_{split_index}"
            new_chunk["title"] = improved_title
            new_chunk["text"] = part
            new_chunk["char_count"] = len(part)
            new_chunk["subpart_index"] = part_index
            new_chunk["subpart_count"] = len(parts)
            final_chunks.append(new_chunk)

    return final_chunks


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"❌ Missing input file: {INPUT_FILE}")
        return

    chunks = chunk_journal_v31()

    OUTPUT_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ JournalOff V3.1 chunking completed.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total chunks: {len(chunks)}")

    types = sorted(set(chunk["chunk_type"] for chunk in chunks))
    for chunk_type in types:
        print(f"{chunk_type}: {sum(1 for chunk in chunks if chunk['chunk_type'] == chunk_type)}")

    print("\nFirst chunks:")
    for chunk in chunks[:10]:
        print("-", chunk["title"], "| pages", chunk["page_start"], "-", chunk["page_end"], "| chars", chunk["char_count"])


if __name__ == "__main__":
    main()
