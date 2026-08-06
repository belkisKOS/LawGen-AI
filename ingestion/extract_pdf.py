from pathlib import Path
import fitz  # PyMuPDF


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def extract_text_from_pdf(pdf_path: Path, output_path: Path) -> None:
    """
    Extract text from one PDF file and save it as a TXT file.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = fitz.open(pdf_path)
    extracted_pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        page_content = (
            f"\n\n--- PAGE {page_number} ---\n\n"
            f"{text.strip()}"
        )

        extracted_pages.append(page_content)

    full_text = "\n".join(extracted_pages)
    output_path.write_text(full_text, encoding="utf-8")

    print(f"✅ Extracted: {pdf_path.name}")
    print(f"   Pages: {len(document)}")
    print(f"   Output: {output_path}")


def extract_all_pdfs() -> None:
    """
    Extract text from all PDF files inside data/raw.
    """

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDF files found in {RAW_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF file(s).")

    for pdf_file in pdf_files:
        output_file = PROCESSED_DIR / f"{pdf_file.stem}_extracted.txt"
        extract_text_from_pdf(pdf_file, output_file)

    print("\n✅ All PDF files extracted successfully.")


if __name__ == "__main__":
    extract_all_pdfs()
