from pathlib import Path
from ingestion.deepseek_ocr import run_ocr


def extract_page_with_ocr(
    pdf_path: Path,
    page_number: int,
) -> str:
    """
    Extract one PDF page using DeepSeek OCR.
    Used only when normal extraction fails.
    """

    temp_image = Path(
        "data/ocr_samples"
    ) / f"ocr_page_{page_number}.png"

    temp_output = Path(
        "data/ocr_samples"
    ) / f"ocr_page_{page_number}.txt"

    text = run_ocr(
        pdf_path=pdf_path,
        page_number=page_number,
        output_image_path=temp_image,
    )

    temp_output.write_text(
        text,
        encoding="utf-8"
    )

    return text

