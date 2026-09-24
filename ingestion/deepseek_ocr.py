import argparse
import base64
import os
from pathlib import Path

import fitz
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def render_pdf_page_to_png(
    pdf_path: Path,
    page_number: int,
    output_image_path: Path,
    zoom: float = 1.5,
    quality: int = 85,
) -> str:
    document = fitz.open(pdf_path)

    if page_number < 1 or page_number > len(document):
        raise ValueError(f"Page number must be between 1 and {len(document)}")

    page = document[page_number - 1]
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(output_image_path)

    image_bytes = output_image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    print("Rendered image:", output_image_path)
    print("Image bytes:", len(image_bytes))
    print("Base64 chars:", len(encoded))

    document.close()

    return f"data:image/png;base64,{encoded}"


def run_ocr(pdf_path: Path, page_number: int, output_image_path: Path) -> str:
    api_key = os.getenv("DEEPSEEK_OCR_API_KEY")
    base_url = os.getenv("DEEPSEEK_OCR_BASE_URL", "https://app.ai-grid.io/v1")
    model = os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-ocr")

    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_OCR_API_KEY in .env")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    image_url = render_pdf_page_to_png(
        pdf_path=pdf_path,
        page_number=page_number,
        output_image_path=output_image_path,
    )

    prompt = "Extract all readable text from this document."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
        max_tokens=1024,
        temperature=0,
    )

    print("=" * 80)
    print("RAW RESPONSE DEBUG")
    print("=" * 80)
    print("finish_reason:", response.choices[0].finish_reason)
    print("content repr:", repr(response.choices[0].message.content))

    return response.choices[0].message.content or ""


def main():
    parser = argparse.ArgumentParser(description="Run DeepSeek OCR on one PDF page.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument("--page", required=True, type=int, help="1-based page number.")
    parser.add_argument("--output", required=True, help="Output text file path.")

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    output_path = Path(args.output)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_image_path = output_path.with_suffix(".png")

    text = run_ocr(
        pdf_path=pdf_path,
        page_number=args.page,
        output_image_path=output_image_path,
    )

    print("=" * 80)
    print("OCR OUTPUT")
    print("=" * 80)
    print(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    print("=" * 80)
    print(f"Saved OCR output to: {output_path}")


def encode_image_bytes(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _call_ocr_api(image_url: str) -> str:
    api_key = os.getenv("DEEPSEEK_OCR_API_KEY")
    base_url = os.getenv("DEEPSEEK_OCR_BASE_URL", "https://app.ai-grid.io/v1")
    model = os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-ocr")

    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_OCR_API_KEY in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = "Extract all readable text from this document."

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        max_tokens=1024,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def run_ocr_on_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """
    OCR a raw photo (camera/gallery) — not a PDF page.
    Nothing here touches disk or Chroma.
    """
    image_url = encode_image_bytes(image_bytes, mime_type=mime_type)
    return _call_ocr_api(image_url)


if __name__ == "__main__":
    main()