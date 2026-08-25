import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def data_url(path: str, mime_type: str = "image/png") -> str:
    image_path = Path(path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    print("Image path:", image_path)
    print("Image bytes:", image_path.stat().st_size)
    print("Base64 chars:", len(encoded))

    return f"data:{mime_type};base64,{encoded}"


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_OCR_API_KEY"),
    base_url=os.getenv("DEEPSEEK_OCR_BASE_URL", "https://app.ai-grid.io/v1"),
)


response = client.chat.completions.create(
    model=os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-ocr"),
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You are an OCR engine. Read the image carefully and return only the exact visible text. Do not add any explanation.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url("data/ocr_samples/big_ocr_test.png"),
                    },
                },
            ],
        }
    ],
    max_tokens=1024,
    temperature=0,
)


print("=" * 80)
print("RAW RESPONSE")
print("=" * 80)
print(response)

print("=" * 80)
print("CONTENT")
print("=" * 80)
print(repr(response.choices[0].message.content))
