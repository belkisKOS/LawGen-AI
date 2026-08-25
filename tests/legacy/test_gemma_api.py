from dotenv import load_dotenv
from llm.gemma_client import GemmaClient


load_dotenv()


gemma = GemmaClient()


response = gemma.generate(
    "قل جملة قصيرة عن الجزائر"
)


print(response)

