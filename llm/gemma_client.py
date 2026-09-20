import os
from dotenv import load_dotenv
from openai import OpenAI
from core.logger import get_logger


load_dotenv()

logger = get_logger(__name__)


class GemmaClientError(Exception):
    """Raised when the Gemma/LLM API cannot be reached or fails to answer."""


class GemmaClient:

    def __init__(self):
        self.api_key = os.getenv("GEMMA_API_KEY")
        self.base_url = os.getenv("GEMMA_BASE_URL")
        self.model = os.getenv("GEMMA_MODEL")

        missing = [
            name
            for name, value in [
                ("GEMMA_API_KEY", self.api_key),
                ("GEMMA_BASE_URL", self.base_url),
                ("GEMMA_MODEL", self.model),
            ]
            if not value
        ]

        if missing:
            logger.error(
                "GemmaClient is misconfigured. Missing env vars: %s",
                ", ".join(missing),
            )
            raise GemmaClientError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            )

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as error:
            logger.exception("Failed to initialize the Gemma/OpenAI client.")
            raise GemmaClientError(f"Could not initialize LLM client: {error}") from error

        logger.info("GemmaClient initialized (model=%s, base_url=%s)", self.model, self.base_url)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content

        except Exception as error:
            logger.exception(
                "Gemma generation failed (model=%s, prompt_len=%d).",
                self.model,
                len(prompt or ""),
            )
            raise GemmaClientError(f"LLM generation failed: {error}") from error

    def check_connection(self) -> tuple[bool, str]:
        try:
            self.client.models.list()
            return True, "ok"
        except Exception as list_error:
            logger.info(
                "models.list() unavailable (%s), falling back to a minimal generate() call.",
                list_error,
            )

        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
            )
            return True, "ok"
        except Exception as error:
            logger.warning("Gemma connectivity check failed: %s", error)
            return False, str(error)