import hashlib
import json
from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)

HASH_LOG = Path("data/upload_hashes.json")
MIN_WORDS = 50
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


class DocumentValidationError(Exception):
    """Raised when validation cannot be completed (e.g. the LLM check itself fails)."""


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_hash_log() -> set:
    if not HASH_LOG.exists():
        return set()
    try:
        return set(json.loads(HASH_LOG.read_text(encoding="utf-8")))
    except Exception as error:
        logger.warning("Could not read %s, treating as empty: %s", HASH_LOG, error)
        return set()


def save_hash(file_hash: str) -> None:
    try:
        hashes = load_hash_log()
        hashes.add(file_hash)
        HASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        HASH_LOG.write_text(json.dumps(list(hashes)), encoding="utf-8")
    except Exception as error:
        logger.error("Failed to save upload hash: %s", error)


def check_basic_rules(file_path: Path, text: str) -> tuple[bool, str, bool]:
    """
    Returns (valid, reason, already_present).
    - already_present=True means: valid file, but we've seen it before.
      Callers should treat it as success and skip ingestion.
    """
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type: {file_path.suffix}", False

    if file_path.stat().st_size > MAX_SIZE_BYTES:
        return False, "File exceeds 10 MB limit", False

    if not text.strip():
        return False, "No extractable text found (possibly a scanned image with no text layer)", False

    word_count = len(text.split())
    if word_count < MIN_WORDS:
        return False, f"Document too short ({word_count} words, minimum {MIN_WORDS})", False

    file_hash = compute_hash(text)
    if file_hash in load_hash_log():
        return True, "", True

    return True, "", False


def classify_with_llm(text: str, llm_generate_fn) -> tuple[bool, str]:
    sample = text[:2000]
    prompt = (
        "Answer with only YES or NO, then a short reason on a new line.\n"
        "Is the following text a legal or law-related document "
        "(e.g. constitution, law, decree, contract, legal code, regulation) "
        "relevant to Algerian law?\n\n"
        f"TEXT:\n{sample}"
    )

    try:
        response = llm_generate_fn(prompt).strip()
    except Exception as error:
        logger.exception("LLM call failed during document classification.")
        raise DocumentValidationError(
            f"Could not verify document with the AI model: {error}"
        ) from error

    first_line = response.splitlines()[0].strip().upper() if response else ""

    is_legal = first_line.startswith("YES")
    reason = response.split("\n", 1)[1].strip() if "\n" in response else response

    return is_legal, reason