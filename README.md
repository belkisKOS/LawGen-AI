# LawGen AI

LawGen AI is an Algerian legal AI assistant based on a Retrieval-Augmented Generation architecture.

The system retrieves relevant legal text from Algerian legal documents, builds a legal context, and uses a generative model to produce an answer with clear legal sources.

Current backend status:

```text
RAG core: working
Gemma generation: working
FastAPI /ask endpoint: working
Evaluation: 20/20 on the current test set
OCR: experimental and paused pending confirmation
```

---

## 1. Project Goal

LawGen AI aims to answer legal questions using Algerian legal sources.

The current version focuses on:

- extracting text from legal PDFs
- creating legal chunks
- indexing chunks in ChromaDB
- retrieving relevant chunks using hybrid retrieval
- generating answers with Gemma through AI Grid
- returning answers with legal sources
- exposing the system through a FastAPI endpoint

This project currently uses RAG, not fine-tuning.

---

## 2. Current Documents

The current indexed legal documents are:

```text
data/raw/Constitution.pdf
data/raw/JournalOff.pdf
data/raw/LoiFinance2026.pdf
```

Processed and indexed sources include:

```text
Constitution_extracted.txt
JournalOff_extracted.txt
LoiFinance2026_extracted.txt
```

Current evaluation covers all three sources:

```text
Constitution: 10 questions
Finance Law 2026: 5 questions
Official Journal: 5 questions
Total: 20 questions
```

Current result:

```text
Evaluation score: 20/20
```

---

## 3. Architecture

Current flow:

```text
PDF documents
    ↓
Text extraction
    ↓
Legal chunking
    ↓
Metadata correction
    ↓
ChromaDB indexing
    ↓
Hybrid retrieval
    ↓
Context building
    ↓
Gemma generation
    ↓
Answer with sources
```

Main modules:

```text
ingestion/       PDF extraction, chunking, indexing, OCR experiments
retrieval/       Hybrid retriever
rag/             Context builder and RAG pipeline
reranking/       Context enhancement experiments
llm/             Gemma API client
eval/            Evaluation questions and runner
api/             FastAPI backend
scripts/         Utility scripts
```

---

## 4. Branches

Main working branches:

```text
phase-1-ingestion
```

Stable branch containing:

- legal chunking
- Chroma indexing
- hybrid retrieval
- Gemma generation
- 20-question evaluation
- FastAPI `/ask` endpoint

```text
phase-2-ocr
```

Experimental branch containing DeepSeek OCR tests.

OCR is not yet integrated into the stable RAG pipeline because the current DeepSeek OCR output is not reliable on Arabic legal PDF pages.

---

## 5. Setup

Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 6. Environment Variables

Create a `.env` file in the project root.

Required Gemma variables:

```env
GEMMA_API_KEY=your_gemma_api_key_here
GEMMA_BASE_URL=https://app.ai-grid.io/v1
GEMMA_MODEL=your_gemma_model_here
```

Experimental OCR variables:

```env
DEEPSEEK_OCR_API_KEY=your_deepseek_ocr_key_here
DEEPSEEK_OCR_BASE_URL=https://app.ai-grid.io/v1
DEEPSEEK_OCR_MODEL=deepseek-ocr
```

Important:

```text
Never commit .env
Never paste API keys into GitHub
Rotate exposed keys immediately
```

---

## 7. Build the Chroma Index

The Chroma vector database is generated locally and is not committed.

To rebuild it:

```bash
rm -rf data/chroma_db
python ingestion/build_chroma_index.py
```

Expected output:

```text
Chroma index created successfully.
Location: data/chroma_db
Collection: lawgen_v3
```

---

## 8. Metadata Fixes

Some extracted legal chunks may contain OCR or extraction errors in article titles.

Example fixed issue:

```text
المادة ٣2
```

was corrected to:

```text
المادة 223
```

To apply metadata fixes:

```bash
python scripts/fix_chunk_metadata.py
```

Then rebuild Chroma:

```bash
rm -rf data/chroma_db
python ingestion/build_chroma_index.py
```

---

## 9. Run the CLI Chatbot

Run:

```bash
python chatbot.py
```

Example question:

```text
ما هو دين الدولة حسب الدستور الجزائري؟
```

Example answer:

```text
الإجابة:
دين الدولة هو الإسلام.

المصدر:
- الوثيقة: Constitution_extracted.txt
- المادة أو النص: المادة 2
- الصفحة: 4-4
```

---

## 10. Run Evaluation

Run:

```bash
python eval/run_eval.py
```

Current expected result:

```text
EVALUATION SCORE: 20/20
```

The evaluation checks whether the retriever brings the expected legal references from the expected documents.

Current evaluation distribution:

```text
Constitution_extracted.txt: 10 questions
LoiFinance2026_extracted.txt: 5 questions
JournalOff_extracted.txt: 5 questions
```

---

## 11. Run the FastAPI Backend

Start the API server:

```bash
uvicorn api.main:app --reload
```

API will run at:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ask endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"ما هو دين الدولة حسب الدستور الجزائري؟"}'
```

Expected response format:

```json
{
  "question": "ما هو دين الدولة حسب الدستور الجزائري؟",
  "answer": "...",
  "sources": [
    {
      "source_file": "Constitution_extracted.txt",
      "title": "المادة 2",
      "page_start": 4,
      "page_end": 4,
      "chunk_type": "article",
      "score": 130,
      "text_preview": "المادة 2\n: الإسلام دين الدولة."
    }
  ]
}
```

---

## 12. OCR Status

DeepSeek OCR has been tested through AI Grid using the OpenAI-compatible `image_url` format.

Current OCR findings:

```text
API connection: works
Simple image OCR: partially works
Full Arabic legal PDF page OCR: unreliable
Cropped Arabic legal page OCR: unreliable
```

Because of this, OCR is currently paused and remains experimental on:

```text
phase-2-ocr
```

OCR will not be merged into the stable RAG pipeline until the correct model format and settings are confirmed.

---

## 13. Current Limitations

Current limitations:

- evaluation checks retrieval correctness more than full legal answer quality
- generated data and Chroma index are local and ignored by Git
- some extraction metadata may still require cleaning
- API currently returns retrieved sources, not only final answer-used sources
- conversation history is not yet implemented
- OCR is not production-ready

---

## 14. Next Roadmap

Next development steps:

```text
1. Improve API response structure
2. Separate main sources from retrieved sources
3. Add API tests
4. Improve evaluation with expected answer phrases
5. Add negative questions to test hallucination resistance
6. Add conversation history
7. Resume OCR after Hichem confirms the correct DeepSeek OCR setup
8. Connect frontend to FastAPI
9. Prepare deployment
```

---

## 15. Current Stable Checkpoint

Stable branch:

```text
phase-1-ingestion
```

Current confirmed capabilities:

```text
RAG retrieval: working
Gemma generation: working
Source citation: working
Evaluation: 20/20
FastAPI endpoint: working
OCR: experimental only
```

