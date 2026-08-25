# LawGen AI

Legal RAG system for Algerian legal documents.

LawGen AI is a Retrieval Augmented Generation (RAG) system that allows users to ask questions about Algerian legal texts and receive answers with legal sources and references.

The system works with:

- Constitution documents
- Finance Law documents
- Official Journal documents

---

# Features

## 1. Document Ingestion

The ingestion pipeline processes legal PDF documents.

Features:

- PDF text extraction using PyMuPDF
- Document preprocessing
- Legal chunking
- Metadata preservation:
  - source document
  - article number
  - page numbers
  - chunk type

Pipeline:

```
PDF
 |
Text Extraction
 |
Cleaning
 |
Chunking
 |
Vector Database
```

---

# 2. OCR Support

LawGen AI integrates DeepSeek OCR through AI-Grid API.

OCR is used as a fallback when PDF text extraction is missing or insufficient.

Pipeline:

```
PDF Page
 |
Normal Text Extraction
 |
Text Quality Check
 |
+----------------+
|                |
Good text        Bad/empty text
|                |
Keep text        DeepSeek OCR
                 |
                 |
              Extracted text
```

OCR preserves:

- Arabic legal text
- Article numbers
- Decree numbers
- Dates
- Legal formatting

---

# 3. Retrieval System

The system uses hybrid retrieval:

- Semantic search using embeddings
- Keyword-based matching
- Chroma vector database

Retrieval flow:

```
User Question
      |
      |
Query Reformulation
      |
      |
Hybrid Search
      |
      |
Relevant Legal Chunks
```

---

# 4. Legal Query Reformulation

User questions can be written in:

- Algerian dialect
- French
- Informal Arabic

The system reformulates them into legal search queries.

It combines:

- Rule-based legal indicators
- AI-based legal reformulation

Example:

User:

```
واش يقول القانون على دين الدولة؟
```

Reformulated:

```
النص الدستوري المتعلق بالإسلام باعتباره دين الدولة في الدستور الجزائري
```

The reformulated query is used for retrieval.

---

# 5. RAG Answer Generation

The answer generation pipeline:

```
Question
 |
Reformulation
 |
Retrieval
 |
Legal Context
 |
Gemma LLM
 |
Answer + Sources
```

The response includes:

- Generated answer
- Source document
- Article/reference
- Page numbers

---

# 6. API

LawGen AI exposes a FastAPI backend.

Endpoint:

```
POST /ask
```

Example request:

```json
{
  "question": "ما هو دين الدولة حسب الدستور الجزائري؟"
}
```

Example response:

```json
{
  "question": "ما هو دين الدولة حسب الدستور الجزائري؟",
  "answer": "الإسلام هو دين الدولة.",
  "sources": [
    {
      "source_file": "Constitution_extracted.txt",
      "title": "المادة 2",
      "page_start": 4,
      "page_end": 4
    }
  ]
}
```

---

# Architecture

```
Client
 |
FastAPI API
 |
LawGen Pipeline
 |
+-------------------------+
|                         |
Query Reformulation       Retrieval
|                         |
AI + Rules                Chroma
                          |
                          |
                    Legal Documents
                          |
                          |
                       Gemma LLM
                          |
                          |
                    Answer + Sources
```

---

# Project Structure

```
lawgen-ai/

api/
 └── main.py

rag/
 ├── pipeline.py
 ├── query_reformulator.py
 └── context_builder.py

retrieval/
 └── retriever.py

ingestion/
 ├── extract_pdf.py
 ├── deepseek_ocr.py
 ├── ocr_fallback.py
 ├── prepare_chunks_v3.py
 └── build_chroma_index.py

llm/
 └── gemma_client.py

data/
 ├── raw/
 ├── processed/
 └── chroma_db/
```

---

# Installation

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the API

Start server:

```bash
uvicorn api.main:app --reload
```

API available at:

```
http://127.0.0.1:8000
```

---

# Testing

Example:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question":"ما هو دين الدولة حسب الدستور الجزائري؟"}'
```

---

# Current Status

Completed:

- ✅ Legal document ingestion
- ✅ PDF extraction
- ✅ DeepSeek OCR fallback integration
- ✅ Document chunking
- ✅ Chroma vector indexing
- ✅ Hybrid retrieval
- ✅ Gemma RAG generation
- ✅ Legal query reformulation
- ✅ FastAPI backend
- ✅ Source citation system

---

# Team Notes

The Python AI system can be integrated with any technology stack.

Example:

```
Next.js / Java / Mobile App
          |
          |
       HTTP API
          |
          |
     FastAPI Python Backend
          |
          |
       LawGen AI Pipeline
```

The external application does not need Python code. It only communicates with the API.