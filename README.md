# LawGen AI

LawGen AI is an Algerian legal AI assistant project based on a RAG architecture.

The goal is to retrieve reliable legal information from Algerian legal documents, then later generate answers using an LLM with clear legal references.

## Current Status

This branch focuses on the ingestion and retrieval pipeline.

Completed so far:

- Project structure setup
- PDF text extraction
- Legal document chunking
- Final cleaned chunks generation
- Embeddings creation
- ChromaDB indexing
- Semantic retrieval testing
- Hybrid retrieval implementation
- RAG context preparation

The LLM/API layer is not connected yet.

## Current RAG Pipeline

PDF documents
→ text extraction
→ legal chunking
→ final chunks
→ embeddings
→ ChromaDB
→ hybrid retrieval
→ RAG context

## Documents Used

Current legal sources:

- Algerian Constitution
- Official Journal document
- Finance Law 2026

## Chunking Strategy

The project uses legal-aware chunking instead of basic text splitting.

Current chunking logic:

- Constitution: article-based chunking
- Finance Law 2026: article and table-based chunking
- Official Journal: legal act-based chunking

Final chunks are stored in:

data/final_chunks_v3/final_chunks.json

Current final chunk count:

491 chunks

## Retrieval Strategy

Pure semantic retrieval was not accurate enough for legal references.

The project now uses hybrid retrieval:

- semantic search
- exact keyword matching
- article/decree number matching
- document source boosting
- chunk type boosting

## Successful Retrieval Tests

The current system correctly retrieves:

- Article 2 of the Constitution for the state religion question
- Article 3 of the Constitution for the national and official language question
- Article 2 of the Finance Law 2026 for estimated revenues
- Presidential Decree 26-239 from the Official Journal

## Main Files

- ingestion/extract_pdf.py
- ingestion/build_chroma_index.py
- retrieval/hybrid_retriever.py
- rag/context_builder.py
- chatbot_cli.py
- rag_prompt_test.py

## Test Retrieval

Run:

python chatbot_cli.py

Example questions:

ما هو دين الدولة حسب الدستور الجزائري؟
ما هي اللغة الوطنية والرسمية في الجزائر حسب الدستور؟
ما قيمة الإيرادات المقدرة في قانون المالية لسنة 2026؟
ما هو المرسوم الرئاسي رقم 26-239؟

## Test RAG Prompt Context

Run:

python rag_prompt_test.py

This shows the legal context that will later be sent to the LLM.

## Next Steps

Planned next steps:

- Add offline chatbot output
- Connect AI Grid or another OpenAI-compatible LLM API
- Generate final legal answers from retrieved context
- Add citations in the final response
- Add FastAPI backend
- Add frontend interface
- Add document upload/admin workflow

## Notes

The current version does not provide final legal advice.

It retrieves legal sources and prepares context for a future LLM answer layer.
