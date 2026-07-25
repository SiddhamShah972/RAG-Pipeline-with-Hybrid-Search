# RAG Pipeline (Zero Cost Edition)

A production-grade, fully free, locally-hosted Retrieval-Augmented Generation pipeline.

## Features
- **Zero Cost**: Every component in this stack is either open-source running locally or covered by a free-tier API.
- **Hybrid Search**: Combines Dense (ChromaDB + BGE Embeddings) and Sparse (BM25) retrieval using Reciprocal Rank Fusion.
- **Reranking**: Uses a cross-encoder (`ms-marco-MiniLM-L-6-v2`) to improve relevance.
- **Grounded Answers**: Uses Google Gemini 1.5 Flash (free tier) to answer questions based strictly on the retrieved context with inline citations.
- **Offline Fallback**: Run fully offline using Ollama and Phi-3 mini.

## Prerequisites
- Docker & Docker Compose
- Windows WSL2 (with NVIDIA Container Toolkit) or Linux for GPU passthrough
- A Gemini API Key from Google AI Studio (Free)

## Quickstart

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`
3. Build and start the containers:
   ```bash
   docker-compose up --build
   ```
4. Ingest a document:
   ```bash
   curl -X POST http://localhost:8080/ingest -F "file=@your_doc.pdf"
   ```
5. Query it:
   ```bash
   curl -X POST http://localhost:8080/query -H "Content-Type: application/json" -d '{"query": "what is this document about?"}'
   ```

## API Reference

- `POST /ingest`: Upload PDF, TXT, HTML, or DOCX files for indexing.
- `POST /query`: Search the corpus and get a grounded answer.
- `GET /health`: Check system component statuses.
