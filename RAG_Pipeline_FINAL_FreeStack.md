**RAG PIPELINE**

Hybrid Search · Zero Cost Edition

_PRD · TRD · Implementation Plan · Tech Stack_

**★ FULLY FREE STACK - \$0 TOTAL COST ★**

Stack: Gemini 1.5 Flash · BGE · ChromaDB · BM25 · FastAPI · Docker

Target Hardware: Ryzen 5 · 16GB RAM · GTX 1650 · Windows 11

_Version 2.0 - Final · Confidential_

# **SECTION 1 - Product Requirements Document (PRD)**

## **1.1 Executive Summary**

A production-grade, fully free, locally-hosted Retrieval-Augmented Generation pipeline that allows a user to query an internal document corpus and receive grounded, cited answers. The system runs entirely on a consumer laptop (Ryzen 5, 16GB RAM, GTX 1650) with zero monthly cost: embeddings and reranking execute locally via GPU, retrieval is handled by ChromaDB and BM25 in-process, and language generation is delegated to the Google Gemini 1.5 Flash API - free up to 1,500 queries per day with no credit card required.

**Zero Cost Guarantee**

Every component in this stack is either open-source running locally or covered by a free-tier API. No credit card. No subscription. No cloud bill. Total monthly cost: \$0.

## **1.2 Problem Statement**

• Information is scattered across PDFs, DOCXs, and text files - finding answers requires manual reading.

• General LLMs hallucinate when asked about internal documents they have never seen.

• Paid RAG APIs (OpenAI, Anthropic) add cost and data privacy concerns.

• Local-only LLMs on a 4GB VRAM GPU produce low-quality answers.

• This system solves all four: grounded answers, free, private, high quality.

## **1.3 Goals & Non-Goals**

### **Goals**

• **✓** Grounded answers with inline citations - every claim traceable to source chunk.

• **✓** 100% free to run - no API costs, no cloud costs, no subscriptions.

• **✓** Runs on Ryzen 5 / 16GB RAM / GTX 1650 / Windows.

• **✓** Ingests PDF, TXT, HTML, DOCX.

• **✓** REST API: /ingest, /query, /health.

• **✓** Single command deploy via docker-compose.

• **✓** Sub-5s p95 query latency on local hardware.

### **Non-Goals (v1)**

• **✗** Real-time document sync or webhook ingestion.

• **✗** Multi-user auth or access control.

• **✗** Frontend UI - API-first, UI is v2.

• **✗** Offline LLM - Gemini Flash requires internet.

## **1.4 User Stories**

| **ID** | **As a…** | **I want to…**                                | **So that…**                          | **Priority** |
| ------ | --------- | --------------------------------------------- | ------------------------------------- | ------------ |
| US-01  | User      | Ask a natural-language question about my docs | I get an instant cited answer         | P0           |
| US-02  | User      | See which source backs each claim             | I can verify and trust the answer     | P0           |
| US-03  | User      | Upload a new document                         | The corpus updates without restarting | P0           |
| US-04  | User      | Run everything for free                       | No monthly bill ever                  | P0           |
| US-05  | Dev       | Swap LLM provider via env var                 | Not locked to Gemini forever          | P1           |
| US-06  | Dev       | Run stack locally in under 5 min              | Fast dev/test cycle                   | P0           |

## **1.5 Success Metrics**

| **Metric**               | **Target**               | **Measurement**               |
| ------------------------ | ------------------------ | ----------------------------- |
| Monthly cost             | \$0                      | Invoice check                 |
| Query latency p95        | < 5s on local hardware   | API response timer            |
| Citation accuracy        | ≥ 85%                    | Human eval on 50 test queries |
| Answer relevance (RAGAS) | ≥ 0.75                   | Automated RAGAS suite         |
| Daily query capacity     | 1500 free / day (Gemini) | API quota dashboard           |
| Ingestion speed          | < 10s per 10-page doc    | Benchmark timer               |

# **SECTION 2 - Technical Requirements Document (TRD)**

## **2.1 System Architecture**

**Architecture Overview**

Client → FastAPI (:8080) │ ├─ POST /ingest → DocLoader → Chunker → BGE Embedder (CUDA) → ChromaDB + BM25 Index │ └─ POST /query → Dense Search (ChromaDB) + Sparse Search (BM25) → RRF Fusion → Cross-Encoder Reranker (CUDA) → Gemini 1.5 Flash API → Answer + Citations → Response

## **2.2 Component Specifications**

### **2.2.1 Document Ingestion**

| **Component**     | **Specification**                                                  |
| ----------------- | ------------------------------------------------------------------ |
| Chunking          | RecursiveCharacterTextSplitter - 512 tokens, 64 overlap            |
| Chunk metadata    | source filename, chunk_index, page_number, char_offset             |
| Supported formats | PDF (pdfminer.six), TXT, HTML (BeautifulSoup4), DOCX (python-docx) |
| Embedding model   | BAAI/bge-base-en-v1.5 - 768-dim, normalized, runs on GTX 1650      |
| Batch size        | 32 chunks per embed call                                           |
| BM25 persistence  | Rebuilt + pickled to /data/bm25_index.pkl after each ingest        |
| VRAM usage        | ~900MB for BGE embedder on GTX 1650                                |

### **2.2.2 Hybrid Retrieval**

| **Component**               | **Specification**                                       |
| --------------------------- | ------------------------------------------------------- |
| Dense retrieval             | ChromaDB cosine similarity, top-15 candidates           |
| Sparse retrieval            | BM25Okapi (rank-bm25), top-15 candidates                |
| Fusion                      | Reciprocal Rank Fusion (RRF) - k=60                     |
| Reranker                    | cross-encoder/ms-marco-MiniLM-L-6-v2 - runs on GTX 1650 |
| VRAM usage                  | ~400MB for reranker                                     |
| Total VRAM (embed + rerank) | ~1.3GB of 4GB - 2.7GB free                              |
| Final top-k                 | Default 5, configurable via TOP_K_DEFAULT env var       |
| Latency budget (local)      | Dense ≤300ms, Sparse ≤50ms, Rerank ≤500ms               |

### **2.2.3 Generation - Gemini 1.5 Flash**

| **Component**       | **Specification**                                                    |
| ------------------- | -------------------------------------------------------------------- |
| Model               | gemini-1.5-flash (free tier)                                         |
| Free quota          | 15 requests/min · 1500 requests/day · 1M tokens/min                  |
| Context window      | 1,000,000 tokens - fits entire retrieval context easily              |
| Max output tokens   | 1024                                                                 |
| Prompt strategy     | Numbered context blocks + strict citation instruction                |
| Citation format     | Inline \[N\] referencing context block index                         |
| Hallucination guard | Returns "Not found in documents" if ungroundable                     |
| Fallback            | Ollama + Phi-3 mini if offline (switchable via LLM_PROVIDER env var) |
| VRAM usage          | 0MB - Gemini runs on Google servers                                  |

## **2.3 Hardware Utilization (GTX 1650)**

| **Process**            | **VRAM**     | **Runs On**  | **Notes**                           |
| ---------------------- | ------------ | ------------ | ----------------------------------- |
| BGE Embedder           | ~900MB       | GPU (CUDA)   | Batch=32, ~150ms/batch              |
| Cross-Encoder Reranker | ~400MB       | GPU (CUDA)   | ~300ms for top-20 pairs             |
| ChromaDB               | 0MB          | CPU + RAM    | In-memory index, volume-persisted   |
| BM25                   | ~50MB RAM    | CPU          | Pickle-persisted, rebuilt on ingest |
| FastAPI + Python       | ~200MB RAM   | CPU          | Async, single worker for local      |
| Gemini 1.5 Flash       | 0MB          | Google Cloud | HTTP call, ~1-2s round trip         |
| TOTAL VRAM USED        | ~1.3GB / 4GB |              | 2.7GB headroom - comfortable        |

## **2.4 API Contract**

### **POST /ingest**

Request: multipart/form-data - file (PDF, TXT, HTML, DOCX)

Response 200: { "chunks_indexed": int, "source": str, "duration_ms": int }

Response 422: { "detail": str } - unsupported file type

### **POST /query**

Request: { "query": str, "top_k": int = 5 }

Response 200: { "answer": str, "sources": \[str\], "chunks": \[{"text": str, "source": str, "score": float}\], "latency_ms": int }

### **GET /health**

Response 200: { "status": "ok", "chromadb": "ok"|"error", "bm25_chunks": int, "gemini": "ok"|"error" }

## **2.5 Non-Functional Requirements**

| **Category**  | **Requirement**                                                       |
| ------------- | --------------------------------------------------------------------- |
| Cost          | \$0 per month - all components free tier or local                     |
| Performance   | p95 query latency < 5s on local hardware (Ryzen 5 + 1650)             |
| Persistence   | ChromaDB volume-mounted; BM25 pickled to /data; survives restart      |
| Privacy       | All document data stays local - only query+context sent to Gemini API |
| Portability   | Docker-compose on Windows (WSL2 backend) - single command             |
| GPU           | CUDA 11+ required for BGE + Reranker; falls back to CPU if no GPU     |
| Testing       | Unit tests for chunker, RRF, reranker; integration test for /query    |
| Observability | Structured JSON logging; /health with component status                |

# **SECTION 3 - Tech Stack (Zero Cost)**

## **3.1 Stack Decision Matrix**

| **Layer**        | **Technology**           | **Version** | **Cost** | **Rationale**                             |
| ---------------- | ------------------------ | ----------- | -------- | ----------------------------------------- |
| LLM              | Gemini 1.5 Flash         | latest      | FREE     | 1500 req/day free, 1M ctx, no card needed |
| LLM Fallback     | Ollama + Phi-3 mini      | 3.x / 3.8B  | FREE     | Offline fallback, fits in 2.3GB VRAM      |
| Embeddings       | BAAI/bge-base-en-v1.5    | -           | FREE     | Best open embedding at 768-dim, CUDA      |
| Reranker         | ms-marco-MiniLM-L-6-v2   | -           | FREE     | Cross-encoder, fast on GTX 1650           |
| Vector Store     | ChromaDB                 | 0.5         | FREE     | Zero-config local + HTTP server mode      |
| Sparse Retrieval | rank-bm25                | 0.2         | FREE     | Pure Python BM25Okapi, no infra           |
| ML Runtime       | sentence-transformers    | 3.0         | FREE     | Handles both embed + rerank models        |
| PDF Parsing      | pdfminer.six             | 20221105    | FREE     | Reliable text + page number extraction    |
| HTML Parsing     | BeautifulSoup4           | 4.12        | FREE     | Standard, robust                          |
| DOCX Parsing     | python-docx              | 1.1         | FREE     | Clean paragraph extraction                |
| Chunking         | langchain-text-splitters | 0.2         | FREE     | RecursiveCharacterTextSplitter            |
| API Framework    | FastAPI                  | 0.111       | FREE     | Async, auto OpenAPI, Pydantic v2          |
| ASGI Server      | Uvicorn                  | 0.30        | FREE     | Production-grade ASGI                     |
| Validation       | Pydantic                 | v2          | FREE     | Request/response models + env config      |
| CUDA Runtime     | PyTorch + CUDA 11        | 2.3         | FREE     | GPU acceleration for embed+rerank         |
| Containerization | Docker Desktop + Compose | 26/2.27     | FREE     | WSL2 backend on Windows                   |
| Eval             | RAGAS                    | 0.1         | FREE     | Faithfulness, relevance, recall metrics   |
| Testing          | pytest + httpx           | 8.x         | FREE     | Async test client for FastAPI             |

## **3.2 LLM Provider Comparison (Why Gemini Flash)**

| **Provider**     | **Model**         | **Free Tier** | **Quality** | **Offline** | **Verdict**  |
| ---------------- | ----------------- | ------------- | ----------- | ----------- | ------------ |
| Google AI Studio | gemini-1.5-flash  | 1500 req/day  | ★★★★        | No          | ✓ CHOSEN     |
| Ollama local     | phi-3 mini        | Unlimited     | ★★★         | Yes         | Fallback     |
| Ollama local     | gemma 2B          | Unlimited     | ★★          | Yes         | Fallback     |
| Anthropic        | claude-sonnet-4-6 | None          | ★★★★★       | No          | Paid - skip  |
| OpenAI           | gpt-4o            | None          | ★★★★★       | No          | Paid - skip  |
| Moonshot         | moonshot-v1-8k    | Limited       | ★★★★        | No          | High latency |

## **3.3 Docker Compose Topology**

**Services**

api → FastAPI on :8080 (builds from Dockerfile) chromadb → ChromaDB server on :8000 (volume: chroma_data) ollama → Ollama server on :11434 (volume: ollama_data, GPU passthrough) Environment variables injected from .env file. LLM_PROVIDER=gemini → uses Gemini Flash API LLM_PROVIDER=ollama → uses local Phi-3 mini (fully offline)

# **SECTION 4 - Implementation Plan**

## **4.1 Phases Overview**

| **Phase** | **Name**           | **Days** | **Deliverable**                          | **Cost** |
| --------- | ------------------ | -------- | ---------------------------------------- | -------- |
| 0         | Environment Setup  | 1        | Repo, Docker, WSL2, CUDA verified        | \$0      |
| 1         | Ingestion Pipeline | 3        | /ingest endpoint + ChromaDB + BM25       | \$0      |
| 2         | Hybrid Retrieval   | 3        | Dense + Sparse + RRF + Reranker          | \$0      |
| 3         | Gemini Generation  | 2        | /query with grounded answers + citations | \$0      |
| 4         | Ollama Fallback    | 1        | Offline mode via Phi-3 mini              | \$0      |
| 5         | Hardening          | 2        | Logging, /health, error handling, retry  | \$0      |
| 6         | Evaluation         | 2        | RAGAS suite, chunk/k tuning              | \$0      |
| 7         | Docs + Cleanup     | 1        | README, .env.example, final test         | \$0      |
| TOTAL     |                    | 15 days  | Production-ready RAG pipeline            | \$0      |

## **4.2 Detailed Phase Breakdown**

### **Phase 0 - Environment Setup (Day 1)**

• Enable WSL2 on Windows - required for Docker GPU passthrough.

• Install Docker Desktop - enable WSL2 backend + NVIDIA Container Toolkit.

• Verify CUDA: run nvidia-smi inside a Docker container.

• Init repo structure: /backend, /tests, /eval, /docker, /data.

• Write .env.example with all required variables.

• Write docker-compose.yml with api + chromadb + ollama services.

• Write Dockerfile - python:3.11-slim + CUDA base.

• GitHub Actions CI: ruff lint + mypy + pytest on push.

### **Phase 1 - Ingestion Pipeline (Days 2-4)**

• Document loaders: pdf_loader.py, txt_loader.py, html_loader.py, docx_loader.py.

• chunker.py - RecursiveCharacterTextSplitter, 512/64, returns list\[Chunk\] with metadata.

• embedder.py - BGE-base-en-v1.5, device=cuda, batch=32, normalize=True.

• dense.py - ChromaDB HttpClient, upsert() + dense_search() returning scored hits.

• sparse.py - BM25Store.build() + search() + pickle persist/load.

• POST /ingest - loader → chunker → embed → upsert → BM25 rebuild → response.

• Unit tests: test_chunker.py, test_ingest.py.

### **Phase 2 - Hybrid Retrieval (Days 5-7)**

• dense_search() and bm25_store.search() both return top-15 with scores.

• RRF fusion in retrieval/\__init_\_.py - dedup by text\[:80\], k=60.

• reranker.py - CrossEncoder ms-marco-MiniLM-L-6-v2, device=cuda.

• hybrid_search() entry: dense + sparse → RRF → rerank → top_k.

• Log sub-step latency - verify embed ≤300ms, rerank ≤500ms on GTX 1650.

• Unit tests: test_rrf.py, test_reranker.py.

### **Phase 3 - Gemini Generation + Citations (Days 8-9)**

• pip install google-generativeai - add to requirements.txt.

• generator.py - genai.GenerativeModel("gemini-1.5-flash"), numbered context prompt.

• Citation parser - extracts \[N\] from answer, maps to source filenames.

• POST /query - hybrid_search → generate → {answer, sources, chunks, latency_ms}.

• Rate limit handler - 429 from Gemini → wait 4s → retry up to 3x.

• Integration test: ingest fixture docs → query → assert \[1\] citation in answer.

### **Phase 4 - Ollama Fallback (Day 10)**

• docker-compose ollama service with NVIDIA GPU passthrough.

• ollama_generator.py - OpenAI-compat client, base_url=<http://ollama:11434/v1>.

• LLM_PROVIDER env var: "gemini" → Gemini Flash, "ollama" → Phi-3 mini.

• provider_factory.py - returns correct generator based on env var.

• Test: switch LLM_PROVIDER=ollama, verify /query still returns grounded answer.

• ollama model pull script: scripts/pull_models.sh → ollama pull phi3:mini.

### **Phase 5 - Hardening (Days 11-12)**

• Structured JSON logging via structlog - query, latency, chunk_count per request.

• GET /health - checks ChromaDB connection, BM25 chunk count, Gemini API ping.

• Global 500 handler - returns {"error": str}, never raw traceback.

• Retry logic on embed calls - tenacity, max 3 attempts, exponential backoff.

• Input validation - max file size 50MB, allowed extensions enforced.

• BM25 index auto-loads from pickle on startup if /data/bm25_index.pkl exists.

### **Phase 6 - Evaluation (Days 13-14)**

• eval/dataset.py - 50 question/answer/source triples from real test documents.

• RAGAS metrics: answer_relevancy, faithfulness, context_precision, context_recall.

• Grid search: chunk_size in {256, 512, 1024} × top_k in {3, 5, 10}.

• Target: RAGAS faithfulness ≥ 0.80, answer_relevancy ≥ 0.75.

• Save results to eval/results/YYYY-MM-DD.json - track regression.

• Tune chunk_size and top_k to hit targets, update defaults in .env.example.

### **Phase 7 - Docs + Cleanup (Day 15)**

• README.md - quickstart (5 steps), env vars table, API reference.

• .env.example with all variables and inline comments.

• scripts/pull_models.sh - one-shot Ollama model download.

• Final integration test run - all endpoints green.

• git tag v2.0.0 - zero cost edition.

## **4.3 Milestone Calendar**

| **Day** | **Milestone**                                    | **Status** |
| ------- | ------------------------------------------------ | ---------- |
| 1       | WSL2 + Docker + CUDA verified, repo scaffolded   | Planned    |
| 4       | /ingest endpoint passing unit tests              | Planned    |
| 7       | Hybrid search benchmarked on GTX 1650            | Planned    |
| 9       | /query returning grounded answers with citations | Planned    |
| 10      | Ollama fallback working offline                  | Planned    |
| 12      | Hardening + /health endpoint complete            | Planned    |
| 14      | RAGAS targets met, chunk/k tuned                 | Planned    |
| 15      | v2.0.0 tagged, README shipped                    | Planned    |

## **4.4 Windows-Specific Setup Notes**

**Windows WSL2 + Docker + CUDA**

1\. Enable WSL2: wsl --install in PowerShell (Admin) 2. Install NVIDIA driver for Windows (not inside WSL) 3. Install Docker Desktop → Settings → Use WSL2 backend 4. Install NVIDIA Container Toolkit inside WSL2: curl -fsSL <https://nvidia.github.io/libnvidia-container/gpgkey> | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 5. Verify: docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi 6. Should show GTX 1650 - you are ready.

## **4.5 Risks & Mitigations**

| **Risk**                                | **Likelihood** | **Impact** | **Mitigation**                                                                |
| --------------------------------------- | -------------- | ---------- | ----------------------------------------------------------------------------- |
| Gemini 1500/day quota exhausted         | Low            | Medium     | Rate-limit logger warns at 1200; fallback to Ollama automatically             |
| GTX 1650 VRAM OOM during embed+rerank   | Low            | High       | BGE and reranker share ~1.3GB - 2.7GB headroom; if OOM, drop batch_size to 16 |
| Docker GPU passthrough fails on Windows | Medium         | High       | Phase 0 verifies this first; CPU fallback always available                    |
| BM25 index lost on restart              | Low            | Medium     | Auto-persists to /data volume; auto-reloads on startup                        |
| Gemini API latency spike (network)      | Medium         | Low        | 3s timeout → retry → if all fail, return top chunks without generation        |
| Chunk boundary breaks mid-sentence      | High           | Low        | 64-token overlap + sentence-aware splitter minimizes this                     |

# **SECTION 5 - Appendix**

## **5.1 Environment Variables**

| **Variable**     | **Required**    | **Default**           | **Description**                                  |
| ---------------- | --------------- | --------------------- | ------------------------------------------------ |
| GEMINI_API_KEY   | Yes (if gemini) | -                     | Get free at aistudio.google.com - no card needed |
| LLM_PROVIDER     | No              | gemini                | Options: gemini \| ollama                        |
| OLLAMA_HOST      | No              | <http://ollama:11434> | Ollama server URL inside Docker network          |
| OLLAMA_MODEL     | No              | phi3:mini             | Ollama model to use for generation               |
| CHROMA_HOST      | No              | chromadb              | ChromaDB hostname                                |
| CHROMA_PORT      | No              | 8000                  | ChromaDB port                                    |
| TOP_K_DEFAULT    | No              | 5                     | Default reranked chunks per query                |
| EMBED_BATCH_SIZE | No              | 32                    | Chunks per embedding batch (reduce to 16 if OOM) |
| EMBED_DEVICE     | No              | cuda                  | cuda \| cpu - auto-detected if unset             |
| LOG_LEVEL        | No              | INFO                  | DEBUG \| INFO \| WARNING                         |
| MAX_FILE_SIZE_MB | No              | 50                    | Max upload size in MB                            |

## **5.2 Quickstart (5 Steps)**

• **Step 1** git clone the repo and cd into it.

• **Step 2** Copy .env.example to .env - paste your GEMINI_API_KEY from aistudio.google.com.

• **Step 3** docker compose up --build (first run pulls images + downloads ML models ~2GB).

• **Step 4** Ingest a doc: curl -X POST <http://localhost:8080/ingest> -F "file=@your_doc.pdf"

• **Step 5** Query it: curl -X POST <http://localhost:8080/query> -H "Content-Type: application/json" -d '{"query": "what is the refund policy?"}'

## **5.3 Glossary**

| **Term**         | **Definition**                                                                      |
| ---------------- | ----------------------------------------------------------------------------------- |
| RAG              | Retrieval-Augmented Generation - grounding LLM output in retrieved documents        |
| RRF              | Reciprocal Rank Fusion - score fusion formula: 1/(k+rank), k=60                     |
| Dense retrieval  | Semantic similarity via embedding vectors (cosine distance in ChromaDB)             |
| Sparse retrieval | Term-frequency matching via BM25 (Best Match 25 algorithm)                          |
| Reranker         | Cross-encoder that scores (query, passage) pairs - higher precision than bi-encoder |
| BGE              | BAAI General Embedding - open-source embedding model family from Beijing AI         |
| RAGAS            | RAG Assessment framework - measures faithfulness, relevance, context recall         |
| Chunk            | Fixed-size text segment with metadata - the atomic unit of retrieval                |
| WSL2             | Windows Subsystem for Linux 2 - required for Docker GPU passthrough on Windows      |
| CUDA             | NVIDIA parallel compute platform - enables GPU acceleration for ML models           |

_RAG Pipeline v2.0 - Zero Cost Edition · \$0/month · All rights reserved_