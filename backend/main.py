from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel  # pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse  # pyrefly: ignore [missing-import]
import uuid
import time
from backend.ingestion.loaders import load_document
from backend.ingestion.chunker import chunk_document
from backend.ingestion.embedder import embed_texts
from backend.retrieval.dense import ChromaDBStore
from backend.retrieval.sparse import BM25Store
from backend.retrieval.fusion import hybrid_search
from backend.generation.provider_factory import get_generator
from backend.core.config import settings
from backend.ingestion.visual_extractor import extract_visual_chunks
from backend.ingestion.table_extractor import extract_tables_from_pdf
from backend.core.memory import ConversationMemory
from backend.retrieval.knowledge_graph import KnowledgeGraph
from backend.retrieval.agent import agentic_query
from fastapi.responses import StreamingResponse  # pyrefly: ignore [missing-import]
from backend.generation.streaming import stream_gemini_response
import structlog  # pyrefly: ignore [missing-import]
from typing import Optional
import shutil
import os
import tempfile

logger = structlog.get_logger()
app = FastAPI(title="Zero Cost RAG Pipeline")

@app.post("/ingest")
def ingest_document(file: UploadFile = File(...)):
    start_time = time.time()
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        # 1. Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_size = os.path.getsize(temp_file_path)
        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB}MB")

        # 2. Extract Text
        try:
            text = load_document(temp_file_path, file.content_type)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to process file: {str(e)}")

        if not text.strip():
            raise HTTPException(status_code=422, detail="Extracted text is empty")

        # 3. Chunk Text
        chunks = chunk_document(text, file.filename)
        chunk_texts = [c.text for c in chunks]
        
        # 4. Embed Chunks
        embeddings = embed_texts(chunk_texts)
        
        # 5. Store in ChromaDB (Dense)
        chroma = ChromaDBStore.get_instance()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [c.metadata for c in chunks]
        
        chroma.upsert(ids, embeddings, chunk_texts, metadatas)
        
        # 6. Store in BM25 (Sparse)
        bm25 = BM25Store.get_instance()
        sparse_chunks = []
        for i in range(len(chunks)):
            sparse_chunks.append({
                "id": ids[i],
                "text": chunk_texts[i],
                "metadata": metadatas[i]
            })

        # 4b. Extract visual content (images/diagrams) — PDF only
        visual_chunks = []
        table_chunks = []
        ext = file.filename.split('.')[-1].lower()
        if ext == 'pdf':
            visual_chunks = extract_visual_chunks(temp_file_path, file.filename)
            table_chunks = extract_tables_from_pdf(temp_file_path, file.filename)

        # 4c. Embed visual + table chunks
        extra_texts = [vc["text"] for vc in visual_chunks] + [tc["text"] for tc in table_chunks]
        extra_metadatas = [vc["metadata"] for vc in visual_chunks] + [tc["metadata"] for tc in table_chunks]
        if extra_texts:
            extra_embeddings = embed_texts(extra_texts)
            extra_ids = [str(uuid.uuid4()) for _ in extra_texts]
            chroma.upsert(extra_ids, extra_embeddings, extra_texts, extra_metadatas)
            # Add to BM25
            for j in range(len(extra_texts)):
                sparse_chunks.append({
                    "id": extra_ids[j],
                    "text": extra_texts[j],
                    "metadata": extra_metadatas[j]
                })

        bm25.build(sparse_chunks)

        # 7. Build Knowledge Graph (async-safe — runs in background)
        kg = KnowledgeGraph.get_instance()
        kg_chunks = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        kg.extract_and_add(kg_chunks)

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info("Ingestion complete", filename=file.filename, chunks=len(chunks), duration_ms=duration_ms)
        
        return {
            "chunks_indexed": len(chunks),
            "visual_chunks": len(visual_chunks),
            "table_chunks": len(table_chunks),
            "source": file.filename,
            "duration_ms": duration_ms
        }
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

class QueryRequest(BaseModel):
    query: str
    top_k: int = settings.TOP_K_DEFAULT
    session_id: Optional[str] = None

@app.post("/session")
def create_session():
    memory = ConversationMemory.get_instance()
    session_id = memory.create_session()
    return {"session_id": session_id}

@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        memory = ConversationMemory.get_instance()
        effective_query = request.query

        if request.session_id:
            history = memory.get_history(request.session_id, last_n=4)
            if history:
                context_hint = " ".join([
                    h["content"][:100] for h in history if h["role"] == "user"
                ])
                effective_query = f"{context_hint} {request.query}"

        # 1. Agentic Retrieval & Generation
        result = agentic_query(effective_query, request.top_k)
        
        if request.session_id:
            memory.add_turn(request.session_id, "user", request.query)
            memory.add_turn(request.session_id, "assistant", result["answer"])
                    
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks": result["chunks"],
            "latency_ms": result["latency_ms"],
            "attempts": result["attempts"]
        }
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during query processing")

@app.post("/query/stream")
def query_documents_stream(request: QueryRequest):
    """SSE streaming endpoint for real-time token output."""
    retrieval_res = hybrid_search(request.query, request.top_k)
    chunks = retrieval_res["chunks"]
    
    def event_stream():
        # First, send the chunks metadata
        import json
        yield f"data: {json.dumps({'type': 'chunks', 'data': chunks})}\n\n"
        
        # Then stream tokens
        for token in stream_gemini_response(request.query, chunks):
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    chroma_status = "ok"
    bm25_chunks = 0
    
    try:
        chroma = ChromaDBStore.get_instance()
        chroma.client.heartbeat()
    except Exception:
        chroma_status = "error"
        
    try:
        bm25 = BM25Store.get_instance()
        bm25_chunks = bm25.get_chunk_count()
    except Exception:
        pass
        
    return {
        "status": "ok",
        "chromadb": chroma_status,
        "bm25_chunks": bm25_chunks,
        "gemini": "ok" if settings.LLM_PROVIDER == "gemini" else "N/A"
    }

@app.get("/documents")
def list_documents():
    """List all ingested documents."""
    chroma = ChromaDBStore.get_instance()
    # Get all unique sources from ChromaDB metadata
    results = chroma.collection.get(include=["metadatas"])
    sources = set()
    for meta in results["metadatas"]:
        if meta:
            sources.add(meta.get("source", "unknown"))
    return {"documents": list(sources), "total": len(sources)}

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete all chunks for a specific document."""
    chroma = ChromaDBStore.get_instance()
    # Find all IDs with this source
    results = chroma.collection.get(
        where={"source": filename},
        include=["metadatas"]
    )
    if results["ids"]:
        chroma.collection.delete(ids=results["ids"])
    
    # Rebuild BM25 without this document
    from backend.retrieval.sparse import _tokenize
    from rank_bm25 import BM25Okapi  # pyrefly: ignore [missing-import]
    bm25 = BM25Store.get_instance()
    bm25.corpus = [c for c in bm25.corpus if c["metadata"].get("source") != filename]
    if bm25.corpus:
        tokenized = [_tokenize(c["text"]) for c in bm25.corpus]
        bm25.bm25 = BM25Okapi(tokenized)
    else:
        bm25.bm25 = None
    bm25.save()
    
    return {"deleted": filename, "chunks_removed": len(results["ids"])}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )

# Mount Frontend UI
from fastapi.staticfiles import StaticFiles  # pyrefly: ignore [missing-import]
import os

ui_path = os.path.join(os.path.dirname(__file__), "ui")
if os.path.exists(ui_path):
    app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")


