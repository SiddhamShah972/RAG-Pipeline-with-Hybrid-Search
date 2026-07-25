from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
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
import structlog
import shutil
import os

logger = structlog.get_logger()
app = FastAPI(title="Zero Cost RAG Pipeline")

@app.post("/ingest")
def ingest_document(file: UploadFile = File(...)):
    start_time = time.time()
    temp_file_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    
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
        bm25.build(sparse_chunks)

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info("Ingestion complete", filename=file.filename, chunks=len(chunks), duration_ms=duration_ms)
        
        return {
            "chunks_indexed": len(chunks),
            "source": file.filename,
            "duration_ms": duration_ms
        }
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

class QueryRequest(BaseModel):
    query: str
    top_k: int = settings.TOP_K_DEFAULT

@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        # 1. Retrieval
        retrieval_res = hybrid_search(request.query, request.top_k)
        chunks = retrieval_res["chunks"]
        
        # 2. Generation
        gen_start = time.time()
        generator = get_generator()
        answer, sources = generator.generate(request.query, chunks)
        gen_ms = int((time.time() - gen_start) * 1000)
        
        total_latency = retrieval_res["latency_ms"] + gen_ms
        
        logger.info("Query complete", 
                    latency_ms=total_latency, 
                    sources=len(sources))
                    
        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
            "latency_ms": total_latency
        }
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during query processing")

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

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )

# Mount Frontend UI
from fastapi.staticfiles import StaticFiles
import os

ui_path = os.path.join(os.path.dirname(__file__), "ui")
if os.path.exists(ui_path):
    app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")


