from typing import List, Dict, Any
from backend.retrieval.dense import ChromaDBStore
from backend.retrieval.sparse import BM25Store
from backend.retrieval.reranker import Reranker
from backend.ingestion.embedder import embed_texts
import time
import structlog

logger = structlog.get_logger()

def reciprocal_rank_fusion(dense_hits: List[Dict[str, Any]], sparse_hits: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Combines dense and sparse results using Reciprocal Rank Fusion.
    """
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}
    
    # Process dense
    for rank, doc in enumerate(dense_hits):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Process sparse
    for rank, doc in enumerate(sparse_hits):
        doc_id = doc["id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Sort by fused score
    fused_hits = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [doc_map[doc_id] for doc_id, _ in fused_hits]

def hybrid_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Executes the full retrieval pipeline:
    1. Embed query
    2. Dense search (ChromaDB)
    3. Sparse search (BM25)
    4. RRF fusion
    5. Cross-Encoder reranking
    """
    start_time = time.time()
    
    # 1. Embed Query
    query_embedding = embed_texts([query])[0]
    embed_ms = int((time.time() - start_time) * 1000)
    
    # 2. Dense Search
    dense_start = time.time()
    chroma = ChromaDBStore.get_instance()
    dense_hits = chroma.dense_search(query_embedding, top_k=15)
    dense_ms = int((time.time() - dense_start) * 1000)
    
    # 3. Sparse Search
    sparse_start = time.time()
    bm25 = BM25Store.get_instance()
    sparse_hits = bm25.search(query, top_k=15)
    sparse_ms = int((time.time() - sparse_start) * 1000)
    
    # 4. RRF Fusion
    fusion_start = time.time()
    fused_candidates = reciprocal_rank_fusion(dense_hits, sparse_hits)
    fusion_ms = int((time.time() - fusion_start) * 1000)
    
    # 5. Reranking
    rerank_start = time.time()
    reranker = Reranker.get_instance()
    final_hits = reranker.rerank(query, fused_candidates, top_k)
    rerank_ms = int((time.time() - rerank_start) * 1000)
    
    total_ms = int((time.time() - start_time) * 1000)
    
    logger.info("Hybrid search complete", 
                dense_hits=len(dense_hits), 
                sparse_hits=len(sparse_hits), 
                fused_candidates=len(fused_candidates),
                total_ms=total_ms,
                embed_ms=embed_ms,
                dense_ms=dense_ms,
                sparse_ms=sparse_ms,
                rerank_ms=rerank_ms)
                
    return {
        "chunks": final_hits,
        "latency_ms": total_ms
    }
