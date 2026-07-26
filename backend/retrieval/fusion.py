from typing import List, Dict, Any
from backend.retrieval.dense import ChromaDBStore
from backend.retrieval.sparse import BM25Store
from backend.retrieval.reranker import Reranker
from backend.ingestion.embedder import embed_texts
from backend.retrieval.knowledge_graph import KnowledgeGraph
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
    
    # 6. Knowledge Graph Augmentation
    try:
        kg = KnowledgeGraph.get_instance()
        # Extract key entities from query (simple: use nouns/capitalized words)
        import re
        query_entities = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', query)
        if not query_entities:
            query_entities = query.split()[:3]  # Fallback: first 3 words
        
        graph_results = kg.query_graph(query_entities, max_hops=2)
        
        if graph_results:
            # Add graph context as additional chunks
            for gr in graph_results[:3]:
                graph_chunk = {
                    "id": f"kg_{gr['subject']}_{gr['object']}",
                    "text": f"[Knowledge Graph] {gr['subject']} {gr['relation']} {gr['object']}. Context: {gr['context']}",
                    "metadata": {"source": gr["source"], "chunk_type": "graph"},
                    "rerank_score": 0.5  # Moderate score
                }
                final_hits.append(graph_chunk)
    except Exception as e:
        logger.warning("KG augmentation failed", error=str(e))
    
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
        "latency_ms": total_ms,
        "trace": {
             "embed_ms": embed_ms,
             "dense_ms": dense_ms,
             "sparse_ms": sparse_ms,
             "fusion_ms": fusion_ms,
             "rerank_ms": rerank_ms,
             "dense_hits": len(dense_hits),
             "sparse_hits": len(sparse_hits),
             "fused_candidates": len(fused_candidates)
         }
    }
