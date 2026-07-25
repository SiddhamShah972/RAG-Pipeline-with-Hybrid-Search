import os
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

DEVICE = os.getenv("EMBED_DEVICE", "cuda")

class Reranker:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Load the CrossEncoder model
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []
            
        # CrossEncoder expects pairs of (query, document_text)
        pairs = [[query, doc["text"]] for doc in candidates]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Attach scores and sort
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[i])
            
        # Sort descending by rerank_score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
