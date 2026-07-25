import os
import pickle
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

BM25_INDEX_PATH = "/app/data/bm25_index.pkl"

class BM25Store:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.bm25 = None
        self.corpus: List[Dict[str, Any]] = []
        self.load()

    def build(self, new_chunks: List[Dict[str, Any]]):
        """
        Rebuilds the BM25 index with new chunks and persists to disk.
        new_chunks: [{"id": str, "text": str, "metadata": dict}]
        """
        self.corpus.extend(new_chunks)
        tokenized_corpus = [chunk["text"].split(" ") for chunk in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump({
                "bm25": self.bm25,
                "corpus": self.corpus
            }, f)

    def load(self):
        if os.path.exists(BM25_INDEX_PATH):
            with open(BM25_INDEX_PATH, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.corpus = data["corpus"]

    def get_chunk_count(self) -> int:
        return len(self.corpus)

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.corpus:
            return []
            
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_n = min(top_k, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        
        hits = []
        for i in top_indices:
            hits.append({
                "id": self.corpus[i]["id"],
                "text": self.corpus[i]["text"],
                "metadata": self.corpus[i]["metadata"],
                "score": float(scores[i])
            })
        return hits
