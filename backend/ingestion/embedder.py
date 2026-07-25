import os
from sentence_transformers import SentenceTransformer
from typing import List

# Setup device and batch size based on env vars
DEVICE = os.getenv("EMBED_DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

class Embedder:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Load the BGE model. Will download on first run.
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEVICE)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts into vectors.
        """
        if not texts:
            return []
            
        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True, # Important for cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    embedder = Embedder.get_instance()
    return embedder.embed(texts)
