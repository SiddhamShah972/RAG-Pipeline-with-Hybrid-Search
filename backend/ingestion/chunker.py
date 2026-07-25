from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from backend.core.config import settings

class Chunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

def chunk_document(text: str, filename: str) -> List[Chunk]:
    """
    Chunks a document text using RecursiveCharacterTextSplitter.
    512 tokens (chars approx for now), 64 overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks_text = splitter.split_text(text)
    
    chunks = []
    for i, chunk_text in enumerate(chunks_text):
        metadata = {
            "source": filename,
            "chunk_index": i
        }
        chunks.append(Chunk(text=chunk_text, metadata=metadata))
        
    return chunks
