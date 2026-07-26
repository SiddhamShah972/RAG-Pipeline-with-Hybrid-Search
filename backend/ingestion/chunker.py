from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from backend.core.config import settings

class Chunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

import re

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
        page_match = re.findall(r'\[PAGE (\d+)\]', chunk_text)
        page_num = int(page_match[-1]) if page_match else None
        clean_text = re.sub(r'\[PAGE \d+\]\n?', '', chunk_text).strip()
        
        # Build contextual header for better retrieval
        context_parts = [f"Document: {filename}"]
        if page_num:
            context_parts.append(f"Page: {page_num}")
        context_parts.append(f"Chunk: {i+1}/{len(chunks_text)}")
        context_header = " | ".join(context_parts)
        
        enriched_text = f"[{context_header}]\n{clean_text}"
        
        metadata = {
            "source": filename,
            "chunk_index": i,
            "chunk_type": "text"
        }
        if page_num is not None:
            metadata["page_number"] = page_num
            
        chunks.append(Chunk(text=enriched_text, metadata=metadata))
        
    return chunks
