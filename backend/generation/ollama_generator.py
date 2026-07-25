from typing import List, Dict, Any, Tuple
from backend.core.config import settings
import re
from ollama import Client

class OllamaGenerator:
    def __init__(self):
        self.client = Client(host=settings.OLLAMA_HOST)
        self.model = settings.OLLAMA_MODEL

    def generate(self, query: str, chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Generates an answer using local Ollama model (Phi-3 mini).
        """
        if not chunks:
            return "Not found in documents.", []

        context_blocks = []
        for i, chunk in enumerate(chunks):
            context_blocks.append(f"[{i+1}] Source: {chunk['metadata'].get('source', 'unknown')}\n{chunk['text']}")
            
        context_str = "\n\n".join(context_blocks)

        prompt = f"""Answer the question based strictly on the context. If not found, say "Not found in documents." Cite using [N].

Context:
{context_str}

Question:
{query}

Answer:"""

        response = self.client.chat(model=self.model, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        
        answer = response['message']['content']
        
        citations = set(map(int, re.findall(r"\[(\d+)\]", answer)))
        sources = set()
        for c in citations:
            if 1 <= c <= len(chunks):
                sources.add(chunks[c-1]["metadata"].get("source", "unknown"))
                
        return answer, list(sources)
