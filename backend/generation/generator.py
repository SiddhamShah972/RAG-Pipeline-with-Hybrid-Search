import google.generativeai as genai
from typing import List, Dict, Any, Tuple
from backend.core.config import settings
import re
import tenacity

class GeminiGenerator:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("models/gemini-3.5-flash")

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, query: str, chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Generates an answer using Gemini 1.5 Flash based on provided context chunks.
        Returns (answer, list of sources used)
        """
        if not chunks:
            return "Not found in documents.", []

        # Build context
        context_blocks = []
        for i, chunk in enumerate(chunks):
            # 1-indexed for citations
            context_blocks.append(f"[{i+1}] Source: {chunk['metadata'].get('source', 'unknown')}\n{chunk['text']}")
            
        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are an expert internal assistant. Answer the user's question based strictly on the provided context blocks.
If the answer is not contained in the context, reply exactly with: "Not found in documents."
Do not hallucinate or use outside knowledge.
For every claim you make, you MUST cite the source using the inline format [N] where N is the context block number.

Context Blocks:
{context_str}

User Question:
{query}

Answer:"""

        response = self.model.generate_content(prompt)
        answer = response.text
        
        # Extract citations
        citations = set(map(int, re.findall(r"\[(\d+)\]", answer)))
        
        # Map citations to sources
        sources = set()
        for c in citations:
            if 1 <= c <= len(chunks):
                sources.add(chunks[c-1]["metadata"].get("source", "unknown"))
                
        return answer, list(sources)
