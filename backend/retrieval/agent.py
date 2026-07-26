from backend.retrieval.fusion import hybrid_search
from backend.generation.provider_factory import get_generator
from typing import Dict, Any, Tuple, List
import structlog
import re

logger = structlog.get_logger()

MAX_RETRIES = 2

def agentic_query(query: str, top_k: int = 5, session_history: str = "") -> Dict[str, Any]:
    """
    Self-correcting RAG pipeline:
    1. Retrieve + Generate
    2. Check if answer is grounded
    3. If not, rewrite query and retry
    """
    generator = get_generator()
    attempt = 0
    current_query = query
    all_chunks = []
    
    while attempt <= MAX_RETRIES:
        # Retrieve
        retrieval_res = hybrid_search(current_query, top_k)
        chunks = retrieval_res["chunks"]
        all_chunks = chunks
        
        # Generate
        answer, sources = generator.generate(query, chunks)  # Always use original query
        
        # Check groundedness: did the LLM refuse or give a generic answer?
        is_grounded = _check_groundedness(answer)
        
        if is_grounded or attempt >= MAX_RETRIES:
            return {
                "answer": answer,
                "sources": sources,
                "chunks": chunks,
                "latency_ms": retrieval_res["latency_ms"],
                "attempts": attempt + 1,
                "final_query": current_query
            }
        
        # Rewrite query for next attempt
        logger.info("Answer not grounded, rewriting query", attempt=attempt)
        current_query = _rewrite_query(query, chunks, generator)
        attempt += 1
    
    return {
        "answer": answer,
        "sources": sources,
        "chunks": all_chunks,
        "latency_ms": retrieval_res["latency_ms"],
        "attempts": attempt,
        "final_query": current_query
    }

def _check_groundedness(answer: str) -> bool:
    """Simple heuristic: check if the answer is a refusal or too short."""
    refusal_phrases = [
        "not found in documents",
        "i don't have enough information",
        "the provided context does not",
        "no relevant information"
    ]
    answer_lower = answer.lower().strip()
    
    if any(phrase in answer_lower for phrase in refusal_phrases):
        return False
    if len(answer_lower) < 30:
        return False
    # Check if citations exist
    if not re.findall(r'\[\d+\]', answer):
        return False
    return True

def _rewrite_query(original_query: str, failed_chunks: list, generator) -> str:
    """Use the LLM to rewrite the query for better retrieval."""
    chunk_summaries = "\n".join([c["text"][:100] for c in failed_chunks[:3]])
    
    rewrite_prompt = f"""The following search query did not return useful results:
Query: {original_query}

The retrieved context was about:
{chunk_summaries}

Rewrite the query to find better results. Use different keywords, synonyms, or rephrase the question.
Return ONLY the rewritten query, nothing else."""
    
    try:
        # Use a simple generate call
        answer, _ = generator.generate(rewrite_prompt, [])
        # The generator will return "Not found in documents" since chunks is empty
        # So we need to call the model directly
        import google.generativeai as genai
        from backend.core.config import settings
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        response = model.generate_content(rewrite_prompt)
        return response.text.strip()
    except Exception:
        # Fallback: just add "explain" to the query
        return f"explain {original_query}"
