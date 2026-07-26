import google.generativeai as genai
from backend.core.config import settings
from typing import List, Dict, Any, Generator

def stream_gemini_response(query: str, chunks: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """Yields tokens one-by-one from Gemini streaming API."""
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(settings.GEMINI_MODEL)
    
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(
            f"[{i+1}] Source: {chunk['metadata'].get('source', 'unknown')}\n{chunk['text']}"
        )
    context_str = "\n\n".join(context_blocks)
    
    prompt = f"""You are an expert internal assistant. Answer based strictly on the context.
If not found, say "Not found in documents." Cite using [N].
If the retrieved context contains a table, or if the user asks for a table, you MUST present your answer as a properly formatted Markdown table.

Context Blocks:
{context_str}

Question: {query}

Answer:"""
    
    try:
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            yield "\n\n⚠️ **Rate Limit Exceeded:** You are using the Gemini Free Tier and have exceeded the limit of 15 requests per minute (likely because the massive document is still being indexed in the background). Please wait a minute before asking another question, or upgrade to a paid API key."
        else:
            yield f"\n\n⚠️ **API Error:** {error_msg}"
