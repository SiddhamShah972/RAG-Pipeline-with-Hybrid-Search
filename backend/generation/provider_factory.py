from backend.core.config import settings
from backend.generation.generator import GeminiGenerator
from backend.generation.ollama_generator import OllamaGenerator
import structlog

logger = structlog.get_logger()

_generator_cache = None

def get_generator():
    global _generator_cache
    if _generator_cache is not None:
        return _generator_cache

    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "gemini":
        logger.info("Using Gemini generator")
        _generator_cache = GeminiGenerator()
    elif provider == "ollama":
        logger.info("Using Ollama generator")
        _generator_cache = OllamaGenerator()
    else:
        logger.warning(f"Unknown provider {provider}, falling back to Gemini")
        _generator_cache = GeminiGenerator()
        
    return _generator_cache
