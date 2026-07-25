from backend.core.config import settings
from backend.generation.generator import GeminiGenerator
from backend.generation.ollama_generator import OllamaGenerator
import structlog

logger = structlog.get_logger()

def get_generator():
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "gemini":
        logger.info("Using Gemini generator")
        return GeminiGenerator()
    elif provider == "ollama":
        logger.info("Using Ollama generator")
        return OllamaGenerator()
    else:
        logger.warning(f"Unknown provider {provider}, falling back to Gemini")
        return GeminiGenerator()
