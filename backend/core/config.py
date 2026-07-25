from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    LLM_PROVIDER: str = "gemini"
    OLLAMA_HOST: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "phi3:mini"
    
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    
    TOP_K_DEFAULT: int = 5
    EMBED_BATCH_SIZE: int = 32
    EMBED_DEVICE: str = "cuda"
    
    LOG_LEVEL: str = "INFO"
    MAX_FILE_SIZE_MB: int = 50
    
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
