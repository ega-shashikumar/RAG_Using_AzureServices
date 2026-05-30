"""
app/core/config.py — All settings loaded from environment variables / .env file.
"""
from functools import lru_cache
from pydantic import AnyHttpUrl, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ────────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"

    # ── Azure OpenAI ───────────────────────────────────────────────────────
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str 
    azure_openai_chat_deployment: str 
    azure_openai_embedding_deployment: str 
    azure_openai_embedding_dimensions: int 

    # ── Azure AI Search ────────────────────────────────────────────────────
    azure_search_endpoint: str
    azure_search_api_key: str
    azure_search_index_name: str

    # ── Azure Blob Storage ─────────────────────────────────────────────────
    azure_storage_connection_string: str
    azure_storage_container_name: str 

    # ── Azure Cosmos DB ────────────────────────────────────────────────────
    azure_cosmos_endpoint: str
    azure_cosmos_key: str
    azure_cosmos_database: str 
    azure_cosmos_container: str

    # ── RAG Tuning ─────────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0)
    top_k_retrieval: int = Field(default=5, ge=1, le=20)
    semantic_ranker_enabled: bool = True
    max_tokens_response: int = Field(default=2048, ge=256, le=16384)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    system_prompt: str = (
        "You are a helpful assistant. Answer only from the provided context. "
        "If the context does not contain the answer, clearly state that you "
        "don't have enough information. Be concise and accurate."
    )

    # ── API Security ───────────────────────────────────────────────────────
    api_key: str = ""
    rate_limit_per_minute: int = 60

    @model_validator(mode="after")
    def _validate_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — loaded once at startup."""
    return Settings()