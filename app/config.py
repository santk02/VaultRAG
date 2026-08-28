from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "VaultRAG"
    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "postgresql://vaultrag:vaultrag@localhost:5432/vaultrag"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"
    qdrant_vector_size: int = 384

    # LLM Configuration
    mode: Literal["cloud", "offline"] = "offline"

    # Cloud LLM
    anthropic_api_key: str = ""
    cloud_model: str = "claude-sonnet-4-6"

    # Local LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    offline_model: str = "mistral"
    alternative_model: str = "llama3.1:8b"

    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"

    # Retrieval Parameters
    bm25_top_k: int = 50
    vector_top_k: int = 50
    rrf_k: int = 60
    rrf_top_n: int = 30
    rerank_top_k: int = 5
    rerank_min_score: float = 0.0

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # Guardrails
    guardrails_api_key: str = ""

    # File Upload
    max_file_size: int = 10485760  # 10MB
    allowed_extensions: str = ".pdf,.docx"

    # Evaluation
    ragas_faithfulness_threshold: float = 0.80
    ragas_relevancy_threshold: float = 0.75

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
