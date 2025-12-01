"""
Application settings loaded from environment variables.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings."""

    # LLM Provider Selection
    llm_provider: str = os.getenv("LLM_PROVIDER", "proxy")

    # LiteLLM Proxy settings
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_endpoint: str = os.getenv("LLM_ENDPOINT", "")
    llm_model: str = os.getenv("LLM_MODEL", "azure/sc-rnd-gpt-4o-mini-01")
    llm_model_secondary: str = os.getenv("LLM_MODEL_SECONDARY", "azure/sc-rnd-gpt-4o-01")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    llm_streaming: bool = os.getenv("LLM_STREAMING", "False").lower() == "true"

    # Azure OpenAI settings
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "sc-rnd-gpt-4o-mini-01")
    azure_openai_deployment_secondary: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_SECONDARY", "sc-rnd-gpt-4o-01")
    azure_openai_model: str = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini-2024-05-13")
    azure_openai_model_secondary: str = os.getenv("AZURE_OPENAI_MODEL_SECONDARY", "gpt-4o-2024-05-13")
    azure_openai_max_tokens: int = int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "4096"))

    # OpenAI settings (fallback)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model_primary: str = os.getenv("OPENAI_MODEL_PRIMARY", "gpt-4o-mini")
    openai_model_secondary: str = os.getenv("OPENAI_MODEL_SECONDARY", "gpt-4o")
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    openai_max_retries: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    openai_timeout: int = int(os.getenv("OPENAI_TIMEOUT", "300"))

    # Concurrency settings (conservative for reliability)
    openai_max_concurrent_requests: int = int(os.getenv("OPENAI_MAX_CONCURRENT_REQUESTS", "2"))
    openai_per_request_timeout: int = int(os.getenv("OPENAI_PER_REQUEST_TIMEOUT", "300"))

    # Redis settings
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    redis_socket_timeout: int = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
    redis_socket_connect_timeout: int = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
    redis_default_ttl: int = int(os.getenv("REDIS_DEFAULT_TTL", "3600"))

    # Logging settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Processing settings
    default_confidence_threshold: float = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.7"))
    target_chunk_tokens: int = int(os.getenv("TARGET_CHUNK_TOKENS", "10000"))
    max_chunk_tokens: int = int(os.getenv("MAX_CHUNK_TOKENS", "16000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "600"))

    # Database settings
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/policy_processor.db")


# Global settings instance
settings = Settings()
