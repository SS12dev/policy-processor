"""
Configuration settings for Policy Document Processor Agent.
Uses Pydantic BaseSettings for type-safe configuration with .env file support.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Policy Document Processor Agent Settings.

    All settings can be configured via environment variables.
    Create a .env file in the root directory for local configuration.
    """

    # ===== Server Configuration =====
    server_host: str = "0.0.0.0"
    server_port: int = 8001
    server_reload: bool = False
    log_level: str = "INFO"

    # ===== Agent Metadata =====
    agent_name: str = "Policy Document Processor Agent"
    agent_version: str = "4.0.0"
    agent_description: str = "Stateless policy document processor with Redis-based temporary storage. Processes PDFs and generates hierarchical decision trees with eligibility questions. Powered by LangGraph state machine."

    # ===== Redis Configuration =====
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_result_ttl_hours: int = 24
    redis_max_connections: int = 10

    # ===== Database Configuration (Optional) =====
    database_url: Optional[str] = None  # e.g., "sqlite:///./data/policy_processor.db"
    database_echo: bool = False

    # ===== LLM Provider Configuration =====
    llm_provider: str = "openai"  # Options: openai, azure, proxy, auto
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4000
    llm_streaming: bool = True
    llm_timeout: int = 120
    llm_max_retries: int = 3

    # ===== OpenAI Configuration =====
    openai_api_key: Optional[str] = None
    openai_model_primary: str = "gpt-4o-mini"
    openai_model_secondary: str = "gpt-4o"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 4000
    openai_timeout: int = 120
    openai_max_retries: int = 3

    # ===== Azure OpenAI Configuration =====
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_deployment: str = "gpt-4o-mini"
    azure_openai_deployment_secondary: str = "gpt-4o"
    azure_openai_model: str = "gpt-4o-mini"
    azure_openai_model_secondary: str = "gpt-4o"
    azure_openai_max_tokens: int = 4000

    # ===== LiteLLM Proxy Configuration =====
    llm_api_key: Optional[str] = None
    llm_endpoint: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_model_secondary: str = "gpt-4o"

    # ===== Processing Configuration =====
    max_file_size_mb: int = 50
    max_file_size_bytes: int = 52428800  # 50 MB
    default_confidence_threshold: float = 0.7
    enable_streaming: bool = True
    stream_chunk_size: int = 1024

    # ===== Processing Limits =====
    max_pages_per_document: int = 500
    max_policies_per_document: int = 100
    max_decision_trees_per_policy: int = 10

    # ===== Retry Configuration =====
    max_retries: int = 3
    retry_delay_seconds: int = 2
    retry_exponential_backoff: bool = True

    # ===== Rate Limiting =====
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 10
    rate_limit_burst: int = 20

    # ===== Logging Configuration =====
    log_format: str = "json"  # Options: json, text
    log_file: Optional[str] = "logs/agent.log"
    log_rotation: str = "10MB"
    log_retention: str = "30 days"
    log_compression: bool = True

    # ===== Metrics Configuration =====
    metrics_enabled: bool = True
    metrics_file: str = "metrics/agent_metrics.json"
    metrics_export_interval_seconds: int = 60

    # ===== A2A Protocol Configuration =====
    a2a_streaming: bool = True
    a2a_multiturn: bool = False
    a2a_push_notifications: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
