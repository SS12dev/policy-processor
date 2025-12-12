"""
Agent Module Settings
Reads all configuration from .env file and provides them as class attributes.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class AgentSettings(BaseSettings):
    """Agent module configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Agent Metadata
    agent_name: str = Field(
        default="PDF Document Analyzer",
        description="Name of the agent"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    agent_description: str = Field(
        default="Extracts headings, keywords, and key information from PDF documents",
        description="Agent description"
    )
    agent_url: str = Field(
        default="http://localhost:8000",
        description="Agent URL endpoint"
    )

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic", "gemini"] = Field(
        default="openai",
        description="LLM provider to use"
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name"
    )
    openai_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )
    openai_max_tokens: int = Field(
        default=2000,
        gt=0,
        description="Maximum tokens for LLM response"
    )
    openai_timeout: int = Field(
        default=60,
        gt=0,
        description="LLM request timeout in seconds"
    )

    # Server Configuration
    server_host: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    server_port: int = Field(
        default=8000,
        gt=0,
        lt=65536,
        description="Server port"
    )
    max_request_size_mb: int = Field(
        default=100,
        gt=0,
        description="Maximum request size in MB"
    )
    request_timeout_seconds: int = Field(
        default=300,
        gt=0,
        description="Request timeout in seconds"
    )
    workers: int = Field(
        default=4,
        gt=0,
        description="Number of worker processes"
    )

    # Processing Configuration
    max_keywords: int = Field(
        default=15,
        gt=0,
        description="Maximum number of keywords to extract"
    )
    max_headings: int = Field(
        default=30,
        gt=0,
        description="Maximum number of headings to extract"
    )
    chunk_size: int = Field(
        default=2000,
        gt=0,
        description="Text chunk size for processing"
    )
    enable_llm_analysis: bool = Field(
        default=True,
        description="Enable LLM-based analysis"
    )

    # Retry Configuration
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries"
    )
    retry_delay_seconds: float = Field(
        default=2.0,
        ge=0,
        description="Delay between retries in seconds"
    )
    exponential_backoff: bool = Field(
        default=True,
        description="Use exponential backoff for retries"
    )

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(
        default=30,
        gt=0,
        description="Maximum requests per minute"
    )
    rate_limit_burst: int = Field(
        default=10,
        gt=0,
        description="Burst size for rate limiting"
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log format"
    )
    log_file: str = Field(
        default="./logs/agent.log",
        description="Log file path"
    )
    log_rotation: str = Field(
        default="10MB",
        description="Log rotation size"
    )
    log_retention_days: int = Field(
        default=30,
        gt=0,
        description="Log retention in days"
    )
    enable_console_log: bool = Field(
        default=True,
        description="Enable console logging"
    )

    # Metrics Configuration
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_file: str = Field(
        default="./metrics/agent_metrics.json",
        description="Metrics file path"
    )

    # A2A Protocol Configuration
    enable_streaming: bool = Field(
        default=True,
        description="Enable streaming responses"
    )
    stream_chunk_size: int = Field(
        default=500,
        gt=0,
        description="Streaming chunk size in characters"
    )

    @validator("openai_api_key")
    def validate_api_key(cls, v, values):
        """Validate that API key is provided when using OpenAI."""
        if values.get("llm_provider") == "openai" and not v:
            raise ValueError("OPENAI_API_KEY must be set when using OpenAI provider")
        return v

    @validator("log_file", "metrics_file")
    def ensure_directory_exists(cls, v):
        """Ensure parent directory exists for file paths."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def max_request_size_bytes(self) -> int:
        """Get maximum request size in bytes."""
        return self.max_request_size_mb * 1024 * 1024

    @property
    def agent_card_dict(self) -> dict:
        """Get agent card configuration as dictionary."""
        return {
            "name": self.agent_name,
            "version": self.agent_version,
            "description": self.agent_description,
            "url": self.agent_url,
            "capabilities": {
                "streaming": self.enable_streaming
            }
        }

    def get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay based on attempt number."""
        if self.exponential_backoff:
            return self.retry_delay_seconds * (2 ** attempt)
        return self.retry_delay_seconds


# Global settings instance
settings = AgentSettings()


# Convenience function to reload settings
def reload_settings() -> AgentSettings:
    """Reload settings from .env file."""
    global settings
    settings = AgentSettings()
    return settings
