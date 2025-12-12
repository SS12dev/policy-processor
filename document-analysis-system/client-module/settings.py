"""
Client Module Settings
Reads all configuration from .env file.
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class ClientSettings(BaseSettings):
    """Client module configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Agent Connection
    agent_url: str = Field(
        default="http://localhost:8000",
        description="Agent server URL"
    )
    agent_timeout_seconds: int = Field(
        default=300,
        gt=0,
        description="Agent request timeout in seconds"
    )

    # Upload Limits
    max_file_size_mb: int = Field(
        default=50,
        gt=0,
        description="Maximum file size in MB"
    )
    allowed_file_types: str = Field(
        default="pdf",
        description="Allowed file types (comma-separated)"
    )

    # Database
    db_path: str = Field(
        default="./data/client.db",
        description="SQLite database path"
    )

    # Streaming Configuration
    prefer_streaming: bool = Field(
        default=True,
        description="Prefer streaming responses"
    )
    poll_interval_seconds: float = Field(
        default=1.0,
        gt=0,
        description="Polling interval in seconds"
    )
    max_poll_attempts: int = Field(
        default=180,
        gt=0,
        description="Maximum polling attempts"
    )
    stream_update_interval: float = Field(
        default=0.5,
        gt=0,
        description="UI update interval for streaming"
    )

    # UI Configuration
    app_title: str = Field(
        default="PDF Document Analysis Client",
        description="Application title"
    )
    app_icon: str = Field(
        default="📄",
        description="Application icon"
    )
    app_port: int = Field(
        default=8501,
        gt=0,
        lt=65536,
        description="Streamlit app port"
    )
    theme: Literal["light", "dark"] = Field(
        default="light",
        description="UI theme"
    )

    # Client Configuration
    client_name: str = Field(
        default="Document Analysis Client",
        description="Client name"
    )
    client_version: str = Field(
        default="1.0.0",
        description="Client version"
    )
    results_per_page: int = Field(
        default=10,
        gt=0,
        description="Results per page in history"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: str = Field(
        default="./logs/client.log",
        description="Log file path"
    )
    enable_console_log: bool = Field(
        default=True,
        description="Enable console logging"
    )

    # Retry Configuration
    max_connection_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum connection retries"
    )
    retry_delay_seconds: float = Field(
        default=2.0,
        ge=0,
        description="Delay between retries"
    )

    @validator("db_path", "log_file")
    def ensure_directory_exists(cls, v):
        """Ensure parent directory exists for file paths."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def allowed_file_types_list(self) -> list[str]:
        """Get allowed file types as list."""
        return [ext.strip() for ext in self.allowed_file_types.split(",")]


# Global settings instance
settings = ClientSettings()
