"""
Configuration settings for Policy Document Processor Streamlit Client.
Uses Pydantic BaseSettings for type-safe configuration with .env file support.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class ClientSettings(BaseSettings):
    """
    Policy Document Processor Client Settings.

    All settings can be configured via environment variables.
    Create a .env file in the streamlit_app directory for local configuration.
    """

    # ===== A2A Agent Connection =====
    agent_url: str = "http://localhost:8001"
    agent_timeout: int = 300  # seconds
    agent_prefer_streaming: bool = True

    # ===== Upload Configuration =====
    max_upload_size_mb: int = 50
    max_upload_size_bytes: int = 52428800  # 50 MB
    allowed_file_types: str = "pdf"  # Comma-separated list

    # ===== UI Configuration =====
    app_title: str = "Policy Document Processor"
    app_icon: str = ":page_facing_up:"
    page_layout: str = "wide"  # wide or centered
    sidebar_state: str = "expanded"  # expanded or collapsed

    # ===== Database Configuration =====
    database_url: str = "sqlite:///./data/policy_processor.db"
    database_echo: bool = False

    # ===== Processing Defaults =====
    default_use_gpt4: bool = False
    default_confidence_threshold: float = 0.7
    default_enable_streaming: bool = True

    # ===== Display Configuration =====
    show_debug_info: bool = False
    results_per_page: int = 10
    tree_max_depth: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = ClientSettings()
