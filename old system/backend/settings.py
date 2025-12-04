"""Backend settings configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load from system root .env file
system_root = Path(__file__).parent.parent
env_path = system_root / ".env"
load_dotenv(env_path)


class BackendSettings:
    """Backend configuration for Prior Authorization POC."""

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///storage/poc_db.sqlite")

    # Storage
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./storage")
    UPLOADS_PATH: str = os.path.join(STORAGE_PATH, "uploads")

    # PDF Processing
    POPPLER_PATH: str = os.getenv("POPPLER_PATH", "")
    
    # Chunking Configuration - Policy Documents
    POLICY_PAGES_PER_CHUNK: int = int(os.getenv("POLICY_PAGES_PER_CHUNK", "3"))
    POLICY_CHUNK_OVERLAP_PAGES: int = int(os.getenv("POLICY_CHUNK_OVERLAP_PAGES", "1"))
    POLICY_MAX_CHARS_PER_CHUNK: int = int(os.getenv("POLICY_MAX_CHARS_PER_CHUNK", "10000"))
    POLICY_CHAR_OVERLAP: int = int(os.getenv("POLICY_CHAR_OVERLAP", "800"))
    
    # Chunking Configuration - Applications
    APP_PAGES_PER_CHUNK: int = int(os.getenv("APP_PAGES_PER_CHUNK", "1"))
    APP_CHUNK_OVERLAP_PAGES: int = int(os.getenv("APP_CHUNK_OVERLAP_PAGES", "0"))
    APP_MAX_PAGES_FOR_IMAGES: int = int(os.getenv("APP_MAX_PAGES_FOR_IMAGES", "1"))
    
    # Chunking Thresholds
    SMALL_DOC_CHAR_THRESHOLD: int = int(os.getenv("SMALL_DOC_CHAR_THRESHOLD", "25000"))
    SMALL_DOC_PAGE_THRESHOLD: int = int(os.getenv("SMALL_DOC_PAGE_THRESHOLD", "8"))
    
    # Memory Optimization Settings
    MAX_CHUNK_SIZE_BYTES: int = int(os.getenv("MAX_CHUNK_SIZE_BYTES", "50000"))  # ~50KB per chunk
    ENABLE_MEMORY_CLEANUP: bool = os.getenv("ENABLE_MEMORY_CLEANUP", "true").lower() == "true"
    
    # Advanced chunking controls
    DISABLE_CHUNK_SPLITTING: bool = os.getenv("DISABLE_CHUNK_SPLITTING", "false").lower() == "true"
    CHUNK_SPLITTING_THRESHOLD_MULTIPLIER: float = float(os.getenv("CHUNK_SPLITTING_THRESHOLD_MULTIPLIER", "1.5"))

    # Agent Configuration
    AGENT_MODE: str = os.getenv("AGENT_MODE", "local").lower()
    USE_DEPLOYED_AGENTS: bool = AGENT_MODE == "deployed"
    
    # Local Agent URLs
    LOCAL_POLICY_AGENT_URL: str = os.getenv("LOCAL_POLICY_AGENT_URL", "http://localhost:10001")
    LOCAL_APPLICATION_AGENT_URL: str = os.getenv("LOCAL_APPLICATION_AGENT_URL", "http://localhost:10002")
    LOCAL_DECISION_AGENT_URL: str = os.getenv("LOCAL_DECISION_AGENT_URL", "http://localhost:10003")
    
    # Deployed Agent URLs
    DEPLOYED_POLICY_AGENT_URL: str = os.getenv(
        "DEPLOYED_POLICY_AGENT_URL", 
        "https://smartops-di05.ustdev.com:8443/paas/smartgenie/smartgenie-policy-analysis-agent/"
    )
    DEPLOYED_APPLICATION_AGENT_URL: str = os.getenv(
        "DEPLOYED_APPLICATION_AGENT_URL",
        "https://smartops-di05.ustdev.com:8443/paas/smartgenie/smartgenie-application-processing-agent/"
    )
    DEPLOYED_DECISION_AGENT_URL: str = os.getenv(
        "DEPLOYED_DECISION_AGENT_URL",
        "https://smartops-di05.ustdev.com:8443/paas/smartgenie/smartgenie-decision-making-agent/"
    )

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")

    @classmethod
    def validate(cls):
        """Validate and initialize required directories."""
        os.makedirs(cls.STORAGE_PATH, exist_ok=True)
        os.makedirs(cls.UPLOADS_PATH, exist_ok=True)

backend_settings = BackendSettings()