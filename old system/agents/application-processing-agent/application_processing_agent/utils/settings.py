import logging
import os
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class Settings:
    """Configuration management for Application Processing Agent"""
    
    def __init__(self):
        """Initialize settings with environment variables and validation."""
        # Agent-specific LLM Configuration
        self.AP_LLM_API_KEY: str = self._get_required_env("AP_LLM_API_KEY")
        self.AP_LLM_ENDPOINT: str = self._get_env_with_default(
            "AP_LLM_ENDPOINT", 
            "https://smartops-llmops.eastus.cloudapp.azure.com/litellm"
        )
        self.AP_LLM_MODEL: str = self._get_env_with_default(
            "AP_LLM_MODEL", 
            "azure/sc-rnd-gpt-4o-mini-01"
        )
        self.AP_LLM_PRICING: list = [
            0.15,
            0.60,
        ]  # [input $/1M tokens, output $/1M tokens for gpt-4o-mini standard pricing]

        # Agent Server Configuration
        self.AGENT_HOST: str = self._get_env_with_default("AP_AGENT_HOST", "localhost")
        self.AGENT_PORT: int = self._get_int_env("AP_AGENT_PORT", 10002)

        # Agent Metadata
        self.AGENT_NAME: str = self._get_env_with_default(
            "AP_AGENT_NAME", 
            "Application Processing Agent"
        )
        self.AGENT_DESCRIPTION: str = self._get_env_with_default(
            "AP_AGENT_DESCRIPTION", 
            "Extracts and validates medical application data from patient documents"
        )
        self.AGENT_VERSION: str = self._get_env_with_default("AP_AGENT_VERSION", "1.0.0")
        
        # Log configuration status (without sensitive data)
        self._log_configuration()

    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(key)
        if not value:
            error_msg = f"Required environment variable {key} is not set"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return value.strip()

    def _get_env_with_default(self, key: str, default: str) -> str:
        """Get environment variable with default fallback."""
        value = os.getenv(key, default)
        return value.strip() if value else default

    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer environment variable with default fallback."""
        value = os.getenv(key)
        if not value:
            return default
        
        try:
            return int(value.strip())
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}. Using default: {default}")
            return default

    def _log_configuration(self) -> None:
        """Log current configuration (excluding sensitive data)."""
        logger.info("Application Processing Agent Configuration:")
        logger.info(f"  Agent Name: {self.AGENT_NAME}")
        logger.info(f"  Agent Version: {self.AGENT_VERSION}")
        logger.info(f"  Server Host: {self.AGENT_HOST}")
        logger.info(f"  Server Port: {self.AGENT_PORT}")
        logger.info(f"  LLM Endpoint: {self.AP_LLM_ENDPOINT}")
        logger.info(f"  LLM Model: {self.AP_LLM_MODEL}")
        logger.info(f"  API Key Status: {'Set' if self.AP_LLM_API_KEY else 'Not Set'}")

    def validate_required_settings(self) -> list[str]:
        """Validate that all required settings are properly configured."""
        errors = []
        
        if not self.AP_LLM_API_KEY:
            errors.append("AP_LLM_API_KEY is required")
        
        if not self.AP_LLM_ENDPOINT:
            errors.append("AP_LLM_ENDPOINT is required")
        
        if not self.AP_LLM_MODEL:
            errors.append("AP_LLM_MODEL is required")
        
        if not self.AGENT_NAME:
            errors.append("AP_AGENT_NAME is required")
        
        if self.AGENT_PORT <= 0 or self.AGENT_PORT > 65535:
            errors.append(f"AP_AGENT_PORT must be between 1-65535, got: {self.AGENT_PORT}")
        
        return errors

# Create global settings instance
try:
    agent_settings = Settings()
    
    # Validate configuration on import
    validation_errors = agent_settings.validate_required_settings()
    if validation_errors:
        error_msg = f"Configuration validation failed: {'; '.join(validation_errors)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Application Processing Agent settings loaded and validated successfully")
    
except Exception as e:
    logger.error(f"Failed to load agent settings: {e}")
    raise