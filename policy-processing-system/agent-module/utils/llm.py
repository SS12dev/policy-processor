"""
Unified LLM client provider for the policy processor.

Supports multiple providers with async operations and structured output.
Providers: OpenAI, Azure OpenAI, LiteLLM Proxy, Auto-detection
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI, OpenAIError
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.language_models import BaseChatModel

from settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    Unified LLM client that supports multiple providers with async operations.

    Supports:
    - OpenAI API (direct)
    - Azure OpenAI
    - LiteLLM Proxy
    - Auto-detection (tries Azure first, falls back to proxy)
    """

    def __init__(self):
        """Initialize LLM client based on provider configuration."""
        self.provider = settings.llm_provider.lower()
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_timeout
        self.max_retries = settings.llm_max_retries

        # Initialize async client for direct API calls
        self._async_client: Optional[AsyncOpenAI] = None

        # Initialize provider
        self._initialize_provider()

        logger.info(
            f"LLM client initialized",
            extra={
                "provider": self.provider,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        )

    def _initialize_provider(self):
        """Initialize the appropriate provider."""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "azure":
            self._init_azure()
        elif self.provider == "proxy":
            self._init_proxy()
        elif self.provider == "auto":
            try:
                self._init_azure()
                logger.info("Auto-detection: Using Azure OpenAI")
            except Exception as e:
                logger.warning(f"Azure failed, falling back to proxy: {e}")
                self._init_proxy()
        else:
            raise ValueError(f"Invalid LLM_PROVIDER: {self.provider}. Must be 'openai', 'azure', 'proxy', or 'auto'.")

    def _init_openai(self):
        """Initialize OpenAI client."""
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")

        self._async_client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout,
            max_retries=settings.openai_max_retries
        )
        self.primary_model = settings.openai_model_primary
        self.secondary_model = settings.openai_model_secondary
        logger.info(f"Initialized OpenAI client: {self.primary_model}")

    def _init_azure(self):
        """Initialize Azure OpenAI client."""
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required for Azure provider")

        self._async_client = AsyncOpenAI(
            api_key=settings.azure_openai_api_key,
            base_url=f"{settings.azure_openai_endpoint}/openai/deployments/{settings.azure_openai_deployment}",
            api_version=settings.azure_openai_api_version,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries
        )
        self.primary_model = settings.azure_openai_deployment
        self.secondary_model = settings.azure_openai_deployment_secondary
        logger.info(f"Initialized Azure OpenAI client: {self.primary_model}")

    def _init_proxy(self):
        """Initialize LiteLLM proxy client."""
        if not settings.llm_api_key or not settings.llm_endpoint:
            raise ValueError("LLM_API_KEY and LLM_ENDPOINT are required for proxy provider")

        self._async_client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_endpoint,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries
        )
        self.primary_model = settings.llm_model
        self.secondary_model = settings.llm_model_secondary
        logger.info(f"Initialized LiteLLM proxy client: {self.primary_model}")

    def get_langchain_model(self, use_gpt4: bool = False, max_tokens: Optional[int] = None) -> BaseChatModel:
        """
        Get a LangChain-compatible chat model.
        Use this for LangGraph integration and tool use.

        Args:
            use_gpt4: If True, use the secondary (more powerful) model
            max_tokens: Override default max_tokens

        Returns:
            Configured LangChain chat model
        """
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "openai":
            model = settings.openai_model_secondary if use_gpt4 else settings.openai_model_primary
            return ChatOpenAI(
                model=model,
                openai_api_key=settings.openai_api_key,
                temperature=settings.openai_temperature,
                max_tokens=tokens,
                request_timeout=settings.openai_timeout,
                max_retries=settings.openai_max_retries,
            )

        elif self.provider == "azure":
            deployment = settings.azure_openai_deployment_secondary if use_gpt4 else settings.azure_openai_deployment
            model = settings.azure_openai_model_secondary if use_gpt4 else settings.azure_openai_model
            return AzureChatOpenAI(
                azure_deployment=deployment,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                model=model,
                temperature=self.temperature,
                max_tokens=tokens,
                streaming=settings.llm_streaming,
                request_timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
            )

        elif self.provider == "proxy":
            model = settings.llm_model_secondary if use_gpt4 else settings.llm_model
            return ChatOpenAI(
                model=model,
                openai_api_key=settings.llm_api_key,
                openai_api_base=settings.llm_endpoint,
                temperature=self.temperature,
                max_tokens=tokens,
                streaming=settings.llm_streaming,
                request_timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
            )

        else:
            raise ValueError(f"Unsupported provider for LangChain: {self.provider}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_gpt4: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text completion from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            use_gpt4: Use secondary (more powerful) model
            **kwargs: Additional parameters

        Returns:
            Generated text

        Raises:
            OpenAIError: If API call fails
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        model = self.secondary_model if use_gpt4 else self.primary_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(
                "Generating LLM completion",
                extra={
                    "model": model,
                    "temperature": temp,
                    "max_tokens": max_tok,
                    "prompt_length": len(prompt)
                }
            )

            response = await self._async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                **kwargs
            )

            content = response.choices[0].message.content
            usage = response.usage

            logger.info(
                "LLM completion generated",
                extra={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "response_length": len(content)
                }
            )

            return content

        except OpenAIError as e:
            logger.error(
                f"LLM generation failed: {str(e)}",
                extra={"error_type": type(e).__name__}
            )
            raise

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_gpt4: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured output (JSON) from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            schema: Optional JSON schema for response format
            temperature: Override default temperature
            max_tokens: Override default max tokens
            use_gpt4: Use secondary (more powerful) model
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response

        Raises:
            OpenAIError: If API call fails
            ValueError: If response is not valid JSON
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        model = self.secondary_model if use_gpt4 else self.primary_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(
                "Generating structured LLM output",
                extra={"model": model, "has_schema": schema is not None}
            )

            response = await self._async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                response_format={"type": "json_object"},
                **kwargs
            )

            content = response.choices[0].message.content

            # Parse JSON
            result = json.loads(content)

            logger.info(
                "Structured LLM output generated",
                extra={
                    "total_tokens": response.usage.total_tokens,
                    "keys": list(result.keys()) if isinstance(result, dict) else None
                }
            )

            return result

        except (OpenAIError, ValueError, json.JSONDecodeError) as e:
            logger.error(
                f"Structured generation failed: {str(e)}",
                extra={"error_type": type(e).__name__}
            )
            raise

    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        use_gpt4: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Generate completions for multiple prompts in parallel.

        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            use_gpt4: Use secondary (more powerful) model
            **kwargs: Additional parameters

        Returns:
            List of generated texts
        """
        logger.info(f"Generating batch of {len(prompts)} completions")

        tasks = [
            self.generate(prompt, system_prompt, use_gpt4=use_gpt4, **kwargs)
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        outputs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Batch generation failed for prompt {i}: {str(result)}"
                )
                outputs.append("")
            else:
                outputs.append(result)

        return outputs


# Global LLM client instance (singleton)
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def reset_llm_client():
    """Reset the global LLM client instance (useful for testing)."""
    global _llm_client
    _llm_client = None


# Legacy compatibility: Keep get_llm() for existing code
def get_llm(use_gpt4: bool = False, max_tokens: Optional[int] = None) -> BaseChatModel:
    """
    Get a LangChain-compatible chat model (legacy compatibility).

    Args:
        use_gpt4: If True, use the secondary (more powerful) model
        max_tokens: Override the default max_tokens for this LLM instance

    Returns:
        Configured LLM client (ChatOpenAI or AzureChatOpenAI)

    Raises:
        ValueError: If provider is not configured or initialization fails
    """
    client = get_llm_client()
    return client.get_langchain_model(use_gpt4=use_gpt4, max_tokens=max_tokens)


if __name__ == "__main__":
    # Simple smoke test
    import logging
    logging.basicConfig(level=logging.INFO)
    try:
        client = get_llm_client()
        logger.info(f"LLM client provider: {client.provider}")
        logger.info(f"Primary model: {client.primary_model}")
        logger.info(f"Secondary model: {client.secondary_model}")
    except Exception as e:
        logger.exception(f"LLM initialization failed: {e}")
        raise
