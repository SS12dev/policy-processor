"""
Unified LLM client provider for the policy processor.
Supports LiteLLM proxy, Azure OpenAI, and OpenAI based on configuration.
"""
import logging
from typing import Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


def get_llm(use_gpt4: bool = False, max_tokens: Optional[int] = None) -> ChatOpenAI:
    """
    Initialize and return the LLM based on the configured provider.
    
    Args:
        use_gpt4: If True, use the secondary (more powerful) model.
        max_tokens: Override the default max_tokens for this LLM instance.
    
    Returns:
        Configured LLM client (ChatOpenAI or AzureChatOpenAI).
    
    Raises:
        ValueError: If provider is not configured or initialization fails.
    """
    provider = settings.llm_provider.lower()
    
    try:
        if provider == "proxy":
            return _get_proxy_llm(use_gpt4, max_tokens)
        elif provider == "azure":
            return _get_azure_llm(use_gpt4, max_tokens)
        elif provider == "openai":
            return _get_openai_llm(use_gpt4, max_tokens)
        elif provider == "auto":
            # Try Azure first, fall back to proxy
            try:
                return _get_azure_llm(use_gpt4, max_tokens)
            except Exception as e:
                logger.warning(f"Azure LLM initialization failed, falling back to proxy: {e}")
                return _get_proxy_llm(use_gpt4, max_tokens)
        else:
            raise ValueError(f"Invalid LLM_PROVIDER: {provider}. Must be 'proxy', 'azure', 'openai', or 'auto'.")
    
    except Exception as e:
        logger.error(f"Failed to initialize LLM with provider '{provider}': {e}")
        raise


def _get_proxy_llm(use_gpt4: bool = False, max_tokens: Optional[int] = None) -> ChatOpenAI:
    """Initialize LLM via LiteLLM proxy."""
    if not settings.llm_api_key or not settings.llm_endpoint:
        raise ValueError("LiteLLM proxy requires LLM_API_KEY and LLM_ENDPOINT to be set.")
    
    model = settings.llm_model_secondary if use_gpt4 else settings.llm_model
    tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
    
    llm = ChatOpenAI(
        model=model,
        openai_api_key=settings.llm_api_key,
        openai_api_base=settings.llm_endpoint,
        temperature=settings.llm_temperature,
        max_tokens=tokens,
        streaming=settings.llm_streaming,
        request_timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )
    
    logger.info(f"Initialized LiteLLM proxy client: model={model}, max_tokens={tokens}")
    return llm


def _get_azure_llm(use_gpt4: bool = False, max_tokens: Optional[int] = None) -> AzureChatOpenAI:
    """Initialize Azure OpenAI LLM."""
    if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
        raise ValueError("Azure OpenAI requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to be set.")
    
    deployment = settings.azure_openai_deployment_secondary if use_gpt4 else settings.azure_openai_deployment
    model = settings.azure_openai_model_secondary if use_gpt4 else settings.azure_openai_model
    tokens = max_tokens if max_tokens is not None else settings.azure_openai_max_tokens
    
    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        model=model,
        temperature=settings.llm_temperature,
        max_tokens=tokens,
        streaming=settings.llm_streaming,
        request_timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )
    
    logger.info(f"Initialized Azure OpenAI client: deployment={deployment}, model={model}, max_tokens={tokens}")
    return llm


def _get_openai_llm(use_gpt4: bool = False, max_tokens: Optional[int] = None) -> ChatOpenAI:
    """Initialize OpenAI LLM."""
    if not settings.openai_api_key:
        raise ValueError("OpenAI requires OPENAI_API_KEY to be set.")
    
    model = settings.openai_model_secondary if use_gpt4 else settings.openai_model_primary
    tokens = max_tokens if max_tokens is not None else settings.openai_max_tokens
    
    llm = ChatOpenAI(
        model=model,
        openai_api_key=settings.openai_api_key,
        temperature=settings.openai_temperature,
        max_tokens=tokens,
        request_timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )
    
    logger.info(f"Initialized OpenAI client: model={model}, max_tokens={tokens}")
    return llm


if __name__ == "__main__":
    # Simple smoke test
    logging.basicConfig(level=logging.INFO)
    try:
        llm = get_llm()
        logger.info(f"LLM instance: {llm.__class__.__name__}")
        logger.info(f"Provider: {settings.llm_provider}")
    except Exception as e:
        logger.exception(f"LLM initialization failed: {e}")
        raise
