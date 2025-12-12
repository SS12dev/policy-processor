"""
LLM Utility Module
Provides a unified interface for LLM interactions.
All LLM-related changes should be made in this file.
"""

import asyncio
import json
from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI, OpenAIError
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from settings import settings
from utils.logger import get_logger
from utils.metrics import track_llm_call

logger = get_logger(__name__)


class LLMClient:
    """
    Unified LLM client that can be used with both direct API calls and LangChain.
    """

    def __init__(self):
        """Initialize LLM client based on provider configuration."""
        self.provider = settings.llm_provider
        self.model = settings.openai_model
        self.temperature = settings.openai_temperature
        self.max_tokens = settings.openai_max_tokens
        self.timeout = settings.openai_timeout

        # Initialize OpenAI client
        if self.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")

            self._async_client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=self.timeout
            )
            logger.info(
                f"Initialized OpenAI LLM client",
                extra={
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def get_langchain_model(self) -> BaseChatModel:
        """
        Get a LangChain-compatible chat model.
        Use this for LangGraph integration.
        """
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.openai_api_key,
                timeout=self.timeout
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text completion from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional parameters

        Returns:
            Generated text

        Raises:
            OpenAIError: If API call fails
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(
                "Generating LLM completion",
                extra={
                    "model": self.model,
                    "temperature": temp,
                    "max_tokens": max_tok,
                    "prompt_length": len(prompt)
                }
            )

            with track_llm_call(self.model):
                response = await self._async_client.chat.completions.create(
                    model=self.model,
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
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response

        Raises:
            OpenAIError: If API call fails
            ValueError: If response is not valid JSON
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(
                "Generating structured LLM output",
                extra={"model": self.model, "has_schema": schema is not None}
            )

            with track_llm_call(self.model):
                response = await self._async_client.chat.completions.create(
                    model=self.model,
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
        **kwargs
    ) -> List[str]:
        """
        Generate completions for multiple prompts in parallel.

        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            List of generated texts
        """
        logger.info(f"Generating batch of {len(prompts)} completions")

        tasks = [
            self.generate(prompt, system_prompt, **kwargs)
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

    async def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10
    ) -> List[str]:
        """
        Extract keywords from text using LLM.

        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of extracted keywords
        """
        system_prompt = (
            "You are a keyword extraction expert. Extract the most important "
            "keywords and key phrases from the given text. Return ONLY a JSON "
            "object with a 'keywords' array containing the keywords."
        )

        prompt = f"""Extract up to {max_keywords} important keywords or key phrases from this text.
Focus on main concepts, technical terms, and significant topics.

Text:
{text}

Return format: {{"keywords": ["keyword1", "keyword2", ...]}}"""

        try:
            result = await self.generate_structured(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=500
            )

            keywords = result.get("keywords", [])
            return keywords[:max_keywords]

        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []

    async def analyze_document(
        self,
        text: str,
        headings: List[str],
        keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive document analysis.

        Args:
            text: Document text
            headings: Extracted headings
            keywords: Extracted keywords

        Returns:
            Analysis results with summary, topics, and insights
        """
        system_prompt = (
            "You are a document analysis expert. Analyze the given document "
            "and provide structured insights. Return ONLY valid JSON."
        )

        prompt = f"""Analyze this document and provide insights.

Document Information:
- Headings found: {len(headings)}
- Keywords: {', '.join(keywords[:10])}

Document excerpt (first 2000 chars):
{text[:2000]}

Provide analysis in this JSON format:
{{
    "summary": "Brief 2-3 sentence summary",
    "main_topics": ["topic1", "topic2", "topic3"],
    "document_type": "type of document (report, article, research, etc.)",
    "key_insights": ["insight1", "insight2"],
    "complexity_level": "beginner|intermediate|advanced"
}}"""

        try:
            result = await self.generate_structured(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1000
            )

            return result

        except Exception as e:
            logger.error(f"Document analysis failed: {str(e)}")
            return {
                "summary": "Analysis unavailable",
                "main_topics": [],
                "document_type": "unknown",
                "key_insights": [],
                "complexity_level": "unknown"
            }


# Global LLM client instance
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
