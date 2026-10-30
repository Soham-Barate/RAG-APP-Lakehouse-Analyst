"""
Multi-provider LLM service with failover.

Primary: OpenAI GPT-4o
Fallback: Groq (Llama-3.1-70B)

Features: structured output (JSON mode), streaming, retry with backoff,
token counting, and cost estimation.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import Settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Production LLM client with multi-provider failover.

    Supports:
    - OpenAI GPT-4o (primary)
    - Groq Llama-3.1-70B (fallback)
    - JSON mode for structured output
    - Streaming via async generators
    - Automatic retry with exponential backoff
    - Token counting and cost tracking
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._openai_client: Any = None
        self._groq_client: Any = None
        self.last_token_count: int = 0
        self._total_tokens: int = 0
        self._primary_model = settings.openai.model
        self._fallback_model = settings.groq.model

    def _get_openai_client(self) -> Any:
        """Lazy-initialize OpenAI client."""
        if self._openai_client is None:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(
                api_key=self.settings.openai.api_key.get_secret_value(),
                timeout=60.0,
                max_retries=2,
            )
        return self._openai_client

    def _get_groq_client(self) -> Any:
        """Lazy-initialize Groq client."""
        if self._groq_client is None:
            try:
                from groq import AsyncGroq

                self._groq_client = AsyncGroq(
                    api_key=self.settings.groq.api_key.get_secret_value(),
                    timeout=60.0,
                    max_retries=2,
                )
            except ImportError:
                logger.warning("Groq client not installed, fallback unavailable")
        return self._groq_client

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a response using the primary LLM with fallback.

        Args:
            prompt: User/generation prompt
            system_prompt: System instructions
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens in response
            json_mode: Enable JSON structured output

        Returns:
            Generated text response
        """
        openai_key = self.settings.openai.api_key.get_secret_value()
        if not openai_key:
            return await self._generate_groq(
                prompt, system_prompt, temperature, max_tokens, json_mode
            )

        try:
            return await self._generate_openai(
                prompt, system_prompt, temperature, max_tokens, json_mode
            )
        except Exception as primary_exc:
            logger.warning(
                "Primary LLM (OpenAI) failed, trying fallback: %s", primary_exc
            )
            try:
                return await self._generate_groq(
                    prompt, system_prompt, temperature, max_tokens, json_mode
                )
            except Exception as fallback_exc:
                logger.error("Fallback LLM (Groq) also failed: %s", fallback_exc)
                raise primary_exc from fallback_exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _generate_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Generate using OpenAI GPT-4o."""
        client = self._get_openai_client()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._primary_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)

        # Track tokens
        if response.usage:
            self.last_token_count = response.usage.total_tokens
            self._total_tokens += response.usage.total_tokens

        content = response.choices[0].message.content or ""
        return content

    async def _generate_groq(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Generate using Groq (Llama-3.1-70B)."""
        client = self._get_groq_client()
        if not client:
            raise RuntimeError("Groq fallback not available")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._fallback_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(max_tokens, 800),
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)

        if response.usage:
            self.last_token_count = response.usage.total_tokens
            self._total_tokens += response.usage.total_tokens

        content = response.choices[0].message.content or ""
        return content

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from OpenAI."""
        client = self._get_openai_client()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await client.chat.completions.create(
            model=self._primary_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimated cost in USD (GPT-4o: ~$5/1M input, $15/1M output)."""
        return self._total_tokens * 10.0 / 1_000_000  # rough average
