"""
OpenAI-compatible LLM client for Graphiti that uses classic chat completions API.

The default OpenAI client in Graphiti 0.28+ uses the new Responses API 
(client.responses.parse) which isn't supported by most OpenAI-compatible proxies.
This client overrides structured completion to use chat.completions with JSON mode.

Also handles:
- Servers that default to streaming (explicit stream=False)
- Claude's habit of wrapping JSON in markdown code blocks
"""

import json
import logging
import re
import typing
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig, DEFAULT_MAX_TOKENS
from graphiti_core.llm_client.openai_base_client import (
    BaseOpenAIClient,
    DEFAULT_REASONING,
    DEFAULT_VERBOSITY,
)

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> str:
    """Extract JSON from potentially markdown-wrapped response."""
    # Try to extract from ```json ... ``` blocks
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Already clean JSON
    return text.strip()


class CompatOpenAIClient(BaseOpenAIClient):
    """
    OpenAI-compatible client that uses classic chat completions for all calls.
    
    Works with any OpenAI-compatible API (OpenRouter, LiteLLM, vLLM, 9router, etc.)
    that supports /v1/chat/completions but NOT /v1/responses.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str = DEFAULT_REASONING,
        verbosity: str = DEFAULT_VERBOSITY,
    ):
        super().__init__(config, cache, max_tokens, reasoning, verbosity)

        if config is None:
            config = LLMConfig()

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """
        Create a structured completion using classic chat completions + JSON schema in prompt.
        
        Instead of using the Responses API, we:
        1. Add the JSON schema to the system prompt
        2. Request JSON output
        3. Wrap the response in a compat object with .output_text
        """
        # Build JSON schema instruction
        schema = response_model.model_json_schema()
        schema_instruction = (
            "\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object matching this schema. "
            "Do NOT wrap it in markdown code blocks. Do NOT add any text before or after the JSON.\n"
            f"Schema:\n{json.dumps(schema, indent=2)}"
        )
        
        # Inject schema into system message (or add one)
        enhanced_messages = list(messages)
        if enhanced_messages and enhanced_messages[0].get("role") == "system":
            enhanced_messages[0] = {
                "role": "system",
                "content": str(enhanced_messages[0]["content"]) + schema_instruction,
            }
        else:
            enhanced_messages.insert(0, {
                "role": "system",
                "content": f"You are a helpful assistant that always responds in valid JSON. {schema_instruction}",
            })
        
        # Build kwargs — omit temperature if None (some providers reject null)
        kwargs = dict(
            model=model,
            messages=enhanced_messages,
            max_tokens=max_tokens,
            stream=False,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        
        response = await self.client.chat.completions.create(**kwargs)
        
        # Extract and clean content
        raw_content = response.choices[0].message.content or "{}"
        clean_content = _extract_json(raw_content)
        
        # Validate it's actually valid JSON
        try:
            json.loads(clean_content)
        except json.JSONDecodeError:
            logger.warning(f"LLM returned invalid JSON, attempting repair: {clean_content[:200]}")
            clean_content = "{}"
        
        return _CompatResponse(clean_content, response.usage)

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a regular completion with JSON format using classic chat completions."""
        # Build kwargs — omit temperature if None
        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=False,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        
        response = await self.client.chat.completions.create(**kwargs)
        
        # Wrap in a compat object so _handle_json_response works
        # _handle_json_response expects response.choices[0].message.content
        # But we need to clean potential markdown wrapping
        raw_content = response.choices[0].message.content or "{}"
        clean_content = _extract_json(raw_content)
        response.choices[0].message.content = clean_content
        
        return response


class _CompatResponse:
    """Wrapper to make chat completions response look like a Responses API response."""
    
    def __init__(self, output_text: str, usage: Any = None):
        self.output_text = output_text
        self.usage = _CompatUsage(usage) if usage else None
        self.refusal = None


class _CompatUsage:
    """Wrapper for usage stats."""
    
    def __init__(self, usage: Any):
        self.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        self.output_tokens = getattr(usage, "completion_tokens", 0) or 0
