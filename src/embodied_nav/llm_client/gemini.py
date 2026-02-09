"""Gemini native API client."""

import asyncio
import base64
import json
import logging
from typing import Optional

import httpx

from .base import BaseLLMClient, ImageInput, LLMResponse

logger = logging.getLogger(__name__)


class GeminiNativeClient(BaseLLMClient):
    """Client for Gemini native v1beta API format."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gemini-3-flash-preview",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize Gemini native client.

        Args:
            base_url: API base URL
            api_key: Gemini API key
            model: Model identifier (default: gemini-3-flash-preview)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        super().__init__(base_url, api_key, model, timeout, max_retries)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def generate(
        self,
        prompt: str,
        image: Optional[ImageInput] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response using Gemini native API.

        Args:
            prompt: User prompt text
            image: Optional image input
            system_prompt: Optional system prompt

        Returns:
            Normalized LLM response
        """
        client = await self._get_client()

        # Build request parts
        parts = []

        # Add image if provided
        if image:
            image_b64 = base64.b64encode(image.data).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": image.mime_type,
                    "data": image_b64,
                }
            })

        # Add text prompt
        parts.append({"text": prompt})

        # Build contents
        contents = [{"role": "user", "parts": parts}]

        # Build request body
        request_body = {"contents": contents}

        # Add system instruction if provided
        if system_prompt:
            request_body["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        # Build URL
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"

        # Make request with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    url,
                    json=request_body,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key,
                    },
                )

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code >= 500:
                    # Server error - retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error {response.status_code}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code >= 400:
                    # Client error - don't retry
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {response.text}",
                        request=response.request,
                        response=response,
                    )

                # Parse successful response
                data = response.json()
                return self._parse_response(data)

            except httpx.TimeoutException as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(f"Timeout, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                break

        raise last_error or Exception("Max retries exceeded")

    def _parse_response(self, data: dict) -> LLMResponse:
        """Parse Gemini native response into normalized format."""
        text = ""
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    text = parts[0]["text"]

        usage = {}
        if "usageMetadata" in data:
            usage = {
                "prompt_tokens": data["usageMetadata"].get("promptTokenCount", 0),
                "completion_tokens": data["usageMetadata"].get("candidatesTokenCount", 0),
                "total_tokens": data["usageMetadata"].get("totalTokenCount", 0),
            }

        return LLMResponse(
            text=text,
            model=self.model,
            usage=usage,
            raw_response=data,
        )

    async def generate_stream(
        self,
        prompt: str,
        image: Optional[ImageInput] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response using Gemini streaming API for reduced latency.

        Uses streamGenerateContent endpoint to accumulate response chunks.

        Args:
            prompt: User prompt text
            image: Optional image input
            system_prompt: Optional system prompt

        Returns:
            Normalized LLM response with complete text
        """
        client = await self._get_client()

        # Build request parts (same as non-streaming)
        parts = []
        if image:
            image_b64 = base64.b64encode(image.data).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": image.mime_type,
                    "data": image_b64,
                }
            })
        parts.append({"text": prompt})

        contents = [{"role": "user", "parts": parts}]
        request_body = {"contents": contents}
        if system_prompt:
            request_body["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        # Use streaming endpoint
        url = f"{self.base_url}/v1beta/models/{self.model}:streamGenerateContent?alt=sse"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                accumulated_text = ""
                async with client.stream(
                    "POST", url, json=request_body,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key,
                    },
                ) as response:
                    if response.status_code == 429:
                        await response.aread()
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited (stream), waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code >= 500:
                        await response.aread()
                        wait_time = 2 ** attempt
                        logger.warning(f"Server error {response.status_code} (stream), retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code >= 400:
                        body = await response.aread()
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}: {body.decode()}",
                            request=response.request,
                            response=response,
                        )

                    # Parse SSE chunks from Gemini streaming
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        try:
                            chunk = json.loads(data_str)
                            candidates = chunk.get("candidates", [])
                            if candidates:
                                content = candidates[0].get("content", {})
                                parts = content.get("parts", [])
                                for part in parts:
                                    if "text" in part:
                                        accumulated_text += part["text"]
                        except json.JSONDecodeError:
                            continue

                return LLMResponse(
                    text=accumulated_text,
                    model=self.model,
                    usage={},
                    raw_response={},
                )

            except httpx.TimeoutException as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(f"Timeout (stream), retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error in stream: {e}")
                break

        raise last_error or Exception("Max retries exceeded (stream)")

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
