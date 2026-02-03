"""OpenAI compatible API client."""

import asyncio
import base64
import logging
from typing import Optional

import httpx

from .base import BaseLLMClient, ImageInput, LLMResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI compatible API format."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gemini-2.5-pro",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize OpenAI compatible client.

        Args:
            base_url: API base URL
            api_key: API key
            model: Model identifier (default: gemini-2.5-pro)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        super().__init__(base_url, api_key, model, timeout, max_retries)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            # Disable proxy - don't read from environment variables
            self._client = httpx.AsyncClient(timeout=self.timeout, trust_env=False)
        return self._client

    async def generate(
        self,
        prompt: str,
        image: Optional[ImageInput] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response using OpenAI compatible API.

        Args:
            prompt: User prompt text
            image: Optional image input
            system_prompt: Optional system prompt

        Returns:
            Normalized LLM response
        """
        client = await self._get_client()

        # Build messages
        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message content
        if image:
            # Vision request with image
            image_b64 = base64.b64encode(image.data).decode("utf-8")
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.mime_type};base64,{image_b64}"
                    }
                },
                {"type": "text", "text": prompt}
            ]
            messages.append({"role": "user", "content": content})
        else:
            # Text-only request
            messages.append({"role": "user", "content": prompt})

        # Build request body
        request_body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }

        # Build URL
        url = f"{self.base_url}/v1/chat/completions"

        # Make request with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                response = await client.post(
                    url,
                    json=request_body,
                    headers=headers,
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
        """Parse OpenAI compatible response into normalized format."""
        text = ""
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                text = choice["message"]["content"]

        usage = {}
        if "usage" in data:
            usage = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }

        model = data.get("model", self.model)

        return LLMResponse(
            text=text,
            model=model,
            usage=usage,
            raw_response=data,
        )

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
