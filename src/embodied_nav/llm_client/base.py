"""Base LLM client interface and common types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    """Normalized response from LLM API."""

    text: str
    model: str
    usage: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict)


@dataclass
class ImageInput:
    """Image input for vision models."""

    data: bytes
    mime_type: str = "image/jpeg"

    @classmethod
    def from_base64(cls, base64_data: str, mime_type: str = "image/jpeg") -> "ImageInput":
        """Create ImageInput from base64 encoded string."""
        import base64
        return cls(data=base64.b64decode(base64_data), mime_type=mime_type)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize LLM client.

        Args:
            base_url: API base URL
            api_key: API key for authentication
            model: Model identifier
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient errors
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        image: Optional[ImageInput] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt text
            image: Optional image input for vision models
            system_prompt: Optional system prompt

        Returns:
            Normalized LLM response
        """
        pass

    @abstractmethod
    async def close(self):
        """Close the client and release resources."""
        pass
