"""LLM client module."""

from .base import BaseLLMClient, ImageInput, LLMResponse
from .gemini import GeminiNativeClient
from .openai_compat import OpenAICompatibleClient
from .image_utils import encode_image_to_base64, decode_base64_to_image, letterbox, LetterboxInfo

__all__ = [
    "BaseLLMClient",
    "ImageInput",
    "LLMResponse",
    "GeminiNativeClient",
    "OpenAICompatibleClient",
    "encode_image_to_base64",
    "decode_base64_to_image",
    "letterbox",
    "LetterboxInfo",
]
