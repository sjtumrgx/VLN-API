"""Unit tests for LLM client module."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from embodied_nav.llm_client.base import BaseLLMClient, ImageInput, LLMResponse
from embodied_nav.llm_client.gemini import GeminiNativeClient
from embodied_nav.llm_client.openai_compat import OpenAICompatibleClient
from embodied_nav.llm_client.image_utils import encode_image_to_base64


class TestImageInput:
    """Tests for ImageInput class."""

    def test_from_base64(self):
        """Test creating ImageInput from base64 string."""
        original = b"test image data"
        b64 = base64.b64encode(original).decode()

        img_input = ImageInput.from_base64(b64)

        assert img_input.data == original
        assert img_input.mime_type == "image/jpeg"

    def test_from_base64_with_mime_type(self):
        """Test creating ImageInput with custom MIME type."""
        original = b"test image data"
        b64 = base64.b64encode(original).decode()

        img_input = ImageInput.from_base64(b64, mime_type="image/png")

        assert img_input.mime_type == "image/png"


class TestLLMResponse:
    """Tests for LLMResponse class."""

    def test_response_creation(self):
        """Test creating LLMResponse."""
        response = LLMResponse(
            text="Hello world",
            model="test-model",
            usage={"total_tokens": 10},
        )

        assert response.text == "Hello world"
        assert response.model == "test-model"
        assert response.usage["total_tokens"] == 10


class TestEncodeImage:
    """Tests for image encoding utilities."""

    def test_encode_numpy_array(self):
        """Test encoding numpy array to ImageInput."""
        # Create a simple test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [255, 0, 0]  # Blue in BGR

        result = encode_image_to_base64(img)

        assert isinstance(result, ImageInput)
        assert result.mime_type == "image/jpeg"
        assert len(result.data) > 0

    def test_encode_bytes(self):
        """Test encoding raw bytes."""
        data = b"fake image data"

        result = encode_image_to_base64(data)

        assert result.data == data
        assert result.mime_type == "image/jpeg"


class TestGeminiNativeClient:
    """Tests for GeminiNativeClient."""

    def test_init(self):
        """Test client initialization."""
        client = GeminiNativeClient(
            base_url="http://example.com",
            api_key="test-key",
            model="gemini-test",
        )

        assert client.base_url == "http://example.com"
        assert client.api_key == "test-key"
        assert client.model == "gemini-test"

    def test_parse_response(self):
        """Test parsing Gemini native response."""
        client = GeminiNativeClient(
            base_url="http://example.com",
            api_key="test-key",
        )

        raw_response = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello from Gemini"}]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            }
        }

        result = client._parse_response(raw_response)

        assert result.text == "Hello from Gemini"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_generate_text_only(self):
        """Test generating text-only response."""
        client = GeminiNativeClient(
            base_url="http://example.com",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Test response"}]
                }
            }]
        }

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            result = await client.generate("Hello")

            assert result.text == "Test response"
            mock_http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing client."""
        client = GeminiNativeClient(
            base_url="http://example.com",
            api_key="test-key",
        )
        mock_http_client = AsyncMock()
        client._client = mock_http_client

        await client.close()

        mock_http_client.aclose.assert_called_once()
        assert client._client is None


class TestOpenAICompatibleClient:
    """Tests for OpenAICompatibleClient."""

    def test_init(self):
        """Test client initialization."""
        client = OpenAICompatibleClient(
            base_url="http://example.com",
            api_key="test-key",
            model="gpt-test",
        )

        assert client.base_url == "http://example.com"
        assert client.api_key == "test-key"
        assert client.model == "gpt-test"

    def test_parse_response(self):
        """Test parsing OpenAI compatible response."""
        client = OpenAICompatibleClient(
            base_url="http://example.com",
            api_key="test-key",
        )

        raw_response = {
            "choices": [{
                "message": {
                    "content": "Hello from OpenAI"
                }
            }],
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        }

        result = client._parse_response(raw_response)

        assert result.text == "Hello from OpenAI"
        assert result.model == "gpt-4"
        assert result.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_generate_text_only(self):
        """Test generating text-only response."""
        client = OpenAICompatibleClient(
            base_url="http://example.com",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Test response"}
            }],
            "model": "test-model",
        }

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            result = await client.generate("Hello")

            assert result.text == "Test response"
            mock_http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing client."""
        client = OpenAICompatibleClient(
            base_url="http://example.com",
            api_key="test-key",
        )
        mock_http_client = AsyncMock()
        client._client = mock_http_client

        await client.close()

        mock_http_client.aclose.assert_called_once()
