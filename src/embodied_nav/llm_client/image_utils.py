"""Image encoding utilities."""

import base64
import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from .base import ImageInput


def encode_image_to_base64(image: Union[np.ndarray, Image.Image, bytes, str, Path]) -> ImageInput:
    """Encode an image to base64 for LLM API.

    Args:
        image: Image as numpy array (BGR), PIL Image, bytes, or file path

    Returns:
        ImageInput with base64 encoded data and MIME type
    """
    if isinstance(image, (str, Path)):
        # Load from file
        path = Path(image)
        mime_type = _get_mime_type(path.suffix)
        with open(path, "rb") as f:
            data = f.read()
        return ImageInput(data=data, mime_type=mime_type)

    if isinstance(image, bytes):
        # Assume JPEG if raw bytes
        return ImageInput(data=image, mime_type="image/jpeg")

    if isinstance(image, np.ndarray):
        # Convert BGR numpy array to JPEG bytes
        import cv2
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode image")
        return ImageInput(data=encoded.tobytes(), mime_type="image/jpeg")

    if isinstance(image, Image.Image):
        # Convert PIL Image to JPEG bytes
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return ImageInput(data=buffer.getvalue(), mime_type="image/jpeg")

    raise TypeError(f"Unsupported image type: {type(image)}")


def _get_mime_type(suffix: str) -> str:
    """Get MIME type from file extension."""
    suffix = suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_types.get(suffix, "image/jpeg")


def decode_base64_to_image(base64_data: str) -> np.ndarray:
    """Decode base64 image to numpy array.

    Args:
        base64_data: Base64 encoded image string

    Returns:
        Image as numpy array (BGR)
    """
    import cv2
    data = base64.b64decode(base64_data)
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
