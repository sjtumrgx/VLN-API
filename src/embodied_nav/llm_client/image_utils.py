"""Image encoding utilities."""

import base64
import io
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .base import ImageInput


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image with letterbox padding to target size.

    Args:
        image: Input image as numpy array (BGR)
        target_size: Target size (width, height)
        color: Padding color (BGR)

    Returns:
        Tuple of (resized_image, scale_ratio, (pad_w, pad_h))
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    # Create padded image
    padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return padded, scale, (pad_w, pad_h)


def encode_image_to_base64(
    image: Union[np.ndarray, Image.Image, bytes, str, Path],
    apply_letterbox: bool = True,
    target_size: Tuple[int, int] = (640, 640),
) -> ImageInput:
    """Encode an image to base64 for LLM API.

    Args:
        image: Image as numpy array (BGR), PIL Image, bytes, or file path
        apply_letterbox: Whether to apply letterbox resize (default True)
        target_size: Target size for letterbox (default 640x640)

    Returns:
        ImageInput with base64 encoded data and MIME type
    """
    # Convert to numpy array first if needed
    if isinstance(image, (str, Path)):
        # Load from file
        path = Path(image)
        img_array = cv2.imread(str(path))
        if img_array is None:
            raise ValueError(f"Failed to load image: {path}")
    elif isinstance(image, bytes):
        # Decode bytes to numpy array
        nparr = np.frombuffer(image, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError("Failed to decode image bytes")
    elif isinstance(image, Image.Image):
        # Convert PIL Image to numpy array (RGB -> BGR)
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        img_array = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Apply letterbox resize
    if apply_letterbox:
        img_array, _, _ = letterbox(img_array, target_size)

    # Encode to JPEG
    success, encoded = cv2.imencode(".jpg", img_array)
    if not success:
        raise ValueError("Failed to encode image")

    return ImageInput(data=encoded.tobytes(), mime_type="image/jpeg")


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
