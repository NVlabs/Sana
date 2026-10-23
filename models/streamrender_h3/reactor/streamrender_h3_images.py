"""Bounded reference-image decoding on the application side."""

import io

import numpy as np
from PIL import Image, ImageOps


def decode_reference(data: bytes) -> np.ndarray:
    """Decode the selected reference without altering its native resolution."""
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("Reference image exceeds 16 MiB")
    try:
        with Image.open(io.BytesIO(data), formats=["PNG", "JPEG"]) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Reference image exceeds 16 megapixels")
            return np.array(ImageOps.exif_transpose(image).convert("RGB"))
    except Image.DecompressionBombError as error:
        raise ValueError("Reference image exceeds the safe pixel budget") from error
