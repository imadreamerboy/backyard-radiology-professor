from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from radiology_trainer.domain import Anatomy, ImageQuality


def prepare_image(image: Image.Image) -> Image.Image:
    """Normalize orientation and convert to RGB for consistent UI rendering."""
    normalized = ImageOps.exif_transpose(image)
    if normalized.mode not in {"RGB", "L"}:
        normalized = normalized.convert("RGB")
    return normalized


def assess_quality(image: Image.Image) -> ImageQuality:
    rgb = prepare_image(image).convert("RGB")
    arr = np.asarray(rgb).astype(np.float32) / 255.0
    channel_delta = float(np.mean(np.max(arr, axis=2) - np.min(arr, axis=2)))
    grayscale = channel_delta < 0.03
    brightness = float(np.mean(arr))
    contrast = float(np.std(arr))

    notes: list[str] = []
    if not grayscale:
        notes.append("Image is not grayscale; confirm this is a radiograph export.")
    if contrast < 0.12:
        notes.append("Low contrast may hide subtle findings.")
    if brightness < 0.18:
        notes.append("Image appears dark; exposure/windowing may need review.")
    if brightness > 0.82:
        notes.append("Image appears bright; exposure/windowing may need review.")
    if rgb.width < 512 or rgb.height < 512:
        notes.append("Resolution is low for subtle radiographic details.")

    if not notes:
        notes.append("Basic image quality checks look acceptable for a demo read.")

    return ImageQuality(
        width=rgb.width,
        height=rgb.height,
        grayscale=grayscale,
        brightness=round(brightness, 3),
        contrast=round(contrast, 3),
        notes=notes,
    )


def route_anatomy(image: Image.Image) -> Anatomy:
    """Chest-first placeholder router.

    The first production upgrade should use MedSigLIP/CXR Foundation embeddings here.
    """
    rgb = prepare_image(image)
    aspect_ratio = rgb.width / max(rgb.height, 1)
    if 0.65 <= aspect_ratio <= 1.45:
        return Anatomy.CHEST
    return Anatomy.OTHER

