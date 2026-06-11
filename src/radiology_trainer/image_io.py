from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from radiology_trainer.preprocessing import prepare_image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DICOM_SUFFIXES = {".dcm", ".dicom", ""}


def load_xray_image(path: str | Path) -> Image.Image:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix in IMAGE_SUFFIXES:
        with Image.open(source) as image:
            return prepare_image(image.copy()).convert("RGB")

    if suffix in DICOM_SUFFIXES:
        return _load_dicom(source)

    raise ValueError(f"Unsupported file type: {suffix or '[none]'}")


def _load_dicom(path: Path) -> Image.Image:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("Install DICOM support with `uv sync --extra dicom`.") from exc

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    arr = _window_array(arr, ds)
    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        arr = 1.0 - arr

    return Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB")


def _window_array(arr: np.ndarray, ds) -> np.ndarray:
    center = _dicom_value(getattr(ds, "WindowCenter", None))
    width = _dicom_value(getattr(ds, "WindowWidth", None))

    if center is not None and width is not None and width > 0:
        low = center - width / 2
        high = center + width / 2
    else:
        low, high = np.percentile(arr, [1, 99])

    if high <= low:
        high = low + 1.0

    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _dicom_value(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        value = list(value)[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

