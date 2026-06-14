from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
from PIL import Image

from radiology_trainer.domain import (
    DicomMetadata,
    ReferenceAnswer,
    Study,
    StudyImage,
    WindowPreset,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DICOM_SUFFIXES = {".dcm", ".dicom", ""}
SUPPORTED_MODALITIES = {"CR", "DX"}


@dataclass
class StudyImageRecord:
    public: StudyImage
    pixels: np.ndarray
    invert_default: bool

    def render(
        self,
        *,
        center: float | None = None,
        width: float | None = None,
        invert: bool = False,
    ) -> Image.Image:
        preset = self.public.window_presets[0]
        level = center if center is not None else preset.center
        window = width if width is not None else preset.width
        low = level - window / 2
        high = level + window / 2
        if high <= low:
            high = low + 1
        normalized = np.clip((self.pixels - low) / (high - low), 0.0, 1.0)
        if self.invert_default ^ invert:
            normalized = 1.0 - normalized
        return Image.fromarray((normalized * 255).astype(np.uint8), mode="L").convert("RGB")


@dataclass
class StudyRecord:
    public: Study
    images: dict[str, StudyImageRecord]

    def primary(self) -> StudyImageRecord:
        return self.images[self.public.primary_image_id]

    def ordered(self) -> list[StudyImageRecord]:
        return [self.images[item.id] for item in self.public.images]


def load_study(
    *,
    study_id: str,
    title: str,
    source: str,
    paths: Iterable[Path],
    work_dir: Path,
    case_id: str | None = None,
    reference: ReferenceAnswer | None = None,
    max_files: int = 32,
    max_uncompressed_mb: int = 500,
) -> StudyRecord:
    expanded = _expand_paths(
        paths,
        work_dir=work_dir,
        max_files=max_files,
        max_uncompressed_bytes=max_uncompressed_mb * 1024 * 1024,
    )
    records: list[StudyImageRecord] = []
    for path in expanded:
        records.extend(_load_path(path))
    if not records:
        raise ValueError("No supported chest radiograph images were found.")
    if len(records) > max_files:
        raise ValueError(f"A study may contain at most {max_files} images.")

    records.sort(
        key=lambda item: (
            _projection_rank(item.public.projection),
            item.public.instance_number,
            item.public.id,
        )
    )
    primary = next(
        (
            item
            for item in records
            if item.public.projection.upper() in {"PA", "AP", "FRONTAL"}
        ),
        records[0],
    )
    public = Study(
        id=study_id,
        title=title,
        source=source,
        case_id=case_id,
        images=[item.public for item in records],
        primary_image_id=primary.public.id,
        reference=reference,
    )
    return StudyRecord(public=public, images={item.public.id: item for item in records})


def _expand_paths(
    paths: Iterable[Path],
    *,
    work_dir: Path,
    max_files: int,
    max_uncompressed_bytes: int,
) -> list[Path]:
    expanded: list[Path] = []
    for source in paths:
        if source.suffix.lower() != ".zip":
            expanded.append(source)
            continue
        extract_dir = work_dir / f"archive-{uuid4().hex}"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            files = [item for item in archive.infolist() if not item.is_dir()]
            if len(files) > max_files:
                raise ValueError(f"A study archive may contain at most {max_files} files.")
            if sum(item.file_size for item in files) > max_uncompressed_bytes:
                raise ValueError("The uncompressed study archive is too large.")
            root = extract_dir.resolve()
            for item in files:
                destination = (extract_dir / item.filename).resolve()
                if root not in destination.parents:
                    raise ValueError("The study archive contains an unsafe path.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source_file, destination.open("wb") as target:
                    target.write(source_file.read())
                expanded.append(destination)
    return expanded


def _load_path(path: Path) -> list[StudyImageRecord]:
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return [_load_standard_image(path)]
    if path.suffix.lower() in DICOM_SUFFIXES:
        return _load_dicom(path)
    return []


def _load_standard_image(path: Path) -> StudyImageRecord:
    with Image.open(path) as source:
        image = source.convert("L")
        pixels = np.asarray(image, dtype=np.float32)
    metadata = DicomMetadata(
        modality="IMAGE",
        rows=int(pixels.shape[0]),
        columns=int(pixels.shape[1]),
        bits_stored=8,
        photometric_interpretation="MONOCHROME2",
    )
    return _record_from_pixels(
        pixels,
        label=path.stem,
        projection="FRONTAL",
        instance_number=1,
        metadata=metadata,
        invert_default=False,
    )


def _load_dicom(path: Path) -> list[StudyImageRecord]:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("Install DICOM support with `uv sync --extra dicom`.") from exc

    try:
        dataset = pydicom.dcmread(path)
        modality = str(getattr(dataset, "Modality", "")).upper()
        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported DICOM modality {modality or '[missing]'}. "
                "Only CR and DX chest radiographs are accepted."
            )
        body_part = str(getattr(dataset, "BodyPartExamined", "")).upper()
        if body_part and body_part not in {"CHEST", "THORAX"}:
            raise ValueError(f"Unsupported body part {body_part}. Only chest studies are accepted.")
        array = np.asarray(dataset.pixel_array, dtype=np.float32)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not decode DICOM file {path.name}: {exc}") from exc

    slope = _float_value(getattr(dataset, "RescaleSlope", 1.0)) or 1.0
    intercept = _float_value(getattr(dataset, "RescaleIntercept", 0.0)) or 0.0
    array = array * slope + intercept
    frames = _split_frames(array, int(getattr(dataset, "NumberOfFrames", 1) or 1))
    records: list[StudyImageRecord] = []
    for index, pixels in enumerate(frames):
        metadata = _metadata(dataset, frame=index + 1 if len(frames) > 1 else None)
        records.append(
            _record_from_pixels(
                pixels,
                label=_image_label(dataset, index, len(frames)),
                projection=metadata.view_position or "FRONTAL",
                instance_number=int(getattr(dataset, "InstanceNumber", 0) or 0) + index,
                metadata=metadata,
                invert_default=metadata.photometric_interpretation.upper() == "MONOCHROME1",
            )
        )
    return records


def _split_frames(array: np.ndarray, number_of_frames: int) -> list[np.ndarray]:
    if array.ndim == 2:
        return [array]
    if array.ndim == 3 and number_of_frames > 1:
        return [array[index] for index in range(min(number_of_frames, array.shape[0]))]
    raise ValueError("Only grayscale single-frame or multi-frame radiographs are supported.")


def _metadata(dataset, *, frame: int | None) -> DicomMetadata:
    spacing = _pair(getattr(dataset, "PixelSpacing", None)) or _pair(
        getattr(dataset, "ImagerPixelSpacing", None)
    )
    return DicomMetadata(
        modality=str(getattr(dataset, "Modality", "")),
        sop_class_uid=str(getattr(dataset, "SOPClassUID", "")),
        patient_name=str(getattr(dataset, "PatientName", "")),
        patient_id=str(getattr(dataset, "PatientID", "")),
        patient_age=str(getattr(dataset, "PatientAge", "")),
        patient_sex=str(getattr(dataset, "PatientSex", "")),
        study_date=str(getattr(dataset, "StudyDate", "")),
        accession_number=str(getattr(dataset, "AccessionNumber", "")),
        institution_name=str(getattr(dataset, "InstitutionName", "")),
        body_part_examined=str(getattr(dataset, "BodyPartExamined", "")),
        view_position=str(
            getattr(dataset, "ViewPosition", "")
            or getattr(dataset, "PatientPosition", "")
        ),
        laterality=str(getattr(dataset, "ImageLaterality", "") or getattr(dataset, "Laterality", "")),
        rows=int(getattr(dataset, "Rows", 0) or 0),
        columns=int(getattr(dataset, "Columns", 0) or 0),
        bits_stored=int(getattr(dataset, "BitsStored", 0) or 0),
        photometric_interpretation=str(
            getattr(dataset, "PhotometricInterpretation", "")
        ),
        window_center=_float_value(getattr(dataset, "WindowCenter", None)),
        window_width=_float_value(getattr(dataset, "WindowWidth", None)),
        pixel_spacing_mm=spacing,
        frame_number=frame,
    )


def _record_from_pixels(
    pixels: np.ndarray,
    *,
    label: str,
    projection: str,
    instance_number: int,
    metadata: DicomMetadata,
    invert_default: bool,
) -> StudyImageRecord:
    values = np.asarray(pixels, dtype=np.float32)
    presets = _window_presets(values, metadata)
    image_id = uuid4().hex
    public = StudyImage(
        id=image_id,
        label=label,
        projection=projection.upper(),
        instance_number=instance_number,
        width=int(values.shape[1]),
        height=int(values.shape[0]),
        image_url="",
        metadata=metadata,
        window_presets=presets,
    )
    return StudyImageRecord(public=public, pixels=values, invert_default=invert_default)


def _window_presets(values: np.ndarray, metadata: DicomMetadata) -> list[WindowPreset]:
    p01, p05, p20, p50, p80, p95, p99 = np.percentile(
        values, [1, 5, 20, 50, 80, 95, 99]
    )
    original_width = metadata.window_width or max(float(p99 - p01), 1.0)
    original_center = metadata.window_center or float((p99 + p01) / 2)
    return [
        WindowPreset(id="original", label="Original", center=original_center, width=original_width),
        WindowPreset(
            id="lung",
            label="Lung",
            center=float((p95 + p05) / 2),
            width=max(float(p95 - p05), 1.0),
        ),
        WindowPreset(
            id="mediastinum",
            label="Mediastinum",
            center=float((p80 + p20) / 2),
            width=max(float(p80 - p20), 1.0),
        ),
        WindowPreset(
            id="bone",
            label="Bone",
            center=float((p99 + p50) / 2),
            width=max(float(p99 - p50), 1.0),
        ),
        WindowPreset(
            id="soft-tissue",
            label="Soft tissue",
            center=float(p50),
            width=max(float(p95 - p20), 1.0),
        ),
    ]


def _image_label(dataset, index: int, frame_count: int) -> str:
    projection = str(getattr(dataset, "ViewPosition", "") or "Radiograph").upper()
    if frame_count > 1:
        return f"{projection} frame {index + 1}"
    return projection


def _projection_rank(projection: str) -> int:
    value = projection.upper()
    if value == "PA":
        return 0
    if value == "AP":
        return 1
    if value in {"FRONTAL", ""}:
        return 2
    if "LAT" in value:
        return 3
    return 4


def _float_value(value) -> float | None:
    if value is None:
        return None
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        value = list(value)[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pair(value) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        items = list(value)
        if len(items) < 2:
            return None
        return float(items[0]), float(items[1])
    except (TypeError, ValueError):
        return None
