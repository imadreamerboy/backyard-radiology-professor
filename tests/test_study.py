from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    DigitalXRayImageStorageForPresentation,
    ExplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGLSLossless,
    generate_uid,
)
from pydicom.pixels import get_decoder

from radiology_trainer.study import load_study


def test_dicom_study_sorts_frontal_before_lateral_and_renders(tmp_path: Path) -> None:
    lateral = _write_dicom(tmp_path / "lateral.dcm", view="LATERAL", instance=2)
    frontal = _write_dicom(tmp_path / "frontal.dcm", view="PA", instance=1)
    study = load_study(
        study_id="study",
        title="Test",
        source="upload",
        paths=[lateral, frontal],
        work_dir=tmp_path,
    )
    assert study.public.images[0].projection == "PA"
    assert study.public.primary_image_id == study.public.images[0].id
    assert study.primary().render().size == (64, 48)
    assert study.public.images[0].metadata.pixel_spacing_mm == (0.2, 0.2)


def test_multiframe_dicom_becomes_ordered_study_images(tmp_path: Path) -> None:
    path = _write_dicom(tmp_path / "multi.dcm", frames=2)
    study = load_study(
        study_id="study",
        title="Multi",
        source="upload",
        paths=[path],
        work_dir=tmp_path,
    )
    assert len(study.public.images) == 2
    assert [item.metadata.frame_number for item in study.public.images] == [1, 2]


def test_non_chest_or_non_xray_dicom_is_rejected(tmp_path: Path) -> None:
    path = _write_dicom(tmp_path / "ct.dcm", modality="CT")
    with pytest.raises(ValueError, match="Only CR and DX"):
        load_study(
            study_id="study",
            title="CT",
            source="upload",
            paths=[path],
            work_dir=tmp_path,
        )


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("../outside.dcm", b"unsafe")
    with pytest.raises(ValueError, match="unsafe path"):
        load_study(
            study_id="study",
            title="Unsafe",
            source="upload",
            paths=[archive],
            work_dir=tmp_path / "work",
        )


def test_monochrome1_inverts_default_render(tmp_path: Path) -> None:
    mono1 = _write_dicom(tmp_path / "mono1.dcm", photometric="MONOCHROME1")
    study = load_study(
        study_id="study",
        title="Mono",
        source="upload",
        paths=[mono1],
        work_dir=tmp_path,
    )
    default = np.asarray(study.primary().render())
    inverted = np.asarray(study.primary().render(invert=True))
    assert np.mean(default) > np.mean(inverted)


@pytest.mark.parametrize(
    "transfer_syntax",
    [JPEGBaseline8Bit, JPEGLSLossless, JPEG2000Lossless],
)
def test_required_compressed_dicom_decoders_are_available(transfer_syntax) -> None:
    assert get_decoder(transfer_syntax).is_available


def _write_dicom(
    path: Path,
    *,
    view: str = "PA",
    instance: int = 1,
    modality: str = "DX",
    photometric: str = "MONOCHROME2",
    frames: int = 1,
) -> Path:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = modality
    dataset.BodyPartExamined = "CHEST"
    dataset.ViewPosition = view
    dataset.InstanceNumber = instance
    dataset.Rows = 48
    dataset.Columns = 64
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.WindowCenter = 1000
    dataset.WindowWidth = 2000
    dataset.PixelSpacing = [0.2, 0.2]
    pixels = np.linspace(0, 2000, dataset.Rows * dataset.Columns, dtype=np.uint16)
    if frames > 1:
        dataset.NumberOfFrames = frames
        pixels = np.stack([pixels.reshape(dataset.Rows, dataset.Columns)] * frames)
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return path
