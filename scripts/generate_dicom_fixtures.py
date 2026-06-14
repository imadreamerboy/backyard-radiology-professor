from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    DigitalXRayImageStorageForPresentation,
    ExplicitVRLittleEndian,
    generate_uid,
)

from radiology_trainer.cases import demo_cases
from radiology_trainer.image_io import load_xray_image


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in demo_cases(args.xraydar_dir):
        source = Path(case.image_path)
        if not source.exists():
            raise FileNotFoundError(f"Missing public demo source: {source}")
        pixels = np.asarray(load_xray_image(source).convert("L"), dtype=np.uint8)
        target = args.output_dir / f"{case.id}-derived-dx.dcm"
        _write_derived_dx(target, pixels, case.title)
        print(f"Wrote derived fixture: {target}")


def _write_derived_dx(path: Path, pixels: np.ndarray, title: str) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "DX"
    dataset.BodyPartExamined = "CHEST"
    dataset.ViewPosition = "PA"
    dataset.SeriesDescription = f"Derived fixture: {title}"
    dataset.ImageComments = "Derived from the public X-Raydar demo pixels for software testing."
    dataset.PatientName = "PUBLIC^DEMO"
    dataset.PatientID = "DERIVED-FIXTURE"
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.WindowCenter = 127.5
    dataset.WindowWidth = 255
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clearly labeled DX DICOM fixtures from public demo pixels."
    )
    parser.add_argument("--xraydar-dir", default="outputs/xraydar-cv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dicom-fixtures"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
