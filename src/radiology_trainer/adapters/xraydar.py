from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from radiology_trainer.domain import Anatomy, EvidenceBundle, FindingScore, ModelRun
from radiology_trainer.preprocessing import assess_quality, prepare_image
from radiology_trainer.runtime_manifest import (
    XRAYDAR_CODE_REVISION,
    XRAYDAR_REPO,
    XRAYDAR_REVISION,
)
from radiology_trainer.xraydar_labels import XRAYDAR_LABELS, display_xraydar_label


@dataclass
class XRaydarEvidenceModel:
    backend_dir: Path
    device_name: str = "cuda"
    name: str = "xraydar-cv"
    model_revision: str = XRAYDAR_REVISION
    offload_after_inference: bool = True
    _runtime: Any = field(default=None, init=False, repr=False)

    def analyze(self, image: Image.Image) -> EvidenceBundle:
        started = time.perf_counter()
        try:
            runtime = self._load_runtime()
            probs, urgency = runtime.predict(np.asarray(prepare_image(image).convert("L")))
        except Exception as exc:
            raise RuntimeError(f"X-Raydar inference failed: {exc}") from exc
        finally:
            if self._runtime is not None and self.offload_after_inference:
                self._runtime.offload()

        findings = [
            FindingScore(
                label=display_xraydar_label(label),
                score=float(score),
                source=self.name,
                explanation="Probability from the local X-Raydar three-model ensemble.",
            )
            for label, score in zip(XRAYDAR_LABELS, probs, strict=True)
        ]
        findings.sort(key=lambda item: item.score, reverse=True)
        return EvidenceBundle(
            anatomy=Anatomy.CHEST,
            quality=assess_quality(prepare_image(image)),
            findings=findings,
            agreement_notes=[f"X-Raydar urgency estimate: {urgency}"],
            model_notes=[
                "X-Raydar research model; independently verify every prediction.",
                f"Device: {runtime.device.type}",
            ],
            model_runs=[
                ModelRun(
                    role="classifier",
                    model_id=XRAYDAR_REPO,
                    model_source=XRAYDAR_REPO,
                    model_revision=self.model_revision,
                    runtime="PyTorch",
                    runtime_revision=XRAYDAR_CODE_REVISION,
                    status="ok",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            ],
        )

    def _load_runtime(self) -> "_XRaydarRuntime":
        if self._runtime is None:
            self._runtime = _XRaydarRuntime(self.backend_dir, self.device_name)
        return self._runtime


class _XRaydarRuntime:
    def __init__(self, backend_dir: Path, device_name: str) -> None:
        import pydicom
        import torch

        self.torch = torch
        self.device = torch.device(
            device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.backend_dir = backend_dir.resolve()
        src_dir = self.backend_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        original_torch_load = torch.load

        def trusted_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = trusted_torch_load
        if not hasattr(pydicom, "read_file"):
            pydicom.read_file = pydicom.dcmread

        import model_20210820_XNet38MS.XNet38_urg as xnet
        import model_20210820_XNet38MS.predict as predict

        self.predict_module = predict
        self.models = {}
        for size in (299, 512, 1024):
            weight_dir = (
                src_dir
                / "model_20210820_XNet38MS"
                / "model_weights"
                / f"direct_multi93_is{size}_Rv10_pre00_imagenet"
            )
            if not (weight_dir / "model_best.pth.tar").exists():
                raise FileNotFoundError(
                    f"Missing X-Raydar weights for {size}px. Run prepare_runtime.py."
                )
            model = xnet.XNet38_urg()
            model.load_state_dict(str(weight_dir))
            model.to(self.device)
            model.eval()
            self.models[size] = model

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        self.ensure_device()
        tensors = self.predict_module.prepare_data(image.astype(np.uint8))
        probabilities = []
        with self.torch.no_grad():
            for size, tensor in tensors.items():
                logits, _ = self.models[size](tensor.to(self.device))
                probabilities.append(logits[0].detach().cpu().sigmoid())

        probs = self.torch.stack(probabilities, 0).mean(0).numpy().ravel()
        urgencies = np.array(
            [1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1,
             2, 2, 3, 1, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 1, 3, 1, 2, 1]
        )
        urgency = ["normal", "non-urgent", "urgent", "critical"][
            int(np.max((probs > 0.5) * urgencies))
        ]
        return probs, urgency

    def ensure_device(self) -> None:
        for model in self.models.values():
            model.to(self.device)

    def offload(self) -> None:
        if self.device.type != "cuda":
            return
        for model in self.models.values():
            model.to("cpu")
        self.torch.cuda.empty_cache()
