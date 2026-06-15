from __future__ import annotations

import os
import subprocess

import modal


APP_NAME = "backyard-radiology-professor"
VOLUME_NAME = "backyard-radiology-professor-cache"
HF_SECRET_NAME = "backyard-radiology-professor-hf"


def _hf_secret() -> modal.Secret:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name(HF_SECRET_NAME)


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.from_dockerfile("Dockerfile.modal", context_dir=".")


@app.function(
    image=image,
    gpu=["L40S", "L4"],
    volumes={"/data": cache_volume},
    secrets=[_hf_secret()],
    timeout=2 * 60 * 60,
    scaledown_window=30 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=20)
@modal.web_server(7860, startup_timeout=30 * 60)
def serve() -> None:
    subprocess.Popen(["bash", "scripts/entrypoint.sh"], cwd="/workspace")
