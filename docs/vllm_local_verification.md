# vLLM Local Verification

Validation target: run the full app with real X-Raydar evidence and a live quantized Nemotron tutor.

## Why This Model

Use `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8` for local verification. It is a quantized NVIDIA Nemotron model, public on Hugging Face, and practical for a 24 GB RTX 4090. Keep `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` as a heavier hosted or multi-GPU target.

## Start vLLM In WSL

```powershell
uv run python scripts/run_vllm_nemotron_wsl.py --detach --enforce-eager
```

Defaults:

- Loaded model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8`
- Served model name: `nemotron3-nano-4B-FP8`
- API URL: `http://127.0.0.1:8000/v1`
- Max context for verification: `4096`
- KV cache dtype: `fp8`
- Attention backend: `TRITON_ATTN`
- FlashInfer sampler: disabled by default for WSL compatibility
- Eager mode: recommended for WSL verification
- vLLM venv: `~/.cache/radiology-trainer-vllm/venv`

Stop it:

```powershell
wsl -d Ubuntu -- bash -lc 'kill $(cat ~/.cache/radiology-trainer-vllm/server.pid)'
```

Docker alternative:

```powershell
uv run python scripts/run_vllm_nemotron.py --detach
```

## Verify vLLM Alone

```powershell
uv run python scripts/smoke_vllm_tutor.py --model nemotron3-nano-4B-FP8
```

This calls `/v1/models` and `/v1/chat/completions`, then fails unless the model returns the expected readiness token.

## Verify Full Local App

Terminal 1:

```powershell
uv run python scripts/run_xraydar_service.py --backend-dir outputs/xraydar-cv --device cpu --port 9000
```

Terminal 2:

```powershell
uv run python scripts/run_vllm_nemotron_wsl.py --detach --enforce-eager
uv run python scripts/smoke_vllm_tutor.py --model nemotron3-nano-4B-FP8
```

Terminal 3:

```powershell
$env:RAD_TRAINER_MODEL_MODE="local"
$env:RAD_TRAINER_TUTOR_PROVIDER="openai"
$env:RAD_TRAINER_OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
$env:RAD_TRAINER_OPENAI_API_KEY="local"
$env:RAD_TRAINER_NEMOTRON_MODEL="nemotron3-nano-4B-FP8"
$env:RAD_TRAINER_CHEST_EVIDENCE_URL="http://127.0.0.1:9000/analyze"
$env:RAD_TRAINER_CHEST_EVIDENCE_TIMEOUT_SECONDS="300"
uv run python app.py
```

Terminal 4:

```powershell
uv run python scripts/smoke_gradio_xraydar.py --app-url http://127.0.0.1:7860 --require-live-tutor
```

Expected proof:

- X-Raydar top finding for the upstream scoliosis demo DICOM: `scoliosis`
- X-Raydar source: `xraydar-cv`
- Tutor provider in the app response: `nemotron-tutor:openai`
- No tutor fallback text

## Verified Local Result

Verified on 2026-06-13:

- vLLM version: `0.23.0`
- GPU: RTX 4090, 24 GB
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8`
- Served model: `nemotron3-nano-4B-FP8`
- vLLM direct smoke: passed with `VLLM_OK local tutor ready.`
- Full app smoke: passed with X-Raydar top finding `scoliosis` score `0.993`
- Live tutor check: passed with provider `nemotron-tutor:openai`
- Observed GPU memory during full stack: about 22.6 GiB used

## Space Verification Later

After the Space is deployed with the same real backend URLs, run:

```powershell
uv run python scripts/smoke_gradio_xraydar.py --app-url https://<space-host> --require-live-tutor
```
