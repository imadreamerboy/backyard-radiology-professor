# Model integration

One llama.cpp router serves two aliases through `http://127.0.0.1:8080/v1/chat/completions`:

- `medgemma-professor`: MedGemma 27B Q4_K_M, 8K context, full GPU offload.
- `medgemma-localizer`: MedGemma 1.5 4B Q4_K_M, 4K context.

`runtime/models.local-wsl.ini` is the local desktop profile: 6K context and full
GPU offload. Docker Compose selects
that profile through `RAD_TRAINER_LLAMA_PRESET`; the Space uses
`runtime/models.ini`.

The router runs with `--models-max 1`. The application explicitly unloads the inactive model before loading the next one.

Inference order:

1. X-Raydar classifies the primary PA/AP image, then returns its models to CPU.
2. MedGemma 1.5 4B generates structured observations and bounding boxes.
3. The localizer unloads and MedGemma 27B loads.
4. MedGemma 27B produces a structured professor review and remains available for streamed chat.

Every accepted model output records model identity, runtime, status, and latency. Chat additionally records time to first token and throughput when llama.cpp reports token usage. Invalid structured localizer output is shown as an explicit failed model run.

The professor receives the current radiographs, the committed blind read, sanitized
study metadata, X-Raydar predictions, MedGemma observations, accepted regions,
reference labels for bundled public cases, and recent chat history.
