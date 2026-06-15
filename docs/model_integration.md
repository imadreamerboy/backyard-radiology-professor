# Model integration

One llama.cpp router serves two aliases through `http://127.0.0.1:8080/v1/chat/completions`:

- `medgemma-professor`: MedGemma 27B Q4_K_M, 8K context, full GPU offload.
- `medgemma-localizer`: MedGemma 1.5 4B Q4_K_M, 4K context.

`runtime/models.local-wsl.ini` is the local desktop profile: 6K context and full
GPU offload. Docker Compose selects that profile through
`RAD_TRAINER_LLAMA_PRESET`; the Modal backend uses `runtime/models.ini`.

The router runs with `--models-max 2`, and both aliases are loaded at startup on
the L40S backend profile. Session JSON and rendered image arrays persist under
the `/data` volume so a chat can resume after Modal scales the container down and
wakes a new one.

Inference order:

1. X-Raydar classifies the primary PA/AP image, then returns its models to CPU.
2. MedGemma 1.5 4B generates structured observations and bounding boxes.
3. MedGemma 27B produces a structured professor review and remains available for streamed chat.

Every accepted model output records model identity, runtime, status, and latency. Chat additionally records time to first token and throughput when llama.cpp reports token usage. Invalid structured localizer output is shown as an explicit failed model run.

The professor receives the current radiographs, the committed blind read, sanitized
study metadata, X-Raydar predictions, MedGemma observations, accepted regions,
reference labels for bundled public cases, and recent chat history.
