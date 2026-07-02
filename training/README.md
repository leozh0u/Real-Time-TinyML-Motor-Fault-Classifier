# training/

Python side of the pipeline: data prep -> model -> baseline -> quantization -> export.

## Layout

- `data_prep/` — turn raw logged sensor windows (`data/raw/`) into labeled, windowed feature tensors (`data/processed/`). Includes FFT/spectrogram band-energy extraction (pipeline step 3).
- `models/` — 1D-CNN definitions in PyTorch. Per-channel (vibration-only, current-only) and fused variants for the ablation study (pipeline step 4).
- `baseline/` — hand-tuned threshold/rule-based classifier. Mandatory: the CNN must beat this or the ML framing doesn't hold (pipeline step 5).
- `quantization/` — PTQ and QAT scripts, swept across 2-3 model sizes, producing the accuracy/latency/RAM tradeoff frontier (pipeline step 6).
- `export/` — ONNX export of the chosen deployment-point model for STM32Cube AI Studio ingestion (pipeline step 7).

## Order to build in

1. `data_prep/` (needs real logged data first — see `data/raw/README.md`)
2. `baseline/` (cheap, and sets the bar the CNN has to clear)
3. `models/` + training loop
4. `quantization/` sweep
5. `export/`

## Setup

```
pip install -r ../requirements.txt
```
