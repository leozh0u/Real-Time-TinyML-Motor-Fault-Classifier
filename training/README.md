# training/

Python side of the pipeline: data prep -> model -> baseline -> quantization -> export.

## Layout

- `data_prep/` — real code, committed:
  - `collect_data.py` — pyserial receiver for the firmware's UART sample stream, writes labeled runs to `data/raw/<class>/<run_id>/`
  - `preprocess.py` — FFT band-energy extraction (must match `firmware/Drivers/BSP/fft_features.c` exactly, see root README "Open items"), group-aware train/val/test split by remount ID
- `models/` — `cnn_model.py`: real PyTorch 1D-CNN definitions (`VibrationOnlyCNN`, `CurrentOnlyCNN`, `FusedCNN`) sharing one backbone for a fair ablation comparison. No `train.py` yet — write that once `data/processed/` has real data.
- `baseline/` — `threshold_baseline.py`: real feature computation + hierarchical threshold classifier, auto-tuned from train-split data. Mandatory: the CNN must beat this or the ML framing doesn't hold (pipeline step 5). Threshold values are meaningless until run against real data.
- `quantization/` — `ptq.py`, `qat.py`: real PyTorch quantization scripts. Read the docstrings — these measure a PyTorch-side proxy for the tradeoff frontier, not literally the model that runs on the F401 (STM32Cube AI Studio does that conversion in step 7).
- `export/` — `export_onnx.py`: real ONNX export + optional onnxruntime validation against the PyTorch model's output.

## What's still missing

No `train.py` exists — none of the quantization/export scripts are runnable until there's a trained float32 checkpoint, which needs real data, which needs firmware step 1 wiring done on actual hardware first. Write `train.py` once `data/processed/` exists.

## Order to build in

1. `data_prep/` (needs real logged data first — see `data/raw/README.md`)
2. `baseline/` (cheap, and sets the bar the CNN has to clear)
3. `models/` + a training loop (not written yet)
4. `quantization/` sweep
5. `export/`

## Setup

```
pip install -r ../requirements.txt
```
