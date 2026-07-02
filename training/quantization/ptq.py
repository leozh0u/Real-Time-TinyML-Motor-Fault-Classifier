"""
ptq.py — post-training quantization study (pipeline step 6, PTQ half).

IMPORTANT — read before treating this script's quantized model as "the
deployed model": it is not. Actual on-device int8 conversion is done by
STM32Cube AI Studio when it turns the exported ONNX model into C code (see
training/export/export_onnx.py). This script's job is to characterize how
much accuracy PTQ alone costs at a given model size, in PyTorch, on your
dev machine, before you commit to a deployment size. It answers "is PTQ
good enough at this size, or do I need QAT (qat.py) instead" — it does not
produce the literal bytes that run on the F401. Don't cite "PyTorch PTQ
accuracy" and "on-device accuracy" as the same number in the README —
measure the real one after step 8, on hardware.

Not runnable yet: needs a trained float32 checkpoint (no train.py exists
yet — write that once data/processed/ has real data) and requires PyTorch's
fuse_modules layout in models/cnn_model.py's _ConvBackbone (conv1/bn1/relu1,
conv2/bn2/relu2) to stay in sync with the fuse list below.

Usage (once a checkpoint exists):
    python ptq.py --checkpoint <path> --variant fused --width 8 \
        --data-dir ../../data/processed
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.ao.quantization as tq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from cnn_model import build_model, to_model_input  # noqa: E402

FUSE_LIST = [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]]


def evaluate(model, X: torch.Tensor, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1).numpy()
    return float((preds == y).mean())


def load_split(data_dir: Path, split: str, variant: str):
    X_raw = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy")
    X = to_model_input(X_raw, variant)
    return X, y


def run_ptq(checkpoint: str, variant: str, width: int, data_dir: Path):
    model = build_model(variant, width=width)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    X_train, y_train = load_split(data_dir, "train", variant)
    X_test, y_test = load_split(data_dir, "test", variant)

    float_acc = evaluate(model, X_test, y_test)
    float_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # --- fuse + prepare + calibrate + convert (eager-mode static PTQ) -----
    tq.fuse_modules(model.backbone, FUSE_LIST, inplace=True)
    model.qconfig = tq.get_default_qconfig("fbgemm")  # x86 dev-machine backend —
    # irrelevant to the ARM target, this is purely for measuring the accuracy
    # delta on your laptop, not for producing deployable weights.
    tq.prepare(model, inplace=True)

    # Calibration: run a representative subset of train data through in eval
    # mode so PTQ can observe real activation ranges. 200 samples is a
    # starting point, not a validated number — revisit once real data exists.
    calib_n = min(200, len(X_train))
    with torch.no_grad():
        model(X_train[:calib_n])

    tq.convert(model, inplace=True)
    quant_acc = evaluate(model, X_test, y_test)

    result = {
        "method": "PTQ",
        "variant": variant,
        "width": width,
        "float_test_acc": float_acc,
        "quant_test_acc": quant_acc,
        "acc_delta": quant_acc - float_acc,
        "float_param_bytes": float_bytes,
        # Rough estimate only — int8 quantized state_dict size isn't a clean
        # 4x-smaller number once you account for scale/zero-point overhead
        # and unquantized layers (BatchNorm folds into conv, but the model
        # head's dtype depends on qconfig). Treat this as a starting
        # estimate, get the real number from STM32Cube AI Studio later.
        "approx_int8_param_bytes": float_bytes // 4,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True, choices=["vibration", "current", "fused"])
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--data-dir", default="../../data/processed")
    parser.add_argument("--out-dir", default="../../benchmarks/results")
    args = parser.parse_args()

    result = run_ptq(args.checkpoint, args.variant, args.width, Path(args.data_dir))
    print(result)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ptq_{args.variant}_w{args.width}.json"
    import json
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
