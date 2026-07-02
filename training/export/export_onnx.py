"""
export_onnx.py — export the chosen deployment-point model to ONNX
(pipeline step 7).

Exports the FLOAT32 model (a QAT-fine-tuned checkpoint from qat.py, or a
plain float checkpoint if PTQ turned out good enough — see ptq.py/qat.py's
"which model actually ships" note). STM32Cube AI Studio does the actual
int8 conversion when it turns this ONNX file into C code — this script
does not attempt to replicate or bypass that.

Input shape must match training/models/cnn_model.py's to_model_input()
output for whichever variant you're deploying: (1, 1, N_BANDS) for
vibration/current-only, (1, 2, N_BANDS) for fused. Batch size 1 is
intentional — the deployed model runs one window at a time, not batched.

Usage (once a checkpoint exists):
    python export_onnx.py --checkpoint <qat_float_checkpoint.pt> --variant fused --width 8 \
        --out ../../firmware/model_c/model.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from cnn_model import N_BANDS, build_model  # noqa: E402


def input_shape_for(variant: str) -> tuple[int, int, int]:
    in_channels = {"vibration": 1, "current": 1, "fused": 2}[variant]
    return (1, in_channels, N_BANDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True, choices=["vibration", "current", "fused"])
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-with", default=None,
                         help="Optional path to a data/processed X_test.npy to validate "
                              "ONNX output matches PyTorch output before handing off.")
    args = parser.parse_args()

    model = build_model(args.variant, width=args.width)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy_input = torch.zeros(input_shape_for(args.variant), dtype=torch.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["features"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes=None,  # fixed batch size 1 — matches the on-device single-window inference
    )
    print(f"Exported {out_path}")

    if args.check_with:
        _validate(model, out_path, args.check_with, args.variant)


def _validate(model, onnx_path: Path, x_test_path: str, variant: str):
    try:
        import onnxruntime as ort
    except ImportError:
        print("Skipping validation: pip install onnxruntime", file=sys.stderr)
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from cnn_model import to_model_input  # noqa: E402

    X_raw = np.load(x_test_path)[:8]  # small batch, checked one row at a time below
    X = to_model_input(X_raw, variant)

    session = ort.InferenceSession(str(onnx_path))
    with torch.no_grad():
        for i in range(len(X)):
            single = X[i : i + 1]
            torch_out = model(single).numpy()
            onnx_out = session.run(None, {"features": single.numpy()})[0]
            if not np.allclose(torch_out, onnx_out, atol=1e-4):
                print(f"MISMATCH at row {i}: torch={torch_out} onnx={onnx_out}", file=sys.stderr)
                sys.exit(1)
    print(f"Validated: ONNX output matches PyTorch output on {len(X)} rows.")


if __name__ == "__main__":
    main()
