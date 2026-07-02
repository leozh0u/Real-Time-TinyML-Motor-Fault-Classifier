"""
qat.py — quantization-aware training study (pipeline step 6, QAT half).

Same "this is not the deployed model" caveat as ptq.py — read that file's
docstring first. What QAT actually buys you here: the fine-tuned FLOAT32
weights this produces are trained with fake-quant noise in the loop, so
they're more robust to whatever quantization STM32Cube AI Studio applies
later. The artifact that matters for deployment is qat_model's float
state_dict after fine-tuning (fed to export_onnx.py), not the fully
`convert()`-ed int8 PyTorch model measured here — that conversion is only
done in this script to measure the accuracy delta for the tradeoff
frontier, on your dev machine's quantization backend, not the MCU's.

Not runnable yet: needs a trained float32 checkpoint as a starting point
(no train.py exists yet) and data/processed/ from preprocess.py.

Usage (once a checkpoint exists):
    python qat.py --checkpoint <path> --variant fused --width 8 --epochs 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.ao.quantization as tq
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from cnn_model import build_model, to_model_input  # noqa: E402

FUSE_LIST = [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]]


def load_split(data_dir: Path, split: str, variant: str):
    X_raw = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy")
    X = to_model_input(X_raw, variant)
    y_t = torch.from_numpy(y.astype(np.int64))
    return X, y, y_t


def evaluate(model, X: torch.Tensor, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1).numpy()
    return float((preds == y).mean())


def run_qat(checkpoint: str, variant: str, width: int, epochs: int, lr: float, data_dir: Path):
    model = build_model(variant, width=width)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    X_train, y_train, y_train_t = load_split(data_dir, "train", variant)
    X_val, y_val, _ = load_split(data_dir, "val", variant)
    X_test, y_test, _ = load_split(data_dir, "test", variant)

    float_acc = evaluate(model, X_test, y_test)

    # --- fuse + prepare_qat (fake-quant nodes inserted) --------------------
    model.train()
    tq.fuse_modules(model.backbone, FUSE_LIST, inplace=True)
    model.qconfig = tq.get_default_qat_qconfig("fbgemm")  # dev-machine backend, see caveat above
    tq.prepare_qat(model, inplace=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, X_val, y_val)
        print(f"epoch {epoch+1}/{epochs}  loss={loss.item():.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save the fine-tuned FLOAT weights before final convert() — this is
    # the artifact export_onnx.py should actually consume for the Cube AI
    # Studio deployment path.
    qat_float_checkpoint = Path(checkpoint).with_name(Path(checkpoint).stem + f"_qat_{variant}_w{width}.pt")
    torch.save(model.state_dict(), qat_float_checkpoint)

    quant_model = tq.convert(model.eval(), inplace=False)
    quant_acc = evaluate(quant_model, X_test, y_test)

    result = {
        "method": "QAT",
        "variant": variant,
        "width": width,
        "epochs": epochs,
        "float_test_acc": float_acc,
        "quant_test_acc": quant_acc,
        "acc_delta": quant_acc - float_acc,
        "best_val_acc_during_training": best_val_acc,
        "qat_float_checkpoint": str(qat_float_checkpoint),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True, choices=["vibration", "current", "fused"])
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="../../data/processed")
    parser.add_argument("--out-dir", default="../../benchmarks/results")
    args = parser.parse_args()

    result = run_qat(args.checkpoint, args.variant, args.width, args.epochs, args.lr, Path(args.data_dir))
    print(result)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qat_{args.variant}_w{args.width}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
