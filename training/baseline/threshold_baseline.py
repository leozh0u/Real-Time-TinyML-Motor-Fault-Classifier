"""
threshold_baseline.py — hand-tuned rule-based fault classifier.

MANDATORY per the project spec, not optional: the CNN must beat this on
held-out data or the ML justification for this project doesn't hold up
under interview scrutiny. This has to stay legitimately "non-ML" —
interpretable thresholds a human could sanity-check, not a second learned
model in disguise.

Uses the same processed features as the CNN (X_{split}.npy from
preprocess.py — FFT band energies) for an apples-to-apples comparison: same
inputs, the difference being hand-set threshold rules here vs learned conv
weights there. If you'd rather this baseline use simpler raw-signal
features (RMS, peak) instead of FFT bands, that's a legitimate alternative
design — just be consistent about what "beats the baseline" is actually
comparing.

Threshold *values* below are tuned automatically from train-split data via
best_threshold_1d() — a brute-force scan over candidate cutoffs on one
feature at a time, maximizing train accuracy at each decision point. This
keeps every threshold traceable to "the value that best separates classes
X and Y on this one feature," which is what makes it a defensible
rule-based baseline rather than an opaque model — but the actual numbers
are meaningless until real data exists. Do not treat any number in this
file as validated before that.

Usage (once data/processed/ has real data):
    python threshold_baseline.py --data-dir ../../data/processed
"""

import argparse
import json
from pathlib import Path

import numpy as np

VALID_CLASSES = ("healthy", "imbalance", "looseness", "overload")
N_BANDS = 8  # must match training/data_prep/preprocess.py's N_BANDS


def derive_features(X: np.ndarray) -> dict[str, np.ndarray]:
    """X: (N, 2*N_BANDS) as saved by preprocess.py — vib bands then current
    bands. Returns simple, interpretable scalar features per window."""
    vib = X[:, :N_BANDS]
    cur = X[:, N_BANDS:]
    return {
        "vib_energy": vib.sum(axis=1),
        "vib_dominant_band": vib.argmax(axis=1).astype(np.float32),
        "cur_energy": cur.sum(axis=1),
        "cur_variance": cur.var(axis=1),
    }


def best_threshold_1d(values: np.ndarray, is_positive: np.ndarray) -> tuple[float, float]:
    """Brute-force scan for the threshold on `values` that best separates
    is_positive (bool array) from the rest, by train accuracy. Returns
    (threshold, train_accuracy_at_that_threshold). O(n log n), fine at
    this dataset size."""
    candidates = np.unique(values)
    best_thresh, best_acc = candidates[0], 0.0
    for t in candidates:
        pred_positive = values >= t
        acc = (pred_positive == is_positive).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, t
    return float(best_thresh), float(best_acc)


def tune(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Greedy hierarchical threshold tuning, mirroring how a human would
    actually build this baseline by hand: pick the split that separates the
    class currently easiest to isolate, then recurse on what's left.
    Order below (healthy -> overload -> imbalance/looseness) is a starting
    assumption based on which classes are physically most distinct, not a
    validated result — revisit once real data shows which split order
    actually works best."""
    feats = derive_features(X_train)
    class_idx = {name: i for i, name in enumerate(VALID_CLASSES)}

    # Split 1: healthy vs everything else, on total vibration energy
    is_healthy = y_train == class_idx["healthy"]
    t_healthy, acc_healthy = best_threshold_1d(feats["vib_energy"], ~is_healthy)

    # Split 2 (within not-healthy): overload vs imbalance/looseness, on current energy
    not_healthy_mask = ~is_healthy
    is_overload = y_train == class_idx["overload"]
    if not_healthy_mask.any():
        t_overload, acc_overload = best_threshold_1d(
            feats["cur_energy"][not_healthy_mask], is_overload[not_healthy_mask]
        )
    else:
        t_overload, acc_overload = 0.0, 0.0

    # Split 3 (within imbalance/looseness): on dominant vibration band —
    # imbalance is typically concentrated at 1x rotation frequency (a low
    # band), looseness tends to spread energy across harmonics (higher
    # bands too). This is a physically-motivated guess, not a measured
    # result — verify against real FFT plots once data exists.
    remaining_mask = not_healthy_mask & ~is_overload
    is_imbalance = y_train == class_idx["imbalance"]
    if remaining_mask.any():
        t_band, acc_band = best_threshold_1d(
            feats["vib_dominant_band"][remaining_mask], is_imbalance[remaining_mask]
        )
    else:
        t_band, acc_band = 0.0, 0.0

    return {
        "t_healthy_vib_energy": t_healthy,
        "t_overload_cur_energy": t_overload,
        "t_imbalance_dominant_band": t_band,
        "train_acc_at_each_split": {
            "healthy_vs_rest": acc_healthy,
            "overload_vs_rest": acc_overload,
            "imbalance_vs_looseness": acc_band,
        },
    }


def classify(X: np.ndarray, thresholds: dict) -> np.ndarray:
    feats = derive_features(X)
    class_idx = {name: i for i, name in enumerate(VALID_CLASSES)}
    preds = np.full(len(X), class_idx["healthy"], dtype=np.int64)

    not_healthy = feats["vib_energy"] >= thresholds["t_healthy_vib_energy"]
    preds[not_healthy] = class_idx["looseness"]  # default within not-healthy, overwritten below

    is_overload = not_healthy & (feats["cur_energy"] >= thresholds["t_overload_cur_energy"])
    preds[is_overload] = class_idx["overload"]

    is_imbalance = (
        not_healthy
        & ~is_overload
        & (feats["vib_dominant_band"] >= thresholds["t_imbalance_dominant_band"])
    )
    preds[is_imbalance] = class_idx["imbalance"]
    # everything else not-healthy/not-overload/not-imbalance stays "looseness" (the default above)

    return preds


def evaluate(preds: np.ndarray, y: np.ndarray) -> dict:
    overall_acc = float((preds == y).mean())
    per_class = {}
    for i, name in enumerate(VALID_CLASSES):
        mask = y == i
        if mask.any():
            per_class[name] = float((preds[mask] == y[mask]).mean())
    return {"overall_acc": overall_acc, "per_class_acc": per_class}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="../../data/processed")
    parser.add_argument("--out-dir", default="../../benchmarks/results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X_train, y_train = np.load(data_dir / "X_train.npy"), np.load(data_dir / "y_train.npy")
    X_test, y_test = np.load(data_dir / "X_test.npy"), np.load(data_dir / "y_test.npy")

    thresholds = tune(X_train, y_train)
    print("Tuned thresholds:", json.dumps(thresholds, indent=2))

    test_preds = classify(X_test, thresholds)
    results = evaluate(test_preds, y_test)
    results["thresholds"] = thresholds
    print("Test results:", json.dumps(results, indent=2))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "threshold_baseline.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path} — this is the number every CNN variant/size must beat.")


if __name__ == "__main__":
    main()
