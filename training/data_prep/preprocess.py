"""
preprocess.py — turn raw logged windows (data/raw/) into feature tensors
(data/processed/), via FFT band-energy extraction.

MUST mirror firmware/Drivers/BSP/fft_features.c exactly: same window size,
same Hann window formula, same band edges. There is no shared source of
truth between the C and Python implementations right now — if you change
WINDOW_SIZE, SAMPLE_RATE_HZ, or BAND_EDGES_HZ here, change fft_features.c
(and fft_features.h's FFT_WINDOW_SIZE) to match, or the offline accuracy
numbers this produces will not transfer to the on-device model. This is a
real footgun, not a hypothetical one — see README "Open items".

Windowing: raw CSVs are one row per sample, produced continuously during a
run. This script segments each run into non-overlapping WINDOW_SIZE-sample
windows (matching how the firmware will do it on-device — it doesn't
overlap windows either).

Splits: stratified by (class, run_id) *group*, not by individual window —
all windows from one physical remount land in the same split. Splitting at
the window level would leak near-identical adjacent windows from the same
remount across train/val/test and produce inflated accuracy that won't
hold up when a fresh remount is fed to the on-device model. This matches
the project's explicit requirement to validate against multiple physical
remounts, not just multiple windows of the same mount.

Usage:
    python preprocess.py --raw-dir ../../data/raw --out-dir ../../data/processed
"""

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from sklearn.model_selection import GroupShuffleSplit
except ImportError as e:
    raise ImportError("Missing dependency: pip install scikit-learn") from e

# --- must match firmware/Drivers/BSP/fft_features.h / .c exactly ----------
SAMPLE_RATE_HZ = 800
WINDOW_SIZE = 256
BAND_EDGES_HZ = [5.0, 8.9, 15.8, 28.1, 50.0, 88.9, 158.1, 281.2, 400.0]
N_BANDS = len(BAND_EDGES_HZ) - 1
# ----------------------------------------------------------------------------

VALID_CLASSES = ("healthy", "imbalance", "looseness", "overload")


def band_energies(signal: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """FFT band-energy extraction for one window of one channel. Must match
    firmware/Drivers/BSP/fft_features.c's ComputeMagnitudeSpectrum +
    BandEnergiesFromMagnitude exactly: same Hann window, same rFFT,
    same |bin|^2 sum per band."""
    assert len(signal) == WINDOW_SIZE, f"expected window of {WINDOW_SIZE}, got {len(signal)}"

    window = np.hanning(WINDOW_SIZE)  # matches firmware's 0.5*(1-cos(2*pi*n/(N-1))) exactly
    windowed = signal * window

    spectrum = np.fft.rfft(windowed)          # bins 0..N/2, complex
    magnitude = np.abs(spectrum)               # matches arm_cmplx_mag_f32 + DC handling

    bin_hz = sample_rate_hz / WINDOW_SIZE
    bands = np.zeros(N_BANDS, dtype=np.float32)
    for i in range(N_BANDS):
        lo_bin = int(BAND_EDGES_HZ[i] / bin_hz)
        hi_bin = min(int(BAND_EDGES_HZ[i + 1] / bin_hz), len(magnitude) - 1)
        bands[i] = np.sum(magnitude[lo_bin:hi_bin + 1] ** 2)
    return bands


def load_run_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (vib_x array, current_mA array) for one logged run file."""
    vib, cur = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vib.append(float(row["vib_x"]))
            cur.append(float(row["current_mA"]))
    return np.array(vib, dtype=np.float32), np.array(cur, dtype=np.float32)


def windows_from_run(vib: np.ndarray, cur: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Non-overlapping WINDOW_SIZE segments. Drops the trailing partial
    window (matches the firmware, which never fires on a partial buffer)."""
    n_windows = len(vib) // WINDOW_SIZE
    out = []
    for i in range(n_windows):
        s = slice(i * WINDOW_SIZE, (i + 1) * WINDOW_SIZE)
        out.append((vib[s], cur[s]))
    return out


def build_dataset(raw_dir: Path):
    """Walks data/raw/<class>/<run_id>/*.csv and returns:
      X: (n_windows, 2*N_BANDS) feature array — vib bands then current bands, concatenated
      y: (n_windows,) integer class labels
      groups: (n_windows,) string "<class>/<run_id>" — for group-aware splitting
    """
    X, y, groups = [], [], []
    for class_idx, class_name in enumerate(VALID_CLASSES):
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            continue
        for run_dir in sorted(class_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            for csv_path in sorted(run_dir.glob("*.csv")):
                vib, cur = load_run_csv(csv_path)
                for vib_win, cur_win in windows_from_run(vib, cur):
                    vib_bands = band_energies(vib_win, SAMPLE_RATE_HZ)
                    cur_bands = band_energies(cur_win, SAMPLE_RATE_HZ)
                    X.append(np.concatenate([vib_bands, cur_bands]))
                    y.append(class_idx)
                    groups.append(f"{class_name}/{run_id}")

    if not X:
        raise RuntimeError(
            f"No windows found under {raw_dir}. Nothing has been collected yet — "
            f"run training/data_prep/collect_data.py against real hardware first."
        )
    return np.stack(X), np.array(y), np.array(groups)


def group_split(X, y, groups, test_frac=0.15, val_frac=0.15, seed=0):
    """Two-stage GroupShuffleSplit: holds whole (class, run_id) groups
    together so no remount leaks across splits."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(X, y, groups))

    val_frac_of_trainval = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=seed)
    train_idx_rel, val_idx_rel = next(gss2.split(X[trainval_idx], y[trainval_idx], groups[trainval_idx]))
    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    return {
        "train": (X[train_idx], y[train_idx]),
        "val": (X[val_idx], y[val_idx]),
        "test": (X[test_idx], y[test_idx]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="../../data/raw")
    parser.add_argument("--out-dir", default="../../data/processed")
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups = build_dataset(raw_dir)
    print(f"Loaded {len(X)} windows across {len(set(groups))} (class, run) groups.")
    for class_idx, class_name in enumerate(VALID_CLASSES):
        n = int((y == class_idx).sum())
        print(f"  {class_name}: {n} windows")

    n_groups = len(set(groups))
    if n_groups < 6:
        print(f"WARNING: only {n_groups} distinct (class, run) groups — group-aware splitting "
              f"needs multiple remounts per class to be meaningful. With this few groups, "
              f"the val/test splits may end up unrepresentative or empty for some classes.")

    splits = group_split(X, y, groups, args.test_frac, args.val_frac, args.seed)
    for split_name, (X_split, y_split) in splits.items():
        np.save(out_dir / f"X_{split_name}.npy", X_split)
        np.save(out_dir / f"y_{split_name}.npy", y_split)
        print(f"  {split_name}: {len(X_split)} windows -> {out_dir}/X_{split_name}.npy")


if __name__ == "__main__":
    main()
