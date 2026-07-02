"""
preprocess.py — turn raw logged windows into feature tensors.

Not yet implemented. Intended behavior:
  - Load raw vibration + current windows from data/raw/.
  - Per-channel FFT / spectrogram band-energy extraction (pipeline step 3).
    Must mirror whatever CMSIS-DSP will do on-device — same window size,
    same band definitions — or the offline accuracy numbers won't transfer.
  - Emit train/val/test splits stratified by remount ID, not just by class,
    per the project's multi-remount requirement (avoid overfitting to one rig).
  - Write processed tensors to data/processed/.

Usage (once implemented):
  python preprocess.py --raw-dir ../../data/raw --out-dir ../../data/processed
"""

raise NotImplementedError("preprocess.py: pipeline step 3 not yet implemented — depends on collect_data.py producing real logged windows first")
