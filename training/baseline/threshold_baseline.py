"""
threshold_baseline.py — hand-tuned rule-based fault classifier.

MANDATORY, not optional (see project constraints). The CNN must beat this
baseline on held-out data or the ML justification for this project collapses
under interview scrutiny.

Not yet implemented. Intended approach:
  - Simple thresholds/rules on interpretable features: RMS vibration energy,
    dominant frequency band energy, current draw mean/variance.
  - No learned parameters beyond manually-set thresholds (tuned on train split,
    evaluated on the same held-out test split the CNN uses — apples to apples).
  - Report accuracy/precision/recall per class into benchmarks/results/,
    same format the CNN results use, so the comparison is direct.

Usage (once implemented):
  python threshold_baseline.py --data-dir ../../data/processed
"""

raise NotImplementedError("threshold_baseline.py: pipeline step 5 not yet implemented — depends on preprocess.py producing real feature tensors first")
