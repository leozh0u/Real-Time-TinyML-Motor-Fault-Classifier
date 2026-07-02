# benchmarks/results/

Metrics tables: per-model accuracy, latency (ms/inference), RAM (tensor arena size), flash footprint, float-vs-int8 accuracy delta, fusion ablation (vibration-only vs current-only vs fused), and margin over the threshold baseline.

Expect one CSV/JSON per sweep run from `training/quantization/ptq.py` and `qat.py`, using a shared schema so they merge into one tradeoff-frontier table. Also holds the on-device profiling numbers from firmware pipeline step 9 once measured on real hardware — no placeholder numbers, only measured ones.

Empty until the quantization sweep runs.
