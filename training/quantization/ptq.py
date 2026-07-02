"""
ptq.py — post-training quantization sweep.

Not yet implemented. Intended behavior (pipeline step 6, PTQ half):
  - Take a trained float32 model checkpoint from training/models/.
  - Apply PyTorch PTQ (static, calibrated on a representative subset of the
    train split) at 2-3 model sizes.
  - Measure: accuracy delta vs float32, estimated latency, estimated RAM
    (activation + weight footprint) for each size.
  - Write results to benchmarks/results/ in the same schema as qat.py's output
    so both feed the same tradeoff-frontier plot.

Usage (once implemented):
  python ptq.py --checkpoint <path> --model-size small|medium|large
"""

raise NotImplementedError("ptq.py: pipeline step 6 not yet implemented — depends on a trained float32 model from training/models/ first")
