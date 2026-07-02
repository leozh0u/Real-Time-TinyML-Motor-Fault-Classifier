"""
qat.py — quantization-aware training sweep.

Not yet implemented. Intended behavior (pipeline step 6, QAT half):
  - Same 2-3 model sizes as ptq.py, but fine-tune with fake-quant nodes
    inserted (PyTorch QAT) instead of pure post-hoc calibration.
  - Same metrics, same output schema as ptq.py, so results merge directly
    into one accuracy/latency/RAM tradeoff frontier across all
    (model size x PTQ/QAT) combinations.
  - The deployment model (fed to training/export/) gets picked off this
    frontier, not assumed in advance.

Usage (once implemented):
  python qat.py --model-size small|medium|large --epochs <n>
"""

raise NotImplementedError("qat.py: pipeline step 6 not yet implemented — depends on training/models/ + a working training loop first")
