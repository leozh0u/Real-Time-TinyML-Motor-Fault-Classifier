"""
cnn_model.py — small 1D-CNN fault classifier definitions.

Not yet implemented. Intended behavior (pipeline step 4):
  - Define three variants sharing a common conv-block builder for a clean ablation:
      VibrationOnlyCNN  — vibration channel features only
      CurrentOnlyCNN    — current channel features only
      FusedCNN          — both channels fused (concat or two-branch + merge)
  - Keep the architecture small and explicit about parameter count — this feeds
    directly into the quantization/model-size sweep in training/quantization/.
  - Output: fault_class_t-compatible logits, class order must match
    firmware/Core/Inc/main.h's fault_class_t enum exactly.

No training loop here — see a future train.py that imports these and
training/baseline/threshold_baseline.py for the accuracy bar to clear.
"""

raise NotImplementedError("cnn_model.py: pipeline step 4 not yet implemented — depends on preprocess.py producing real feature tensors first")
