"""
export_onnx.py — export the chosen deployment-point model to ONNX.

Not yet implemented. Intended behavior (pipeline step 7):
  - Take the model checkpoint corresponding to the chosen point on the
    accuracy/latency/RAM tradeoff frontier (from ptq.py / qat.py results).
  - Export to ONNX with a fixed input shape matching what firmware feature
    extraction (CMSIS-DSP FFT output) will actually produce.
  - Validate the ONNX model's outputs match the PyTorch model's outputs
    on a held-out batch before handing off to STM32Cube AI Studio.

Usage (once implemented):
  python export_onnx.py --checkpoint <path> --out ../../firmware/model_c/model.onnx
"""

raise NotImplementedError("export_onnx.py: pipeline step 7 not yet implemented — depends on the quantization sweep (ptq.py/qat.py) picking a deployment point first")
