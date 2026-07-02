# Real-Time TinyML Motor Fault Classifier

Real-time motor fault classification on an STM32F401RE (96KB RAM) using a quantized int8 CNN over fused vibration and current sensing. Centered on a PTQ vs QAT efficiency study with a measured accuracy-latency-RAM tradeoff frontier, benchmarked against a mandatory rule-based baseline.

On-device predictive maintenance on an STM32F401RE (Nucleo, Cortex-M4, 96KB RAM, 512KB flash). Classifies motor fault conditions in real time from fused vibration (ADXL345, SPI) + current (ACS712, ADC) sensing, using a quantized int8 CNN running fully on-device — no PC in the loop at inference time.

This is not a "run a model on a microcontroller" demo. The centerpiece is a quantization/efficiency study: PTQ vs QAT across multiple model sizes, with a measured accuracy-latency-RAM tradeoff frontier, and a mandatory non-ML threshold baseline the CNN has to beat.

## Status

Scaffold stage. Hardware is on hand except the load resistor (see Open Items). No data collected yet, no model trained yet.

## Why this project (context for reviewers)

This is project 2 of 3 on the author's resume. Project 1 is a standalone STM32 motor controller (bare-metal C, closed-loop PID, encoder) — that one proves embedded/systems C. This project is deliberately scoped to prove a disjoint capability: the full ML lifecycle (data → train → quantize → deploy → measure). It reuses the motor rig but the point of the project is the ML pipeline, not "I did more embedded stuff."

## Fault classes

Healthy, imbalance (added mass on shaft), mechanical looseness (loosened mount), graded overload (via coupled load motor + resistive braking). 4-5 classes total.

## Repo structure

```
firmware/           STM32CubeIDE project: PID motor control + sensor sampling + on-device FFT + CMSIS-NN inference
  Core/Src, Core/Inc   Application code
  Drivers/CMSIS-DSP    FFT / DSP kernels
  Drivers/CMSIS-NN     Quantized conv kernels
  Drivers/BSP          ADXL345 / ACS712 / encoder drivers
  model_c/             Generated C model code (STM32Cube AI Studio output) — not hand-written

training/            Python: data prep, model, quantization, baseline, export
  data_prep/           Windowing, labeling, FFT/spectrogram feature extraction
  models/               1D-CNN definitions (per-channel + fused variants)
  baseline/             Hand-tuned threshold/rule-based classifier (mandatory baseline)
  quantization/         PTQ and QAT scripts, sweep across model sizes
  export/               ONNX export for STM32Cube AI Studio ingestion

data/
  raw/                  Raw logged sensor windows per run/remount
  processed/            Extracted features / windowed tensors ready for training
  sample/               Small representative dataset sample committed to the repo

benchmarks/
  results/              Metrics tables (accuracy, latency, RAM, flash, fusion ablation)
  plots/                 Pareto/tradeoff frontier plots

docs/                  System diagram, wiring notes, design decisions
```

## Pipeline (build order)

1. Wire ADXL345 (SPI) + ACS712 (ADC) + encoder (timer) alongside existing PID motor control
2. Collect labeled vibration + current windows per class, across multiple physical remounts
3. Feature extraction: FFT/spectrogram band energies per channel
4. Model: small 1D-CNN in PyTorch — per-channel and fused (vibration-only vs current-only vs fused) ablation
5. Build and beat a hand-tuned threshold baseline (mandatory — CNN must win or the ML framing doesn't hold)
6. Quantization sweep: 2-3 model sizes x PTQ vs QAT -> accuracy/latency/RAM tradeoff frontier; pick a deployment point off the frontier
7. Convert (ONNX) -> STM32Cube AI Studio -> C code
8. On-device: CMSIS-DSP FFT + CMSIS-NN inference, size the tensor arena, real-time classification while the motor runs, stream label over UART
9. Profile: flash size, RAM/arena size, ms/inference, float-vs-int8 accuracy delta, fusion ablation results, margin over baseline

## Toolchain decisions (locked)

- Firmware IDE: **STM32CubeIDE** (not PlatformIO) — chosen because STM32Cube AI Studio plugs directly into CubeIDE/CubeMX, no manual integration work.
- Model deployment runtime: **TBD — pick STM32Cube AI Studio or TFLite Micro, not both.** Not yet decided; decide before step 7.
- Explicitly excluded: NanoEdge AI Studio (AutoML black-box, indefensible in interviews), any LLM-on-device framing, upgrading off the F401RE

## Open items

- **Load resistor is wrong.** Currently have a 10K signal potentiometer, which draws negligible current at 12V and won't create real braking torque. Need a low-ohm (5-50 ohm) high-wattage rheostat or cement power resistor before the graded-overload fault class (via coupled load motor + resistive braking) can be collected.
- **ACS712 output voltage may exceed the F401's ADC range.** Not caught in the original hardware list. Most ACS712 breakout boards run their sensor supply off 5V, centering 0A output at ~2.5V with a +-925mV swing at the rated +-5A — the high end (~3.425V) is above the F401's 3.3V VDDA. Confirm the breakout can run its sensor supply off 3.3V, or add a resistor divider / op-amp buffer before wiring OUT to the ADC pin. See `firmware/Drivers/BSP/acs712.h` for detail.
- Verify shaft coupler torque rating (0.5Nm) against JGB37-520 stall torque before running the load class.
- Toolchain decision above (Cube AI Studio vs TFLite Micro) not yet made.
- STM32CubeMX `.ioc` not yet generated — `firmware/Core` peripheral init is all TODO placeholders until that exists.
- **FFT config duplicated by hand between C and Python.** `firmware/Drivers/BSP/fft_features.c` (on-device) and `training/data_prep/preprocess.py` (offline) both hardcode the same window size, Hann window, and band edges — there is no shared source of truth. If you change one, change the other, or offline accuracy silently stops matching on-device accuracy.
- **UART baud rate duplicated by hand.** `firmware/Drivers/BSP/uart_stream.h`'s protocol assumes 115200 baud; `training/data_prep/collect_data.py`'s `BAUD_RATE` must match whatever CubeMX's USART2 config actually uses. Same footgun as above, different pair of files.
- `training/quantization/ptq.py` and `qat.py` measure accuracy on a PyTorch-side quantized model as a proxy for the tradeoff frontier — the model that actually runs on the F401 is whatever STM32Cube AI Studio produces from the exported ONNX float model in step 7. Don't conflate the two numbers; see the docstrings in those files.

## Timeline

Realistic full-time compressed floor: ~2.5-3 weeks (16-20 working days). Evenings-only: ~5-6 weeks. Data collection (3-4 days min) and on-device deployment debugging (4-6 days) are the two bottlenecks — don't compress these first.
