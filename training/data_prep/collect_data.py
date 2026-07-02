"""
collect_data.py — serial logging tool for labeled vibration + current windows.

Not yet implemented. Intended behavior:
  - Open UART connection to the Nucleo (pyserial).
  - Firmware streams raw ADXL345 + ACS712 samples (see firmware/Core/Src/main.c step 1).
  - Tag incoming windows with a class label + run/remount ID passed via CLI args,
    so training/val/test splits can be stratified by remount (not just by class)
    to avoid overfitting to one physical rig setup.
  - Write raw windows to data/raw/<class>/<run_id>/*.csv or .npy.

Usage (once implemented):
  python collect_data.py --port /dev/ttyACM0 --class healthy --run-id remount01
"""

raise NotImplementedError("collect_data.py: pipeline step 2 not yet implemented — write after firmware step 1 (sensor wiring) is verified on the bench")
