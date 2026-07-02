# data/raw/

Raw logged sensor windows, one subfolder per class, one subfolder per physical remount run within that.

Expected structure once data collection starts:

```
data/raw/<class>/<run_id>/*.csv (or .npy)
```

Classes: `healthy`, `imbalance`, `looseness`, `overload` (must match `fault_class_t` in `firmware/Core/Inc/main.h`).

Multiple remount runs per class are required — the model must not overfit to one physical rig setup. This directory is gitignored (raw logs are large); only a small representative slice goes in `data/sample/` and is committed.

Populated by `training/data_prep/collect_data.py` (not yet implemented — blocked on firmware step 1, sensor wiring, plus the load resistor fix for the overload class).
