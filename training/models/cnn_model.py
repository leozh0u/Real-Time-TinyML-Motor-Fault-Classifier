"""
cnn_model.py — small 1D-CNN fault classifier definitions (pipeline step 4).

Input is the FFT band-energy feature vector produced by
training/data_prep/preprocess.py: N_BANDS values per channel, ordered
low-to-high frequency. The 1D conv operates along that band axis — adjacent
bands are correlated (a real fault's energy usually spills into neighboring
bands, not one isolated bin), so 1D locality is a meaningful inductive bias
here, not just a name.

Three variants share one backbone so the vibration-only / current-only /
fused ablation in the project's "core technical pipeline" step 4 is a fair
comparison — same architecture family, only the input channel count and
which columns of X get fed in differ.

`width` scales every conv layer's channel count. Used later by
training/quantization/{ptq,qat}.py to produce the 2-3 model sizes the
accuracy/latency/RAM tradeoff frontier needs, without hand-writing a
separate class per size.

Not trained yet — no train.py exists here. That needs real data
(data/processed/, from preprocess.py) before it means anything to write.
"""

import numpy as np
import torch
import torch.nn as nn

N_BANDS = 8  # must match training/data_prep/preprocess.py's N_BANDS


class _ConvBackbone(nn.Module):
    def __init__(self, in_channels: int, width: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(width * 2)
        # Two distinct ReLU instances, not one reused twice — torch.ao.quantization's
        # fuse_modules (used in quantization/ptq.py and qat.py) fuses each
        # conv+bn+relu triplet by module reference, and a shared ReLU module used
        # in two places breaks that. Cheap to get right now, painful to debug later.
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pool over the band axis
        self.out_features = width * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, N_BANDS)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.pool(x).squeeze(-1)  # (batch, out_features)


class _FaultClassifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 4, width: int = 8):
        super().__init__()
        self.backbone = _ConvBackbone(in_channels, width)
        self.head = nn.Linear(self.backbone.out_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class VibrationOnlyCNN(_FaultClassifier):
    """Input: (batch, 1, N_BANDS) — vibration channel band energies only."""
    def __init__(self, n_classes: int = 4, width: int = 8):
        super().__init__(in_channels=1, n_classes=n_classes, width=width)


class CurrentOnlyCNN(_FaultClassifier):
    """Input: (batch, 1, N_BANDS) — current channel band energies only."""
    def __init__(self, n_classes: int = 4, width: int = 8):
        super().__init__(in_channels=1, n_classes=n_classes, width=width)


class FusedCNN(_FaultClassifier):
    """Input: (batch, 2, N_BANDS) — vibration and current band energies
    stacked as two input channels, band-aligned: index i of channel 0 is
    the vibration energy for the same frequency band as index i of
    channel 1's current energy."""
    def __init__(self, n_classes: int = 4, width: int = 8):
        super().__init__(in_channels=2, n_classes=n_classes, width=width)


def split_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """preprocess.py writes each row of X as
    concat([vib_bands (N_BANDS,), cur_bands (N_BANDS,)]) — see
    build_dataset() there. Splits that back into the two channels."""
    return X[:, :N_BANDS], X[:, N_BANDS:]


def to_model_input(X: np.ndarray, variant: str) -> torch.Tensor:
    """X: (N, 2*N_BANDS) as saved by preprocess.py (X_train.npy etc).
    variant: 'vibration', 'current', or 'fused'. Returns a tensor shaped
    to match the corresponding model class's expected input."""
    vib, cur = split_features(X)
    if variant == "vibration":
        arr = vib[:, None, :]
    elif variant == "current":
        arr = cur[:, None, :]
    elif variant == "fused":
        arr = np.stack([vib, cur], axis=1)
    else:
        raise ValueError(f"unknown variant '{variant}', expected vibration/current/fused")
    return torch.from_numpy(arr.astype(np.float32))


def build_model(variant: str, n_classes: int = 4, width: int = 8) -> nn.Module:
    return {
        "vibration": VibrationOnlyCNN,
        "current": CurrentOnlyCNN,
        "fused": FusedCNN,
    }[variant](n_classes=n_classes, width=width)


if __name__ == "__main__":
    # Sanity check only — confirms shapes and param counts, not accuracy.
    # Real training needs data/processed/ from preprocess.py first.
    for variant in ("vibration", "current", "fused"):
        model = build_model(variant)
        dummy_X = np.zeros((4, 2 * N_BANDS), dtype=np.float32)
        x = to_model_input(dummy_X, variant)
        out = model(x)
        print(f"{variant:10s} input={tuple(x.shape)} output={tuple(out.shape)} params={model.param_count()}")
