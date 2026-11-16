"""Load both ECG200 partitions and represent heartbeats as 2D measures."""
from pathlib import Path

import numpy as np

from wb_bary_algorithms import Measure


def load_ecg200(data_dir: Path):
    """Return amplitudes, numeric labels, partition names, and partition indices.

    Keep the archive's amplitudes unchanged. Concatenation order is training
    followed by testing, with original order preserved within each partition.
    """
    from aeon.datasets import load_classification

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    waveforms, labels, splits, indices = [], [], [], []
    for split in ("train", "test"):
        X, y = load_classification(
            "ECG200", split=split, extract_path=str(data_dir), return_metadata=False
        )
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3 or X.shape[1:] != (1, 96):
            raise ValueError(f"ECG200 {split}: expected (N, 1, 96), got {X.shape}")
        numeric_labels = np.asarray(y, dtype=np.float64)
        if numeric_labels.shape != (len(X),) or not np.isin(numeric_labels, [-1, 1]).all():
            raise ValueError(f"ECG200 {split}: expected one label in {{-1, 1}} per heartbeat")
        if not np.isfinite(X).all():
            raise ValueError(f"ECG200 {split}: amplitudes must be finite")
        waveforms.append(X[:, 0, :])
        labels.append(numeric_labels.astype(np.int64))
        splits.append(np.full(len(X), split))
        indices.append(np.arange(len(X)))
    return tuple(np.concatenate(parts) for parts in (waveforms, labels, splits, indices))


def ecg_support_indices(points=96):
    """Select evenly spaced original indices, retaining both endpoints."""
    if not isinstance(points, (int, np.integer)) or not 2 <= points <= 96:
        raise ValueError("points must be an integer between 2 and 96")
    return np.rint(np.linspace(0, 95, points)).astype(np.int64)


def waveforms_to_measures(waveforms, points=96):
    """Use retained (j/95, amplitude[j]) supports with uniform mass 1/points.

    The default retains all 96 points. A smaller count approximates each
    heartbeat by selecting original amplitudes without interpolation.
    """
    waveforms = np.asarray(waveforms, dtype=np.float64)
    if waveforms.ndim != 2 or waveforms.shape[1] != 96 or not len(waveforms):
        raise ValueError("Expected a nonempty (N, 96) waveform array")
    if not np.isfinite(waveforms).all():
        raise ValueError("ECG amplitudes must be finite")
    indices = ecg_support_indices(points)
    times = np.linspace(0.0, 1.0, 96)[indices]
    return [Measure(np.column_stack((times, amplitudes[indices])), np.full(points, 1.0 / points))
            for amplitudes in waveforms]
