"""Finance OHLCV -> Measure converter for d=3 Wasserstein barycenter.

Reads OHLCV CSV, computes three non-timestamp features per bar:

    Dim 0: Return          = ln(close_t / close_{t-1})
    Dim 1: Normalized Volume = volume_t / mean(volume over rolling window)
    Dim 2: Volatility      = rolling std of returns

The `date` column is used only for sorting/windowing, then dropped — no
timestamps enter the measure support coordinates.

Each input measure P_i = (X_i, w_i) represents one window of n bars embedded
in R^3, with X.shape == (n, 3) and w = uniform 1/n.  Scaling parameters are
returned so barycenter atoms can be inverse-transformed back into
interpretable financial units (bps return, relative volume multiple,
annualized volatility %).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

from wb_bary_algorithms import Measure


@dataclass
class FinanceScaler:
    """Per-feature mean/std for inverse-transform of barycenter atoms."""
    mean: np.ndarray   # shape (3,)
    std: np.ndarray    # shape (3,)


def load_ohlcv(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    return df


def compute_features(
    df: pd.DataFrame,
    vol_window: int = 20,
) -> Tuple[np.ndarray, FinanceScaler]:
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    # Dim 0: Log return  r_t = ln(C_t / C_{t-1})
    ret = np.empty_like(close)
    ret[0] = 0.0
    ret[1:] = np.log(close[1:] / close[:-1])

    # Dim 1: Normalized volume = V_t / rolling mean volume
    vol_roll = (
        pd.Series(volume)
        .rolling(window=vol_window, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    norm_vol = volume / np.maximum(vol_roll, 1e-9)

    # Dim 2: Volatility = rolling std of returns (window-sized)
    ret_std = (
        pd.Series(ret)
        .rolling(window=vol_window, min_periods=2)
        .std()
        .to_numpy(dtype=np.float64)
    )
    ret_std = np.nan_to_num(ret_std, nan=0.0)

    features = np.column_stack([ret, norm_vol, ret_std]).astype(np.float64)

    # Winsorize extreme tails (0.5% / 99.5%) per feature
    lo = np.percentile(features, 0.5, axis=0)
    hi = np.percentile(features, 99.5, axis=0)
    features = np.clip(features, lo[None, :], hi[None, :])

    # Z-score standardize per feature so no axis dominates the transport cost
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    scaler = FinanceScaler(mean=feat_mean, std=feat_std)
    safe_std = np.where(feat_std > 1e-9, feat_std, 1.0)
    features = (features - feat_mean[None, :]) / safe_std[None, :]

    return features, scaler


def window_measures(
    features: np.ndarray,
    k: int,
    n_per_window: int | None = None,
) -> List[Measure]:
    """Split the feature matrix into k contiguous windows, each a Measure."""
    m = len(features)
    if k < 1 or k > m:
        raise ValueError(f"Require 1 <= k <= {m} rows, got k={k}, m={m}")

    if n_per_window is None:
        n_per_window = m // k
    if n_per_window < 1:
        n_per_window = 1

    used = k * n_per_window
    if used < m:
        print(
            f"window_measures: using first {used} of {m} rows "
            f"(dropping {m - used} trailing rows) for equal-size windows",
            file=sys.stderr,
        )
        features = features[:used]

    measures: List[Measure] = []
    for i in range(k):
        start = i * n_per_window
        end = start + n_per_window
        X = features[start:end]
        w = np.full(len(X), 1.0 / len(X))
        measures.append(Measure(X=X, w=w))
    return measures


def finance_to_measures(
    path: Path | str,
    k: int = 10,
    vol_window: int = 20,
    n_per_window: int | None = None,
) -> Tuple[List[Measure], FinanceScaler, pd.DataFrame]:
    """Full pipeline: OHLCV CSV -> k Measures(X, w) in R^3, no timestamps."""
    df = load_ohlcv(path)
    features, scaler = compute_features(df, vol_window=vol_window)
    measures = window_measures(features, k=k, n_per_window=n_per_window)
    return measures, scaler, df
