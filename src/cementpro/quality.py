"""Data-quality detection: spikes, stuck (flat-line) sensors, gaps and per-tag quality metrics.

All detectors work on a single tag resampled to its native period (a
``pd.Series`` with a regular ``DatetimeIndex`` where missing samples are NaN),
and return boolean masks aligned to that index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import TagSpec

_MAD_TO_SIGMA = 1.4826


def robust_sigma_of_diff(s: pd.Series) -> float:
    """Robust noise scale from the first difference (immune to isolated spikes)."""
    d = s.diff().dropna()
    if d.empty:
        return 0.0
    return float(_MAD_TO_SIGMA * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def detect_spikes(s: pd.Series, window: int = 9, sigmas: float = 6.0) -> pd.Series:
    """Hampel-style filter: flag points deviating from the rolling median by more than ``sigmas`` noise units.

    One noise unit is the robust sigma of the first difference of the whole
    series (see :func:`robust_sigma_of_diff`). Using a global noise scale
    instead of the rolling MAD keeps the detector sensitive on slowly
    drifting signals, where a windowed MAD is inflated by the drift itself.
    """
    med = s.rolling(window, center=True, min_periods=max(3, window // 3)).median()
    sigma = max(robust_sigma_of_diff(s), 1e-9)
    return ((s - med).abs() > sigmas * sigma).fillna(False)


def detect_flatline(s: pd.Series, min_samples: int) -> pd.Series:
    """Flag runs of ``>= min_samples`` identical consecutive readings (stuck sensor / frozen value).

    NaN breaks a run: a gap is not evidence of a flat line.
    """
    valid = s.notna()
    # run id increments whenever the value changes or a NaN is hit
    change = (s != s.shift()) | ~valid | ~valid.shift(fill_value=False)
    run_id = change.cumsum()
    run_len = s.groupby(run_id).transform("size")
    return (valid & (run_len >= min_samples)).fillna(False)


def gap_runs(is_missing: pd.Series) -> pd.DataFrame:
    """Return one row per maximal run of missing samples: ``start, end, n_samples``."""
    m = is_missing.to_numpy()
    if not m.any():
        return pd.DataFrame(columns=["start", "end", "n_samples"])
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    idx = is_missing.index
    return pd.DataFrame({"start": idx[starts], "end": idx[ends], "n_samples": ends - starts + 1})


def tag_quality(
    spec: TagSpec,
    raw_n: int,
    n_duplicates: int,
    n_out_of_range: int,
    native: pd.Series,
    spikes: pd.Series,
    flat: pd.Series,
    long_gap_minutes: float = 60.0,
) -> dict:
    """Summarize the quality of one tag over the observed time span (before cleaning)."""
    n_slots = len(native)
    observed = native.notna()
    gaps = gap_runs(~observed)
    period_min = spec.sample_period_s / 60.0
    gap_minutes = gaps["n_samples"] * period_min if len(gaps) else pd.Series(dtype=float)
    return {
        "tag": spec.name,
        "unit": spec.unit,
        "period_s": spec.sample_period_s,
        "raw_rows": int(raw_n),
        "expected_slots": int(n_slots),
        "completeness_pct": round(100.0 * observed.sum() / n_slots, 2) if n_slots else 0.0,
        "duplicates": int(n_duplicates),
        "out_of_range": int(n_out_of_range),
        "spikes": int(spikes.sum()),
        "flatline_samples": int(flat.sum()),
        "gaps": int(len(gaps)),
        "max_gap_min": float(gap_minutes.max()) if len(gaps) else 0.0,
        "long_gaps": int((gap_minutes >= long_gap_minutes).sum()) if len(gaps) else 0,
        "noise_sigma": round(robust_sigma_of_diff(native), 4),
    }
