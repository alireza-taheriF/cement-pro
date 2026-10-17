"""Time alignment: native-period resampling, common-grid synchronization and lab joining."""

from __future__ import annotations

import pandas as pd


def to_native_series(df: pd.DataFrame, period_s: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Bin one tag's rows to its native period (nearest-slot, mean for collisions) on a regular index.

    Missing slots are NaN, so gaps are explicit.
    """
    freq = f"{period_s}s"
    s = df.set_index(df["timestamp"].dt.round(freq))["value"]
    s = s.groupby(level=0).mean()
    idx = pd.date_range(start.ceil(freq), end.floor(freq), freq=freq, tz="UTC")
    return s.reindex(idx)


def to_common_grid(native: dict[str, pd.Series], period_s: int = 60) -> pd.DataFrame:
    """Place all native series on one shared grid.

    Slower tags are linearly interpolated *between* real samples only, never
    across a missing native sample, so no data is invented here.
    """
    freq = f"{period_s}s"
    start = min(s.index.min() for s in native.values())
    end = max(s.index.max() for s in native.values())
    grid = pd.date_range(start.ceil(freq), end.floor(freq), freq=freq, tz="UTC")
    cols = {}
    for tag, s in native.items():
        native_step = (s.index[1] - s.index[0]).total_seconds() if len(s) > 1 else period_s
        on_grid = s.reindex(grid)
        if native_step > period_s:
            # interpolate only between two adjacent *valid* native samples
            valid = s.dropna()
            filled = on_grid.interpolate(method="time", limit_area="inside")
            adjacent = valid.index.to_series().diff().reindex(grid, method="bfill")
            gap_too_long = adjacent > pd.Timedelta(seconds=native_step * 1.5)
            on_grid = filled.where(~gap_too_long.fillna(False))
        cols[tag] = on_grid
    return pd.DataFrame(cols, index=grid)


def align_lab(lab: pd.DataFrame, grid_index: pd.DatetimeIndex, lag_minutes: dict[str, float] | None = None) -> pd.DataFrame:
    """Return the lab samples as a wide frame indexed by *sampling* time, optionally shifted by a known lag.

    ``lag_minutes`` maps lab parameter -> minutes between the process state a
    sample represents and its logged timestamp (positive: logged later).
    The frame keeps only samples inside the signal span.
    """
    lag_minutes = lag_minutes or {}
    lab = lab.copy()
    lab["sample_time"] = lab["timestamp"] - pd.to_timedelta(lab["tag"].map(lambda t: lag_minutes.get(t, 0.0)), unit="m")
    wide = lab.pivot_table(index="sample_time", columns="tag", values="value", aggfunc="mean")
    return wide[(wide.index >= grid_index.min()) & (wide.index <= grid_index.max())]


def join_lab_to_grid(grid: pd.DataFrame, lab_wide: pd.DataFrame, tolerance_min: float = 30.0) -> pd.DataFrame:
    """Attach each lab sample to the nearest grid row (within tolerance); other rows stay NaN."""
    out = grid.copy()
    for col in lab_wide.columns:
        out[f"LAB_{col}"] = float("nan")
        s = lab_wide[col].dropna()
        if s.empty:
            continue
        pos = grid.index.get_indexer(s.index, method="nearest", tolerance=pd.Timedelta(minutes=tolerance_min))
        ok = pos >= 0
        out.iloc[pos[ok], out.columns.get_loc(f"LAB_{col}")] = s.to_numpy()[ok]
    return out
