"""Ingestion and validation of raw long-format CSV exports.

Raw rows that cannot be trusted are *rejected into a report* instead of
silently dropped, so every discarded row is traceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SIGNAL_COLUMNS = ["timestamp", "tag_name", "value"]
LAB_COLUMNS = ["timestamp", "parameter", "value", "sample_id"]


@dataclass
class IngestResult:
    """Valid rows plus a rejection log (reason per rejected row)."""

    data: pd.DataFrame
    rejected: pd.DataFrame
    n_raw: int

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def _validate(df: pd.DataFrame, required: list[str], tz: str, source: str) -> IngestResult:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing required column(s) {missing}; expected {required}")
    n_raw = len(df)
    df = df[required].copy()
    df["_row"] = range(len(df))

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=False, format="mixed")
    # Naive timestamps are interpreted in the source timezone; aware ones are kept.
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    df["timestamp"] = ts.dt.tz_convert("UTC")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    name_col = required[1]
    reason = pd.Series("", index=df.index, dtype=object)
    reason[df["timestamp"].isna()] = "bad_timestamp"
    reason[(reason == "") & df[name_col].isna()] = "missing_name"
    reason[(reason == "") & df["value"].isna()] = "non_numeric_value"
    reason[(reason == "") & ~df["value"].apply(lambda v: pd.notna(v) and abs(v) != float("inf"))] = "non_finite_value"

    bad = reason != ""
    rejected = df.loc[bad].assign(reason=reason[bad]).drop(columns="_row")
    good = df.loc[~bad].drop(columns="_row")
    return IngestResult(data=good.reset_index(drop=True), rejected=rejected.reset_index(drop=True), n_raw=n_raw)


def load_signals(path: str | Path, tz: str = "UTC") -> IngestResult:
    """Load and validate a long-format signal CSV (``timestamp, tag_name, value``)."""
    return _validate(pd.read_csv(path, low_memory=False), SIGNAL_COLUMNS, tz, str(path))


def load_lab(path: str | Path, tz: str = "UTC") -> IngestResult:
    """Load and validate a long-format lab CSV (``timestamp, parameter, value, sample_id``)."""
    df = pd.read_csv(path, low_memory=False)
    if "sample_id" not in df.columns:
        df["sample_id"] = None
    return _validate(df, LAB_COLUMNS, tz, str(path))
