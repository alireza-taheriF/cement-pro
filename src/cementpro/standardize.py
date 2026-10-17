"""Standardization: canonical tag names, canonical units, physical-range checks."""

from __future__ import annotations

import pandas as pd

from .registry import Registry

# (from_unit, to_unit) -> function. Only conversions the registry actually needs.
_CONVERSIONS = {
    ("degF", "degC"): lambda v: (v - 32.0) * 5.0 / 9.0,
    ("K", "degC"): lambda v: v - 273.15,
    ("kg/s", "t/h"): lambda v: v * 3.6,
    ("t/h", "kg/s"): lambda v: v / 3.6,
    ("kW", "MW"): lambda v: v / 1000.0,
}


def convert_unit(values: pd.Series, src: str, dst: str) -> pd.Series:
    """Convert ``values`` from ``src`` to ``dst``; identity when units already match."""
    if src == dst:
        return values
    try:
        return _CONVERSIONS[(src, dst)](values)
    except KeyError:
        raise ValueError(f"No conversion registered for {src} -> {dst}") from None


def standardize(df: pd.DataFrame, registry: Registry, name_col: str = "tag_name") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map raw names to canonical names and convert values to canonical units.

    Returns ``(standardized, unknown_tags)``. Rows of unregistered tags are
    separated into ``unknown_tags`` (never silently used by the pipeline).
    Output column for the name is always ``tag``.
    """
    resolved = df[name_col].map(registry.resolve)
    known = resolved.notna()
    unknown = df.loc[~known].copy()
    out = df.loc[known].copy()
    out["tag"] = resolved[known].map(lambda t: t[0])
    out["src_unit"] = resolved[known].map(lambda t: t[1])

    out["value"] = out["value"].astype(float)
    for (tag, src), idx in out.groupby(["tag", "src_unit"]).groups.items():
        dst = registry.tags[tag].unit
        out.loc[idx, "value"] = convert_unit(out.loc[idx, "value"], src, dst)
    out["converted"] = out["src_unit"] != out["tag"].map(lambda t: registry.tags[t].unit)
    return out.drop(columns=[name_col]).reset_index(drop=True), unknown.reset_index(drop=True)


def apply_physical_limits(df: pd.DataFrame, registry: Registry) -> pd.DataFrame:
    """Flag values outside the physically plausible range of their tag (``out_of_range``)."""
    lo = df["tag"].map(lambda t: registry.tags[t].phys_min)
    hi = df["tag"].map(lambda t: registry.tags[t].phys_max)
    out = df.copy()
    out["out_of_range"] = (out["value"] < lo) | (out["value"] > hi)
    return out
