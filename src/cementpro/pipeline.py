"""End-to-end preprocessing pipeline: ingest -> standardize -> detect -> clean -> align -> impute."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .align import align_lab, join_lab_to_grid, to_common_grid, to_native_series
from .impute import ImputeConfig, impute_tag
from .ingest import IngestResult, load_lab, load_signals
from .quality import detect_flatline, detect_spikes, tag_quality
from .registry import Registry, load_registry
from .standardize import apply_physical_limits, standardize


@dataclass
class PipelineConfig:
    grid_period_s: int = 60
    impute: ImputeConfig = field(default_factory=ImputeConfig)
    impute_method: str = "hybrid"
    lab_tolerance_min: float = 30.0
    lab_lag_minutes: dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """All pipeline outputs. ``clean`` is the model-ready wide frame on the common grid."""

    clean: pd.DataFrame                 # imputed values on the common grid (+ LAB_* columns)
    raw_native: dict[str, pd.Series]    # per-tag native series before cleaning
    cause: dict[str, pd.Series]         # per-tag cause of invalid sample: missing/spike/flatline/out_of_range/""
    imputed: pd.DataFrame               # bool, True where a value was filled
    unrecoverable: pd.DataFrame         # bool, True where a gap was too long to fill
    quality: pd.DataFrame               # per-tag quality summary (computed before cleaning)
    rejected: pd.DataFrame              # rows rejected at ingestion (with reason)
    unknown_tags: pd.DataFrame          # rows with tags missing from the registry
    ingest_stats: dict


def _prepare_predictors(raw_grid: pd.DataFrame) -> pd.DataFrame:
    """Predictor matrix for cross-tag regression: short-gap-bridged copy of the cleaned grid."""
    return raw_grid.interpolate(method="time", limit=5, limit_area="inside")


def run_pipeline(
    signals: IngestResult,
    lab: IngestResult | None,
    registry: Registry,
    cfg: PipelineConfig = PipelineConfig(),
) -> PipelineResult:
    sig, unknown = standardize(signals.data, registry)
    sig = apply_physical_limits(sig, registry)
    n_dup_by_tag = sig.duplicated(["tag", "timestamp"], keep="last").groupby(sig["tag"]).sum()
    sig = sig.drop_duplicates(["tag", "timestamp"], keep="last")

    start, end = sig["timestamp"].min(), sig["timestamp"].max()
    raw_native: dict[str, pd.Series] = {}
    cause: dict[str, pd.Series] = {}
    cleaned_native: dict[str, pd.Series] = {}
    quality_rows = []

    for tag in registry.process_tags:
        spec = registry.tags[tag]
        rows = sig[sig["tag"] == tag]
        if rows.empty:
            continue
        native = to_native_series(rows, spec.sample_period_s, start, end)
        oor_rows = rows[rows["out_of_range"]]
        oor_idx = pd.DatetimeIndex(oor_rows["timestamp"].dt.round(f"{spec.sample_period_s}s"))
        valid = native.copy()
        oor = valid.index.isin(oor_idx)
        valid[oor] = float("nan")

        # Stuck runs are removed first so their edges are not mistaken for spikes.
        flat = detect_flatline(valid, spec.flatline_min_samples)
        spikes = detect_spikes(valid.where(~flat), spec.hampel_window, spec.hampel_sigmas)

        c = pd.Series("", index=native.index, dtype=object)
        c[native.isna()] = "missing"
        c[oor] = "out_of_range"
        c[spikes & (c == "")] = "spike"
        c[flat & (c == "")] = "flatline"
        cause[tag] = c
        raw_native[tag] = native
        cleaned_native[tag] = native.where(c == "")

        quality_rows.append(
            tag_quality(spec, len(rows), int(n_dup_by_tag.get(tag, 0)), int(oor.sum()), valid, spikes, flat)
        )

    # Common-grid copy of cleaned (un-imputed) data used as regression predictors.
    pre_grid = to_common_grid(cleaned_native, cfg.grid_period_s)
    predictors = _prepare_predictors(pre_grid)

    filled_native, imputed_native, unrec_native = {}, {}, {}
    for tag, s in cleaned_native.items():
        others = predictors.drop(columns=[tag])
        # Native-period predictor rows: take nearest grid rows.
        others_native = others.reindex(s.index, method="nearest", tolerance=pd.Timedelta(seconds=cfg.grid_period_s))
        f, i, u = impute_tag(s, registry.tags[tag].sample_period_s, others_native, cfg.impute, cfg.impute_method)
        filled_native[tag], imputed_native[tag], unrec_native[tag] = f, i, u

    grid = to_common_grid(filled_native, cfg.grid_period_s)
    imputed = pd.DataFrame(False, index=grid.index, columns=grid.columns)
    for tag, flag in imputed_native.items():
        tol = pd.Timedelta(seconds=registry.tags[tag].sample_period_s)
        prev = flag.reindex(grid.index, method="ffill", tolerance=tol).astype("boolean").fillna(False).astype(bool)
        nxt = flag.reindex(grid.index, method="bfill", tolerance=tol).astype("boolean").fillna(False).astype(bool)
        imputed[tag] = prev | nxt
    unrecoverable = grid.isna()

    if lab is not None and len(lab.data):
        lab_std, _ = standardize(lab.data.rename(columns={"parameter": "tag_name"}), registry)
        lab_std = apply_physical_limits(lab_std, registry)
        lab_std = lab_std[~lab_std["out_of_range"]]
        lab_wide = align_lab(lab_std, grid.index, cfg.lab_lag_minutes)
        grid = join_lab_to_grid(grid, lab_wide, cfg.lab_tolerance_min)

    return PipelineResult(
        clean=grid,
        raw_native=raw_native,
        cause=cause,
        imputed=imputed,
        unrecoverable=unrecoverable,
        quality=pd.DataFrame(quality_rows),
        rejected=pd.concat([signals.rejected] + ([lab.rejected] if lab else []), ignore_index=True),
        unknown_tags=unknown,
        ingest_stats={
            "signal_rows_raw": signals.n_raw,
            "signal_rows_rejected": signals.n_rejected,
            "unknown_tag_rows": int(len(unknown)),
            "lab_rows_raw": lab.n_raw if lab else 0,
            "duplicates_removed": int(n_dup_by_tag.sum()),
        },
    )


def run_from_files(
    signals_csv: str | Path,
    lab_csv: str | Path | None,
    registry_path: str | Path,
    cfg: PipelineConfig = PipelineConfig(),
) -> PipelineResult:
    """Convenience wrapper: read CSVs with source timezones taken from the registry."""
    reg = load_registry(registry_path)
    sig = load_signals(signals_csv, reg.source_timezones.get("signals", "UTC"))
    lab = load_lab(lab_csv, reg.source_timezones.get("lab", "UTC")) if lab_csv else None
    return run_pipeline(sig, lab, reg, cfg)
