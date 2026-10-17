"""Scores the pipeline against the ground truth of the synthetic dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .impute import ImputeConfig
from .ingest import IngestResult, load_lab, load_signals
from .pipeline import PipelineConfig, PipelineResult, run_pipeline
from .quality import gap_runs
from .registry import Registry
from .synth import SynthTruth

BUCKETS = [("short (<=5 min)", 0, 5), ("medium (6-60 min)", 5, 60), ("long (>60 min)", 60, 1e9)]


def _mark(idx: pd.DatetimeIndex, runs: list[tuple[pd.Timestamp, int]], period: int) -> pd.Series:
    m = pd.Series(False, index=idx)
    for st, ln in runs:
        m.loc[st: st + pd.Timedelta(seconds=period * (ln - 1))] = True
    return m


def detection_scores(res: PipelineResult, truth: SynthTruth, registry: Registry) -> pd.DataFrame:
    """Precision/recall of spike, flat-line and gap detection, per fault type (pooled over tags)."""
    tp = {"spike": 0, "flatline": 0, "missing": 0}
    fp = {k: 0 for k in tp}
    fn = {k: 0 for k in tp}
    for tag, cause in res.cause.items():
        period = registry.tags[tag].sample_period_s
        idx = cause.index
        spike_times = pd.DatetimeIndex([t.round(f"{period}s") for t in truth.spikes.get(tag, [])])
        true_spike = pd.Series(idx.isin(spike_times), index=idx)
        true_flat = _mark(idx, [(st, ln) for t, st, ln in truth.flatlines if t == tag], period)
        true_gap = _mark(idx, [(st, ln) for t, st, ln in truth.gaps if t == tag], period)
        for kind, truth_mask in (("spike", true_spike & ~true_gap & ~true_flat), ("flatline", true_flat), ("missing", true_gap)):
            pred = cause == kind
            tp[kind] += int((pred & truth_mask).sum())
            fp[kind] += int((pred & ~truth_mask).sum())
            fn[kind] += int((~pred & truth_mask).sum())
    rows = []
    for k in tp:
        p = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else float("nan")
        r = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else float("nan")
        rows.append({"fault": k, "true_positive": tp[k], "false_positive": fp[k], "false_negative": fn[k],
                     "precision": round(p, 3), "recall": round(r, 3)})
    return pd.DataFrame(rows)


def imputation_scores(res: PipelineResult, truth: SynthTruth, registry: Registry) -> pd.DataFrame:
    """Normalized RMSE (RMSE / std of truth) of filled values per gap-length bucket, pooled over tags.

    Gap length is the run length of *invalid* samples (missing, spike, stuck, out-of-range).
    """
    sq: dict[str, list[float]] = {b[0]: [] for b in BUCKETS}
    filled_n = {b[0]: 0 for b in BUCKETS}
    total_n = {b[0]: 0 for b in BUCKETS}
    for tag, cause in res.cause.items():
        spec = registry.tags[tag]
        sd = float(truth.clean[tag].std())
        for g in gap_runs(cause != "").itertuples():
            minutes = g.n_samples * spec.sample_period_s / 60
            name = next(b[0] for b in BUCKETS if b[1] < minutes <= b[2])
            sl = cause.index[(cause.index >= g.start) & (cause.index <= g.end)]
            total_n[name] += len(sl)
            est = res.clean[tag].reindex(sl)
            ok = est.notna()
            filled_n[name] += int(ok.sum())
            err = (est[ok] - truth.clean[tag].reindex(sl)[ok]) / sd
            sq[name].extend((err ** 2).tolist())
    rows = []
    for name, *_ in BUCKETS:
        vals = np.array(sq[name])
        rows.append({
            "gap_class": name,
            "invalid_samples": total_n[name],
            "filled_pct": round(100 * filled_n[name] / total_n[name], 1) if total_n[name] else float("nan"),
            "nrmse": round(float(np.sqrt(vals.mean())), 3) if len(vals) else float("nan"),
        })
    return pd.DataFrame(rows)


def compare_imputers(sig: IngestResult, lab: IngestResult, registry: Registry, truth: SynthTruth) -> pd.DataFrame:
    """NRMSE by gap class for hybrid vs linear vs hold-last (forward fill)."""
    out = []
    for method in ("hybrid", "linear", "ffill"):
        res = run_pipeline(sig, lab, registry, PipelineConfig(impute_method=method, impute=ImputeConfig()))
        sc = imputation_scores(res, truth, registry)
        sc.insert(0, "method", method)
        out.append(sc)
    return pd.concat(out, ignore_index=True)


def unit_scores(res: PipelineResult, truth: SynthTruth) -> pd.DataFrame:
    """Median absolute error after standardization for tags that arrived under an alias and a different unit."""
    rows = []
    for tag in ("BURNING_ZONE_TEMP", "FUEL_RATE", "KILN_FEED_RATE"):
        t = truth.clean[tag]
        err = (res.clean[tag].reindex(t.index) - t).abs().dropna()
        rows.append({"tag": tag, "median_abs_error": round(float(err.median()), 4),
                     "relative_to_mean_pct": round(100 * float(err.median() / t.mean()), 4)})
    return pd.DataFrame(rows)


def lab_alignment_score(res: PipelineResult, truth: SynthTruth) -> dict:
    """Share of lab samples in the output that sit on the correct timestamp with the correct value."""
    hit = total = 0
    for param in ("FREE_LIME", "BLAINE"):
        t = truth.lab_clean[truth.lab_clean["parameter"] == param].set_index("timestamp")["value"]
        got = res.clean[f"LAB_{param}"].dropna()
        total += len(got)
        common = got.index.intersection(t.index)
        hit += int(np.isclose(got.loc[common].to_numpy(), t.loc[common].to_numpy(), atol=1e-6).sum())
    return {"lab_samples_in_output": total, "correctly_aligned": hit,
            "alignment_rate_pct": round(100 * hit / total, 2) if total else float("nan")}


def lab_alignment_without_tz(sig_path: str, lab_path: str, registry: Registry, truth: SynthTruth) -> dict:
    """Counterfactual: same pipeline but lab timestamps treated as UTC (source timezone ignored)."""
    res = run_pipeline(load_signals(sig_path, "UTC"), load_lab(lab_path, "UTC"), registry)
    return lab_alignment_score(res, truth)


def correlation_drift(res: PipelineResult, truth: SynthTruth, tags: list[str]) -> float:
    """Mean absolute difference between truth and pipeline correlation matrices (lower is better)."""
    t = pd.DataFrame({k: truth.clean[k].reindex(res.clean.index) for k in tags}).dropna()
    mask = ~np.eye(len(tags), dtype=bool)
    return float(np.abs(t.corr().to_numpy() - res.clean[tags].corr().to_numpy())[mask].mean())
