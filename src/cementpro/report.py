"""Runs the full synthetic evaluation, writes figures, metrics JSON and the generated Persian report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import (compare_imputers, correlation_drift, detection_scores, imputation_scores,
                         lab_alignment_score, lab_alignment_without_tz, unit_scores)
from .ingest import load_lab, load_signals
from .pipeline import PipelineResult, run_pipeline
from .registry import load_registry
from .synth import SynthTruth, generate_dataset

NAVY, TEAL, AMBER, RED, GREY = "#0f3d5e", "#14919b", "#f2a900", "#d64545", "#9aa5b1"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 9, "axes.titleweight": "bold",
    })


def fig_cleaning_example(res: PipelineResult, truth: SynthTruth, path: Path, tag: str = "BURNING_ZONE_TEMP") -> None:
    """Raw vs repaired signal around the first medium-length gap, with injected spikes marked."""
    gaps = [(st, ln) for t, st, ln in truth.gaps if t == tag and 8 <= ln <= 60]
    st, ln = gaps[0] if gaps else (truth.clean[tag].index[3000], 20)
    a, b = st - pd.Timedelta(minutes=90), st + pd.Timedelta(minutes=ln + 90)
    raw, clean, true = res.raw_native[tag].loc[a:b], res.clean[tag].loc[a:b], truth.clean[tag].loc[a:b]
    imp = res.imputed[tag].loc[a:b]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(true.index, true, "--", color=GREY, lw=1, label="Ground truth")
    ax.plot(raw.index, raw, ".", color=AMBER, ms=3, label="Raw (after standardization)")
    ax.plot(clean.index, clean, color=NAVY, lw=1.4, label="Pipeline output")
    ax.fill_between(imp.index, clean.min(), clean.max(), where=imp, color=TEAL, alpha=0.15, label="Imputed region")
    ax.set_title(f"{tag}: gap repair")
    ax.set_ylabel("degC")
    ax.legend(loc="best", fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_completeness(res: PipelineResult, path: Path) -> None:
    q = res.quality.set_index("tag")
    after = 100 * (1 - res.unrecoverable[q.index].mean())
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(q))
    ax.bar(x - 0.2, q["completeness_pct"], 0.4, color=GREY, label="Raw completeness")
    ax.bar(x + 0.2, after.reindex(q.index), 0.4, color=TEAL, label="After pipeline")
    ax.set_xticks(x, q.index, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(85, 100.5); ax.set_ylabel("% of time slots with a value")
    ax.set_title("Completeness before / after"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_imputers(cmp_df: pd.DataFrame, path: Path) -> None:
    d = cmp_df[cmp_df["gap_class"] != "long (>60 min)"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    classes = list(dict.fromkeys(d["gap_class"]))
    for i, (m, col) in enumerate((("ffill", GREY), ("linear", AMBER), ("hybrid", TEAL))):
        vals = [float(d[(d["method"] == m) & (d["gap_class"] == c)]["nrmse"].iloc[0]) for c in classes]
        ax.bar(np.arange(len(classes)) + (i - 1) * 0.26, vals, 0.26, color=col, label=m)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_ylabel("NRMSE (lower is better)"); ax.set_title("Imputation error by gap length")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _lab_relation(res: PipelineResult) -> float:
    """Correlation between lab free lime and burning-zone temperature at the lab timestamp."""
    d = res.clean[["LAB_FREE_LIME", "BURNING_ZONE_TEMP"]].dropna()
    return float(d.corr().iloc[0, 1]) if len(d) > 5 else float("nan")


def fig_lab_alignment(res_ok: PipelineResult, res_bad: PipelineResult, path: Path) -> tuple[float, float]:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    out = []
    for ax, res, title, col in ((axes[0], res_ok, "Source timezone handled", TEAL), (axes[1], res_bad, "Timezone ignored", RED)):
        d = res.clean[["LAB_FREE_LIME", "BURNING_ZONE_TEMP"]].dropna()
        ax.scatter(d["BURNING_ZONE_TEMP"], d["LAB_FREE_LIME"], s=10, color=col, alpha=0.7)
        r = _lab_relation(res); out.append(r)
        ax.set_title(f"{title}\nr = {r:.2f}", fontsize=9); ax.set_xlabel("Burning zone temp (degC)")
    axes[0].set_ylabel("Lab free lime (%)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return out[0], out[1]


def run_evaluation(root: str | Path = ".", seed: int = 42, days: int = 7) -> dict:
    """Generate data, run the pipeline, score it and write all artifacts. Returns the metrics dict."""
    root = Path(root)
    _style()
    reg = load_registry(root / "config/tags.yaml")
    data_dir = root / "data/synthetic"
    sig_raw, lab_raw, truth = generate_dataset(days, seed, data_dir)
    sig = load_signals(data_dir / "signals_raw.csv", reg.source_timezones["signals"])
    lab = load_lab(data_dir / "lab_raw.csv", reg.source_timezones["lab"])
    res = run_pipeline(sig, lab, reg)
    res_bad = run_pipeline(load_signals(data_dir / "signals_raw.csv", "UTC"), load_lab(data_dir / "lab_raw.csv", "UTC"), reg)

    det = detection_scores(res, truth, reg)
    imp = imputation_scores(res, truth, reg)
    cmp_df = compare_imputers(sig, lab, reg, truth)
    units = unit_scores(res, truth)
    lab_ok = lab_alignment_score(res, truth)
    lab_bad = lab_alignment_without_tz(str(data_dir / "signals_raw.csv"), str(data_dir / "lab_raw.csv"), reg, truth)
    drift = correlation_drift(res, truth, reg.process_tags)

    assets = root / "docs/phase1/assets"
    assets.mkdir(parents=True, exist_ok=True)
    fig_cleaning_example(res, truth, assets / "fig_cleaning_example.png")
    fig_completeness(res, assets / "fig_completeness.png")
    fig_imputers(cmp_df, assets / "fig_imputers.png")
    r_ok, r_bad = fig_lab_alignment(res, res_bad, assets / "fig_lab_alignment.png")

    out_dir = root / "reports"
    out_dir.mkdir(exist_ok=True)
    res.quality.to_csv(out_dir / "quality_report.csv", index=False)
    res.clean.to_csv(out_dir / "clean_output_sample.csv.gz", compression="gzip")
    res.rejected.to_csv(out_dir / "rejected_rows.csv", index=False)

    metrics = {
        "config": {"seed": seed, "days": days, "grid_period_s": 60},
        "ingest": res.ingest_stats,
        "detection": det.to_dict("records"),
        "imputation": imp.to_dict("records"),
        "imputer_comparison": cmp_df.to_dict("records"),
        "units": units.to_dict("records"),
        "lab_alignment": lab_ok,
        "lab_alignment_timezone_ignored": lab_bad,
        "lab_relation_r": {"timezone_handled": round(r_ok, 3), "timezone_ignored": round(r_bad, 3)},
        "correlation_drift": round(drift, 4),
        "unrecoverable_pct": round(100 * float(res.unrecoverable[reg.process_tags].mean().mean()), 2),
        "quality": res.quality.to_dict("records"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    from .docgen import write as write_doc
    write_doc(metrics, root / "docs/phase1/06-lab-prototype-evaluation.md")
    return metrics
