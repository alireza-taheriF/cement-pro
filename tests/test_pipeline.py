"""Unit and end-to-end tests for the Phase 1 pipeline (synthetic data only)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cementpro.evaluation import detection_scores, lab_alignment_score
from cementpro.impute import ImputeConfig, impute_tag
from cementpro.ingest import load_lab, load_signals
from cementpro.pipeline import run_pipeline
from cementpro.quality import detect_flatline, detect_spikes, gap_runs
from cementpro.registry import load_registry
from cementpro.standardize import convert_unit, standardize
from cementpro.synth import generate_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registry():
    return load_registry(ROOT / "config/tags.yaml")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, registry):
    out = tmp_path_factory.mktemp("syn")
    _, _, truth = generate_dataset(days=3, seed=7, out_dir=out)
    sig = load_signals(out / "signals_raw.csv", registry.source_timezones["signals"])
    lab = load_lab(out / "lab_raw.csv", registry.source_timezones["lab"])
    return sig, lab, truth, out


def test_registry_resolves_aliases_and_units(registry):
    assert registry.resolve("bzt_f") == ("BURNING_ZONE_TEMP", "degF")
    assert registry.resolve("KLN_FUEL_KGS") == ("FUEL_RATE", "kg/s")
    assert registry.resolve("NOPE") is None


def test_convert_unit():
    assert convert_unit(pd.Series([212.0]), "degF", "degC").iloc[0] == pytest.approx(100.0)
    assert convert_unit(pd.Series([1.0]), "kg/s", "t/h").iloc[0] == pytest.approx(3.6)
    with pytest.raises(ValueError):
        convert_unit(pd.Series([1.0]), "bar", "degC")


def test_ingest_rejects_bad_rows_with_reason(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("timestamp,tag_name,value\n2026-01-01 00:00:00,O2,2.5\nnope,O2,2.5\n2026-01-01 00:01:00,O2,ERR\n")
    res = load_signals(p)
    assert len(res.data) == 1
    assert sorted(res.rejected["reason"]) == ["bad_timestamp", "non_numeric_value"]


def test_ingest_missing_column(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("time,tag,val\n1,2,3\n")
    with pytest.raises(ValueError, match="missing required"):
        load_signals(p)


def test_naive_lab_timestamps_use_source_timezone(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("timestamp,parameter,value,sample_id\n2026-01-01 12:00:00,FREE_LIME,1.0,A\n")
    res = load_lab(p, "Asia/Tehran")
    assert res.data["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 08:30:00", tz="UTC")


def test_standardize_separates_unknown_tags(registry):
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01"] * 2, utc=True),
                       "tag_name": ["BZT_F", "MYSTERY"], "value": [2552.0, 1.0]})
    out, unknown = standardize(df, registry)
    assert out["tag"].tolist() == ["BURNING_ZONE_TEMP"]
    assert out["value"].iloc[0] == pytest.approx(1400.0)
    assert len(unknown) == 1


def test_spike_detection_flags_isolated_spike_only():
    rng = np.random.default_rng(0)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 0.2, 500)), index=pd.date_range("2026-01-01", periods=500, freq="60s"))
    s.iloc[250] += 15
    flagged = detect_spikes(s)
    assert flagged.iloc[250] and flagged.sum() <= 2


def test_flatline_requires_min_run_and_nan_breaks_runs():
    s = pd.Series([1, 2, 5, 5, 5, 5, 5, 3, 4, 4, 4], dtype=float)
    assert detect_flatline(s, 5).tolist() == [False] * 2 + [True] * 5 + [False] * 4
    s2 = pd.Series([5, 5, np.nan, 5, 5], dtype=float)
    assert not detect_flatline(s2, 4).any()


def test_gap_runs():
    m = pd.Series([False, True, True, False, True], index=pd.date_range("2026-01-01", periods=5, freq="60s"))
    g = gap_runs(m)
    assert g["n_samples"].tolist() == [2, 1]


def test_impute_policy_short_filled_long_left_missing():
    idx = pd.date_range("2026-01-01", periods=400, freq="60s", tz="UTC")
    s = pd.Series(np.linspace(0, 40, 400), index=idx)
    s.iloc[50:52] = np.nan        # short
    s.iloc[100:200] = np.nan      # 100 min -> long
    filled, imputed, unrec = impute_tag(s, 60, None, ImputeConfig())
    assert filled.iloc[50:52].notna().all() and imputed.iloc[50:52].all()
    assert filled.iloc[50] == pytest.approx(s.iloc[49] + (s.iloc[52] - s.iloc[49]) / 3)
    assert filled.iloc[100:200].isna().all() and unrec.iloc[100:200].all()


def test_impute_never_touches_observed_values():
    idx = pd.date_range("2026-01-01", periods=100, freq="60s", tz="UTC")
    s = pd.Series(np.arange(100, dtype=float), index=idx)
    s.iloc[10:15] = np.nan
    filled, imputed, _ = impute_tag(s, 60, None)
    assert (filled[~imputed] .dropna() == s[~imputed].dropna()).all()


def test_pipeline_end_to_end(dataset, registry):
    sig, lab, truth, _ = dataset
    res = run_pipeline(sig, lab, registry)
    assert res.ingest_stats["signal_rows_rejected"] == 3
    assert res.ingest_stats["unknown_tag_rows"] == 50
    assert res.clean.index.is_monotonic_increasing and res.clean.index.is_unique
    # unit standardization: degF alias lands in degC range
    assert 1300 < res.clean["BURNING_ZONE_TEMP"].mean() < 1600
    # all injected gaps detected, no invented gap
    det = detection_scores(res, truth, registry).set_index("fault")
    assert det.loc["missing", "recall"] == 1.0 and det.loc["missing", "precision"] == 1.0
    assert det.loc["flatline", "recall"] >= 0.95
    assert det.loc["spike", "recall"] >= 0.9
    # lab timestamps land on the right grid row with the right value
    assert lab_alignment_score(res, truth)["alignment_rate_pct"] == 100.0


def test_pipeline_is_deterministic(dataset, registry):
    sig, lab, _, _ = dataset
    a = run_pipeline(sig, lab, registry).clean
    b = run_pipeline(sig, lab, registry).clean
    pd.testing.assert_frame_equal(a, b)


def test_ignoring_lab_timezone_breaks_alignment(dataset, registry):
    _, _, truth, out = dataset
    res = run_pipeline(load_signals(out / "signals_raw.csv"), load_lab(out / "lab_raw.csv", "UTC"), registry)
    assert lab_alignment_score(res, truth)["alignment_rate_pct"] < 10
