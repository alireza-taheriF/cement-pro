"""Synthetic plant-data generator with *injected, labelled faults*.

Real plant data is not available for Phase 1, so this module simulates
a kiln/mill data stream whose statistics are plausible, then corrupts it the
way real historian exports are corrupted. Every injected fault is recorded,
which gives the pipeline an objective ground truth to be scored against.

IMPORTANT: results on this data demonstrate that the pipeline mechanics work.
They are NOT evidence of performance on a real plant (UNVERIFIED on real data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SynthTruth:
    """Ground truth for scoring: clean data and the exact faults that were injected."""

    clean: dict[str, pd.Series]                      # native-period clean series (UTC index)
    gaps: list[tuple[str, pd.Timestamp, int]] = field(default_factory=list)       # tag, start, n_samples
    spikes: dict[str, list[pd.Timestamp]] = field(default_factory=dict)
    flatlines: list[tuple[str, pd.Timestamp, int]] = field(default_factory=list)  # tag, start, n_samples
    lab_clean: pd.DataFrame | None = None            # true lab values at true sample time
    lab_offset_hours: float = 0.0


PERIODS = {"CO": 300, "NOX": 300}


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def generate_clean(days: int, seed: int, start: str = "2026-01-01") -> dict[str, pd.Series]:
    """Simulate physically coupled process tags on the 1-minute grid (5-minute tags subsampled)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="60s", tz="UTC")
    n = len(idx)
    t = np.arange(n)
    daily = np.sin(2 * np.pi * t / 1440)

    feed = 210 + 6 * daily + _ar1(n, 0.995, 0.25, rng)
    fuel = 11.2 + 0.05 * (feed - 210) + _ar1(n, 0.99, 0.02, rng)
    speed = 3.8 + 0.004 * (feed - 210) + _ar1(n, 0.995, 0.003, rng)
    idfan = 72 + 0.3 * (feed - 210) + _ar1(n, 0.99, 0.12, rng)
    bzt = 1445 + 9 * (fuel - 11.2) / 0.05 * 0.1 + 0.5 * (feed - 210) * -0.1 + _ar1(n, 0.995, 0.9, rng)
    o2 = 2.8 - 0.12 * (fuel - 11.2) / 0.1 + 0.02 * (idfan - 72) + _ar1(n, 0.98, 0.04, rng)
    co = np.maximum(30, 210 - 80 * (o2 - 2.8) + _ar1(n, 0.97, 6, rng))
    nox = np.maximum(100, 520 + 0.9 * (bzt - 1445) * 4 + _ar1(n, 0.98, 5, rng))
    sec_air = 1020 + 0.6 * (bzt - 1445) + _ar1(n, 0.995, 0.6, rng)
    clinker = 118 + 0.25 * (feed - 210) + 0.03 * (bzt - 1445) + _ar1(n, 0.995, 0.5, rng)
    mill = 5200 + 80 * daily + _ar1(n, 0.99, 15, rng)

    series = {
        "KILN_FEED_RATE": feed, "FUEL_RATE": fuel, "KILN_SPEED": speed, "BURNING_ZONE_TEMP": bzt,
        "ID_FAN_SPEED": idfan, "O2": o2, "CO": co, "NOX": nox, "SECONDARY_AIR_TEMP": sec_air,
        "CLINKER_TEMP": clinker, "MILL_POWER": mill,
    }
    out = {}
    for tag, v in series.items():
        s = pd.Series(v, index=idx)
        step = PERIODS.get(tag, 60) // 60
        out[tag] = s.iloc[::step]
    return out


def _lab_from_process(clean: dict[str, pd.Series], rng: np.random.Generator) -> pd.DataFrame:
    """Hourly free lime (driven by burning-zone temperature and feed) and 2-hourly Blaine (mill power)."""
    bzt, feed, mill = clean["BURNING_ZONE_TEMP"], clean["KILN_FEED_RATE"], clean["MILL_POWER"]
    rows = []
    for ts in bzt.index[::60]:
        fl = 1.1 - 0.012 * (bzt.loc[ts] - 1445) + 0.015 * (feed.loc[ts] - 210) + rng.normal(0, 0.05)
        rows.append((ts, "FREE_LIME", max(0.05, fl), f"LAB-FL-{len(rows):05d}"))
    for ts in mill.index[::120]:
        bl = 3500 + 0.12 * (mill.loc[ts] - 5200) + rng.normal(0, 15)
        rows.append((ts, "BLAINE", bl, f"LAB-BL-{len(rows):05d}"))
    return pd.DataFrame(rows, columns=["timestamp", "parameter", "value", "sample_id"])


def corrupt(
    clean: dict[str, pd.Series],
    seed: int,
    gap_rate: float = 1.0,
    spike_rate: float = 1.0,
    flatline_rate: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, SynthTruth]:
    """Turn clean series into messy historian/LIMS exports and return the truth record."""
    rng = np.random.default_rng(seed + 1)
    truth = SynthTruth(clean={k: v.copy() for k, v in clean.items()})
    frames = []

    for tag, s in clean.items():
        v = s.copy()
        period = PERIODS.get(tag, 60)
        n = len(v)
        per_day = 1440 * 60 / period

        # --- stuck sensor (flat-line) ---
        for _ in range(int(round(rng.poisson(1.0 * flatline_rate * n / per_day / 7 + 0.4)))):
            length = int(rng.integers(25, 70) * 60 / period) if period == 60 else int(rng.integers(10, 20))
            a = int(rng.integers(100, n - length - 100))
            v.iloc[a:a + length] = v.iloc[a]
            truth.flatlines.append((tag, v.index[a], length))
        stuck_mask = pd.Series(False, index=v.index)
        for t, st, ln in truth.flatlines:
            if t == tag:
                stuck_mask.loc[st: st + pd.Timedelta(seconds=period * (ln - 1))] = True

        # --- spikes (only on points that are not part of a stuck run) ---
        sigma = float(np.std(np.diff(s.to_numpy())) / np.sqrt(2)) or 1.0
        n_sp = int(rng.poisson(3.0 * spike_rate * n / per_day / 7 + 1))
        cand = np.where(~stuck_mask.to_numpy())[0]
        picks = rng.choice(cand, size=min(n_sp, len(cand)), replace=False)
        for a in picks:
            v.iloc[a] += rng.choice([-1, 1]) * rng.uniform(10, 25) * sigma
        truth.spikes[tag] = [v.index[a] for a in picks]

        # --- gaps: mix of short / medium / long outages ---
        keep = np.ones(n, dtype=bool)
        for lo, hi, cnt in ((1, 3, 6), (8, 25, 4), (30, 55, 2), (120, 360, 1)):
            for _ in range(int(rng.poisson(cnt * gap_rate))):
                ln = max(1, int(rng.integers(lo, hi + 1) * 60 / period))
                a = int(rng.integers(50, n - ln - 50))
                keep[a:a + ln] = False
                truth.gaps.append((tag, v.index[a], ln))

        d = pd.DataFrame({"timestamp": v.index[keep], "tag_name": tag, "value": v.to_numpy()[keep]})
        # timestamp jitter of a couple of seconds, like a real historian
        d["timestamp"] = d["timestamp"] + pd.to_timedelta(rng.integers(-2, 3, len(d)), unit="s")
        frames.append(d)

    sig = pd.concat(frames, ignore_index=True)

    # --- alias + unit differences ---
    alias = {"BURNING_ZONE_TEMP": ("BZT_F", lambda x: x * 9 / 5 + 32),
             "FUEL_RATE": ("KLN_FUEL_KGS", lambda x: x / 3.6),
             "KILN_FEED_RATE": ("KLN_FEED_TPH", lambda x: x)}
    for canon, (name, fn) in alias.items():
        m = sig["tag_name"] == canon
        sig.loc[m, "value"] = fn(sig.loc[m, "value"])
        sig.loc[m, "tag_name"] = name

    # --- duplicates, out-of-order rows, bad rows, an unregistered tag ---
    sig["timestamp"] = sig["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    oor_rows = sig.sample(n=5, random_state=seed + 7).index
    sig.loc[oor_rows, "value"] = 99999.0  # transmitter glitch: physically impossible value
    dup = sig.sample(frac=0.002, random_state=seed)
    bad = pd.DataFrame({
        "timestamp": ["not-a-time", "2026-01-02T03:00:00+00:00", "2026-01-02T04:00:00+00:00"],
        "tag_name": ["O2", "O2", ""],
        "value": [2.7, "ERR", 3.0],
    })
    unk = pd.DataFrame({"timestamp": sig["timestamp"].iloc[:50], "tag_name": "PLC_SPARE_AI_7", "value": 0.0})
    sig = pd.concat([sig, dup, bad, unk], ignore_index=True)
    sig = sig.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # out-of-order

    # --- lab export: stamped in local time (Asia/Tehran, UTC+3:30), naive ---
    lab_true = _lab_from_process(clean, np.random.default_rng(seed + 2))
    truth.lab_clean = lab_true.copy()
    truth.lab_offset_hours = 3.5
    lab = lab_true.copy()
    lab["timestamp"] = (lab["timestamp"] + pd.Timedelta(hours=3.5)).dt.tz_localize(None)
    lab["timestamp"] = lab["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    lab = lab.drop(lab.sample(frac=0.05, random_state=seed).index).reset_index(drop=True)  # missed samples

    return sig, lab, truth


def generate_dataset(days: int = 7, seed: int = 42, out_dir: str | Path | None = None):
    """Generate clean data, corrupt it, optionally write CSVs. Returns ``(signals, lab, truth)``."""
    clean = generate_clean(days, seed)
    sig, lab, truth = corrupt(clean, seed)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sig.to_csv(out / "signals_raw.csv", index=False)
        lab.to_csv(out / "lab_raw.csv", index=False)
    return sig, lab, truth
