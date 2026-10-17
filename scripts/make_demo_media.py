"""Builds a demo screenshot (PNG) and a short video (MP4) of the pipeline repairing a messy signal.

Usage: python scripts/make_demo_media.py   (needs ffmpeg for the MP4)
All data is synthetic with injected faults (see docs/phase1/06-lab-prototype-evaluation.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cementpro.ingest import load_lab, load_signals  # noqa: E402
from cementpro.pipeline import run_pipeline  # noqa: E402
from cementpro.registry import load_registry  # noqa: E402
from cementpro.synth import generate_dataset  # noqa: E402

BG, FG, NAVY, TEAL, AMBER, RED, GREY = "#0b1f2e", "#e8eef3", "#5ec8d4", "#14919b", "#f2a900", "#ff5c5c", "#7f8b97"
HOURS = 8
OUT = ROOT / "docs/phase1/assets"


def pick_window(res) -> tuple[str, pd.Timestamp]:
    """(tag, start) of the HOURS-long window richest in distinct fault types across all 1-minute tags."""
    best, pick = -1, None
    for tag, c in res.cause.items():
        if len(c) < 5000:
            continue  # skip slower (5-minute) tags: too few points for a smooth animation
        for st in c.index[::30][:-HOURS * 2]:
            w = c.loc[st: st + pd.Timedelta(hours=HOURS)]
            kinds = set(w[w != ""]) - {"out_of_range"}
            score = len(kinds) * 1000 + int((w != "").sum())
            if score > best:
                best, pick = score, (tag, st)
    return pick


def build() -> None:
    reg = load_registry(ROOT / "config/tags.yaml")
    generate_dataset(7, 42, ROOT / "data/synthetic")
    res = run_pipeline(load_signals(ROOT / "data/synthetic/signals_raw.csv", "UTC"),
                       load_lab(ROOT / "data/synthetic/lab_raw.csv", "Asia/Tehran"), reg)
    TAG, st = pick_window(res)
    unit = reg.tags[TAG].unit
    sl = slice(st, st + pd.Timedelta(hours=HOURS))
    raw, clean, cause = res.raw_native[TAG].loc[sl], res.clean[TAG].loc[sl], res.cause[TAG].loc[sl]
    imp = res.imputed[TAG].loc[sl]
    x = (raw.index - raw.index[0]).total_seconds() / 3600
    pad = 0.4 * (clean.max() - clean.min())
    ylo, yhi = clean.min() - pad, clean.max() + pad
    # clip absurd raw readings out of the view (they are removed by the pipeline anyway)
    raw_view = raw.where((raw > ylo) & (raw < yhi))  # drop absurd readings from the view

    colors = {"spike": RED, "flatline": GREY, "missing": AMBER, "out_of_range": RED}

    def draw(ax, k: int):
        """Draw the first k samples."""
        ax.clear()
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_color("#2a4256")
        ax.tick_params(colors=FG)
        ax.set_xlim(0, HOURS); ax.set_ylim(ylo, yhi)
        ax.set_xlabel("hours", color=FG); ax.set_ylabel(f"{TAG} ({unit})", color=FG)
        ax.grid(alpha=0.15)
        xs = x[:k]
        # fault shading
        for kind, col in (("missing", AMBER), ("flatline", GREY)):
            m = (cause.iloc[:k] == kind).to_numpy()
            ax.fill_between(xs, ylo, yhi, where=m, color=col, alpha=0.18, step="mid", linewidth=0)
        ax.plot(xs, raw_view.iloc[:k], ".", color=AMBER, ms=3, alpha=0.9, label="raw input")
        sp = (cause.iloc[:k] == "spike").to_numpy()
        ax.plot(xs[sp], raw_view.iloc[:k][sp], "x", color=RED, ms=9, mew=2, label="spike detected")
        done = clean.iloc[:max(0, k - 25)]  # repaired output trails the raw stream
        xd = x[:len(done)]
        ax.plot(xd, done, color=NAVY, lw=2, label="pipeline output")
        impd = imp.iloc[:len(done)].to_numpy()
        ax.fill_between(xd, ylo, yhi, where=impd, color=TEAL, alpha=0.28, step="mid", linewidth=0, label="imputed")
        leg = ax.legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor="#2a4256", labelcolor=FG, ncol=4)
        n_sp = int(sp.sum()); n_gap = int((cause.iloc[:k] == "missing").sum()); n_fl = int((cause.iloc[:k] == "flatline").sum())
        ax.set_title(f"cement-pro | detected: {n_sp} spikes, {n_gap} missing samples, {n_fl} stuck samples | imputed: {int(impd.sum())}",
                     color=FG, fontsize=10, loc="left")

    def new_fig():
        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=120)
        fig.patch.set_facecolor(BG)
        fig.text(0.99, 0.01, "SYNTHETIC DATA with injected faults", color=AMBER, ha="right", fontsize=8)
        return fig, ax

    # ---- screenshot: full window, finished state ----
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = new_fig()
    draw(ax, len(x) + 25)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "demo_screenshot.png", facecolor=BG); plt.close(fig)

    # ---- video ----
    fig, ax = new_fig()
    n = len(x)
    frames = list(range(5, n + 26, 6)) + [n + 25] * 30  # hold last frame ~2 s
    anim = FuncAnimation(fig, lambda k: draw(ax, min(k, n + 25)), frames=frames)
    anim.save(OUT / "demo_video.mp4", writer=FFMpegWriter(fps=15, bitrate=2500, extra_args=["-pix_fmt", "yuv420p"]),
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", OUT / "demo_screenshot.png", OUT / "demo_video.mp4", "frames:", len(frames))


if __name__ == "__main__":
    build()
