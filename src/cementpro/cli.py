"""Command line interface: ``python -m cementpro {generate,run,evaluate}``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_from_files
from .synth import generate_dataset


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cementpro", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a synthetic raw dataset with injected faults")
    g.add_argument("--days", type=int, default=7)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="data/synthetic")

    r = sub.add_parser("run", help="run the pipeline on raw CSV exports")
    r.add_argument("--signals", required=True)
    r.add_argument("--lab")
    r.add_argument("--registry", default="config/tags.yaml")
    r.add_argument("--out", default="reports")

    e = sub.add_parser("evaluate", help="full synthetic evaluation (figures + metrics + report)")
    e.add_argument("--root", default=".")
    e.add_argument("--seed", type=int, default=42)

    a = p.parse_args(argv)
    if a.cmd == "generate":
        generate_dataset(a.days, a.seed, a.out)
        print(f"wrote synthetic dataset to {a.out}")
    elif a.cmd == "run":
        res = run_from_files(a.signals, a.lab, a.registry)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        res.clean.to_csv(out / "clean.csv")
        res.quality.to_csv(out / "quality_report.csv", index=False)
        res.rejected.to_csv(out / "rejected_rows.csv", index=False)
        print(f"clean grid {res.clean.shape}, rejected rows {len(res.rejected)}, unknown-tag rows {len(res.unknown_tags)}")
    else:
        from .report import run_evaluation
        m = run_evaluation(a.root, a.seed)
        print("evaluation done; lab alignment", m["lab_alignment"], "unrecoverable %", m["unrecoverable_pct"])


if __name__ == "__main__":
    main()
