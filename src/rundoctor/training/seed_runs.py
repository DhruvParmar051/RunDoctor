"""Reset the DB and create the 4 planted demo runs with known problems.

Usage: python -m rundoctor.training.seed_runs [--db PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from rundoctor import db
from rundoctor.config import get_settings
from rundoctor.models import IssueCode, RunConfig
from rundoctor.training.train import train_run

GROUND_TRUTH_PATH = Path(__file__).resolve().parents[1] / "evals" / "ground_truth.json"

# name -> (config, expected issue codes)
PLANTED_RUNS: dict[str, tuple[RunConfig, list[IssueCode]]] = {
    "diverging": (
        RunConfig(lr=1.0, epochs=15, batch_size=512, hidden=64, dropout=0.0, seed=1),
        ["nan_or_inf", "divergence"],
    ),
    "overfitting": (
        RunConfig(
            lr=0.02,
            epochs=40,
            batch_size=16,
            hidden=512,
            dropout=0.0,
            weight_decay=0.0,
            train_size=48,
            noise=1.0,
            seed=2,
        ),
        ["overfitting"],
    ),
    "plateau": (
        RunConfig(lr=1e-6, epochs=15, hidden=64, dropout=0.2, seed=3),
        ["plateau"],
    ),
    "healthy": (
        RunConfig(lr=0.05, epochs=15, hidden=64, dropout=0.2, train_size=2000, seed=4),
        ["healthy"],
    ),
}


def _fmt(values: list[float | None]) -> str:
    return " ".join("nan" if v is None or not math.isfinite(v) else f"{v:.3f}" for v in values)


def seed(db_path: Path, verbose: bool = True, write_ground_truth: bool = True) -> dict[str, int]:
    ids: dict[str, int] = {}
    with db.connection(db_path) as conn:
        db.reset_db(conn)
        for name, (cfg, _) in PLANTED_RUNS.items():
            run_id = db.create_run(conn, name, "signal1d", cfg)
            start = time.perf_counter()
            train_run(conn, run_id)
            elapsed = time.perf_counter() - start
            ids[name] = run_id
            run = db.get_run(conn, run_id)
            epochs = db.get_epochs(conn, run_id)
            status = run.status if run else "?"
            if not verbose:
                continue
            print(f"[{run_id}] {name}: {status}, {len(epochs)} epochs, {elapsed:.1f}s")
            print(f"    train_loss: {_fmt([e.train_loss for e in epochs])}")
            print(f"    val_loss:   {_fmt([e.val_loss for e in epochs])}")
            print(f"    val_acc:    {_fmt([e.val_acc for e in epochs])}")

    if not write_ground_truth:
        return ids
    truth = {
        name: {"run_id": ids[name], "expected_issues": codes}
        for name, (_, codes) in PLANTED_RUNS.items()
    }
    GROUND_TRUTH_PATH.write_text(json.dumps(truth, indent=2) + "\n")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the 4 planted demo runs.")
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args()
    seed(args.db or get_settings().db_path)


if __name__ == "__main__":
    main()
