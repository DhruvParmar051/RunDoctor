"""Training entrypoint. Runs as a detached subprocess launched by the MCP server.

Usage: python -m rundoctor.training.train --run-id N [--db PATH]
"""

from __future__ import annotations

import argparse
import math
import signal
import sqlite3
import sys
import warnings
from pathlib import Path
from types import FrameType

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from rundoctor import db  # noqa: E402
from rundoctor.config import get_settings  # noqa: E402
from rundoctor.logging import get_logger  # noqa: E402
from rundoctor.models import Epoch  # noqa: E402
from rundoctor.training.tasks import build_task  # noqa: E402

log = get_logger(__name__)

MOMENTUM = 0.9


class _StopFlag:
    def __init__(self) -> None:
        self.stop = False

    def handle(self, signum: int, frame: FrameType | None) -> None:
        log.warning("received signal %d, stopping", signum)
        self.stop = True


def _evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = float(loss_fn(logits, y))
        acc = float((logits.argmax(dim=1) == y).float().mean())
    model.train()
    return loss, acc


def train_run(conn: sqlite3.Connection, run_id: int, stop: _StopFlag | None = None) -> None:
    """Train a run to completion, writing each epoch's metrics and the final status."""
    run = db.get_run(conn, run_id)
    if run is None:
        raise ValueError(f"run_id {run_id} not found")
    cfg = run.config
    stop = stop or _StopFlag()
    try:
        model, (x_train, y_train), (x_val, y_val) = build_task(run.task, cfg)
        # Plain SGD+momentum (not Adam) so a too-high lr genuinely diverges.
        opt = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=MOMENTUM, weight_decay=cfg.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(cfg.seed)
        n = x_train.shape[0]

        for epoch in range(1, cfg.epochs + 1):
            perm = torch.randperm(n, generator=gen)
            total_loss, total_norm, batches = 0.0, 0.0, 0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(x_train[idx]), y_train[idx])
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=math.inf)
                opt.step()
                total_loss += float(loss.detach())
                total_norm += float(norm)
                batches += 1
            train_loss = total_loss / batches
            val_loss, val_acc = _evaluate(model, x_val, y_val, loss_fn)
            db.insert_epoch(
                conn,
                Epoch(
                    run_id=run_id,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    grad_norm=total_norm / batches,
                ),
            )
            log.info(
                "run %d epoch %d train=%.4f val=%.4f acc=%.3f",
                run_id,
                epoch,
                train_loss,
                val_loss,
                val_acc,
            )
            if stop.stop:
                db.set_status(conn, run_id, "killed")
                return
            if not math.isfinite(train_loss):
                # A numeric blow-up is a training outcome, not a crash: the run completed.
                log.warning("run %d: non-finite loss at epoch %d, stopping", run_id, epoch)
                break
        db.set_status(conn, run_id, "completed")
    except Exception as exc:
        log.exception("run %d failed", run_id)
        db.set_status(conn, run_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a RunDoctor run.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args(argv)

    stop = _StopFlag()
    signal.signal(signal.SIGTERM, stop.handle)
    signal.signal(signal.SIGINT, stop.handle)
    torch.set_num_threads(2)

    with db.connection(args.db or get_settings().db_path) as conn:
        try:
            train_run(conn, args.run_id, stop)
        except Exception:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
