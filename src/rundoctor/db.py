"""SQLite storage for runs and per-epoch metrics.

The server and training subprocesses write concurrently, so connections use WAL mode.

SQLite stores NaN as NULL. The trainer always writes every metric, so a NULL metric
read back from ``epochs`` means the value was NaN, and ``get_epochs`` returns it as NaN.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from rundoctor.models import Epoch, Run, RunConfig, RunStatus, TaskName

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  task TEXT NOT NULL,
  config_json TEXT NOT NULL,
  status TEXT NOT NULL,
  pid INTEGER,
  created_at TEXT NOT NULL,
  finished_at TEXT,
  error TEXT
);
CREATE TABLE IF NOT EXISTS epochs (
  run_id INTEGER NOT NULL REFERENCES runs(id),
  epoch INTEGER NOT NULL,
  train_loss REAL,
  val_loss REAL,
  val_acc REAL,
  grad_norm REAL,
  PRIMARY KEY (run_id, epoch)
);
"""

StatusFilter = Literal["all", "running", "completed", "failed", "killed"]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection with WAL, foreign keys, and a busy timeout. Creates the schema."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    init_db(conn)
    return conn


@contextmanager
def connection(path: Path | str) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    """Delete all runs and epochs and reset the id counter."""
    with conn:
        conn.execute("DELETE FROM epochs")
        conn.execute("DELETE FROM runs")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'runs'")


def create_run(
    conn: sqlite3.Connection,
    name: str,
    task: TaskName,
    config: RunConfig,
    status: RunStatus = "running",
) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO runs (name, task, config_json, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, task, config.model_dump_json(), status, now_iso()),
        )
    run_id = cur.lastrowid
    assert run_id is not None
    return run_id


def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        id=row["id"],
        name=row["name"],
        task=row["task"],
        config=RunConfig.model_validate(json.loads(row["config_json"])),
        status=row["status"],
        pid=row["pid"],
        created_at=row["created_at"],
        finished_at=row["finished_at"],
        error=row["error"],
    )


def get_run(conn: sqlite3.Connection, run_id: int) -> Run | None:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_run(row) if row is not None else None


def list_runs(conn: sqlite3.Connection, status: StatusFilter = "all", limit: int = 10) -> list[Run]:
    if status == "all":
        rows = conn.execute("SELECT * FROM runs ORDER BY id LIMIT ?", (limit,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM runs WHERE status = ? ORDER BY id LIMIT ?", (status, limit)
        ).fetchall()
    return [_row_to_run(r) for r in rows]


def run_id_range(conn: sqlite3.Connection) -> tuple[int, int] | None:
    row = conn.execute("SELECT MIN(id), MAX(id) FROM runs").fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0]), int(row[1])


def set_pid(conn: sqlite3.Connection, run_id: int, pid: int | None) -> None:
    with conn:
        conn.execute("UPDATE runs SET pid = ? WHERE id = ?", (pid, run_id))


def set_status(
    conn: sqlite3.Connection, run_id: int, status: RunStatus, error: str | None = None
) -> None:
    finished_at = None if status == "running" else now_iso()
    with conn:
        conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE id = ?",
            (status, finished_at, error, run_id),
        )


def insert_epoch(conn: sqlite3.Connection, epoch: Epoch) -> None:
    with conn:
        conn.execute(
            "INSERT INTO epochs (run_id, epoch, train_loss, val_loss, val_acc, grad_norm) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                epoch.run_id,
                epoch.epoch,
                epoch.train_loss,
                epoch.val_loss,
                epoch.val_acc,
                epoch.grad_norm,
            ),
        )


def _nan_if_null(value: float | None) -> float:
    return math.nan if value is None else float(value)


def get_epochs(conn: sqlite3.Connection, run_id: int) -> list[Epoch]:
    rows = conn.execute(
        "SELECT * FROM epochs WHERE run_id = ? ORDER BY epoch", (run_id,)
    ).fetchall()
    return [
        Epoch(
            run_id=r["run_id"],
            epoch=r["epoch"],
            train_loss=_nan_if_null(r["train_loss"]),
            val_loss=_nan_if_null(r["val_loss"]),
            val_acc=_nan_if_null(r["val_acc"]),
            grad_norm=_nan_if_null(r["grad_norm"]),
        )
        for r in rows
    ]
