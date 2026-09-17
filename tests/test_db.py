from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

import db
from models import Epoch, RunConfig


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    with db.connection(tmp_path / "test.db") as c:
        yield c


def test_schema_and_wal(conn: sqlite3.Connection) -> None:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "epochs"} <= tables
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_connect_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "test.db"
    with db.connection(path) as c:
        db.create_run(c, "a", "signal1d", RunConfig())
    with db.connection(path) as c:
        assert len(db.list_runs(c)) == 1


def test_create_and_get_run(conn: sqlite3.Connection) -> None:
    cfg = RunConfig(lr=0.1, epochs=3, hidden=32)
    run_id = db.create_run(conn, "demo", "signal1d", cfg)
    run = db.get_run(conn, run_id)
    assert run is not None
    assert run.name == "demo"
    assert run.status == "running"
    assert run.config == cfg
    assert run.finished_at is None
    assert db.get_run(conn, 999) is None


def test_list_runs_filter_and_limit(conn: sqlite3.Connection) -> None:
    ids = [db.create_run(conn, f"r{i}", "signal1d", RunConfig()) for i in range(5)]
    db.set_status(conn, ids[0], "completed")
    db.set_status(conn, ids[1], "failed", error="boom")
    assert [r.id for r in db.list_runs(conn)] == ids
    assert [r.id for r in db.list_runs(conn, limit=2)] == ids[:2]
    assert [r.id for r in db.list_runs(conn, status="completed")] == [ids[0]]
    failed = db.list_runs(conn, status="failed")
    assert failed[0].error == "boom"
    assert failed[0].finished_at is not None
    assert len(db.list_runs(conn, status="running")) == 3
    assert db.list_runs(conn, status="killed") == []
    assert db.run_id_range(conn) == (ids[0], ids[-1])


def test_set_pid(conn: sqlite3.Connection) -> None:
    run_id = db.create_run(conn, "p", "signal1d", RunConfig())
    db.set_pid(conn, run_id, 4242)
    run = db.get_run(conn, run_id)
    assert run is not None and run.pid == 4242


def test_epochs_ordering_and_nan(conn: sqlite3.Connection) -> None:
    run_id = db.create_run(conn, "e", "signal1d", RunConfig())
    for ep in (3, 1, 2):
        db.insert_epoch(
            conn,
            Epoch(
                run_id=run_id,
                epoch=ep,
                train_loss=float(ep),
                val_loss=1.0,
                val_acc=0.5,
                grad_norm=0.1,
            ),
        )
    db.insert_epoch(
        conn,
        Epoch(
            run_id=run_id,
            epoch=4,
            train_loss=math.nan,
            val_loss=math.inf,
            val_acc=0.3,
            grad_norm=math.nan,
        ),
    )
    epochs = db.get_epochs(conn, run_id)
    assert [e.epoch for e in epochs] == [1, 2, 3, 4]
    assert epochs[0].train_loss == 1.0
    assert epochs[3].train_loss is not None and math.isnan(epochs[3].train_loss)
    assert epochs[3].val_loss == math.inf


def test_duplicate_epoch_rejected(conn: sqlite3.Connection) -> None:
    run_id = db.create_run(conn, "d", "signal1d", RunConfig())
    db.insert_epoch(conn, Epoch(run_id=run_id, epoch=1, train_loss=1.0))
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_epoch(conn, Epoch(run_id=run_id, epoch=1, train_loss=2.0))


def test_epoch_requires_existing_run(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_epoch(conn, Epoch(run_id=123, epoch=1, train_loss=1.0))


def test_reset_db(conn: sqlite3.Connection) -> None:
    run_id = db.create_run(conn, "x", "signal1d", RunConfig())
    db.insert_epoch(conn, Epoch(run_id=run_id, epoch=1, train_loss=1.0))
    db.reset_db(conn)
    assert db.list_runs(conn) == []
    assert db.get_epochs(conn, run_id) == []
    assert db.create_run(conn, "y", "signal1d", RunConfig()) == 1
