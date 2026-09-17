from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.mcpserver.exceptions import ToolError

from rundoctor import db, server
from rundoctor.models import Epoch, RunConfig


def _add_run(conn: object, name: str, cfg: RunConfig, train: list[float], val: list[float]) -> int:
    assert hasattr(conn, "execute")
    run_id = db.create_run(conn, name, "signal1d", cfg)  # type: ignore[arg-type]
    for i, (t, v) in enumerate(zip(train, val, strict=True), start=1):
        db.insert_epoch(
            conn,  # type: ignore[arg-type]
            Epoch(run_id=run_id, epoch=i, train_loss=t, val_loss=v, val_acc=0.5, grad_norm=0.3),
        )
    db.set_status(conn, run_id, "completed")  # type: ignore[arg-type]
    return run_id


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "server.db"
    monkeypatch.setenv("RUNDOCTOR_DB", str(path))
    monkeypatch.setenv("RUNDOCTOR_LOG_DIR", str(tmp_path / "logs"))
    with db.connection(path) as conn:
        _add_run(
            conn,
            "overfit",
            RunConfig(hidden=512, dropout=0.0),
            [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05],
            [1.0, 0.85, 0.7, 0.65, 0.7, 0.8, 0.95, 1.1],
        )
        _add_run(
            conn,
            "good",
            RunConfig(),
            [1.1, 0.8, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05],
            [1.0, 0.75, 0.5, 0.32, 0.22, 0.15, 0.1, 0.08],
        )
        db.create_run(conn, "stale", "signal1d", RunConfig())  # running, no pid
    yield path


# --- list_runs ---------------------------------------------------------------------------


def test_list_runs(db_path: Path) -> None:
    out = server.list_runs()
    assert "3 run(s)" in out
    assert "id=1 name=overfit" in out and "hidden=512" in out
    assert "final_val_loss=1.1" in out


def test_list_runs_filter(db_path: Path) -> None:
    out = server.list_runs(status="completed")
    assert "2 run(s)" in out and "stale" not in out
    assert "No runs with status 'killed'" in server.list_runs(status="killed")


def test_list_runs_bad_limit(db_path: Path) -> None:
    with pytest.raises(ToolError, match="limit must be between 1 and 50"):
        server.list_runs(limit=0)


# --- get_training_curve ------------------------------------------------------------------


def test_curve_downsamples(db_path: Path) -> None:
    out = server.get_training_curve(1, max_points=3)
    assert "showing 3 of 8" in out
    rows = out.splitlines()[2:]
    assert [r.split(" | ")[0] for r in rows] == ["1", "5", "8"]


def test_curve_unknown_run(db_path: Path) -> None:
    with pytest.raises(ToolError, match=r"run_id 99 not found\. Valid ids: 1-3\. Call list_runs"):
        server.get_training_curve(99)


def test_curve_bad_max_points(db_path: Path) -> None:
    with pytest.raises(ToolError, match="max_points"):
        server.get_training_curve(1, max_points=1)


def test_curve_no_epochs(db_path: Path) -> None:
    assert "No epochs yet" in server.get_training_curve(3)


# --- compare_runs / diagnose_run ---------------------------------------------------------


def test_compare_runs_shows_only_differences(db_path: Path) -> None:
    out = server.compare_runs([1, 2])
    assert "hidden: 1: 512 | 2: 64" in out
    assert "dropout: 1: 0 | 2: 0.2" in out
    assert "lr:" not in out  # identical, so omitted
    assert "diagnosis: 1: overfitting | 2: healthy" in out


def test_compare_runs_arg_errors(db_path: Path) -> None:
    with pytest.raises(ToolError, match="2-5 distinct ids"):
        server.compare_runs([1])
    with pytest.raises(ToolError, match="2-5 distinct ids"):
        server.compare_runs([1, 1])
    with pytest.raises(ToolError, match="run_id 42 not found"):
        server.compare_runs([1, 42])


def test_diagnose_run(db_path: Path) -> None:
    out = server.diagnose_run(1)
    assert out.startswith("Run 1: overfitting")
    assert "config: lr=0.05 hidden=512" in out
    assert "Run 2: healthy" in server.diagnose_run(2)


def test_outputs_are_small(db_path: Path) -> None:
    for out in (
        server.list_runs(limit=50),
        server.get_training_curve(1, max_points=50),
        server.compare_runs([1, 2, 3]),
        server.diagnose_run(1),
    ):
        assert len(out.encode()) < 2048


# --- launch_run / kill_run ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr": 100.0}, "lr=100 is out of bounds"),
        ({"lr": 0.0}, "lr=0 is out of bounds"),
        ({"epochs": 0}, "epochs=0 is out of bounds"),
        ({"epochs": 500}, "between 1 and 50"),
        ({"dropout": 0.95}, "dropout"),
        ({"name": "x" * 100}, "name is too long"),
    ],
)
def test_launch_run_validates(db_path: Path, kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ToolError, match=message):
        server.launch_run(**kwargs)  # type: ignore[arg-type]
    with db.connection(db_path) as conn:
        assert len(db.list_runs(conn)) == 3  # nothing created


def test_kill_requires_confirmation(db_path: Path) -> None:
    out = server.kill_run(3)
    assert "Confirmation required" in out and "confirm=true" in out
    with db.connection(db_path) as conn:
        run = db.get_run(conn, 3)
        assert run is not None and run.status == "running"


def test_kill_non_running(db_path: Path) -> None:
    assert "is not running" in server.kill_run(1, confirm=True)


def test_kill_without_live_process(db_path: Path) -> None:
    out = server.kill_run(3, confirm=True)
    assert "no live training process" in out
    with db.connection(db_path) as conn:
        run = db.get_run(conn, 3)
        assert run is not None and run.status == "failed"


def test_kill_refuses_foreign_pid(db_path: Path) -> None:
    with db.connection(db_path) as conn:
        db.set_pid(conn, 3, 1)  # launchd/init: alive but not our trainer
    assert "no live training process" in server.kill_run(3, confirm=True)


def _wait_for(pred: object, timeout: float = 60.0) -> None:
    assert callable(pred)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.2)
    raise AssertionError("timed out")


def _status(path: Path, run_id: int) -> str:
    with db.connection(path) as conn:
        run = db.get_run(conn, run_id)
        assert run is not None
        return run.status


def _epoch_count(path: Path, run_id: int) -> int:
    with db.connection(path) as conn:
        return len(db.get_epochs(conn, run_id))


def test_launch_returns_immediately_and_trains(db_path: Path) -> None:
    start = time.monotonic()
    out = server.launch_run(lr=0.05, epochs=3, batch_size=256, hidden=16, name="bg")
    assert time.monotonic() - start < 2.0
    assert "Launched run_id=4" in out and "status=running" in out
    _wait_for(lambda: _status(db_path, 4) != "running")
    assert _status(db_path, 4) == "completed"
    assert "3/3 epochs recorded" in server.get_training_curve(4)
    assert (db_path.parent / "logs" / "run_4.log").exists()


def test_partial_curve_then_kill(db_path: Path) -> None:
    server.launch_run(epochs=50, name="long")
    _wait_for(lambda: _epoch_count(db_path, 4) >= 1)
    partial = server.get_training_curve(4)
    assert "status=running" in partial and "/50 epochs recorded" in partial
    assert "Confirmation required" in server.kill_run(4)
    assert "marked killed" in server.kill_run(4, confirm=True)
    _wait_for(lambda: all(p.poll() is not None for p in server._CHILDREN.values()), 30)
    assert _status(db_path, 4) == "killed"
    assert _epoch_count(db_path, 4) < 50


# --- resource and stdio integration ------------------------------------------------------


def test_run_resource(db_path: Path) -> None:
    out = server.run_resource("1")
    assert '"name": "overfit"' in out and '"diagnosis": "overfitting"' in out


def test_stdio_server(db_path: Path) -> None:
    async def run() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "rundoctor.server"],
            env={"RUNDOCTOR_DB": str(db_path), "RUNDOCTOR_LOG_LEVEL": "WARNING"},
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == set(server.TOOL_FUNCTIONS)
            ok = await session.call_tool("diagnose_run", {"run_id": 1})
            assert not ok.is_error
            assert "overfitting" in ok.content[0].text  # type: ignore[union-attr]
            bad = await session.call_tool("diagnose_run", {"run_id": 99})
            assert bad.is_error
            assert "Valid ids: 1-3" in bad.content[0].text  # type: ignore[union-attr]
            res = await session.read_resource("runs://2")
            assert '"name": "good"' in res.contents[0].text  # type: ignore[union-attr]

    asyncio.run(run())
