from __future__ import annotations

import json
from pathlib import Path

import pytest

import db
from evals.runner import EvalPaths, completed_keys, entry_key, reset_work_db
from models import RunConfig


def test_raw_file_name_is_safe(tmp_path: Path) -> None:
    paths = EvalPaths(tmp_path)
    assert paths.raw_file("qwen3:8b", "good").name == "qwen3_8b__good.jsonl"
    assert paths.raw_file("hf.co/x/y:Q4", "naive").name == "hf.co_x_y_Q4__naive.jsonl"


def test_completed_keys_skips_partial_lines(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rec = {"task_id": "t01", "model": "m", "variant": "good", "repeat": 2}
    (raw / "m__good.jsonl").write_text(json.dumps(rec) + '\n{"task_id": "t02", "mod\n')
    assert completed_keys(raw) == {entry_key("t01", "m", "good", 2)}


def test_reset_work_db_restores_template(tmp_path: Path) -> None:
    paths = EvalPaths(tmp_path)
    paths.work.mkdir(parents=True)
    with db.connection(paths.template_db) as conn:
        db.create_run(conn, "seeded", "signal1d", RunConfig())
    reset_work_db(paths.template_db, paths.work / "eval.db")
    with db.connection(paths.work / "eval.db") as conn:
        db.create_run(conn, "launched-by-model", "signal1d", RunConfig())  # status running
        assert len(db.list_runs(conn)) == 2
    reset_work_db(paths.template_db, paths.work / "eval.db")
    with db.connection(paths.work / "eval.db") as conn:
        assert [r.name for r in db.list_runs(conn)] == ["seeded"]


def test_parallel_run_writes_each_unit_once_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    from collections.abc import Sequence
    from typing import Any

    from client.host import AssistantTurn, Message, ToolCallRequest
    from evals import runner
    from evals.scoring import Task

    class FakeModel:
        def __init__(self, name: str, **_: Any) -> None:
            self.name = name

        async def complete(
            self, messages: Sequence[Message], tools: Sequence[Message]
        ) -> AssistantTurn:
            if messages[-1]["role"] == "tool":
                return AssistantTurn(content="Run 1 looks fine.")
            return AssistantTurn(
                content=None, tool_calls=[ToolCallRequest("c1", "diagnose_run", '{"run_id": 1}')]
            )

    monkeypatch.setattr(runner, "OpenAIChatModel", FakeModel)
    paths = EvalPaths(tmp_path / "results")
    paths.work.mkdir(parents=True)
    with db.connection(paths.template_db) as conn:
        db.create_run(conn, "seeded", "signal1d", RunConfig())

    tasks = [
        Task.model_validate(
            {
                "id": f"x{i}",
                "category": "single_tool",
                "prompt": "diagnose run 1",
                "answer_check": {"type": "contains_any", "values": ["fine"]},
            }
        )
        for i in range(3)
    ]
    plan = runner.RunPlan(
        tasks=tasks,
        models=["m1", "m2"],
        variants=["good", "naive"],
        repeats=2,
        max_iterations=4,
        temperature=0.0,
        workers_per_model=2,
    )
    assert asyncio.run(runner.run_eval(plan, paths)) == 24
    keys = [
        entry_key(r["task_id"], r["model"], r["variant"], r["repeat"])
        for r in runner.iter_records(paths.raw)
    ]
    assert len(keys) == 24 and len(set(keys)) == 24
    assert {p.name for p in paths.work.iterdir() if p.is_dir()} == {"w0", "w1", "w2", "w3"}

    plan.repeats = 3  # resume: only the new repeat runs
    assert asyncio.run(runner.run_eval(plan, paths)) == 12
    assert len(completed_keys(paths.raw)) == 36
