from __future__ import annotations

import json
from pathlib import Path

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
    reset_work_db(paths)
    with db.connection(paths.work_db) as conn:
        db.create_run(conn, "launched-by-model", "signal1d", RunConfig())  # status running
        assert len(db.list_runs(conn)) == 2
    reset_work_db(paths)
    with db.connection(paths.work_db) as conn:
        assert [r.name for r in db.list_runs(conn)] == ["seeded"]
