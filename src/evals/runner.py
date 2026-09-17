"""Eval runner: tasks x models x schema variants x repeats, resumable.

Usage:
    uv run rundoctor-eval --models all --variants good,naive --repeats 3
    uv run rundoctor-eval --report-only

Each trajectory is appended to ``results/raw/<model>__<variant>.jsonl`` as soon as it
finishes; completed (task, model, variant, repeat) entries are skipped on restart.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TextIO

import typer

import db
from client.host import Agent, OpenAIChatModel, connect, server_parameters
from config import get_settings
from evals.scoring import Task, load_tasks, score
from log import get_logger
from server import _pid_belongs_to_run
from training.seed_runs import seed

log = get_logger(__name__)

VARIANTS = ("good", "naive")
app = typer.Typer(add_completion=False)


class OllamaUnavailableError(RuntimeError):
    pass


# --- paths & state -------------------------------------------------------------------


@dataclass(frozen=True)
class EvalPaths:
    results: Path

    @property
    def raw(self) -> Path:
        return self.results / "raw"

    @property
    def work(self) -> Path:
        return self.results / "raw" / "work"

    @property
    def template_db(self) -> Path:
        return self.work / "template.db"

    @property
    def work_db(self) -> Path:
        return self.work / "eval.db"

    @property
    def summary(self) -> Path:
        return self.results / "summary.md"

    def raw_file(self, model: str, variant: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
        return self.raw / f"{safe}__{variant}.jsonl"


def entry_key(task_id: str, model: str, variant: str, repeat: int) -> str:
    return f"{task_id}|{model}|{variant}|{repeat}"


def iter_records(raw_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open() as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A partial line from an interrupted write; it will be re-run.
                    log.warning("%s:%d: skipping unreadable record", path.name, lineno)


def completed_keys(raw_dir: Path) -> set[str]:
    return {
        entry_key(r["task_id"], r["model"], r["variant"], r["repeat"])
        for r in iter_records(raw_dir)
    }


def ensure_template(paths: EvalPaths, reseed: bool) -> None:
    if paths.template_db.exists() and not reseed:
        return
    paths.work.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{paths.template_db}{suffix}").unlink(missing_ok=True)
    log.info("seeding eval template DB (trains the 4 planted runs once)")
    seed(paths.template_db, verbose=False, write_ground_truth=False)


def stop_launched_runs(db_path: Path) -> None:
    """Kill trainers that a model launched during the previous task."""
    if not db_path.exists():
        return
    with db.connection(db_path) as conn:
        running = db.list_runs(conn, status="running", limit=1000)
    for run in running:
        if run.pid is not None and _pid_belongs_to_run(run.pid, run.id):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(run.pid, signal.SIGKILL)


def reset_work_db(paths: EvalPaths) -> None:
    """Restore the work DB to the seeded state."""
    stop_launched_runs(paths.work_db)
    src = sqlite3.connect(paths.template_db)
    dst = sqlite3.connect(paths.work_db)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()


# --- ollama --------------------------------------------------------------------------


def available_models(base_url: str) -> set[str]:
    tags_url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise OllamaUnavailableError(f"cannot reach Ollama at {tags_url}: {exc}") from exc
    names: set[str] = set()
    for m in data.get("models", []):
        name = str(m.get("name", ""))
        names.add(name)
        if name.endswith(":latest"):
            names.add(name.removesuffix(":latest"))
    return names


# --- running -------------------------------------------------------------------------


@dataclass
class RunPlan:
    tasks: list[Task]
    models: list[str]
    variants: list[str]
    repeats: int
    max_iterations: int
    temperature: float


async def _run_block(
    plan: RunPlan,
    model: str,
    variant: str,
    todo: list[tuple[int, Task]],
    paths: EvalPaths,
    progress: dict[str, int],
    server_log: TextIO,
) -> None:
    """Run all pending (repeat, task) pairs for one model and variant in one server session."""
    params = server_parameters(
        extra_args=["--variant", variant],
        env={
            "RUNDOCTOR_DB": str(paths.work_db),
            "RUNDOCTOR_LOG_DIR": str(paths.work / "logs"),
        },
    )
    out_path = paths.raw_file(model, variant)
    async with connect(params, errlog=server_log) as session:
        for repeat, task in todo:
            reset_work_db(paths)
            llm = OpenAIChatModel(model, temperature=plan.temperature, seed=repeat)
            agent = await Agent.create(session, llm, max_iterations=plan.max_iterations)
            traj = await agent.run(task.prompt)
            if traj.stop_reason == "llm_error":
                raise OllamaUnavailableError(
                    f"{model} failed on {task.id}: {traj.error}. Fix it and rerun to resume."
                )
            record = {
                "task_id": task.id,
                "category": task.category,
                "model": model,
                "variant": variant,
                "repeat": repeat,
                "seed": repeat,
                "temperature": plan.temperature,
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "trajectory": traj.model_dump(mode="json"),
            }
            with out_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            progress["done"] += 1
            s = score(task, traj)
            mark = "✓" if s.task_success else f"✗ {s.failure}"
            typer.echo(
                f"[{progress['done']}/{progress['total']}] {model} {variant} r{repeat} "
                f"{task.id}: {mark} ({traj.stop_reason}, {len(traj.tool_calls)} calls, "
                f"{traj.total_latency_s:.1f}s)",
                err=True,
            )
    stop_launched_runs(paths.work_db)


async def run_eval(plan: RunPlan, paths: EvalPaths) -> int:
    paths.raw.mkdir(parents=True, exist_ok=True)
    done = completed_keys(paths.raw)
    blocks: list[tuple[str, str, list[tuple[int, Task]]]] = []
    for model in plan.models:
        for variant in plan.variants:
            todo = [
                (r, t)
                for r in range(plan.repeats)
                for t in plan.tasks
                if entry_key(t.id, model, variant, r) not in done
            ]
            if todo:
                blocks.append((model, variant, todo))
    total = sum(len(t) for _, _, t in blocks)
    planned = len(plan.tasks) * len(plan.models) * len(plan.variants) * plan.repeats
    typer.echo(
        f"{planned} trajectories planned, {planned - total} already done, {total} to run.",
        err=True,
    )
    if not total:
        return 0

    progress = {"done": 0, "total": total}
    with (paths.work / "server.log").open("a") as server_log:
        for model, variant, todo in blocks:
            start = time.perf_counter()
            await _run_block(plan, model, variant, todo, paths, progress, server_log)
            typer.echo(
                f"finished {model} / {variant}: {len(todo)} trajectories in "
                f"{time.perf_counter() - start:.0f}s",
                err=True,
            )
    return total


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@app.command()
def main(
    models: Annotated[
        str, typer.Option(help="'all' (from config.toml) or a comma-separated list")
    ] = "all",
    variants: Annotated[str, typer.Option(help="comma-separated: good,naive")] = "good,naive",
    repeats: Annotated[int | None, typer.Option(help="repeats per task (default: config)")] = None,
    tasks_file: Annotated[Path | None, typer.Option("--tasks", help="tasks.jsonl path")] = None,
    task_ids: Annotated[str, typer.Option(help="only these task ids (comma-separated)")] = "",
    results_dir: Annotated[Path | None, typer.Option(help="output directory")] = None,
    max_iterations: Annotated[int | None, typer.Option(help="agent loop limit")] = None,
    reseed: Annotated[bool, typer.Option(help="rebuild the seeded template DB")] = False,
    report_only: Annotated[bool, typer.Option(help="skip running; just write the report")] = False,
    no_report: Annotated[bool, typer.Option(help="don't write summary.md after running")] = False,
) -> None:
    """Run the RunDoctor tool-use eval and write results/summary.md."""
    from evals.report import write_report

    settings = get_settings()
    paths = EvalPaths(results_dir or settings.results_dir)
    task_path = tasks_file or None
    tasks = load_tasks(task_path) if task_path else load_tasks()

    if not report_only:
        if wanted := set(_split(task_ids)):
            missing = wanted - {t.id for t in tasks}
            if missing:
                raise typer.BadParameter(f"unknown task ids: {sorted(missing)}")
            tasks = [t for t in tasks if t.id in wanted]
        model_list = settings.models.eval if models == "all" else _split(models)
        variant_list = _split(variants)
        if bad := set(variant_list) - set(VARIANTS):
            raise typer.BadParameter(f"unknown variants: {sorted(bad)}")
        try:
            have = available_models(settings.ollama.base_url)
        except OllamaUnavailableError as exc:
            typer.secho(str(exc), fg="red", err=True)
            raise typer.Exit(1) from exc
        if missing_models := [m for m in model_list if m not in have]:
            typer.secho(
                f"models not pulled in Ollama: {', '.join(missing_models)}\n"
                + "".join(f"  ollama pull {m}\n" for m in missing_models),
                fg="red",
                err=True,
            )
            raise typer.Exit(1)

        plan = RunPlan(
            tasks=tasks,
            models=model_list,
            variants=variant_list,
            repeats=repeats if repeats is not None else settings.eval.repeats,
            max_iterations=max_iterations or settings.eval.max_iterations,
            temperature=settings.eval.temperature,
        )
        ensure_template(paths, reseed)
        try:
            asyncio.run(run_eval(plan, paths))
        except OllamaUnavailableError as exc:
            typer.secho(str(exc), fg="red", err=True)
            raise typer.Exit(1) from exc
        except KeyboardInterrupt:
            typer.secho("\ninterrupted; rerun the same command to resume", fg="yellow", err=True)
            stop_launched_runs(paths.work_db)
            raise typer.Exit(130) from None

    if not no_report:
        out = write_report(paths.raw, paths.summary, tasks_path=task_path)
        typer.echo(f"wrote {out}", err=True)


def run() -> None:
    app()


if __name__ == "__main__":
    sys.exit(app())
