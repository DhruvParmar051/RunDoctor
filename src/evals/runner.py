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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from mcp.client.session import ClientSession

import db
from client.host import (
    Agent,
    Message,
    OpenAIChatModel,
    connect,
    mcp_tools_to_openai,
    server_parameters,
)
from config import get_settings
from evals.scoring import Task, load_tasks, score
from log import get_logger
from server import VARIANTS, _pid_belongs_to_run
from training.seed_runs import planted_fingerprint, seed

log = get_logger(__name__)

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
    """Seed the template DB once; rebuild it if the planted runs changed."""
    stamp = paths.work / "template.fingerprint"
    current = planted_fingerprint()
    fresh = paths.template_db.exists() and stamp.exists() and stamp.read_text().strip() == current
    if fresh and not reseed:
        return
    paths.work.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{paths.template_db}{suffix}").unlink(missing_ok=True)
    log.info("seeding eval template DB (trains the 4 planted runs once)")
    seed(paths.template_db, verbose=False, write_ground_truth=False)
    stamp.write_text(current + "\n")


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


def reset_work_db(template_db: Path, work_db: Path) -> None:
    """Restore a worker's DB to the seeded state."""
    stop_launched_runs(work_db)
    src = sqlite3.connect(template_db)
    dst = sqlite3.connect(work_db)
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
    reasoning_effort: dict[str, str] = field(default_factory=dict)
    workers_per_model: int = 1
    max_tokens: int | None = None

    @property
    def labels(self) -> dict[str, str]:
        return {
            m: f"{m}[reasoning={self.reasoning_effort[m]}]" if self.reasoning_effort.get(m) else m
            for m in self.models
        }


@dataclass(frozen=True)
class WorkerSlot:
    """Private state for one parallel worker: its own DB copy, logs, and server processes."""

    index: int
    root: Path

    @property
    def db(self) -> Path:
        return self.root / "eval.db"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


Unit = tuple[str, int, Task]  # (variant, repeat, task)


async def _worker(
    plan: RunPlan,
    model: str,
    queue: asyncio.Queue[Unit],
    slot: WorkerSlot,
    paths: EvalPaths,
    progress: dict[str, int],
    stop: asyncio.Event,
) -> None:
    """Pull (variant, repeat, task) units for one model until the queue is empty."""
    label = plan.labels[model]
    slot.root.mkdir(parents=True, exist_ok=True)
    async with contextlib.AsyncExitStack() as stack:
        server_log = stack.enter_context((slot.root / "server.log").open("a"))
        sessions: dict[str, ClientSession] = {}
        tools: dict[str, list[Message]] = {}
        try:
            while not stop.is_set():
                try:
                    variant, repeat, task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                if variant not in sessions:
                    params = server_parameters(
                        extra_args=["--variant", variant],
                        env={"RUNDOCTOR_DB": str(slot.db), "RUNDOCTOR_LOG_DIR": str(slot.logs)},
                    )
                    session = await stack.enter_async_context(connect(params, errlog=server_log))
                    sessions[variant] = session
                    tools[variant] = mcp_tools_to_openai((await session.list_tools()).tools)

                reset_work_db(paths.template_db, slot.db)
                llm = OpenAIChatModel(
                    model,
                    temperature=plan.temperature,
                    seed=repeat,
                    reasoning_effort=plan.reasoning_effort.get(model),
                    max_tokens=plan.max_tokens,
                )
                agent = Agent(
                    sessions[variant], llm, tools[variant], max_iterations=plan.max_iterations
                )
                traj = await agent.run(task.prompt)
                if traj.stop_reason == "llm_error":
                    raise OllamaUnavailableError(
                        f"{model} failed on {task.id}: {traj.error}. Fix it and rerun to resume."
                    )
                record = {
                    "task_id": task.id,
                    "category": task.category,
                    "model": label,
                    "ollama_model": model,
                    "reasoning_effort": plan.reasoning_effort.get(model),
                    "variant": variant,
                    "repeat": repeat,
                    "seed": repeat,
                    "temperature": plan.temperature,
                    "max_tokens": plan.max_tokens,
                    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                    "trajectory": traj.model_dump(mode="json"),
                }
                # Single event loop, synchronous write: lines from workers never interleave.
                with paths.raw_file(label, variant).open("a") as f:
                    f.write(json.dumps(record) + "\n")
                progress["done"] += 1
                s = score(task, traj)
                mark = "✓" if s.task_success else f"✗ {s.failure}"
                typer.echo(
                    f"[{progress['done']}/{progress['total']}] {label} {variant} r{repeat} "
                    f"{task.id}: {mark} ({traj.stop_reason}, {len(traj.tool_calls)} calls, "
                    f"{traj.total_latency_s:.1f}s)",
                    err=True,
                )
        finally:
            stop_launched_runs(slot.db)


async def run_eval(plan: RunPlan, paths: EvalPaths) -> int:
    paths.raw.mkdir(parents=True, exist_ok=True)
    done = completed_keys(paths.raw)
    queues: dict[str, asyncio.Queue[Unit]] = {}
    for model in plan.models:
        queue: asyncio.Queue[Unit] = asyncio.Queue()
        # Repeat-first order, so an interrupted run still has complete repeats to report.
        for r in range(plan.repeats):
            for variant in plan.variants:
                for t in plan.tasks:
                    if entry_key(t.id, plan.labels[model], variant, r) not in done:
                        queue.put_nowait((variant, r, t))
        if not queue.empty():
            queues[model] = queue
    total = sum(q.qsize() for q in queues.values())
    planned = len(plan.tasks) * len(plan.models) * len(plan.variants) * plan.repeats
    typer.echo(
        f"{planned} trajectories planned, {planned - total} already done, {total} to run "
        f"({plan.workers_per_model} worker(s) per model).",
        err=True,
    )
    if not total:
        return 0

    progress = {"done": 0, "total": total}
    start = time.perf_counter()
    stop = asyncio.Event()
    slots = [
        WorkerSlot(i, paths.work / f"w{i}") for i in range(len(queues) * plan.workers_per_model)
    ]

    def on_signal() -> None:
        if not stop.is_set():
            stop.set()
            typer.secho(
                "\nstopping after in-flight tasks finish (press Ctrl-C again to abort now)",
                fg="yellow",
                err=True,
            )
            return
        # Second signal: abort. Server processes exit when our pipes close.
        for slot in slots:
            stop_launched_runs(slot.db)
        typer.secho("aborted; rerun the same command to resume", fg="yellow", err=True)
        os._exit(130)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, on_signal)
    try:
        async with asyncio.TaskGroup() as group:
            workers = iter(slots)
            for model, queue in queues.items():
                for _ in range(plan.workers_per_model):
                    group.create_task(
                        _worker(plan, model, queue, next(workers), paths, progress, stop)
                    )
    except* OllamaUnavailableError as eg:
        raise eg.exceptions[0] from None
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
    if stop.is_set():
        raise KeyboardInterrupt
    typer.echo(f"finished {total} trajectories in {time.perf_counter() - start:.0f}s", err=True)
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
    workers_per_model: Annotated[
        int | None,
        typer.Option(
            help="parallel workers per model (default: config). Models always run in "
            "parallel; >1 only helps if Ollama's OLLAMA_NUM_PARALLEL is raised too."
        ),
    ] = None,
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
            workers_per_model=max(1, workers_per_model or settings.eval.workers_per_model),
            max_tokens=settings.eval.max_tokens,
            reasoning_effort={
                m: e for m, e in settings.models.reasoning_effort.items() if m in model_list
            },
        )
        ensure_template(paths, reseed)
        try:
            asyncio.run(run_eval(plan, paths))
        except OllamaUnavailableError as exc:
            typer.secho(str(exc), fg="red", err=True)
            raise typer.Exit(1) from exc
        except KeyboardInterrupt:
            typer.secho("\ninterrupted; rerun the same command to resume", fg="yellow", err=True)
            for slot_db in paths.work.glob("w*/eval.db"):
                stop_launched_runs(slot_db)
            raise typer.Exit(130) from None

    if not no_report:
        out = write_report(paths.raw, paths.summary, tasks_path=task_path)
        typer.echo(f"wrote {out}", err=True)


def run() -> None:
    app()


if __name__ == "__main__":
    sys.exit(app())
