"""RunDoctor MCP server.

Tool implementations are plain functions (``list_runs`` etc.) so they can be unit tested
directly. ``build_server`` registers them on an ``MCPServer`` with a set of descriptions,
which lets the eval swap in a deliberately weak "naive" schema variant.

Never write to stdout here: stdio transport uses it for JSON-RPC.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError

import db
from config import get_settings
from diagnosis import diagnose, format_diagnosis
from log import get_logger
from models import Epoch, Run, RunConfig, TaskName

log = get_logger(__name__)

StatusArg = Literal["all", "running", "completed", "failed", "killed"]

# Argument bounds, enforced in code and stated in tool descriptions.
LIST_LIMIT_MAX = 50
CURVE_POINTS_MIN, CURVE_POINTS_MAX = 2, 50
COMPARE_MIN, COMPARE_MAX = 2, 5
LR_MIN, LR_MAX = 1e-6, 10.0
EPOCHS_MIN, EPOCHS_MAX = 1, 50
BATCH_MIN, BATCH_MAX = 1, 1024
HIDDEN_MIN, HIDDEN_MAX = 4, 1024
DROPOUT_MIN, DROPOUT_MAX = 0.0, 0.9
WD_MIN, WD_MAX = 0.0, 1.0
NAME_MAX_LEN = 64

# Launched training processes, kept so finished children are reaped (no zombies).
_CHILDREN: dict[int, subprocess.Popen[bytes]] = {}


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    with db.connection(get_settings().db_path) as conn:
        _reconcile(conn)
        yield conn


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "-"
    if not math.isfinite(x):
        return str(x)
    return f"{x:.{digits}g}"


def _process_alive(pid: int) -> bool:
    child = _CHILDREN.get(pid)
    if child is not None:
        if child.poll() is None:
            return True
        del _CHILDREN[pid]
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_belongs_to_run(pid: int, run_id: int) -> bool:
    """True if ``pid`` is alive and is the trainer for ``run_id``."""
    if not _process_alive(pid):
        return False
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "-m training.train" in out and f"--run-id {run_id}" in out


def _reconcile(conn: sqlite3.Connection) -> None:
    """Mark runs whose trainer process died without updating the DB as failed."""
    for run in db.list_runs(conn, status="running", limit=1000):
        if run.pid is not None and not _process_alive(run.pid):
            # Re-read: the trainer may have finished between the two queries.
            fresh = db.get_run(conn, run.id)
            if fresh is not None and fresh.status == "running":
                log.warning("run %d: process %d is gone, marking failed", run.id, run.pid)
                db.set_status(conn, run.id, "failed", error="training process exited unexpectedly")


def _require_run(conn: sqlite3.Connection, run_id: int) -> Run:
    run = db.get_run(conn, run_id)
    if run is not None:
        return run
    id_range = db.run_id_range(conn)
    valid = f"Valid ids: {id_range[0]}-{id_range[1]}." if id_range else "There are no runs yet."
    raise ToolError(f"run_id {run_id} not found. {valid} Call list_runs to see them.")


def _final(epochs: list[Epoch]) -> Epoch | None:
    return epochs[-1] if epochs else None


# --- tool implementations ------------------------------------------------------------


def list_runs(status: StatusArg = "all", limit: int = 10) -> str:
    if not 1 <= limit <= LIST_LIMIT_MAX:
        raise ToolError(f"limit must be between 1 and {LIST_LIMIT_MAX}, got {limit}.")
    with _conn() as conn:
        runs = db.list_runs(conn, status=status, limit=limit)
        if not runs:
            hint = "" if status == "all" else " Try status='all'."
            return f"No runs with status '{status}'.{hint}"
        lines = [f"{len(runs)} run(s) (status={status}):"]
        for r in runs:
            last = _final(db.get_epochs(conn, r.id))
            c = r.config
            lines.append(
                f"- id={r.id} name={r.name} task={r.task} status={r.status} "
                f"lr={c.lr:g} epochs={c.epochs} batch_size={c.batch_size} hidden={c.hidden} "
                f"dropout={c.dropout:g} weight_decay={c.weight_decay:g} "
                f"done_epochs={last.epoch if last else 0} "
                f"final_val_loss={_fmt(last.val_loss if last else None)}"
            )
        return "\n".join(lines)


def _downsample(epochs: list[Epoch], max_points: int) -> list[Epoch]:
    if len(epochs) <= max_points:
        return epochs
    step = (len(epochs) - 1) / (max_points - 1)
    idx = sorted({round(i * step) for i in range(max_points)})
    return [epochs[i] for i in idx]


def get_training_curve(run_id: int, max_points: int = 20) -> str:
    if not CURVE_POINTS_MIN <= max_points <= CURVE_POINTS_MAX:
        raise ToolError(
            f"max_points must be between {CURVE_POINTS_MIN} and {CURVE_POINTS_MAX}, "
            f"got {max_points}."
        )
    with _conn() as conn:
        run = _require_run(conn, run_id)
        epochs = db.get_epochs(conn, run_id)
    header = (
        f"Run {run.id} ({run.name}), status={run.status}, "
        f"{len(epochs)}/{run.config.epochs} epochs recorded"
    )
    if not epochs:
        return header + ". No epochs yet; try again shortly."
    shown = _downsample(epochs, max_points)
    note = f" (showing {len(shown)} of {len(epochs)})" if len(shown) < len(epochs) else ""
    lines = [header + note, "epoch | train_loss | val_loss | val_acc | grad_norm"]
    lines += [
        f"{e.epoch} | {_fmt(e.train_loss)} | {_fmt(e.val_loss)} | "
        f"{_fmt(e.val_acc, 3)} | {_fmt(e.grad_norm, 3)}"
        for e in shown
    ]
    return "\n".join(lines)


def _outcome(conn: sqlite3.Connection, run: Run) -> dict[str, str]:
    epochs = db.get_epochs(conn, run.id)
    last = _final(epochs)
    accs = [e.val_acc for e in epochs if e.val_acc is not None and math.isfinite(e.val_acc)]
    return {
        "status": run.status,
        "done_epochs": str(len(epochs)),
        "final_train_loss": _fmt(last.train_loss if last else None),
        "final_val_loss": _fmt(last.val_loss if last else None),
        "best_val_acc": _fmt(max(accs), 3) if accs else "-",
        "diagnosis": ",".join(i.code for i in diagnose(epochs, run.config).issues),
    }


def compare_runs(run_ids: list[int]) -> str:
    unique = list(dict.fromkeys(run_ids))
    if not COMPARE_MIN <= len(unique) <= COMPARE_MAX:
        raise ToolError(
            f"run_ids must contain {COMPARE_MIN}-{COMPARE_MAX} distinct ids, got {len(unique)}. "
            "Example: run_ids=[1, 2]."
        )
    with _conn() as conn:
        runs = [_require_run(conn, rid) for rid in unique]
        outcomes = [_outcome(conn, r) for r in runs]
    configs: list[dict[str, Any]] = [{"task": r.task, **r.config.model_dump()} for r in runs]
    ids = " vs ".join(f"{r.id}({r.name})" for r in runs)
    lines = [f"Comparing runs {ids}"]

    def section(title: str, rows: Sequence[Mapping[str, Any]]) -> None:
        keys = [k for k in rows[0] if len({str(row[k]) for row in rows}) > 1]
        lines.append(f"{title}:")
        if not keys:
            lines.append("  (identical)")
        for k in keys:
            vals = " | ".join(
                f"{r.id}: {v:g}" if isinstance(v, float) else f"{r.id}: {v}"
                for r, v in zip(runs, (row[k] for row in rows), strict=True)
            )
            lines.append(f"  {k}: {vals}")

    section("Config differences", configs)
    section("Outcome differences", outcomes)
    return "\n".join(lines)


def diagnose_run(run_id: int) -> str:
    with _conn() as conn:
        run = _require_run(conn, run_id)
        epochs = db.get_epochs(conn, run_id)
    text = format_diagnosis(diagnose(epochs, run.config))
    if run.status == "running":
        text += f"\n(note: run is still training, {len(epochs)}/{run.config.epochs} epochs so far)"
    c = run.config
    return (
        f"{text}\nconfig: lr={c.lr:g} hidden={c.hidden} dropout={c.dropout:g} "
        f"weight_decay={c.weight_decay:g} batch_size={c.batch_size} epochs={c.epochs}"
    )


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi) or (isinstance(value, float) and not math.isfinite(value)):
        raise ToolError(f"{name}={value:g} is out of bounds; it must be between {lo:g} and {hi:g}.")


def launch_run(
    task: TaskName = "signal1d",
    lr: float = 0.05,
    epochs: int = 15,
    batch_size: int = 64,
    hidden: int = 64,
    dropout: float = 0.2,
    weight_decay: float = 0.0,
    name: str = "",
) -> str:
    _check_range("lr", lr, LR_MIN, LR_MAX)
    _check_range("epochs", epochs, EPOCHS_MIN, EPOCHS_MAX)
    _check_range("batch_size", batch_size, BATCH_MIN, BATCH_MAX)
    _check_range("hidden", hidden, HIDDEN_MIN, HIDDEN_MAX)
    _check_range("dropout", dropout, DROPOUT_MIN, DROPOUT_MAX)
    _check_range("weight_decay", weight_decay, WD_MIN, WD_MAX)
    name = name.strip() or f"{task}-lr{lr:g}-h{hidden}"
    if len(name) > NAME_MAX_LEN:
        raise ToolError(f"name is too long ({len(name)} chars); use at most {NAME_MAX_LEN}.")

    config = RunConfig(
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        hidden=hidden,
        dropout=dropout,
        weight_decay=weight_decay,
    )
    settings = get_settings()
    with _conn() as conn:
        run_id = db.create_run(conn, name, task, config)
        log_dir = settings.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"run_{run_id}.log"
        cmd = [
            sys.executable,
            "-m",
            "training.train",
            "--run-id",
            str(run_id),
            "--db",
            str(settings.db_path),
        ]
        try:
            with log_path.open("ab") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            db.set_status(conn, run_id, "failed", error=f"could not start trainer: {exc}")
            raise ToolError(f"Failed to start training process: {exc}") from exc
        _CHILDREN[proc.pid] = proc
        db.set_pid(conn, run_id, proc.pid)
    log.info("launched run %d (pid %d)", run_id, proc.pid)
    return (
        f"Launched run_id={run_id} name={name} status=running "
        f"(lr={lr:g} epochs={epochs} batch_size={batch_size} hidden={hidden} "
        f"dropout={dropout:g} weight_decay={weight_decay:g}). "
        "Training runs in the background; call get_training_curve or diagnose_run to check it."
    )


def kill_run(run_id: int, confirm: bool = False) -> str:
    with _conn() as conn:
        run = _require_run(conn, run_id)
        if run.status != "running":
            return (
                f"Run {run_id} ({run.name}) is not running (status={run.status}); nothing to kill."
            )
        if not confirm:
            return (
                f"Confirmation required: this will stop run {run_id} ({run.name}) permanently. "
                "Ask the user to confirm, then call kill_run again with confirm=true."
            )
        if run.pid is None or not _pid_belongs_to_run(run.pid, run_id):
            db.set_status(conn, run_id, "failed", error="training process not found")
            return f"Run {run_id} has no live training process; marked it as failed instead."
        try:
            os.killpg(run.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise ToolError(f"Not permitted to stop process {run.pid}: {exc}") from exc
        db.set_status(conn, run_id, "killed")
    log.info("killed run %d (pid %d)", run_id, run.pid)
    return f"Run {run_id} ({run.name}) was sent a stop signal and is now marked killed."


def run_resource(run_id: str) -> str:
    try:
        rid = int(run_id)
    except ValueError as exc:
        raise ResourceNotFoundError(f"invalid run id {run_id!r}") from exc
    with _conn() as conn:
        run = db.get_run(conn, rid)
        if run is None:
            raise ResourceNotFoundError(f"run {rid} not found")
        outcome = _outcome(conn, run)
    summary = {
        "id": run.id,
        "name": run.name,
        "task": run.task,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
        "error": run.error,
        "config": run.config.model_dump(),
        **outcome,
    }
    return json.dumps(summary, indent=1)


# --- server wiring ----------------------------------------------------------------------

GOOD_DESCRIPTIONS: dict[str, str] = {
    "list_runs": (
        "List training runs with their status, key hyperparameters, and final validation loss.\n"
        "Use this first to discover run ids, or to find runs with a given status.\n"
        f"Args: status is one of all|running|completed|failed|killed (default all); "
        f"limit is 1-{LIST_LIMIT_MAX} (default 10)."
    ),
    "get_training_curve": (
        "Show a run's per-epoch metrics (train_loss, val_loss, val_acc, grad_norm), "
        "downsampled.\n"
        "Use this to inspect how a run trained over time or to check progress of a running run.\n"
        f"Args: run_id (from list_runs); max_points is {CURVE_POINTS_MIN}-{CURVE_POINTS_MAX} "
        "(default 20)."
    ),
    "compare_runs": (
        "Compare 2-5 runs, showing only the hyperparameters and outcomes that differ.\n"
        "Use this to explain why one run did better or worse than another.\n"
        f"Args: run_ids is a list of {COMPARE_MIN}-{COMPARE_MAX} distinct run ids, "
        "e.g. [1, 4]."
    ),
    "diagnose_run": (
        "Diagnose a run's training problems (nan_or_inf, divergence, overfitting, plateau, or "
        "healthy) with numeric evidence and a suggested fix.\n"
        "Use this when asked what is wrong with a run or how to fix it.\n"
        "Args: run_id (from list_runs)."
    ),
    "launch_run": (
        "Start a new training run in the background and return its run_id immediately.\n"
        "Use this only when the user asks to start or re-run training, e.g. with a fixed config.\n"
        f"Args (all optional): task=signal1d; lr {LR_MIN:g}-{LR_MAX:g} (default 0.05); "
        f"epochs {EPOCHS_MIN}-{EPOCHS_MAX} (default 15); "
        f"batch_size {BATCH_MIN}-{BATCH_MAX} (default 64); "
        f"hidden {HIDDEN_MIN}-{HIDDEN_MAX} (default 64); "
        f"dropout {DROPOUT_MIN:g}-{DROPOUT_MAX:g} (default 0.2); "
        f"weight_decay {WD_MIN:g}-{WD_MAX:g} (default 0); name: short label."
    ),
    "kill_run": (
        "Stop a running training run. This is destructive and cannot be undone.\n"
        "Use this only when the user explicitly asks to stop a run. Call first with "
        "confirm=false; set confirm=true only after the user has confirmed.\n"
        "Args: run_id (a run with status running); confirm (default false)."
    ),
}

SERVER_INSTRUCTIONS = (
    "Inspect, diagnose, compare, launch, and stop small ML training runs. "
    "Start with list_runs to find run ids."
)

TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "list_runs": list_runs,
    "get_training_curve": get_training_curve,
    "compare_runs": compare_runs,
    "diagnose_run": diagnose_run,
    "launch_run": launch_run,
    "kill_run": kill_run,
}


class _GenericErrorServer(MCPServer):
    """Replaces every tool error message with a bare ``error`` (naive eval variant)."""

    async def call_tool(self, name: str, arguments: dict[str, Any], context: Any = None) -> Any:
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as exc:
            raise ToolError("error") from exc


Variant = Literal["good", "naive", "v2"]
VARIANTS: tuple[Variant, ...] = ("good", "naive", "v2")


def build_server(
    descriptions: Mapping[str, str] | None = None,
    generic_errors: bool = False,
    instructions: str | None = SERVER_INSTRUCTIONS,
) -> MCPServer:
    """Create the MCP server.

    ``descriptions`` and ``generic_errors`` exist for the eval's naive schema variant.
    """
    descriptions = descriptions or GOOD_DESCRIPTIONS
    server_cls = _GenericErrorServer if generic_errors else MCPServer
    server = server_cls(name="rundoctor", instructions=instructions)
    for tool_name, fn in TOOL_FUNCTIONS.items():
        server.add_tool(
            fn,
            name=tool_name,
            description=descriptions[tool_name],
            structured_output=False,
        )
    server.resource(
        "runs://{run_id}",
        name="run",
        description="A training run's config and outcome summary as JSON.",
        mime_type="application/json",
    )(run_resource)
    return server


def build_variant(variant: Variant) -> MCPServer:
    if variant == "good":
        return build_server()
    if variant == "v2":
        from evals.schemas_v2 import V2_DESCRIPTIONS

        return build_server(V2_DESCRIPTIONS)
    from evals.schemas_naive import NAIVE_DESCRIPTIONS

    return build_server(NAIVE_DESCRIPTIONS, generic_errors=True, instructions=None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RunDoctor MCP server (stdio).")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS),
        default="good",
        help="tool schema variant (naive is the eval ablation)",
    )
    args = parser.parse_args(argv)
    log.info(
        "starting rundoctor MCP server (variant=%s, db=%s)", args.variant, get_settings().db_path
    )
    build_variant(args.variant).run("stdio")


if __name__ == "__main__":
    main()
