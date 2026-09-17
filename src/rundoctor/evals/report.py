"""Turn raw eval trajectories into ``results/summary.md``."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rundoctor.client.host import Trajectory
from rundoctor.evals.runner import iter_records
from rundoctor.evals.scoring import (
    CATEGORIES,
    FAILURE_CLASSES,
    GroupMetrics,
    ScoredResult,
    Task,
    aggregate,
    load_tasks,
    score,
)
from rundoctor.logging import get_logger

log = get_logger(__name__)

TARGET_TASKS = 40
RESULT_SNIPPET = 220

MAIN_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("task_success", "Task success", True),
    ("tool_selection_acc", "Tool selection", True),
    ("arg_acc", "Arg acc", True),
    ("safety_pass", "Safety pass", True),
    ("malformed_call_rate", "Malformed calls", True),
    ("loop_rate", "Loop rate", True),
    ("avg_iterations", "Avg iters", False),
)

FAILURE_EXPLANATIONS = {
    "wrong_tool": "skipped a required tool, or called a tool the task forbids "
    "(e.g. launched or killed a run unprompted)",
    "bad_args": "called the right tool with wrong or invalid arguments "
    "(including confirm=true without asking)",
    "hallucinated_tool": "called a tool that does not exist",
    "malformed_json": "emitted tool arguments that were not a JSON object",
    "loop": "hit the iteration limit or kept making failing calls",
    "gave_up": "produced no answer, or answered without calling any tool when one was needed",
    "wrong_answer": "used the tools correctly but the final answer failed the check",
}


def load_results(
    raw_dir: Path, tasks: Sequence[Task]
) -> tuple[list[ScoredResult], dict[str, Trajectory]]:
    by_id = {t.id: t for t in tasks}
    results: list[ScoredResult] = []
    trajectories: dict[str, Trajectory] = {}
    unknown: Counter[str] = Counter()
    seen: set[str] = set()
    for rec in iter_records(raw_dir):
        task = by_id.get(rec["task_id"])
        if task is None:
            unknown[rec["task_id"]] += 1
            continue
        key = f"{rec['task_id']}|{rec['model']}|{rec['variant']}|{rec['repeat']}"
        if key in seen:  # duplicate from a crash between write and exit; keep the first
            continue
        seen.add(key)
        traj = Trajectory.model_validate(rec["trajectory"])
        results.append(
            ScoredResult(
                task=task,
                model=rec["model"],
                variant=rec["variant"],
                repeat=rec["repeat"],
                score=score(task, traj),
            )
        )
        trajectories[key] = traj
    if unknown:
        log.warning("ignored records for tasks not in tasks.jsonl: %s", dict(unknown))
    return results, trajectories


def _key(r: ScoredResult) -> str:
    return f"{r.task.id}|{r.model}|{r.variant}|{r.repeat}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _fmt_s(x: float | None) -> str:
    return "–" if x is None else f"{x:.1f}s"


def _main_table(groups: Sequence[GroupMetrics]) -> list[str]:
    header = ["Model", "Variant", "n", *[c[1] for c in MAIN_COLUMNS], "p50 latency", "p95 latency"]
    rows = [
        [
            f"`{g.model}`",
            g.variant,
            str(g.n_trajectories),
            *[g.metrics[key].fmt(pct=pct) for key, _, pct in MAIN_COLUMNS],
            _fmt_s(g.p50_latency_s),
            _fmt_s(g.p95_latency_s),
        ]
        for g in groups
    ]
    return _table(header, rows)


def _delta_table(groups: Sequence[GroupMetrics]) -> list[str]:
    by = {(g.model, g.variant): g for g in groups}
    models = sorted({g.model for g in groups})
    rows = []
    for m in models:
        good, naive = by.get((m, "good")), by.get((m, "naive"))
        if good is None or naive is None:
            continue
        cells = []
        for key, _, pct in MAIN_COLUMNS:
            a, b = good.metrics[key].mean, naive.metrics[key].mean
            if a is None or b is None:
                cells.append("–")
            elif pct:
                cells.append(f"{(a - b) * 100:+.0f} pp")
            else:
                cells.append(f"{a - b:+.2f}")
        rows.append([f"`{m}`", *cells])
    if not rows:
        return ["_Needs both `good` and `naive` results for at least one model._"]
    return _table(["Model", *[c[1] for c in MAIN_COLUMNS]], rows)


def _category_table(groups: Sequence[GroupMetrics]) -> list[str]:
    rows = [
        [f"`{g.model}`", g.variant, *[g.by_category[c].fmt(pct=True) for c in CATEGORIES]]
        for g in groups
    ]
    return _table(["Model", "Variant", *CATEGORIES], rows)


def _failure_table(groups: Sequence[GroupMetrics]) -> list[str]:
    rows = [
        [
            f"`{g.model}`",
            g.variant,
            *[str(g.failures[f]) for f in FAILURE_CLASSES],
            str(sum(g.failures.values())),
        ]
        for g in groups
    ]
    return _table(["Model", "Variant", *FAILURE_CLASSES, "Total failed"], rows)


def _snip(text: str, limit: int = RESULT_SNIPPET) -> str:
    one = " ".join(text.split())
    return one if len(one) <= limit else one[: limit - 1] + "…"


def _render_example(title: str, r: ScoredResult, traj: Trajectory, annotation: str) -> list[str]:
    lines = [
        f"#### {title}",
        "",
        f"`{r.model}` · variant `{r.variant}` · task `{r.task.id}` ({r.task.category}) · "
        f"repeat {r.repeat} · {traj.iterations} iteration(s) · stop: `{traj.stop_reason}`",
        "",
        f"> **User:** {r.task.prompt}",
        "",
    ]
    for c in traj.tool_calls:
        args = json.dumps(c.arguments) if c.arguments is not None else repr(c.raw_arguments)
        flag = f" ⚠️ {c.error_kind}" if c.error_kind else ""
        lines.append(f"- `{c.name}({args})`{flag} → {_snip(c.result)}")
    if not traj.tool_calls:
        lines.append("- _(no tool calls)_")
    lines += ["", f"> **Answer:** {_snip(traj.final_answer or '(none)', 600)}", ""]
    lines += [f"**Annotation:** {annotation}", ""]
    return lines


def _pick_examples(
    results: Sequence[ScoredResult], trajectories: dict[str, Trajectory]
) -> list[str]:
    def sort_key(r: ScoredResult) -> tuple[str, str, str, int]:
        return (r.variant, r.model, r.task.id, r.repeat)

    ordered = sorted(results, key=sort_key)
    successes = [r for r in ordered if r.score.task_success]
    success = next(
        (
            r
            for r in successes
            if r.task.category in ("multi_step", "reasoning") and r.score.calls >= 2
        ),
        successes[0] if successes else None,
    )

    failures = [r for r in ordered if r.score.failure is not None]
    common = [f for f, _ in Counter(r.score.failure for r in failures).most_common()]
    picked: list[ScoredResult] = []
    for cls in common:
        # Prefer failures on the good variant: they show limits the schema didn't fix.
        candidates = [r for r in failures if r.score.failure == cls]
        candidates.sort(key=lambda r: (r.variant != "good", sort_key(r)))
        picked.append(candidates[0])
        if len(picked) == 2:
            break

    lines: list[str] = []
    if success is not None:
        s = success.score
        lines += _render_example(
            "Success",
            success,
            trajectories[_key(success)],
            f"Called {', '.join(dict.fromkeys(trajectories[_key(success)].tools_called))} "
            f"and passed every answer check in {s.iterations} iterations "
            f"({s.latency_s:.1f}s).",
        )
    for i, r in enumerate(picked, start=1):
        cls = r.score.failure or "wrong_answer"
        expected = ", ".join(r.task.expected_tools) or "none required"
        note = (
            f"Classified as `{cls}`: {FAILURE_EXPLANATIONS[cls]}. "
            f"Expected tools: {expected}. Tool selection "
            f"{'ok' if r.score.tool_selection else 'wrong'}"
            + (f", arg accuracy {r.score.arg_acc:.0%}" if r.score.arg_acc is not None else "")
            + (f". Task note: {r.task.notes}" if r.task.notes else ".")
        )
        lines += _render_example(f"Failure {i}: `{cls}`", r, trajectories[_key(r)], note)
    if not lines:
        lines = ["_No trajectories yet._"]
    return lines


def _section(
    title: str,
    results: Sequence[ScoredResult],
    trajectories: dict[str, Trajectory],
    with_examples: bool,
) -> list[str]:
    groups = aggregate(results)
    lines = [
        f"## {title}",
        "",
        "### Main results",
        "",
        "Mean ± std across repeats. Rates are fractions of trajectories; malformed calls are "
        "a fraction of all tool calls.",
        "",
        *_main_table(groups),
        "",
        "### Good − naive",
        "",
        *_delta_table(groups),
        "",
        "### Task success by category",
        "",
        *_category_table(groups),
        "",
        "### Failure taxonomy (counts of failed trajectories)",
        "",
        *_failure_table(groups),
        "",
    ]
    if with_examples:
        lines += ["### Example trajectories", "", *_pick_examples(results, trajectories)]
    return lines


def build_report(
    results: Sequence[ScoredResult], trajectories: dict[str, Trajectory], tasks: Sequence[Task]
) -> str:
    main = [r for r in results if not r.task.holdout]
    holdout = [r for r in results if r.task.holdout]
    task_counts = Counter(t.category for t in tasks if not t.holdout)
    models = sorted({r.model for r in results})
    variants = sorted({r.variant for r in results})
    repeats = sorted({r.repeat for r in results})

    lines = [
        "# RunDoctor eval results",
        "",
        f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`rundoctor-eval`. Numbers are as measured; nothing is hand-edited._",
        "",
        f"- **Tasks:** {sum(task_counts.values())} "
        f"({', '.join(f'{c}: {task_counts.get(c, 0)}' for c in CATEGORIES)})"
        + (
            f", plus {sum(t.holdout for t in tasks)} held-out"
            if any(t.holdout for t in tasks)
            else ""
        ),
        f"- **Models:** {', '.join(f'`{m}`' for m in models) or '–'}",
        f"- **Variants:** {', '.join(variants) or '–'} "
        "(good = frozen descriptions + informative errors; naive = terse descriptions + "
        "`error`)",
        f"- **Repeats:** {len(repeats)} · **Trajectories scored:** {len(results)}",
        "- **Scoring:** deterministic checks only (no LLM judge). See "
        "`src/rundoctor/evals/scoring.py`.",
        "",
    ]
    n_main = sum(task_counts.values())
    if n_main < TARGET_TASKS:
        lines += [
            f"> ⚠️ Only {n_main} of the planned {TARGET_TASKS} tasks exist, so these numbers "
            "are preliminary.",
            "",
        ]
    expected = n_main * len(models) * len(variants) * len(repeats)
    if len(main) < expected:
        lines += [
            f"> ⚠️ Incomplete grid: {len(main)}/{expected} trajectories. Rerun "
            "`rundoctor-eval` to resume.",
            "",
        ]
    lines += _section("Results", main, trajectories, with_examples=True)
    if holdout:
        lines += _section("Held-out tasks", holdout, trajectories, with_examples=False)
    lines += [
        "## Metric definitions",
        "",
        "- **Task success:** the loop ended with an answer and every `answer_check` passed.",
        "- **Tool selection:** every expected tool was called, and `launch_run`/`kill_run` "
        "were not called unless the task expected or allowed them.",
        "- **Arg acc:** fraction of `expected_args` entries matched by at least one call "
        "that did not error (only tasks that define them).",
        "- **Safety pass:** task success on `safety` tasks.",
        "- **Loop rate:** fraction of trajectories that hit the iteration limit.",
        "- **Failure classes** are assigned in priority order: malformed_json, "
        "hallucinated_tool, loop, gave_up, wrong_tool, bad_args, wrong_answer.",
        "",
    ]
    return "\n".join(lines)


def write_report(raw_dir: Path, out_path: Path, tasks_path: Path | None = None) -> Path:
    tasks = load_tasks(tasks_path) if tasks_path else load_tasks()
    results, trajectories = load_results(raw_dir, tasks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(results, trajectories, tasks))
    return out_path
