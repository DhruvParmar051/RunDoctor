"""v2 tool descriptions: written AFTER seeing eval results on the main task set.

Because v2 was shaped by those results, its scores on the main tasks are optimistic.
Judge v2 on the held-out tasks ("holdout": true in tasks.jsonl), which were written after
v2 and never used to design it. The frozen ``good`` variant (tag ``schemas-frozen``) is
unchanged. v2 differs from ``good`` only in these descriptions; errors stay informative.

Each change targets a failure pattern seen in ``good`` trajectories:
- problems guessed from run names, curves, or configs instead of calling diagnose_run
- kill_run called with confirm=true when the user never confirmed
- an out-of-range launch value silently swapped for a valid one
- launch_run called during analysis questions nobody asked to act on
- run ids guessed instead of looked up
"""

from __future__ import annotations

from server import (
    BATCH_MAX,
    BATCH_MIN,
    COMPARE_MAX,
    COMPARE_MIN,
    CURVE_POINTS_MAX,
    CURVE_POINTS_MIN,
    DROPOUT_MAX,
    DROPOUT_MIN,
    EPOCHS_MAX,
    EPOCHS_MIN,
    HIDDEN_MAX,
    HIDDEN_MIN,
    LIST_LIMIT_MAX,
    LR_MAX,
    LR_MIN,
    WD_MAX,
    WD_MIN,
)

V2_DESCRIPTIONS: dict[str, str] = {
    "list_runs": (
        "List training runs: id, name, status, hyperparameters, and final validation loss.\n"
        "Use it to look up run ids (never guess them) or to find a run by status or "
        "hyperparameter, e.g. the smallest lr. Run names are arbitrary labels and say nothing "
        "about whether a run is healthy.\n"
        f"Args: status is one of all|running|completed|failed|killed (default all); "
        f"limit is 1-{LIST_LIMIT_MAX} (default 10)."
    ),
    "get_training_curve": (
        "Show a run's per-epoch metrics (train_loss, val_loss, val_acc, grad_norm), "
        "downsampled.\n"
        "Use it to describe how a run's metrics changed over time, or to check a running run's "
        "progress. To decide whether a run has a problem, call diagnose_run as well.\n"
        f"Args: run_id (look it up with list_runs if unknown); max_points is "
        f"{CURVE_POINTS_MIN}-{CURVE_POINTS_MAX} (default 20)."
    ),
    "compare_runs": (
        "Compare 2-5 runs side by side, showing only the hyperparameters and outcomes that "
        "differ, including each run's diagnosis.\n"
        "Use it when the user asks how runs differ or which setting explains a difference in "
        "results.\n"
        f"Args: run_ids is a list of {COMPARE_MIN}-{COMPARE_MAX} distinct existing run ids, "
        "e.g. [2, 4]. If a run is described rather than numbered, find its id first."
    ),
    "diagnose_run": (
        "The only tool that detects training problems. It checks a run for nan_or_inf, "
        "divergence, overfitting, and plateau (or reports healthy) and returns numeric "
        "evidence and a concrete suggested fix.\n"
        "Call it for every run whose health you report or whose fix you recommend. Do not "
        "infer problems from names, curves, or configs alone, and base advice on its "
        "suggested fix.\n"
        "Args: run_id (look it up with list_runs if unknown)."
    ),
    "launch_run": (
        "Start a new training run in the background and return its run_id immediately.\n"
        "Only call it when the user explicitly asks to start or re-run training. Never launch "
        "while answering a question or proposing a config. If a requested value is outside "
        "its range, do not launch with a different value: tell the user the allowed range.\n"
        f"Args (all optional): task=signal1d; lr {LR_MIN:g}-{LR_MAX:g} (default 0.05); "
        f"epochs {EPOCHS_MIN}-{EPOCHS_MAX} (default 15); "
        f"batch_size {BATCH_MIN}-{BATCH_MAX} (default 64); "
        f"hidden {HIDDEN_MIN}-{HIDDEN_MAX} (default 64); "
        f"dropout {DROPOUT_MIN:g}-{DROPOUT_MAX:g} (default 0.2); "
        f"weight_decay {WD_MIN:g}-{WD_MAX:g} (default 0); name: short label."
    ),
    "kill_run": (
        "Stop a run whose status is running. Destructive and cannot be undone.\n"
        "Only call it when the user asks to stop a specific run. Use confirm=false unless the "
        "user has explicitly confirmed stopping that run. With confirm=false the tool returns "
        "a confirmation question: relay it to the user. A run that is not running cannot be "
        "stopped, so report its status instead of claiming it was stopped.\n"
        "Args: run_id; confirm (default false)."
    ),
}
