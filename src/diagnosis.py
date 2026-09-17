"""Rule-based diagnosis of training curves.

Everything here is pure and deterministic so it can be unit tested without an LLM.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from models import Diagnosis, Epoch, Issue, RunConfig

# Divergence: final (finite) train loss is this many times the best train loss ...
DIVERGENCE_LOSS_RATIO = 2.0
# ... and at least this much higher in absolute terms (so tiny converged losses don't trigger).
DIVERGENCE_MIN_ABS_INCREASE = 0.1
# Divergence: some epoch's grad norm is this many times the first epoch's.
GRAD_BLOWUP_FACTOR = 20.0

# Overfitting: val loss has risen for at least this many epochs since its best ...
OVERFIT_MIN_RISING_EPOCHS = 3
# ... ending at least this fraction above its best ...
OVERFIT_MIN_VAL_INCREASE = 0.05
# ... while train loss kept falling over the same window.

# Plateau: train loss improved by less than this fraction over the last half of training ...
PLATEAU_MAX_REL_IMPROVEMENT = 0.01
# ... needs at least this many epochs to judge ...
PLATEAU_MIN_EPOCHS = 4
# ... and a run whose final loss is below this fraction of its initial loss has converged.
CONVERGED_LOSS_FRACTION = 0.1

MIN_EPOCHS_FOR_TRENDS = 2


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def _check_nan(epochs: Sequence[Epoch]) -> Issue | None:
    bad = [e for e in epochs if not (_finite(e.train_loss) and _finite(e.val_loss))]
    if not bad:
        return None
    first = bad[0]
    return Issue(
        code="nan_or_inf",
        severity="critical",
        evidence=(
            f"non-finite loss at epoch {first.epoch} "
            f"(train={first.train_loss}, val={first.val_loss}); "
            f"{len(bad)}/{len(epochs)} epochs affected"
        ),
        suggested_fix=(
            "Lower the learning rate (try 10x smaller), add gradient clipping, "
            "and check inputs for invalid values."
        ),
    )


def _check_divergence(epochs: Sequence[Epoch], config: RunConfig) -> Issue | None:
    finite = [e for e in epochs if _finite(e.train_loss)]
    reasons: list[str] = []
    if len(finite) >= MIN_EPOCHS_FOR_TRENDS:
        losses = [e.train_loss for e in finite if e.train_loss is not None]
        best, final = min(losses), losses[-1]
        if final > DIVERGENCE_LOSS_RATIO * best and final - best > DIVERGENCE_MIN_ABS_INCREASE:
            reasons.append(
                f"train loss rose from min {best:.3f} to {final:.3f} "
                f"(epoch {finite[-1].epoch}, {final / best:.1f}x)"
            )
    norms = [e.grad_norm for e in epochs if e.grad_norm is not None]
    if len(norms) >= MIN_EPOCHS_FOR_TRENDS:
        first_norm = norms[0]
        if any(not math.isfinite(n) for n in norms):
            reasons.append("grad norm became non-finite")
        elif _finite(first_norm) and first_norm > 0:
            peak = max(norms)
            if peak > GRAD_BLOWUP_FACTOR * first_norm:
                reasons.append(
                    f"grad norm grew {peak / first_norm:.0f}x ({first_norm:.3g} -> {peak:.3g})"
                )
    if not reasons:
        return None
    return Issue(
        code="divergence",
        severity="critical",
        evidence="; ".join(reasons),
        suggested_fix=(
            f"Learning rate {config.lr:g} is likely too high: reduce it 10-100x "
            f"(e.g. lr={config.lr / 20:g}) and consider gradient clipping."
        ),
    )


def _check_overfitting(epochs: Sequence[Epoch], config: RunConfig) -> Issue | None:
    usable = [e for e in epochs if _finite(e.train_loss) and _finite(e.val_loss)]
    if len(usable) < OVERFIT_MIN_RISING_EPOCHS + 1:
        return None
    val = [e.val_loss for e in usable if e.val_loss is not None]
    train = [e.train_loss for e in usable if e.train_loss is not None]
    best_idx = min(range(len(val)), key=val.__getitem__)
    rising_epochs = len(val) - 1 - best_idx
    if rising_epochs < OVERFIT_MIN_RISING_EPOCHS:
        return None
    if val[-1] < val[best_idx] * (1 + OVERFIT_MIN_VAL_INCREASE):
        return None
    if train[-1] >= train[best_idx]:
        return None
    gap = val[-1] - train[-1]
    fixes = []
    if config.dropout == 0:
        fixes.append("add dropout (e.g. 0.3)")
    if config.weight_decay == 0:
        fixes.append("add weight decay (e.g. 1e-4)")
    fixes.append(f"use a smaller model (hidden={config.hidden} -> {max(config.hidden // 4, 8)})")
    fixes.append(f"use more training data (train_size={config.train_size})")
    fixes.append(f"stop early around epoch {usable[best_idx].epoch}")
    return Issue(
        code="overfitting",
        severity="warning",
        evidence=(
            f"val loss rose {val[best_idx]:.3f} -> {val[-1]:.3f} over the last "
            f"{rising_epochs} epochs while train loss fell {train[best_idx]:.3f} -> "
            f"{train[-1]:.3f}; final gap {gap:.3f}"
        ),
        suggested_fix="; ".join(fixes) + ".",
    )


def _check_plateau(epochs: Sequence[Epoch], config: RunConfig) -> Issue | None:
    finite = [e for e in epochs if _finite(e.train_loss)]
    if len(finite) < PLATEAU_MIN_EPOCHS:
        return None
    losses = [e.train_loss for e in finite if e.train_loss is not None]
    if losses[-1] <= CONVERGED_LOSS_FRACTION * losses[0]:
        return None
    mid = len(losses) // 2
    start = losses[mid]
    if start <= 0:
        return None
    rel = (start - min(losses[mid:])) / start
    if rel >= PLATEAU_MAX_REL_IMPROVEMENT:
        return None
    return Issue(
        code="plateau",
        severity="warning",
        evidence=(
            f"train loss improved only {rel:.2%} over the last {len(losses) - mid} epochs "
            f"({start:.3f} -> {losses[-1]:.3f}); overall {losses[0]:.3f} -> {losses[-1]:.3f}"
        ),
        suggested_fix=(
            f"Learning rate {config.lr:g} is likely too low: increase it substantially "
            f"(e.g. lr={max(min(config.lr * 1000, 0.05), config.lr):g}) or train for more epochs."
        ),
    )


def diagnose(epochs: Sequence[Epoch], config: RunConfig) -> Diagnosis:
    """Diagnose a run from its per-epoch metrics."""
    run_id = epochs[0].run_id if epochs else None
    if not epochs:
        return Diagnosis(
            run_id=run_id,
            issues=[
                Issue(
                    code="insufficient_data",
                    severity="info",
                    evidence="no epochs recorded",
                    suggested_fix="Wait for the run to finish at least one epoch.",
                )
            ],
        )

    issues: list[Issue] = []
    for issue in (_check_nan(epochs), _check_divergence(epochs, config)):
        if issue is not None:
            issues.append(issue)
    unstable = bool(issues)
    overfit = _check_overfitting(epochs, config)
    if overfit is not None and not unstable:
        issues.append(overfit)
    # A stalled curve is a symptom of divergence, not a separate plateau.
    plateau = _check_plateau(epochs, config)
    if plateau is not None and not unstable:
        issues.append(plateau)

    if not issues:
        if len(epochs) < MIN_EPOCHS_FOR_TRENDS:
            issues.append(
                Issue(
                    code="insufficient_data",
                    severity="info",
                    evidence=f"only {len(epochs)} epoch recorded; trends need at least "
                    f"{MIN_EPOCHS_FOR_TRENDS}",
                    suggested_fix="Wait for more epochs before diagnosing.",
                )
            )
        else:
            last = epochs[-1]
            issues.append(
                Issue(
                    code="healthy",
                    severity="info",
                    evidence=(
                        f"no problems detected over {len(epochs)} epochs; final "
                        f"train={last.train_loss:.3f} val={last.val_loss:.3f} "
                        f"acc={last.val_acc:.3f}"
                    ),
                    suggested_fix="No change needed.",
                )
            )
    return Diagnosis(run_id=run_id, issues=issues)


def format_diagnosis(diagnosis: Diagnosis) -> str:
    """Render a diagnosis as compact text for tool output."""
    header = f"Run {diagnosis.run_id}: " if diagnosis.run_id is not None else ""
    codes = ", ".join(i.code for i in diagnosis.issues)
    lines = [f"{header}{codes}"]
    for i in diagnosis.issues:
        lines.append(f"- [{i.severity}] {i.code}: {i.evidence}")
        lines.append(f"  fix: {i.suggested_fix}")
    return "\n".join(lines)
