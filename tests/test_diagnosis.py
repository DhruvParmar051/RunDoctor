from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rundoctor import db, diagnosis
from rundoctor.diagnosis import diagnose, format_diagnosis
from rundoctor.models import Epoch, RunConfig
from rundoctor.training.seed_runs import GROUND_TRUTH_PATH, PLANTED_RUNS, seed

CFG = RunConfig()


def curve(
    train: list[float],
    val: list[float] | None = None,
    grad: list[float] | None = None,
) -> list[Epoch]:
    val = val if val is not None else train
    grad = grad if grad is not None else [0.5] * len(train)
    return [
        Epoch(run_id=7, epoch=i + 1, train_loss=t, val_loss=v, val_acc=0.5, grad_norm=g)
        for i, (t, v, g) in enumerate(zip(train, val, grad, strict=True))
    ]


def codes(epochs: list[Epoch], cfg: RunConfig = CFG) -> list[str]:
    return [i.code for i in diagnose(epochs, cfg).issues]


HEALTHY_TRAIN = [1.1, 0.8, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.04, 0.03]
HEALTHY_VAL = [1.0, 0.75, 0.5, 0.32, 0.22, 0.15, 0.1, 0.08, 0.07, 0.06]


def test_healthy() -> None:
    d = diagnose(curve(HEALTHY_TRAIN, HEALTHY_VAL), CFG)
    assert [i.code for i in d.issues] == ["healthy"]
    assert d.run_id == 7


# --- edge cases ------------------------------------------------------------------------


def test_empty() -> None:
    d = diagnose([], CFG)
    assert [i.code for i in d.issues] == ["insufficient_data"]
    assert d.run_id is None


def test_single_epoch() -> None:
    assert codes(curve([1.0])) == ["insufficient_data"]


def test_single_nan_epoch() -> None:
    assert codes(curve([math.nan])) == ["nan_or_inf"]


def test_all_nan() -> None:
    nan = math.nan
    result = codes(curve([nan] * 5, grad=[nan] * 5))
    assert result[0] == "nan_or_inf"
    assert "plateau" not in result and "healthy" not in result


# --- nan_or_inf ------------------------------------------------------------------------


def test_nan_mid_run() -> None:
    d = diagnose(curve([1.0, 0.9, math.nan], [1.0, 0.9, 0.8]), CFG)
    nan_issue = d.issues[0]
    assert nan_issue.code == "nan_or_inf"
    assert nan_issue.severity == "critical"
    assert "epoch 3" in nan_issue.evidence


def test_inf_val_loss() -> None:
    assert "nan_or_inf" in codes(curve([1.0, 0.9, 0.8], [1.0, math.inf, 0.8]))


# --- divergence ------------------------------------------------------------------------


def test_divergence_by_loss() -> None:
    result = codes(curve([1.0, 0.9, 1.5, 3.0, 9.0]))
    assert result == ["divergence"]


def test_divergence_by_grad_norm() -> None:
    result = codes(
        curve(HEALTHY_TRAIN, HEALTHY_VAL, grad=[0.1] * 9 + [5.0]),
    )
    assert result == ["divergence"]


def test_divergence_by_nonfinite_grad() -> None:
    assert "divergence" in codes(curve(HEALTHY_TRAIN, HEALTHY_VAL, grad=[0.1] * 9 + [math.inf]))


def test_small_absolute_rise_is_not_divergence() -> None:
    # 3x relative rise, but on an already-converged loss.
    train = [1.0, 0.5, 0.1, 0.02, 0.01, 0.03]
    assert "divergence" not in codes(curve(train))


def test_divergence_suppresses_plateau() -> None:
    result = codes(curve([1.0, 1.0, 1.0, 1.0, 5.0, 5.0]))
    assert "divergence" in result and "plateau" not in result


def test_divergence_fix_mentions_lr() -> None:
    d = diagnose(curve([1.0, 0.9, 5.0]), RunConfig(lr=1.0))
    assert "lr=0.05" in d.issues[0].suggested_fix


# --- overfitting -----------------------------------------------------------------------


OVERFIT_TRAIN = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05]
OVERFIT_VAL = [1.0, 0.85, 0.7, 0.65, 0.7, 0.8, 0.95, 1.1]


def test_overfitting() -> None:
    cfg = RunConfig(dropout=0.0, weight_decay=0.0, hidden=512)
    d = diagnose(curve(OVERFIT_TRAIN, OVERFIT_VAL), cfg)
    assert [i.code for i in d.issues] == ["overfitting"]
    issue = d.issues[0]
    assert "gap 1.050" in issue.evidence
    assert "dropout" in issue.suggested_fix
    assert "epoch 4" in issue.suggested_fix


def test_overfitting_needs_three_rising_epochs() -> None:
    val = [1.0, 0.85, 0.7, 0.6, 0.55, 0.5, 0.6, 0.7]
    assert "overfitting" not in codes(curve(OVERFIT_TRAIN, val))


def test_overfitting_needs_falling_train() -> None:
    train = [1.0, 0.8, 0.6, 0.4, 0.45, 0.5, 0.55, 0.6]
    assert "overfitting" not in codes(curve(train, OVERFIT_VAL))


def test_noisy_val_is_not_overfitting() -> None:
    val = [1.0, 0.8, 0.6, 0.4, 0.31, 0.3, 0.305, 0.31, 0.312]
    train = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.18, 0.17]
    assert "overfitting" not in codes(curve(train, val))


# --- plateau ---------------------------------------------------------------------------


def test_plateau() -> None:
    train = [1.10, 1.10, 1.099, 1.10, 1.099, 1.10, 1.10, 1.099]
    d = diagnose(curve(train), RunConfig(lr=1e-6))
    assert [i.code for i in d.issues] == ["plateau"]
    assert "too low" in d.issues[0].suggested_fix


def test_plateau_after_initial_progress() -> None:
    train = [2.0, 1.2, 1.0, 1.0, 1.0, 0.999, 1.0, 0.998]
    assert codes(curve(train)) == ["plateau"]


def test_converged_run_is_not_plateau() -> None:
    train = [1.0, 0.3, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]
    assert codes(curve(train)) == ["healthy"]


def test_plateau_needs_min_epochs() -> None:
    assert codes(curve([1.0, 1.0, 1.0])) == ["healthy"]


def test_steady_improvement_is_not_plateau() -> None:
    train = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93]
    assert codes(curve(train)) == ["healthy"]


# --- formatting ------------------------------------------------------------------------


def test_format_is_compact() -> None:
    text = format_diagnosis(diagnose(curve(OVERFIT_TRAIN, OVERFIT_VAL), CFG))
    assert text.startswith("Run 7: overfitting")
    assert len(text.encode()) < 2048


def test_thresholds_are_named_constants() -> None:
    assert diagnosis.PLATEAU_MAX_REL_IMPROVEMENT == 0.01
    assert diagnosis.OVERFIT_MIN_RISING_EPOCHS == 3
    assert diagnosis.DIVERGENCE_LOSS_RATIO == 2.0


# --- planted runs ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def planted_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("planted") / "planted.db"
    seed(path, write_ground_truth=False)
    return path


def test_planted_runs_match_ground_truth(planted_db: Path) -> None:
    truth = json.loads(GROUND_TRUTH_PATH.read_text())
    assert set(truth) == set(PLANTED_RUNS)
    with db.connection(planted_db) as conn:
        for name, entry in truth.items():
            run = db.get_run(conn, entry["run_id"])
            assert run is not None and run.name == name
            assert run.status == "completed"
            result = [i.code for i in diagnose(db.get_epochs(conn, run.id), run.config).issues]
            assert result == entry["expected_issues"], name
