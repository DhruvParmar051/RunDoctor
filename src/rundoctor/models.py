"""Pydantic models shared across the DB, diagnosis engine, and server."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TaskName = Literal["signal1d"]
RunStatus = Literal["running", "completed", "failed", "killed"]
Severity = Literal["info", "warning", "critical"]
IssueCode = Literal["nan_or_inf", "divergence", "overfitting", "plateau", "healthy"]


class RunConfig(BaseModel):
    lr: float = Field(default=0.05, gt=0)
    epochs: int = Field(default=15, ge=1)
    batch_size: int = Field(default=64, ge=1)
    hidden: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0)
    seed: int = 0
    train_size: int = Field(default=2000, ge=1)
    noise: float = Field(default=0.3, ge=0)


class Run(BaseModel):
    id: int
    name: str
    task: TaskName
    config: RunConfig
    status: RunStatus
    pid: int | None = None
    created_at: str
    finished_at: str | None = None
    error: str | None = None


class Epoch(BaseModel):
    run_id: int
    epoch: int
    train_loss: float | None = None
    val_loss: float | None = None
    val_acc: float | None = None
    grad_norm: float | None = None


class Issue(BaseModel):
    code: IssueCode
    severity: Severity
    evidence: str
    suggested_fix: str


class Diagnosis(BaseModel):
    run_id: int | None = None
    issues: list[Issue]
