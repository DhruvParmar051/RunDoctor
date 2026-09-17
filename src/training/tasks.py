"""Toy tasks: synthetic data and small models that train in seconds on CPU."""

from __future__ import annotations

import math

import torch
from torch import nn

from models import RunConfig, TaskName

SIGNAL_LENGTH = 128
NUM_CLASSES = 3  # sine, square, sawtooth
VAL_SIZE = 600
VAL_SEED_OFFSET = 10_000


def make_signal1d(n: int, noise: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``n`` noisy waveforms of shape (n, 1, SIGNAL_LENGTH) with labels in {0,1,2}."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    t = torch.linspace(0.0, 1.0, SIGNAL_LENGTH).unsqueeze(0)
    freq = 2.0 + 4.0 * torch.rand(n, 1, generator=g)
    phase = torch.rand(n, 1, generator=g)
    amp = 0.5 + torch.rand(n, 1, generator=g)
    cycles = freq * t + phase
    sine = torch.sin(2 * math.pi * cycles)
    square = torch.sign(sine)
    saw = 2.0 * (cycles - torch.floor(cycles)) - 1.0
    waves = torch.stack([sine, square, saw], dim=1)  # (n, 3, L)
    x = waves[torch.arange(n), labels] * amp
    x = x + noise * torch.randn(x.shape, generator=g)
    return x.unsqueeze(1), labels


class Signal1DNet(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32 * 8, hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(self.features(x))
        return out


def build_task(
    task: TaskName, config: RunConfig
) -> tuple[nn.Module, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Return (model, train_data, val_data) for a task."""
    if task == "signal1d":
        torch.manual_seed(config.seed)
        model = Signal1DNet(config.hidden, config.dropout)
        train = make_signal1d(config.train_size, config.noise, config.seed)
        val = make_signal1d(VAL_SIZE, config.noise, config.seed + VAL_SEED_OFFSET)
        return model, train, val
    raise ValueError(f"unknown task {task!r}")
