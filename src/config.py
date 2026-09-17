"""Load ``config.toml`` into typed settings."""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH_ENV = "RUNDOCTOR_CONFIG"
DB_PATH_ENV = "RUNDOCTOR_DB"
LOG_DIR_ENV = "RUNDOCTOR_LOG_DIR"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsSettings(_Frozen):
    db: Path = Path("data/rundoctor.db")
    log_dir: Path = Path("runs")
    results_dir: Path = Path("results")


class OllamaSettings(_Frozen):
    base_url: str = "http://localhost:11434/v1"


class ModelsSettings(_Frozen):
    eval: list[str] = Field(default_factory=lambda: ["qwen3:8b", "llama3.1:8b", "mistral"])
    chat_default: str = "qwen3:8b"
    # Per-model reasoning_effort sent to Ollama ("none" turns off qwen3's thinking mode).
    reasoning_effort: dict[str, str] = Field(default_factory=dict)


class EvalSettings(_Frozen):
    temperature: float = 0.2
    repeats: int = 3
    max_iterations: int = 8
    workers_per_model: int = 1
    # Per-response token cap; stops runaway generations from stalling a worker.
    max_tokens: int = 4096


class Settings(_Frozen):
    paths: PathsSettings = PathsSettings()
    ollama: OllamaSettings = OllamaSettings()
    models: ModelsSettings = ModelsSettings()
    eval: EvalSettings = EvalSettings()

    @property
    def db_path(self) -> Path:
        override = os.environ.get(DB_PATH_ENV)
        return _resolve(Path(override) if override else self.paths.db)

    @property
    def log_dir(self) -> Path:
        override = os.environ.get(LOG_DIR_ENV)
        return _resolve(Path(override) if override else self.paths.log_dir)

    @property
    def results_dir(self) -> Path:
        return _resolve(self.paths.results_dir)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read settings from ``$RUNDOCTOR_CONFIG`` or ``<project>/config.toml``."""
    path = Path(os.environ.get(CONFIG_PATH_ENV, PROJECT_ROOT / "config.toml"))
    if not path.exists():
        return Settings()
    with path.open("rb") as f:
        return Settings.model_validate(tomllib.load(f))
