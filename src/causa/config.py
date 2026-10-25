"""Runtime settings, read once from environment variables.

The CLI, the API and the benchmark read paths, seeds and limits from here, so they agree on where the
targeting model and the experiment store live. Every value has an offline-safe default; ``CAUSA_`` prefixes
the environment overrides, except the API key, which uses the conventional ``APP_API_KEY``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration with ``CAUSA_`` environment overrides."""

    model_config = SettingsConfigDict(env_prefix="CAUSA_", extra="ignore")

    data_dir: Path = Field(default=Path("data"), description="Experiment store and generated CSVs")
    artifacts_dir: Path = Field(default=Path("artifacts"), description="Trained targeting model")
    benchmark_path: Path = Field(default=Path("benchmarks/results.json"), description="Benchmark results")
    seed: int = Field(default=42, ge=0, description="Seed for generators and models")
    train_rows: int = Field(
        default=50_000, ge=1_000, description="Campaign history size for the targeting model"
    )
    max_rows: int = Field(default=200_000, ge=100, description="Largest experiment upload accepted")
    max_ate_rows: int = Field(default=20_000, ge=100, description="Largest observational upload accepted")
    max_covariates: int = Field(default=12, ge=1, description="Most covariates in an observational upload")
    log_level: str = Field(default="INFO", description="Python logging level name")
    api_key: str | None = Field(
        default=None, validation_alias="APP_API_KEY", description="API key header value"
    )

    @property
    def model_path(self) -> Path:
        """Location of the serialised targeting model."""
        return self.artifacts_dir / "targeting_model.joblib"

    @property
    def store_path(self) -> Path:
        """Location of the sqlite experiment store."""
        return self.data_dir / "experiments.sqlite"


def get_settings() -> Settings:
    """Build settings from the current environment (not cached: tests change the environment)."""
    return Settings()
