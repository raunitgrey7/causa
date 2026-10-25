"""Shared fixtures: small seeded datasets, a small targeting model and an API client on a temp directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from threadpoolctl import threadpool_limits

from causa.api.app import create_app
from causa.config import Settings
from causa.data.campaign import CampaignConfig, CampaignData, simulate_campaign
from causa.data.experiments import ExperimentConfig, simulate_experiment
from causa.targeting import TargetingModel, train_targeting_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _two_threads() -> None:
    """Other builds share this machine: keep OpenMP and BLAS pools at two threads for the whole run."""
    with threadpool_limits(2):
        yield


@pytest.fixture(scope="session")
def campaign() -> CampaignData:
    """A confounded campaign of 4000 customers."""
    return simulate_campaign(CampaignConfig(n=4000, seed=11))


@pytest.fixture(scope="session")
def experiment_frame() -> pd.DataFrame:
    """A 20000-user experiment with an 8 percent lift."""
    return simulate_experiment(ExperimentConfig(n_users=20_000, lift=0.08, seed=5))


@pytest.fixture(scope="session")
def small_model() -> TargetingModel:
    """A targeting model trained on a few thousand customers (seconds, not minutes)."""
    return train_targeting_model(n_train=4000, n_holdout=3000, seed=3)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that keep every file the app writes inside ``tmp_path``."""
    return Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        benchmark_path=tmp_path / "results.json",
        max_rows=60_000,
    )


@pytest.fixture
def client(settings: Settings, small_model: TargetingModel) -> TestClient:
    """An API client with the small model injected."""
    with TestClient(create_app(settings, model=small_model)) as c:
        yield c


def linear_world(n: int, seed: int, randomized: bool = True) -> dict[str, np.ndarray]:
    """A closed-form world: y = 1 + x1 - x2 + t * (1 + 2 x1) + noise, optionally confounded through x1."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    e = np.full(n, 0.5) if randomized else 1.0 / (1.0 + np.exp(-0.8 * X[:, 0]))
    t = (rng.random(n) < e).astype(int)
    tau = 1.0 + 2.0 * X[:, 0]
    y = 1.0 + X[:, 0] - X[:, 1] + t * tau + 0.5 * rng.standard_normal(n)
    return {"X": X, "t": t, "y": y, "tau": tau, "e": e}
