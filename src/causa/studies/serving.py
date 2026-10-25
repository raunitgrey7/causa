"""In-process latency of the three main API calls, measured through the full FastAPI stack.

Requests go through Starlette's test client, so the timings include validation, JSON encoding, the
middleware and the analysis, but not a network hop. They answer the usability question (is a decision
report interactive?) rather than a capacity one; no concurrency is measured.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from threadpoolctl import threadpool_limits

from causa.api.app import create_app
from causa.config import Settings
from causa.data.campaign import CampaignConfig, simulate_campaign
from causa.data.experiments import ExperimentConfig, simulate_experiment
from causa.features import FEATURE_NAMES, feature_matrix
from causa.targeting import train_targeting_model


@dataclass(frozen=True)
class ServingStudyConfig:
    """Payload sizes and repetitions."""

    experiment_rows: int = 40_000
    experiment_requests: int = 15
    customers: int = 1_000
    targeting_requests: int = 40
    ate_rows: int = 4_000
    ate_boot: int = 100
    ate_requests: int = 8
    train_rows: int = 20_000
    seed: int = 42


def _timed(send: Callable[[], int], repeats: int) -> dict[str, float]:
    times, statuses = [], set()
    for _ in range(repeats):
        started = time.perf_counter()
        statuses.add(send())
        times.append((time.perf_counter() - started) * 1000.0)
    if statuses not in ({200}, {201}):
        raise RuntimeError(f"unexpected status codes {statuses}")
    arr = np.asarray(times)
    return {
        "requests": repeats,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def _ate_payload(cfg: ServingStudyConfig) -> dict[str, object]:
    data = simulate_campaign(CampaignConfig(n=cfg.ate_rows, seed=cfg.seed))
    X = feature_matrix(data.customers)
    columns = {name: X[:, j].tolist() for j, name in enumerate(FEATURE_NAMES)}
    columns |= {"coupon": data.treatment.tolist(), "revenue": data.revenue.tolist()}
    return {
        "covariates": list(FEATURE_NAMES),
        "treatment": "coupon",
        "outcome": "revenue",
        "data": columns,
        "n_boot": cfg.ate_boot,
    }


def serving_study(cfg: ServingStudyConfig) -> dict[str, object]:
    """p50 and p95 latency of a decision report, a targeting call and an observational ATE."""
    experiment = simulate_experiment(ExperimentConfig(n_users=cfg.experiment_rows, seed=cfg.seed))
    upload = {"spec": {"name": "latency"}, "data": experiment.to_dict("list")}
    ate = _ate_payload(cfg)
    with tempfile.TemporaryDirectory() as tmp, threadpool_limits(2):
        model = train_targeting_model(n_train=cfg.train_rows, n_holdout=cfg.train_rows // 2, seed=cfg.seed)
        root = Path(tmp)
        settings = Settings(data_dir=root / "data", artifacts_dir=root / "artifacts", log_level="WARNING")
        with TestClient(create_app(settings, model=model)) as client:
            customers = client.get(f"/v1/customers/sample?n={cfg.customers}&seed=1").json()
            target = {"customers": customers, "budget": 0.2}
            return {
                "client": "in-process starlette TestClient, sequential requests, 2 threads",
                "experiment_report": {
                    "rows": cfg.experiment_rows,
                    **_timed(
                        lambda: client.post("/v1/experiments", json=upload).status_code,
                        cfg.experiment_requests,
                    ),
                },
                "targeting": {
                    "customers": cfg.customers,
                    **_timed(
                        lambda: client.post("/v1/targeting", json=target).status_code, cfg.targeting_requests
                    ),
                },
                "observational_ate": {
                    "rows": cfg.ate_rows,
                    "n_boot": cfg.ate_boot,
                    **_timed(
                        lambda: client.post("/v1/observational/ate", json=ate).status_code, cfg.ate_requests
                    ),
                },
            }
