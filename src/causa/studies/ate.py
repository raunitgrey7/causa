"""Monte Carlo study of the observational ATE estimators against the generator's ground truth.

Each replication draws a fresh confounded campaign of ``n`` customers, runs every estimator in
:data:`causa.ate.METHODS` with ``n_boot`` refit bootstrap resamples, and records the estimate, the interval
and the overlap diagnostics. The estimand is the population ATE, ``E[tau(X)]``, computed by Monte Carlo
over four million fresh customers; its own standard error is reported and is two orders of magnitude below
any estimator's.

Replications are independent and seeded from ``SeedSequence([seed, scenario, r])``, so results do not
depend on how many worker processes run them or in which order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from causa.ate import METHODS, ATEConfig, estimate_ate
from causa.data.campaign import CampaignConfig, population_ate, simulate_campaign
from causa.features import feature_matrix


@dataclass(frozen=True)
class ATEStudyConfig:
    """One overlap scenario of the study."""

    scenario: str
    scenario_id: int
    targeting_strength: float
    replications: int = 500
    n: int = 4000
    n_boot: int = 200
    seed: int = 42


def _replication(cfg: ATEStudyConfig, r: int) -> dict[str, list[float]]:
    data_seed, est_seed = np.random.SeedSequence([cfg.seed, cfg.scenario_id, r]).generate_state(2)
    data = simulate_campaign(
        CampaignConfig(n=cfg.n, targeting_strength=cfg.targeting_strength, seed=int(data_seed))
    )
    result = estimate_ate(
        feature_matrix(data.customers),
        data.treatment,
        data.revenue,
        ATEConfig(n_boot=cfg.n_boot, seed=int(est_seed)),
    )
    out = {m: [e.estimate, e.ci_low, e.ci_high, e.se] for m, e in result.estimates.items()}
    ov = result.overlap
    if ov is None:
        raise RuntimeError("estimate_ate returned no overlap report")
    out["overlap"] = [
        ov.auc,
        ov.share_outside_band,
        ov.ess_treated / ov.n_treated,
        ov.ess_control / ov.n_control,
        ov.max_abs_smd_before,
        ov.max_abs_smd_after,
    ]
    return out


def _chunk(cfg: ATEStudyConfig, reps: list[int]) -> list[dict[str, list[float]]]:
    with threadpool_limits(1):
        return [_replication(cfg, r) for r in reps]


def _summarise(values: np.ndarray, truth: float) -> dict[str, float]:
    est, lo, hi, se = values.T
    covered = (lo <= truth) & (truth <= hi)
    reps = len(est)
    sd = float(est.std(ddof=1))
    coverage = float(covered.mean())
    return {
        "bias": float(est.mean() - truth),
        "bias_mcse": sd / np.sqrt(reps),
        "sd": sd,
        "rmse": float(np.sqrt(np.mean((est - truth) ** 2))),
        "coverage": coverage,
        "coverage_mcse": float(np.sqrt(coverage * (1.0 - coverage) / reps)),
        "mean_width": float((hi - lo).mean()),
        "mean_se": float(se.mean()),
        "se_to_sd": float(se.mean() / sd),
    }


def run_ate_study(cfg: ATEStudyConfig, truth: float, workers: int = 2) -> dict[str, object]:
    """Run all replications of one scenario and summarise every estimator against ``truth``."""
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    chunks = [list(c) for c in np.array_split(np.arange(cfg.replications), max(workers * 4, 1)) if len(c)]
    parts = Parallel(n_jobs=workers, backend="loky")(delayed(_chunk)(cfg, c) for c in chunks)
    rows = [row for part in parts for row in part]
    estimators = {m: _summarise(np.array([row[m] for row in rows]), truth) for m in METHODS}
    overlap = np.array([row["overlap"] for row in rows]).mean(axis=0)
    keys = (
        "auc",
        "share_outside_band",
        "ess_frac_treated",
        "ess_frac_control",
        "max_smd_before",
        "max_smd_after",
    )
    return {
        "scenario": cfg.scenario,
        "targeting_strength": cfg.targeting_strength,
        "n": cfg.n,
        "replications": cfg.replications,
        "n_boot": cfg.n_boot,
        "true_ate": truth,
        "estimators": estimators,
        "overlap_mean": dict(zip(keys, overlap.tolist(), strict=True)),
    }


def true_ate(seed: int, draws: int = 4_000_000) -> dict[str, float]:
    """Population ATE (it does not depend on the targeting strength, which only moves assignment)."""
    return population_ate(CampaignConfig(seed=seed), n_draws=draws)
