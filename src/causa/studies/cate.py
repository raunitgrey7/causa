"""Study of the CATE learners: PEHE, Qini, AUUC and the value of budgeted targeting against the oracle.

Each seed trains every learner on a confounded campaign history of ``n_train`` customers and scores it on a
separate randomised holdout of ``n_holdout`` customers drawn from the same population:

* PEHE, ``sqrt(mean((tau_hat - tau)^2))``, uses the holdout's true CATE;
* Qini coefficient and AUUC use only the holdout's observed revenue and assignment, the way a team would
  evaluate a model in production; the Qini coefficient is also reported relative to ranking by the true
  CATE on the same holdout;
* the true gain area is the noise-free counterpart of AUUC (cumulative true CATE in score order);
* policy value is the true incremental revenue per customer of coupons sent to the top ``budget`` share by
  predicted uplift (only where predicted uplift exceeds the contact cost), relative to the oracle policy
  that ranks by the true CATE. The IPW estimate of the same value from the randomised holdout shows how
  well a team could measure it without ground truth.

Baselines: the oracle, a constant prediction equal to the DR-learner's ATE (what an ATE-only analysis can
offer), and the historical targeting rule (rank by the marketers' propensity), which is how the coupon was
allocated before.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from causa.data.campaign import CampaignConfig, CampaignData, simulate_campaign, true_propensity
from causa.features import feature_matrix
from causa.learners import LEARNERS, DRLearner, make_learner
from causa.uplift import (
    auuc,
    budget_policy,
    ipw_policy_value,
    policy_value,
    qini_coefficient,
    true_gain_curve,
)

BUDGETS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
METRICS: tuple[str, ...] = (
    "pehe",
    "pehe_rel",
    "ate_error",
    "qini",
    "qini_rel",
    "auuc",
    "gain_area_rel",
    "policy_value",
    "policy_value_rel",
    "policy_value_ipw",
    "treated_share",
    "fit_seconds",
)


@dataclass(frozen=True)
class CATEStudyConfig:
    """Sizes and seeds of the learner study."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_train: int = 50_000
    n_holdout: int = 40_000
    budget: float = 0.2
    cost: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class _Holdout:
    data: CampaignData
    tau: np.ndarray
    oracle_value: float
    oracle_qini: float
    oracle_gain: float


def _score(pred: np.ndarray, h: _Holdout, cfg: CATEStudyConfig, is_cate: bool = True) -> dict[str, float]:
    """Ranking and policy metrics for any score; effect-scale ones only when ``pred`` estimates the CATE."""
    y, t, tau = h.data.revenue, h.data.treatment, h.tau
    mask = budget_policy(pred, cfg.budget, cfg.cost)
    value = policy_value(mask, tau, cfg.cost)
    qini = qini_coefficient(pred, y, t)
    pehe = float(np.sqrt(np.mean((pred - tau) ** 2)))
    effect_scale = (
        {"pehe": pehe, "pehe_rel": pehe / float(tau.std()), "ate_error": float(pred.mean() - tau.mean())}
        if is_cate
        else {}
    )
    return {
        **effect_scale,
        "qini": qini,
        "qini_rel": qini / h.oracle_qini,
        "auuc": auuc(pred, y, t),
        "gain_area_rel": true_gain_curve(pred, tau).area() / h.oracle_gain,
        "policy_value": value,
        "policy_value_rel": value / h.oracle_value,
        "policy_value_ipw": ipw_policy_value(mask, y, t, 0.5, cfg.cost),
        "treated_share": float(mask.mean()),
    }


def _value_by_budget(pred: np.ndarray, tau: np.ndarray, cost: float) -> list[float]:
    return [policy_value(budget_policy(pred, b, cost), tau, cost) for b in BUDGETS]


@dataclass(frozen=True)
class _SeedResult:
    rows: dict[str, dict[str, float]]
    curves: dict[str, list[float]]
    oracle_value: float
    sd_tau: float


def _holdout(cfg: CATEStudyConfig, seed: int) -> _Holdout:
    hold = simulate_campaign(CampaignConfig(n=cfg.n_holdout, seed=seed, randomized=True))
    tau = hold.cate
    return _Holdout(
        data=hold,
        tau=tau,
        oracle_value=policy_value(budget_policy(tau, cfg.budget, cfg.cost), tau, cfg.cost),
        oracle_qini=qini_coefficient(tau, hold.revenue, hold.treatment),
        oracle_gain=true_gain_curve(tau, tau).area(),
    )


def _one_seed(cfg: CATEStudyConfig, s: int) -> _SeedResult:
    train_seed, hold_seed = (int(v) for v in np.random.SeedSequence([cfg.seed, 7, s]).generate_state(2))
    train = simulate_campaign(CampaignConfig(n=cfg.n_train, seed=train_seed))
    h = _holdout(cfg, hold_seed)
    X, Xh = feature_matrix(train.customers), feature_matrix(h.data.customers)
    rows: dict[str, dict[str, float]] = {}
    curves: dict[str, list[float]] = {"oracle": _value_by_budget(h.tau, h.tau, cfg.cost)}
    ate_hat = 0.0
    with threadpool_limits(1):
        for name in LEARNERS:
            started = time.perf_counter()
            learner = make_learner(name, seed=train_seed % 10_000).fit(X, train.treatment, train.revenue)
            pred = learner.predict(Xh)
            rows[name] = {**_score(pred, h, cfg), "fit_seconds": time.perf_counter() - started}
            curves[name] = _value_by_budget(pred, h.tau, cfg.cost)
            if isinstance(learner, DRLearner) and learner.pseudo_outcomes_ is not None:
                ate_hat = float(learner.pseudo_outcomes_.mean())
    rows["constant_ate"] = _score(np.full(len(h.tau), ate_hat), h, cfg)
    rows["historical_rule"] = _score(true_propensity(h.data.customers), h, cfg, is_cate=False)
    rows["oracle"] = _score(h.tau, h, cfg)
    return _SeedResult(rows=rows, curves=curves, oracle_value=h.oracle_value, sd_tau=float(h.tau.std()))


def _aggregate(per_seed: list[_SeedResult]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in per_seed[0].rows:
        out[name] = {}
        for metric in METRICS:
            if metric not in per_seed[0].rows[name]:
                continue
            arr = np.asarray([p.rows[name][metric] for p in per_seed], dtype=float)
            out[name][metric] = float(arr.mean())
            out[name][f"{metric}_sd"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return out


def run_cate_study(cfg: CATEStudyConfig, workers: int = 2) -> dict[str, object]:
    """Train and score every learner on every seed; report means and across-seed standard deviations."""
    per_seed: list[_SeedResult] = Parallel(n_jobs=workers, backend="loky")(
        delayed(_one_seed)(cfg, s) for s in cfg.seeds
    )
    curves = {
        name: np.mean([p.curves[name] for p in per_seed], axis=0).tolist() for name in per_seed[0].curves
    }
    return {
        "n_train": cfg.n_train,
        "n_holdout": cfg.n_holdout,
        "seeds": len(cfg.seeds),
        "budget": cfg.budget,
        "cost": cfg.cost,
        "sd_tau": float(np.mean([p.sd_tau for p in per_seed])),
        "oracle_value": float(np.mean([p.oracle_value for p in per_seed])),
        "learners": _aggregate(per_seed),
        "value_by_budget": {"budgets": list(BUDGETS), "values": curves},
    }
