"""Seeded generator of randomised A/B experiments with pre-period metrics, guardrails and SRM faults.

The scenario is a checkout redesign tested on a 50/50 split for ``days`` days. Every user has a latent
spending rate drawn from a gamma distribution (heterogeneous customers) and a lognormal typical basket. The
same rate drives two windows:

* ``pre_revenue``: revenue in the 28 days before the experiment, which CUPED uses as its covariate;
* ``revenue``: revenue in the experiment window, the primary metric. Treatment multiplies the order rate by
  ``1 + lift``, so the true average effect is ``lift`` times the mean control revenue.

Order counts are Poisson given the rate and order values are gamma, so revenue is zero-inflated and right
skewed, and its correlation with the pre-period comes from the shared rate, as it does in real stores.
Two guardrails come with each user: ``support_contact`` (did the user contact support, a binary harm
metric) and ``page_load_ms`` (mean page latency). ``srm_drop`` simulates a classic logging fault: that
share of non-converting treatment users never reaches the analytics table, which both unbalances the arms
(a sample ratio mismatch) and inflates the treatment's apparent effect.

:func:`expected_effects` gives the closed-form true effects so tests and the benchmark can score estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRE_DAYS = 28.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Knobs of one simulated A/B test; defaults give a realistic small revenue lift."""

    n_users: int = 20_000
    days: int = 14
    split: float = 0.5
    lift: float = 0.03
    base_orders_per_month: float = 1.5
    rate_shape: float = 0.6
    basket_mean: float = 1400.0
    basket_sigma: float = 0.45
    order_shape: float = 4.0
    window_days: float = 14.0
    support_rate: float = 0.04
    support_harm: float = 0.0
    latency_shift_ms: float = 0.0
    srm_drop: float = 0.0
    seed: int = 7

    def __post_init__(self) -> None:
        """Reject impossible settings early with a specific message."""
        if self.n_users < 2 or self.days < 1:
            raise ValueError("need at least 2 users and 1 day")
        if not 0 < self.split < 1:
            raise ValueError("split must be in (0, 1)")
        if self.lift <= -1 or not 0 <= self.srm_drop < 1:
            raise ValueError("lift must exceed -1 and srm_drop must be in [0, 1)")
        if not 0 <= self.support_rate + max(self.support_harm, 0.0) <= 1:
            raise ValueError("support rate plus harm must be a probability")


@dataclass(frozen=True)
class ExperimentArrays:
    """``reps`` independent experiments as ``(reps, n_users)`` arrays; ``logged`` marks rows that reach
    the analytics table (all True unless ``srm_drop`` is set)."""

    treated: np.ndarray
    day: np.ndarray
    pre_revenue: np.ndarray
    revenue: np.ndarray
    converted: np.ndarray
    support_contact: np.ndarray
    page_load_ms: np.ndarray
    logged: np.ndarray


def _revenue(orders: np.ndarray, basket: np.ndarray, shape: float, rng: np.random.Generator) -> np.ndarray:
    """Sum of ``orders`` gamma order values with mean ``basket``: a gamma with shape ``orders * shape``."""
    total = np.zeros(orders.shape)
    buying = orders > 0
    total[buying] = rng.gamma(orders[buying] * shape, basket[buying] / shape)
    return total


def simulate_arrays(config: ExperimentConfig, reps: int, rng: np.random.Generator) -> ExperimentArrays:
    """Vectorised draw of ``reps`` experiments; the Monte Carlo studies call this directly."""
    shape = (reps, config.n_users)
    c = config
    rate = rng.gamma(c.rate_shape, c.base_orders_per_month / c.rate_shape, shape)
    basket = c.basket_mean * np.exp(c.basket_sigma * rng.standard_normal(shape) - c.basket_sigma**2 / 2)
    treated = rng.random(shape) < c.split
    pre_orders = rng.poisson(rate * PRE_DAYS / 30.0)
    orders = rng.poisson(rate * (c.window_days / 30.0) * np.where(treated, 1.0 + c.lift, 1.0))
    converted = orders > 0
    logged = ~(treated & ~converted & (rng.random(shape) < c.srm_drop))
    return ExperimentArrays(
        treated=treated,
        day=rng.integers(1, c.days + 1, shape),
        pre_revenue=_revenue(pre_orders, basket, c.order_shape, rng),
        revenue=_revenue(orders, basket, c.order_shape, rng),
        converted=converted,
        support_contact=rng.random(shape) < c.support_rate + c.support_harm * treated,
        page_load_ms=800.0 * np.exp(0.3 * rng.standard_normal(shape)) + c.latency_shift_ms * treated,
        logged=logged,
    )


def simulate_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """One experiment as the analytics table would hold it: one row per logged user, ordered by day."""
    a = simulate_arrays(config, 1, np.random.default_rng(config.seed))
    keep = a.logged[0]
    frame = pd.DataFrame(
        {
            "user_id": np.arange(1, config.n_users + 1)[keep],
            "day": a.day[0][keep],
            "variant": np.where(a.treated[0][keep], "treatment", "control"),
            "pre_revenue": np.round(a.pre_revenue[0][keep], 2),
            "revenue": np.round(a.revenue[0][keep], 2),
            "converted": a.converted[0][keep].astype(np.int64),
            "support_contact": a.support_contact[0][keep].astype(np.int64),
            "page_load_ms": np.round(a.page_load_ms[0][keep], 1),
        }
    )
    return frame.sort_values(["day", "user_id"], kind="stable").reset_index(drop=True)


def expected_effects(config: ExperimentConfig) -> dict[str, float]:
    """Closed-form true effects of the treatment on each metric (before any SRM fault)."""
    month_orders = config.base_orders_per_month * config.window_days / 30.0
    control_revenue = month_orders * config.basket_mean
    # P(orders > 0) averages exp(-rate * w) over the gamma rate: the gamma Laplace transform.
    scale = config.base_orders_per_month / config.rate_shape * config.window_days / 30.0
    p0 = 1.0 - (1.0 + scale) ** -config.rate_shape
    p1 = 1.0 - (1.0 + scale * (1.0 + config.lift)) ** -config.rate_shape
    return {
        "revenue": control_revenue * config.lift,
        "revenue_control_mean": control_revenue,
        "converted": p1 - p0,
        "converted_control": p0,
        "support_contact": config.support_harm,
        "page_load_ms": config.latency_shift_ms,
    }
