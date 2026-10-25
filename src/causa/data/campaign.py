"""Seeded generator of customers, a confounded historical campaign and known treatment effects.

The scenario is an online retailer that sent a discount coupon to part of its customer base last quarter.
Marketers did not randomise: they targeted customers who were already engaged (many recent sessions, high
email open rates) and lapsed high-value customers they wanted to win back. Engagement also drives buying,
so a naive comparison of treated and untreated customers mixes the coupon's effect with the targeting
rule. That is the confounding the estimators in this package have to remove.

Each customer has closed-form potential outcome means, so the conditional average treatment effect (CATE)
is known exactly:

    p_t(x)   probability of buying in the 30 days after the campaign; logistic in the covariates, and the
             coupon shifts the logit by delta(x)
    v_t(x)   expected basket value; the coupon cuts the price by ``discount`` and lifts basket size by
             ``basket_lift``
    mu_t(x)  p_t(x) * v_t(x), the expected 30-day revenue under treatment t
    tau(x)   mu_1(x) - mu_0(x), the CATE in currency units (INR) per customer

delta(x) encodes the four uplift segments marketers talk about: price-sensitive persuadables (large positive
effect), sure things who would buy anyway and now pay less (negative revenue effect), lost causes (tiny
effect) and sleeping dogs, frequent buyers who rarely open email and are put off by one more promotion
(negative effect). Realised revenue is a purchase indicator times a gamma basket value, which gives the
zero-inflated, right-skewed outcome that real revenue data has.

The historical targeting logit is linear in the standardised signals, their squares and one product term,
so a logistic model on :class:`causa.features.Basis` features is correctly specified for the propensity
while the outcome surface (a logistic probability times a value) is not linear in any basis. That split is
deliberate: it lets the benchmark show double robustness instead of assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

COVARIATES: tuple[str, ...] = (
    "tenure_months",
    "orders_12m",
    "recency_days",
    "sessions_30d",
    "email_open_rate",
    "avg_order_value",
    "discount_affinity",
    "app_user",
    "loyalty_member",
    "city_tier",
)
TIER_PROBS: tuple[float, float, float] = (0.45, 0.35, 0.20)
TIER_VALUE: tuple[float, float, float] = (1.15, 1.0, 0.85)


@dataclass(frozen=True)
class CampaignConfig:
    """Knobs of the campaign world; defaults are the benchmark's main scenario."""

    n: int = 20_000
    targeting_strength: float = 1.0
    discount: float = 0.15
    basket_lift: float = 0.10
    basket_shape: float = 2.5
    randomized: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject settings that would make the world degenerate."""
        if self.n < 1:
            raise ValueError("n must be positive")
        if self.targeting_strength < 0:
            raise ValueError("targeting_strength must be non-negative")
        if not 0 <= self.discount < 1 or self.basket_lift <= -1 or self.basket_shape <= 0:
            raise ValueError("discount must be in [0, 1), basket_lift > -1 and basket_shape > 0")


@dataclass(frozen=True)
class PotentialOutcomes:
    """Closed-form conditional means of both arms for every customer."""

    p0: np.ndarray
    p1: np.ndarray
    v0: np.ndarray
    v1: np.ndarray

    @property
    def mu0(self) -> np.ndarray:
        """Expected revenue without the coupon."""
        return self.p0 * self.v0

    @property
    def mu1(self) -> np.ndarray:
        """Expected revenue with the coupon."""
        return self.p1 * self.v1

    @property
    def cate(self) -> np.ndarray:
        """True conditional average treatment effect on revenue."""
        return self.mu1 - self.mu0


@dataclass(frozen=True)
class CampaignData:
    """One simulated campaign: what an analyst would see plus the ground truth they never get."""

    customers: pd.DataFrame
    treatment: np.ndarray
    revenue: np.ndarray
    converted: np.ndarray
    propensity: np.ndarray
    truth: PotentialOutcomes

    @property
    def cate(self) -> np.ndarray:
        """Ground-truth CATE per customer."""
        return self.truth.cate

    def __len__(self) -> int:
        """Number of customers."""
        return len(self.treatment)


def simulate_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw CRM covariates from three latent factors: engagement, customer value and price sensitivity.

    Shared latent factors give the covariates the correlation structure seen in practice (frequent buyers
    also browse and open email more) without hand-writing a covariance matrix.
    """
    engagement, value, price = rng.standard_normal((3, n))
    noise = rng.standard_normal((3, n))
    tenure = np.minimum(1.0 + rng.gamma(1.6, 18.0, n), 120.0)
    orders = rng.poisson(np.exp(0.8 + 0.6 * engagement + 0.15 * value))
    recency = np.clip(np.round(1.0 + rng.exponential(300.0 / (1.0 + orders))), 1, 540)
    tier = rng.choice(3, size=n, p=TIER_PROBS)
    aov = np.exp(7.0 + 0.4 * value + 0.25 * noise[1]) * np.asarray(TIER_VALUE)[tier]
    return pd.DataFrame(
        {
            "tenure_months": np.round(tenure, 1),
            "orders_12m": orders.astype(np.int64),
            "recency_days": recency.astype(np.int64),
            "sessions_30d": rng.poisson(np.exp(1.3 + 0.75 * engagement)).astype(np.int64),
            "email_open_rate": np.round(expit(-0.9 + 0.7 * engagement + 0.6 * noise[0]), 4),
            "avg_order_value": np.round(aov, 2),
            "discount_affinity": np.round(expit(-1.0 + 1.1 * price + 0.4 * noise[2]), 4),
            "app_user": rng.binomial(1, expit(-0.3 + 0.7 * engagement)).astype(np.int64),
            "loyalty_member": rng.binomial(1, expit(-1.6 + 0.025 * tenure + 0.5 * value)).astype(np.int64),
            "city_tier": (tier + 1).astype(np.int64),
        }
    )


def signals(customers: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fixed, roughly standardised transforms of the raw covariates that the response surfaces use."""
    c = customers
    return {
        "orders": (np.log1p(c["orders_12m"].to_numpy(float)) - 1.2) / 0.6,
        "recency": (np.log(c["recency_days"].to_numpy(float)) - 4.0) / 1.0,
        "sessions": (np.log1p(c["sessions_30d"].to_numpy(float)) - 1.6) / 0.7,
        "open": (c["email_open_rate"].to_numpy(float) - 0.3) / 0.17,
        "aov": (np.log(c["avg_order_value"].to_numpy(float)) - 7.0) / 0.5,
        "discount": (c["discount_affinity"].to_numpy(float) - 0.3) / 0.2,
        "app": c["app_user"].to_numpy(float),
        "loyalty": c["loyalty_member"].to_numpy(float),
        "tier1": (c["city_tier"].to_numpy() == 1).astype(float),
    }


def targeting_logit(customers: pd.DataFrame) -> np.ndarray:
    """Logit of the marketers' historical targeting rule at unit strength."""
    s = signals(customers)
    return (
        -0.5
        + 0.7 * s["sessions"]
        + 0.6 * s["open"]
        + 0.35 * s["recency"]
        + 0.3 * s["aov"]
        + 0.25 * s["recency"] * s["aov"]
        - 0.3 * s["loyalty"]
    )


def true_propensity(customers: pd.DataFrame, strength: float = 1.0) -> np.ndarray:
    """Probability that the historical campaign targeted each customer."""
    return expit(strength * targeting_logit(customers))


def coupon_logit_shift(customers: pd.DataFrame) -> np.ndarray:
    """delta(x): how much the coupon moves the purchase logit, by uplift segment."""
    s = signals(customers)
    persuadable = 0.45 * s["discount"]
    win_back = 0.3 * expit(2.0 * (s["recency"] - 0.5))
    sleeping_dog = 0.7 * expit(3.0 * (-s["open"] - 0.8)) * expit(3.0 * (s["orders"] - 0.3))
    return 0.15 + persuadable + win_back - sleeping_dog


def potential_outcomes(customers: pd.DataFrame, config: CampaignConfig) -> PotentialOutcomes:
    """Closed-form purchase probabilities and basket values under both arms."""
    s = signals(customers)
    base_logit = (
        -1.2
        + 0.6 * s["orders"]
        - 0.55 * s["recency"]
        + 0.45 * np.tanh(s["sessions"])
        + 0.3 * s["open"]
        + 0.2 * s["app"]
        + 0.25 * s["loyalty"]
        + 0.1 * s["tier1"]
    )
    v0 = customers["avg_order_value"].to_numpy(float) * (1.0 + 0.15 * s["loyalty"])
    v1 = v0 * (1.0 - config.discount) * (1.0 + config.basket_lift)
    return PotentialOutcomes(
        p0=expit(base_logit), p1=expit(base_logit + coupon_logit_shift(customers)), v0=v0, v1=v1
    )


def simulate_campaign(config: CampaignConfig) -> CampaignData:
    """Draw customers, the historical assignment and realised revenue from one seed.

    Both potential outcomes share the purchase uniform and the basket draw (a monotone coupling), so the
    individual effect of a customer is as close to the CATE as zero-inflated revenue allows.
    """
    rng = np.random.default_rng(config.seed)
    customers = simulate_customers(config.n, rng)
    truth = potential_outcomes(customers, config)
    if config.randomized:
        propensity = np.full(config.n, 0.5)
    else:
        propensity = true_propensity(customers, config.targeting_strength)
    treatment = (rng.random(config.n) < propensity).astype(np.int64)
    buys = rng.random(config.n)
    basket = rng.gamma(config.basket_shape, 1.0 / config.basket_shape, config.n)
    p = np.where(treatment == 1, truth.p1, truth.p0)
    v = np.where(treatment == 1, truth.v1, truth.v0)
    converted = (buys < p).astype(np.int64)
    return CampaignData(
        customers=customers,
        treatment=treatment,
        revenue=np.round(converted * v * basket, 2),
        converted=converted,
        propensity=propensity,
        truth=truth,
    )


def population_ate(
    config: CampaignConfig, n_draws: int = 2_000_000, chunk: int = 250_000
) -> dict[str, float]:
    """Population ATE E[tau(X)] by Monte Carlo over fresh customers, with its Monte Carlo standard error,
    the spread of the CATE and the share of customers the coupon hurts.

    The covariate draw uses its own seed stream so the estimand does not depend on which campaign seed a
    benchmark replication uses.
    """
    rng = np.random.default_rng([config.seed, 20_251_025])
    total, total_sq, negative, done = 0.0, 0.0, 0, 0
    while done < n_draws:
        size = min(chunk, n_draws - done)
        tau = potential_outcomes(simulate_customers(size, rng), config).cate
        total += float(tau.sum())
        total_sq += float(np.square(tau).sum())
        negative += int((tau < 0).sum())
        done += size
    mean = total / done
    var = max(total_sq / done - mean**2, 0.0)
    return {
        "ate": mean,
        "mc_se": float(np.sqrt(var / done)),
        "sd_tau": float(np.sqrt(var)),
        "share_negative": negative / done,
        "draws": done,
    }
