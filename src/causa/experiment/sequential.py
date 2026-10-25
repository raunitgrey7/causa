"""Always-valid inference for experiments that are watched every day.

A fixed-horizon test assumes the analyst looks once. Teams look daily and stop when the dashboard turns
green, which inflates the false positive rate well past the nominal 5 percent. Two remedies live here.

Mixture SPRT for a difference in means (Robbins 1970; Johari, Koomen, Pekelis and Walsh 2017). With an
estimate ``d`` of the effect and its variance ``V`` at a look, a normal mixing distribution of standard
deviation ``tau`` on the effect gives the likelihood ratio against ``effect = theta0``

    Lambda = sqrt(V / (V + tau^2)) * exp(tau^2 (d - theta0)^2 / (2 V (V + tau^2)))

which is a nonnegative martingale under the null, so by Ville's inequality it exceeds ``1 / alpha`` at any
look with probability at most ``alpha``. The always-valid p-value is the running minimum of ``1 / Lambda``
and inverting the test gives a confidence sequence, intervals that hold simultaneously at every look:

    d +/- sqrt(V (V + tau^2) / tau^2 * (2 log(1 / alpha) + log((V + tau^2) / V)))

intersected over looks. ``V`` is estimated from the data, so validity is asymptotic, the same footing as
the z-test; the benchmark measures the false positive rate it actually delivers on skewed revenue.
``tau`` should be on the scale of the effects one hopes to detect; the report sets it to the planned
minimum detectable effect.

Sequential SRM test (Lindon and Malek 2020). Arm counts are multinomial with the planned split under the
null; a Dirichlet mixture over the split gives a Bayes factor that is an exact martingale, so its running
maximum can be checked after every batch of users with no inflation at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from causa.experiment.stats import VAR_FLOOR


@dataclass(frozen=True)
class MixtureSPRT:
    """Normal-mixture SPRT for ``effect = theta0`` with mixing standard deviation ``tau``."""

    alpha: float = 0.05
    tau: float = 1.0
    theta0: float = 0.0

    def __post_init__(self) -> None:
        """Reject settings without a valid test."""
        if not 0 < self.alpha < 1 or not self.tau > 0:
            raise ValueError("alpha must be in (0, 1) and tau positive")

    def log_likelihood_ratio(self, effect: np.ndarray, variance: np.ndarray) -> np.ndarray:
        """Log of the mixture likelihood ratio at each look."""
        v = np.maximum(np.asarray(variance, dtype=float), VAR_FLOOR)
        tau2 = self.tau**2
        diff2 = (np.asarray(effect, dtype=float) - self.theta0) ** 2
        return 0.5 * np.log(v / (v + tau2)) + tau2 * diff2 / (2.0 * v * (v + tau2))

    def p_values(self, effect: np.ndarray, variance: np.ndarray) -> np.ndarray:
        """Always-valid p-values: running minimum of ``min(1, 1 / Lambda)`` along the last axis."""
        raw = np.exp(np.minimum(0.0, -self.log_likelihood_ratio(effect, variance)))
        return np.minimum.accumulate(raw, axis=-1)

    def radius(self, variance: np.ndarray) -> np.ndarray:
        """Half-width of the confidence sequence at each look (before intersecting over looks)."""
        v = np.maximum(np.asarray(variance, dtype=float), VAR_FLOOR)
        tau2 = self.tau**2
        return np.sqrt(v * (v + tau2) / tau2 * (2.0 * np.log(1.0 / self.alpha) + np.log((v + tau2) / v)))

    def confidence_sequence(self, effect: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Running intersection of the per-look intervals along the last axis."""
        effect = np.asarray(effect, dtype=float)
        r = self.radius(variance)
        return np.maximum.accumulate(effect - r, axis=-1), np.minimum.accumulate(effect + r, axis=-1)


def sequential_srm_p_values(
    counts: np.ndarray, expected: np.ndarray, concentration: float = 1000.0
) -> np.ndarray:
    """Always-valid SRM p-values from cumulative arm counts ``(..., looks, arms)``.

    The Dirichlet prior is centred on the planned split ``expected`` with total mass ``concentration``.
    At the default of 1000 its standard deviation on a 50/50 split is about 1.6 points of share, so the test
    is quickest on faults of a point or more; smaller faults need many users, and the decision report pairs
    this test with the chi-square check for them.
    """
    counts = np.asarray(counts, dtype=float)
    p0 = np.asarray(expected, dtype=float)
    if counts.shape[-1] != len(p0) or (counts < 0).any():
        raise ValueError("counts must be non-negative with one column per arm")
    if not np.isclose(p0.sum(), 1.0) or (p0 <= 0).any():
        raise ValueError("expected split must be positive and sum to 1")
    prior = concentration * p0
    n = counts.sum(axis=-1)
    log_bf = (
        gammaln(prior.sum())
        - gammaln(prior.sum() + n)
        + (gammaln(prior + counts) - gammaln(prior)).sum(axis=-1)
        - (counts * np.log(p0)).sum(axis=-1)
    )
    return np.minimum.accumulate(np.exp(np.minimum(0.0, -log_bf)), axis=-1)
