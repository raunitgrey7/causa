"""Power and sample size for a two-arm comparison of means.

Power answers the planning question: with this many users per arm and this much noise, how likely is the
test to detect an effect of the size the business cares about (the minimum detectable effect, MDE)? The
report quotes power at the pre-registered MDE, never "observed power" at the estimated effect, which is a
transformation of the p-value and says nothing new. CUPED enters through the standard deviation: an
adjustment that removes the share ``r2`` of the variance multiplies the standard deviation by
``sqrt(1 - r2)``, which is how it shortens experiments.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _check(sd: float, alpha: float) -> None:
    if not sd > 0:
        raise ValueError("standard deviation must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def power_two_sample(
    effect: float, sd: float, n_control: float, n_treatment: float, alpha: float = 0.05
) -> float:
    """Power of the two-sided z-test at level ``alpha`` against a true difference ``effect``."""
    _check(sd, alpha)
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("arm sizes must be positive")
    se = sd * math.sqrt(1.0 / n_control + 1.0 / n_treatment)
    z = abs(effect) / se
    crit = norm.ppf(1.0 - alpha / 2.0)
    return float(norm.cdf(z - crit) + norm.cdf(-z - crit))


def sample_size_per_arm(mde: float, sd: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Users per arm (equal split) for the two-sided z-test to reach ``power`` at effect ``mde``."""
    _check(sd, alpha)
    if mde == 0 or not 0 < power < 1:
        raise ValueError("mde must be non-zero and power in (0, 1)")
    z = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)
    return math.ceil(2.0 * (z * sd / mde) ** 2)


def minimum_detectable_effect(
    sd: float, n_control: float, n_treatment: float, power: float = 0.8, alpha: float = 0.05
) -> float:
    """Smallest effect detected with probability ``power`` (ignores the negligible opposite tail)."""
    _check(sd, alpha)
    if n_control <= 0 or n_treatment <= 0 or not 0 < power < 1:
        raise ValueError("arm sizes must be positive and power in (0, 1)")
    z = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)
    return float(z * sd * math.sqrt(1.0 / n_control + 1.0 / n_treatment))
