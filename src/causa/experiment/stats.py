"""Difference in means and CUPED from per-arm sufficient statistics.

Both estimators need only six running sums per arm: ``n``, sums of ``y``, ``y^2``, ``x``, ``x^2`` and
``x*y``. Working from sums means one code path serves a single uploaded experiment, every daily look of a
sequential analysis (cumulative sums) and the Monte Carlo studies (sums over a ``(reps, looks)`` grid).

CUPED (Deng, Xu, Kohavi and Walker 2013) subtracts ``theta * (x - mean(x))`` from the outcome, where ``x``
is a pre-experiment covariate and ``theta = cov(x, y) / var(x)`` is pooled over both arms. Randomisation
makes ``x`` independent of the assignment, so the adjustment leaves the effect unbiased and removes the
share ``rho^2`` of the variance that ``x`` explains. Using the pooled ``theta`` (rather than one per arm)
is the version that is equivalent to ANCOVA with a common slope.

Standard errors are Welch's (unequal variances), and tests and intervals use the normal approximation,
which is what large online experiments rely on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

VAR_FLOOR = 1e-12


@dataclass(frozen=True)
class ArmSums:
    """Running sums for one arm; every field broadcasts over leading axes such as (reps, looks)."""

    n: np.ndarray
    y: np.ndarray
    yy: np.ndarray
    x: np.ndarray
    xx: np.ndarray
    xy: np.ndarray

    @classmethod
    def from_masked(cls, y: np.ndarray, x: np.ndarray | None, mask: np.ndarray, axis: int = -1) -> ArmSums:
        """Sums over ``axis`` of the rows where ``mask`` is True (no covariate means ``x = 0``)."""
        m = np.asarray(mask, dtype=float)
        y = np.asarray(y, dtype=float) * m
        x = np.zeros_like(y) if x is None else np.asarray(x, dtype=float) * m
        return cls(
            n=m.sum(axis=axis),
            y=y.sum(axis=axis),
            yy=(y * y).sum(axis=axis),
            x=x.sum(axis=axis),
            xx=(x * x).sum(axis=axis),
            xy=(x * y).sum(axis=axis),
        )

    def cumulative(self, axis: int = -1) -> ArmSums:
        """Running totals along ``axis``: per-look increments become sums up to each look."""
        fields = (self.n, self.y, self.yy, self.x, self.xx, self.xy)
        return ArmSums(*(np.cumsum(f, axis=axis) for f in fields))

    def mean_y(self) -> np.ndarray:
        """Arm mean of the outcome."""
        return self.y / np.maximum(self.n, 1.0)

    def var_y(self) -> np.ndarray:
        """Unbiased variance of the outcome."""
        return _variance(self.n, self.y, self.yy)


def _variance(n: np.ndarray, s: np.ndarray, ss: np.ndarray) -> np.ndarray:
    return np.maximum(ss - s * s / np.maximum(n, 1.0), 0.0) / np.maximum(n - 1.0, 1.0)


def _covariance(n: np.ndarray, sx: np.ndarray, sy: np.ndarray, sxy: np.ndarray) -> np.ndarray:
    return (sxy - sx * sy / np.maximum(n, 1.0)) / np.maximum(n - 1.0, 1.0)


@dataclass(frozen=True)
class Contrast:
    """Treatment minus control on any array shape, with the variance of each arm mean."""

    control_mean: np.ndarray
    treatment_mean: np.ndarray
    var_control_mean: np.ndarray
    var_treatment_mean: np.ndarray

    @property
    def effect(self) -> np.ndarray:
        """Treatment mean minus control mean."""
        return self.treatment_mean - self.control_mean

    @property
    def variance(self) -> np.ndarray:
        """Variance of the effect (the arms are independent under randomisation)."""
        return self.var_control_mean + self.var_treatment_mean

    @property
    def se(self) -> np.ndarray:
        """Standard error of the effect."""
        return np.sqrt(self.variance)

    def p_value(self) -> np.ndarray:
        """Two-sided normal p-value of ``effect = 0``; a zero-variance contrast gives 0 or 1."""
        var = self.variance
        z = np.abs(self.effect) / np.sqrt(np.maximum(var, VAR_FLOOR))
        return np.where(var > VAR_FLOOR, 2.0 * norm.sf(z), np.where(self.effect == 0, 1.0, 0.0))

    def interval(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Two-sided ``1 - alpha`` normal interval."""
        z = norm.ppf(1.0 - alpha / 2.0)
        return self.effect - z * self.se, self.effect + z * self.se

    def relative(self, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Relative effect ``treatment / control - 1`` with a delta-method interval (NaN if control is 0).

        ``var(r) = var(m_t) / m_c^2 + m_t^2 var(m_c) / m_c^4`` for independent arm means.
        """
        c, t = self.control_mean, self.treatment_mean
        safe = np.where(np.abs(c) > VAR_FLOOR, c, np.nan)
        rel = t / safe - 1.0
        var = self.var_treatment_mean / safe**2 + t**2 * self.var_control_mean / safe**4
        half = norm.ppf(1.0 - alpha / 2.0) * np.sqrt(var)
        return rel, rel - half, rel + half


def difference_in_means(control: ArmSums, treatment: ArmSums) -> Contrast:
    """Difference of arm means with the Welch variance."""
    return Contrast(
        control_mean=control.mean_y(),
        treatment_mean=treatment.mean_y(),
        var_control_mean=control.var_y() / np.maximum(control.n, 1.0),
        var_treatment_mean=treatment.var_y() / np.maximum(treatment.n, 1.0),
    )


def cuped_theta(control: ArmSums, treatment: ArmSums) -> np.ndarray:
    """Pooled regression slope of ``y`` on ``x`` over both arms (zero when ``x`` has no variance)."""
    n = control.n + treatment.n
    sx, sy = control.x + treatment.x, control.y + treatment.y
    var_x = _variance(n, sx, control.xx + treatment.xx)
    cov = _covariance(n, sx, sy, control.xy + treatment.xy)
    return np.where(var_x > VAR_FLOOR, cov / np.maximum(var_x, VAR_FLOOR), 0.0)


def _adjusted_variance(arm: ArmSums, theta: np.ndarray) -> np.ndarray:
    var_y = _variance(arm.n, arm.y, arm.yy)
    var_x = _variance(arm.n, arm.x, arm.xx)
    cov = _covariance(arm.n, arm.x, arm.y, arm.xy)
    return np.maximum(var_y - 2.0 * theta * cov + theta**2 * var_x, 0.0)


def cuped(control: ArmSums, treatment: ArmSums) -> tuple[Contrast, np.ndarray]:
    """CUPED-adjusted contrast and the pooled ``theta`` it used."""
    theta = cuped_theta(control, treatment)
    grand_x = (control.x + treatment.x) / np.maximum(control.n + treatment.n, 1.0)
    shift_c = theta * (control.x / np.maximum(control.n, 1.0) - grand_x)
    shift_t = theta * (treatment.x / np.maximum(treatment.n, 1.0) - grand_x)
    contrast = Contrast(
        control_mean=control.mean_y() - shift_c,
        treatment_mean=treatment.mean_y() - shift_t,
        var_control_mean=_adjusted_variance(control, theta) / np.maximum(control.n, 1.0),
        var_treatment_mean=_adjusted_variance(treatment, theta) / np.maximum(treatment.n, 1.0),
    )
    return contrast, theta
