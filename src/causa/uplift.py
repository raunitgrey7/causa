"""Uplift evaluation and targeting under a budget.

A CATE model is only useful for targeting if it ranks customers well, so the evaluation is about rankings:

* Qini curve (Radcliffe 2007). Sort a randomised holdout by predicted uplift; after the top ``k`` rows,
  ``Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)`` is the incremental outcome among the treated customers in
  the top ``k``, with the control sum scaled to the treated count. With a 50/50 split it is about half of
  what treating all ``k`` would add. Curves are divided by the holdout size.
* Qini coefficient: area between the Qini curve and the straight line of random targeting, over the
  targeted fraction from 0 to 1. Zero means no better than random.
* Uplift curve and AUUC: ``U(k) = (Y_t(k) / N_t(k) - Y_c(k) / N_c(k)) * k``, the difference of the two
  arm means in the top ``k`` scaled to all ``k`` customers, so it estimates the incremental outcome of
  treating the top ``k``; AUUC is its area. For a perfect ranking of standard normal effects the AUUC
  tends to ``1 / (2 sqrt(pi))`` and the Qini coefficient to half of that, which the tests check.
* True gain curve: with ground truth, ``G(k)`` is the sum of the true CATE of the top ``k``, which is what
  the Qini and uplift curves estimate.

Targeting: with a budget of ``b`` coupons, treat the top ``b`` customers by predicted uplift among those
whose predicted uplift exceeds the cost of a contact. The value of a policy ``pi`` is the incremental
outcome per customer, ``mean(pi(x) * (tau(x) - cost))``; with a randomised holdout it can be estimated
without ground truth by inverse propensity weighting, which :func:`ipw_policy_value` implements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Curve:
    """A curve over the targeted fraction of the base, starting at (0, 0)."""

    fraction: np.ndarray
    value: np.ndarray

    def area(self) -> float:
        """Trapezoidal area over the fraction axis."""
        return float(np.trapezoid(self.value, self.fraction))


def _check(score: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    score = np.asarray(score, dtype=float)
    out = [score] + [np.asarray(a, dtype=float) for a in arrays]
    if score.ndim != 1 or any(a.shape != score.shape for a in out):
        raise ValueError("all inputs must be 1-D arrays of the same length")
    if not all(np.isfinite(a).all() for a in out):
        raise ValueError("inputs must be finite")
    return tuple(out)


def _cumulatives(score: np.ndarray, y: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, ...]:
    """Running treated and control counts and outcome sums in descending score order (stable on ties)."""
    order = np.argsort(-score, kind="stable")
    t, y = t[order], y[order]
    return np.cumsum(t), np.cumsum(1.0 - t), np.cumsum(y * t), np.cumsum(y * (1.0 - t))


def qini_curve(score: np.ndarray, y: np.ndarray, t: np.ndarray) -> Curve:
    """Qini curve of a ranking on a randomised holdout, per customer of the holdout."""
    score, y, t = _check(score, y, t)
    if not np.isin(t, (0.0, 1.0)).all():
        raise ValueError("treatment must be 0 or 1")
    n_t, n_c, y_t, y_c = _cumulatives(score, y, t)
    q = y_t - np.divide(y_c * n_t, n_c, out=np.zeros_like(y_c), where=n_c > 0)
    n = len(score)
    return Curve(np.r_[0.0, np.arange(1, n + 1) / n], np.r_[0.0, q / n])


def uplift_curve(score: np.ndarray, y: np.ndarray, t: np.ndarray) -> Curve:
    """Uplift (cumulative gain) curve: arm-mean difference in the top ``k`` scaled to ``k`` customers."""
    score, y, t = _check(score, y, t)
    n_t, n_c, y_t, y_c = _cumulatives(score, y, t)
    mean_t = np.divide(y_t, n_t, out=np.zeros_like(y_t), where=n_t > 0)
    mean_c = np.divide(y_c, n_c, out=np.zeros_like(y_c), where=n_c > 0)
    k = np.arange(1, len(score) + 1)
    both = (n_t > 0) & (n_c > 0)
    n = len(score)
    return Curve(np.r_[0.0, k / n], np.r_[0.0, np.where(both, (mean_t - mean_c) * k, 0.0) / n])


def true_gain_curve(score: np.ndarray, tau: np.ndarray) -> Curve:
    """Cumulative true CATE of the top ``k`` per customer of the base: what the other curves estimate."""
    score, tau = _check(score, tau)
    order = np.argsort(-score, kind="stable")
    n = len(score)
    return Curve(np.r_[0.0, np.arange(1, n + 1) / n], np.r_[0.0, np.cumsum(tau[order]) / n])


def qini_coefficient(score: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    """Area between the Qini curve and random targeting; zero means no better than random."""
    curve = qini_curve(score, y, t)
    return curve.area() - 0.5 * float(curve.value[-1])


def auuc(score: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    """Area under the uplift curve."""
    return uplift_curve(score, y, t).area()


def budget_policy(score: np.ndarray, budget: float, cost: float = 0.0) -> np.ndarray:
    """Boolean mask: the top ``budget`` share of customers by score, keeping only scores above ``cost``.

    ``budget`` is a fraction of the base in (0, 1]. Ties are broken by row order, so equal scores never
    make the selection depend on the sort implementation.
    """
    (score,) = _check(score)
    if not 0 < budget <= 1:
        raise ValueError("budget must be a fraction in (0, 1]")
    k = int(np.floor(budget * len(score)))
    chosen = np.argsort(-score, kind="stable")[:k]
    mask = np.zeros(len(score), dtype=bool)
    mask[chosen[score[chosen] > cost]] = True
    return mask


def policy_value(mask: np.ndarray, tau: np.ndarray, cost: float = 0.0) -> float:
    """True incremental value per customer of treating ``mask``, net of the contact cost."""
    mask = np.asarray(mask, dtype=bool)
    return float(np.mean(mask * (np.asarray(tau, dtype=float) - cost)))


def ipw_policy_value(
    mask: np.ndarray, y: np.ndarray, t: np.ndarray, e: np.ndarray | float, cost: float = 0.0
) -> float:
    """Estimated incremental value of ``mask`` from logged outcomes with assignment probability ``e``."""
    mask = np.asarray(mask, dtype=bool)
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    e = np.broadcast_to(np.asarray(e, dtype=float), y.shape)
    if ((e <= 0) | (e >= 1)).any():
        raise ValueError("assignment probabilities must be strictly between 0 and 1")
    contrast = t * y / e - (1.0 - t) * y / (1.0 - e)
    return float(np.mean(mask * (contrast - cost)))
