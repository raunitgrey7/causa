"""Sample ratio mismatch (SRM) check.

If an experiment planned a 50/50 split and the analytics table holds 50.6/49.4 after 100,000 users, the
randomisation or the logging is broken, and every metric comparison is suspect: the users who went missing
are rarely missing at random. A Pearson chi-square goodness-of-fit test of the arm counts against the
planned split catches it. The conventional alarm threshold is very small (p < 0.001) because the check runs
on every experiment, and a false alarm stops a healthy test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

SRM_ALPHA = 0.001


@dataclass(frozen=True)
class SRMResult:
    """Outcome of the chi-square SRM test."""

    counts: list[int]
    expected_share: list[float]
    observed_share: list[float]
    statistic: float
    p_value: float
    alpha: float
    mismatch: bool


def chi_square_statistic(counts: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Pearson statistic and p-value for count rows ``(..., arms)`` against shares ``expected``."""
    counts = np.asarray(counts, dtype=float)
    p0 = np.asarray(expected, dtype=float)
    n = counts.sum(axis=-1, keepdims=True)
    exp_counts = n * p0
    stat = np.where(exp_counts > 0, (counts - exp_counts) ** 2 / np.maximum(exp_counts, 1e-300), 0.0).sum(
        axis=-1
    )
    return stat, chi2.sf(stat, df=len(p0) - 1)


def srm_test(
    counts: list[int] | np.ndarray, expected: list[float] | np.ndarray, alpha: float = SRM_ALPHA
) -> SRMResult:
    """Chi-square SRM test of observed arm counts against the planned shares."""
    counts_arr = np.asarray(counts, dtype=float)
    p0 = np.asarray(expected, dtype=float)
    if counts_arr.ndim != 1 or len(counts_arr) != len(p0) or len(p0) < 2:
        raise ValueError("need one count and one planned share per arm, at least two arms")
    if (counts_arr < 0).any() or counts_arr.sum() == 0:
        raise ValueError("counts must be non-negative and not all zero")
    if not np.isclose(p0.sum(), 1.0) or (p0 <= 0).any():
        raise ValueError("planned shares must be positive and sum to 1")
    stat, p = chi_square_statistic(counts_arr, p0)
    return SRMResult(
        counts=[int(c) for c in counts_arr],
        expected_share=[float(x) for x in p0],
        observed_share=(counts_arr / counts_arr.sum()).round(6).tolist(),
        statistic=float(stat),
        p_value=float(p),
        alpha=alpha,
        mismatch=bool(p < alpha),
    )
