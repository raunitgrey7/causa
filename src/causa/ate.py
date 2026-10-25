"""Average treatment effect estimators for observational data, with nonparametric bootstrap intervals.

Estimators, from naive to doubly robust:

* ``difference_in_means``: what a dashboard shows. Biased whenever assignment depended on covariates that
  also drive the outcome.
* ``outcome_regression`` (G-computation): ridge regressions of the outcome on the basis, one per arm,
  averaged over everyone. Unbiased only if the outcome model is right.
* ``ipw``: Hajek (self-normalised) inverse propensity weighting. Unbiased if the propensity model is right;
  its variance grows as overlap weakens.
* ``aipw``: augmented IPW, the doubly robust estimator, with nuisance models fitted on the full sample.
  Consistent if either nuisance model is right. With a fixed-dimension basis this needs no sample
  splitting.
* ``aipw_crossfit``: the same score with K-fold cross-fitting (Chernozhukov et al. 2018), the form that
  stays valid when the nuisance models are flexible learners. With a small parametric basis it pays for
  that robustness with variance, because each nuisance model sees only (K - 1) / K of the rows; the
  benchmark measures how much.

Every estimator gets a nonparametric bootstrap percentile interval in which each resample refits every
nuisance model: the propensity model, the per-arm outcome regressions and, for the cross-fitted score, all
of them inside each fold (fold membership follows the original rows, so a duplicated row never sits on both
sides of a fold). :mod:`causa.linear` makes those refits one batched solve. Both AIPW forms also report the
Wald interval from their estimated influence function (suffix ``_wald``), which is what most libraries ship.

Outcome-model predictions are clipped to the range of the outcomes the model was fitted on. A conditional
mean of revenue cannot be negative or exceed the largest basket seen, and without the bound a polynomial
basis extrapolates wildly on the few customers with extreme covariates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm

from causa.features import Basis
from causa.linear import predict_linear, predict_logistic, row_outer, weighted_logistic, weighted_ridge
from causa.propensity import DEFAULT_CLIP, OverlapReport, overlap_report

METHODS: tuple[str, ...] = (
    "difference_in_means",
    "outcome_regression",
    "ipw",
    "aipw",
    "aipw_wald",
    "aipw_crossfit",
    "aipw_crossfit_wald",
)
MIN_PER_ARM = 10
MAX_CACHED_OUTER = 16_000_000


@dataclass(frozen=True)
class ATEConfig:
    """Estimation settings; the defaults are the ones the benchmark validates."""

    n_boot: int = 200
    folds: int = 2
    alpha: float = 0.05
    ridge: float = 1.0
    clip: float = DEFAULT_CLIP
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject settings the estimators cannot honour."""
        if self.n_boot < 20:
            raise ValueError("n_boot must be at least 20 for a percentile interval")
        if self.folds < 2:
            raise ValueError("cross-fitting needs at least 2 folds")
        if not 0 < self.alpha < 1 or self.ridge <= 0 or not 0 <= self.clip < 0.5:
            raise ValueError("alpha must be in (0, 1), ridge positive and clip in [0, 0.5)")


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its standard error and a two-sided interval."""

    method: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    interval: str

    def covers(self, value: float) -> bool:
        """Whether the interval contains ``value``."""
        return self.ci_low <= value <= self.ci_high

    @property
    def width(self) -> float:
        """Interval width."""
        return self.ci_high - self.ci_low


@dataclass(frozen=True)
class ATEResult:
    """All estimates for one dataset, plus overlap diagnostics from the full-sample propensity fit."""

    estimates: dict[str, Estimate]
    overlap: OverlapReport | None
    n: int
    n_boot: int

    def to_dict(self) -> dict[str, object]:
        """Plain-JSON form."""
        return {
            "n": self.n,
            "n_boot": self.n_boot,
            "estimates": {k: asdict(v) for k, v in self.estimates.items()},
            "overlap": self.overlap.to_dict() if self.overlap else None,
        }


@dataclass(frozen=True)
class _Design:
    Z: np.ndarray
    ZZ: np.ndarray | None
    t: np.ndarray
    y: np.ndarray
    folds: np.ndarray
    ridge: float
    clip: float

    def outer(self, rows: np.ndarray) -> np.ndarray | None:
        """Cached outer products of ``rows``, or None when the cache was too large to build."""
        return None if self.ZZ is None else self.ZZ[rows]


def bootstrap_weights(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Row 0 is the original sample (all ones); rows 1..n_boot are multinomial resample counts."""
    idx = rng.integers(0, n, size=(n_boot, n))
    offsets = (np.arange(n_boot) * n)[:, None]
    counts = np.bincount((idx + offsets).ravel(), minlength=n_boot * n).reshape(n_boot, n)
    return np.vstack([np.ones(n), counts.astype(float)])


def _difference_in_means(d: _Design, W: np.ndarray) -> np.ndarray:
    wt, wc = W * d.t, W * (1.0 - d.t)
    return (wt @ d.y) / wt.sum(axis=1) - (wc @ d.y) / wc.sum(axis=1)


def _arm_means(d: _Design, W: np.ndarray, rows: np.ndarray, Z_eval: np.ndarray) -> tuple[np.ndarray, ...]:
    """Ridge outcome model per arm fitted on ``rows``, predicted at ``Z_eval``: returns (mu0, mu1)."""
    out = []
    for arm in (0.0, 1.0):
        fit_rows = rows & (d.t == arm)
        y = d.y[fit_rows]
        beta = weighted_ridge(d.Z[fit_rows], y, W[:, fit_rows], d.ridge, d.outer(fit_rows))
        out.append(np.clip(predict_linear(Z_eval, beta), y.min(), y.max()))
    return out[0], out[1]


def _propensity(d: _Design, W: np.ndarray, rows: np.ndarray, Z_eval: np.ndarray) -> np.ndarray:
    """Clipped propensity ``(B, n_eval)``; resamples start Newton from the original-sample solution."""
    Z, t, ZZ = d.Z[rows], d.t[rows], d.outer(rows)
    start = weighted_logistic(Z, t, W[:1, rows], d.ridge, ZZ)[0]
    beta = weighted_logistic(Z, t, W[:, rows], d.ridge, ZZ, init=start)
    return np.clip(predict_logistic(Z_eval, beta), d.clip, 1.0 - d.clip)


def _ipw(d: _Design, W: np.ndarray, e: np.ndarray) -> np.ndarray:
    a, c = W * d.t / e, W * (1.0 - d.t) / (1.0 - e)
    return (a @ d.y) / a.sum(axis=1) - (c @ d.y) / c.sum(axis=1)


def _aipw_score(t: np.ndarray, y: np.ndarray, e: np.ndarray, mu0: np.ndarray, mu1: np.ndarray) -> np.ndarray:
    """Efficient influence score of the ATE; its mean is the AIPW estimate."""
    return mu1 - mu0 + t * (y - mu1) / e - (1.0 - t) * (y - mu0) / (1.0 - e)


def _crossfit_scores(d: _Design, W: np.ndarray) -> np.ndarray:
    """Cross-fitted scores ``(B, n)``: every row is scored by nuisance models that never saw it."""
    phi = np.empty_like(W)
    for k in np.unique(d.folds):
        test = d.folds == k
        e = _propensity(d, W, ~test, d.Z[test])
        mu0, mu1 = _arm_means(d, W, ~test, d.Z[test])
        phi[:, test] = _aipw_score(d.t[test], d.y[test], e, mu0, mu1)
    return phi


def _weighted_mean(W: np.ndarray, values: np.ndarray) -> np.ndarray:
    return (W * values).sum(axis=1) / W.sum(axis=1)


def _wald(method: str, phi: np.ndarray, alpha: float) -> Estimate:
    point = float(phi.mean())
    se = float(phi.std(ddof=1) / np.sqrt(len(phi)))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return Estimate(method, point, se, point - z * se, point + z * se, "wald")


def _percentile(method: str, values: np.ndarray, alpha: float) -> Estimate:
    boot = values[1:]
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Estimate(method, float(values[0]), float(boot.std(ddof=1)), float(lo), float(hi), "bootstrap")


def _validate(X: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, t, y = np.asarray(X, dtype=float), np.asarray(t), np.asarray(y, dtype=float)
    if X.ndim != 2 or t.shape != (len(X),) or y.shape != (len(X),):
        raise ValueError("X must be (n, p) with t and y of length n")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("covariates and outcomes must be finite")
    if not np.isin(t, (0, 1)).all():
        raise ValueError("treatment must be coded 0 or 1")
    if min(int((t == 1).sum()), int((t == 0).sum())) < MIN_PER_ARM:
        raise ValueError(f"each arm needs at least {MIN_PER_ARM} units")
    return X, t.astype(float), y


def estimate_ate(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    config: ATEConfig | None = None,
    feature_names: tuple[str, ...] | list[str] | None = None,
    diagnostics: bool = True,
) -> ATEResult:
    """Estimate the ATE of ``t`` on ``y`` by every method in ``METHODS``, bootstrapping all refits at once.

    ``feature_names`` label the balance table; they default to ``x0, x1, ...``.
    """
    config = config or ATEConfig()
    X, t, y = _validate(X, t, y)
    names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
    if len(names) != X.shape[1]:
        raise ValueError(f"{len(names)} feature names for {X.shape[1]} covariate columns")
    rng = np.random.default_rng(config.seed)
    Z = Basis.fit(X).transform(X)
    n = len(t)
    ZZ = row_outer(Z) if n * Z.shape[1] ** 2 <= MAX_CACHED_OUTER else None
    d = _Design(Z, ZZ, t, y, rng.permutation(n) % config.folds, config.ridge, config.clip)
    W = bootstrap_weights(n, config.n_boot, rng)
    everyone = np.ones(n, dtype=bool)
    e = _propensity(d, W, everyone, Z)
    mu0, mu1 = _arm_means(d, W, everyone, Z)
    phi = _aipw_score(t, y, e, mu0, mu1)
    phi_cf = _crossfit_scores(d, W)
    a = config.alpha
    estimates = {
        "difference_in_means": _percentile("difference_in_means", _difference_in_means(d, W), a),
        "outcome_regression": _percentile("outcome_regression", _weighted_mean(W, mu1 - mu0), a),
        "ipw": _percentile("ipw", _ipw(d, W, e), a),
        "aipw": _percentile("aipw", _weighted_mean(W, phi), a),
        "aipw_wald": _wald("aipw_wald", phi[0], a),
        "aipw_crossfit": _percentile("aipw_crossfit", _weighted_mean(W, phi_cf), a),
        "aipw_crossfit_wald": _wald("aipw_crossfit_wald", phi_cf[0], a),
    }
    overlap = overlap_report(X, t.astype(int), e[0], names) if diagnostics else None
    return ATEResult(estimates=estimates, overlap=overlap, n=n, n_boot=config.n_boot)
