"""Propensity modelling and overlap diagnostics.

Weighting estimators divide by the estimated probability of the arm a unit actually received. Where that
probability is close to 0 or 1 the data hold almost no comparable units, weights explode and every
estimate that leans on them becomes fragile. Before trusting an observational ATE an analyst should know:

* how well the propensity model separates the arms (AUC near 0.5 means little confounding by observables,
  near 1 means the arms barely overlap);
* what share of units sit outside the overlap band (default 0.05 to 0.95, Crump et al. 2009);
* the Kish effective sample size of the inverse propensity weights in each arm;
* whether weighting balances the covariates: standardised mean differences (SMD) before and after
  weighting, with |SMD| > 0.1 the usual warning line (Austin and Stuart 2015). The denominator is the
  unweighted pooled standard deviation in both cases, so the two columns are comparable.

The propensity model is a ridge-penalised logistic regression on :class:`causa.features.Basis` columns,
fitted by :func:`causa.linear.weighted_logistic`, and predictions are clipped to ``[clip, 1 - clip]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from causa.features import Basis
from causa.linear import predict_logistic, weighted_logistic

DEFAULT_CLIP = 0.01
OVERLAP_BAND = (0.05, 0.95)
SMD_WARNING = 0.1


@dataclass(frozen=True)
class PropensityModel:
    """Logistic propensity model on a polynomial basis with clipped predictions."""

    basis: Basis
    coef: np.ndarray
    clip: float = DEFAULT_CLIP

    @classmethod
    def fit(
        cls, X: np.ndarray, t: np.ndarray, lam: float = 1.0, clip: float = DEFAULT_CLIP
    ) -> PropensityModel:
        """Fit on covariates ``X`` and binary treatment ``t``."""
        if not 0 <= clip < 0.5:
            raise ValueError("clip must be in [0, 0.5)")
        basis = Basis.fit(X)
        Z = basis.transform(X)
        coef = weighted_logistic(Z, np.asarray(t, dtype=float), np.ones((1, len(Z))), lam=lam)[0]
        return cls(basis=basis, coef=coef, clip=clip)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Clipped propensity scores for new rows."""
        raw = predict_logistic(self.basis.transform(X), self.coef)[0]
        return np.clip(raw, self.clip, 1.0 - self.clip)


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2``."""
    w = np.asarray(weights, dtype=float)
    denom = float(np.square(w).sum())
    return float(w.sum() ** 2 / denom) if denom > 0 else 0.0


def ipw_weights(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Inverse probability of the received arm: ``1/e`` for treated, ``1/(1 - e)`` for control."""
    t = np.asarray(t)
    return np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))


def standardized_mean_differences(
    X: np.ndarray, t: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Per-column SMD between arms, optionally after weighting, over the unweighted pooled SD."""
    X = np.asarray(X, dtype=float)
    t = np.asarray(t)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)
    treated, control = t == 1, t == 0
    mean_t = np.average(X[treated], axis=0, weights=w[treated])
    mean_c = np.average(X[control], axis=0, weights=w[control])
    pooled = np.sqrt((X[treated].var(axis=0, ddof=1) + X[control].var(axis=0, ddof=1)) / 2.0)
    return (mean_t - mean_c) / np.where(pooled > 1e-12, pooled, 1.0)


@dataclass(frozen=True)
class OverlapReport:
    """Diagnostics an analyst reads before trusting a weighting estimator."""

    n_treated: int
    n_control: int
    auc: float
    share_outside_band: float
    band: tuple[float, float]
    propensity_quantiles: dict[str, list[float]]
    ess_treated: float
    ess_control: float
    max_abs_smd_before: float
    max_abs_smd_after: float
    smd: list[dict[str, float | str]]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Plain-JSON form for the API and the benchmark."""
        return asdict(self)


def _overlap_warnings(report: dict[str, float], band: tuple[float, float]) -> list[str]:
    warnings = []
    if report["share_outside"] > 0.05:
        warnings.append(
            f"{report['share_outside']:.1%} of units have propensity outside [{band[0]}, {band[1]}]: "
            "weak overlap, weighting estimates lean on few units"
        )
    for arm in ("treated", "control"):
        if report[f"ess_frac_{arm}"] < 0.2:
            warnings.append(
                f"effective sample size of the {arm} weights is {report[f'ess_frac_{arm}']:.1%} of the arm"
            )
    if report["smd_after"] > SMD_WARNING:
        warnings.append(f"max |SMD| after weighting is {report['smd_after']:.3f}, above {SMD_WARNING}")
    return warnings


def overlap_report(
    X: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    band: tuple[float, float] = OVERLAP_BAND,
) -> OverlapReport:
    """Summarise overlap and balance for covariates ``X``, treatment ``t`` and propensity scores ``e``."""
    t = np.asarray(t)
    e = np.asarray(e, dtype=float)
    weights = ipw_weights(t, e)
    before = standardized_mean_differences(X, t)
    after = standardized_mean_differences(X, t, weights)
    treated, control = t == 1, t == 0
    qs = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    summary = {
        "share_outside": float(np.mean((e < band[0]) | (e > band[1]))),
        "ess_frac_treated": effective_sample_size(weights[treated]) / max(int(treated.sum()), 1),
        "ess_frac_control": effective_sample_size(weights[control]) / max(int(control.sum()), 1),
        "smd_after": float(np.abs(after).max()),
    }
    return OverlapReport(
        n_treated=int(treated.sum()),
        n_control=int(control.sum()),
        auc=float(roc_auc_score(t, e)),
        share_outside_band=summary["share_outside"],
        band=band,
        propensity_quantiles={
            "q": list(qs),
            "treated": np.quantile(e[treated], qs).round(6).tolist(),
            "control": np.quantile(e[control], qs).round(6).tolist(),
        },
        ess_treated=effective_sample_size(weights[treated]),
        ess_control=effective_sample_size(weights[control]),
        max_abs_smd_before=float(np.abs(before).max()),
        max_abs_smd_after=summary["smd_after"],
        smd=[
            {"feature": name, "before": float(b), "after": float(a)}
            for name, b, a in zip(feature_names, before, after, strict=True)
        ],
        warnings=_overlap_warnings(summary, band),
    )
