"""Meta-learners for the conditional average treatment effect (CATE): S, T, X and DR.

They differ in how they turn outcome regressions into an effect estimate, which matters most when the
effect is small next to the outcome's noise, as it is for a coupon's effect on revenue:

* S-learner: one model of ``y`` on ``(x, t)``; the effect is the difference of its two predictions. Trees
  often ignore ``t`` when its signal is weak, which shrinks every effect toward zero.
* T-learner: one model per arm; the effect is the difference. Each model's errors land in the effect, so
  it is noisy, and it is confounded wherever one arm is thin.
* X-learner (Kunzel et al. 2019): imputes individual effects with the other arm's model, regresses them per
  arm, and blends the two effect models with the propensity score so the better-populated arm dominates.
* DR-learner (Kennedy 2023): regresses the cross-fitted AIPW pseudo-outcome on ``x``. Its error is second
  order in the nuisance errors, so it stays close to an oracle regression of the true effect.

The base regressor is scikit-learn's histogram gradient boosting with early stopping on an internal 10
percent validation split (fixed seed), so each stage adapts its number of trees to how noisy its target
is. The propensity model is :class:`causa.propensity.PropensityModel`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from causa.propensity import PropensityModel

RegressorFactory = Callable[[int], RegressorMixin]
LEARNERS: tuple[str, ...] = ("s_learner", "t_learner", "x_learner", "dr_learner")


def default_regressor(seed: int) -> RegressorMixin:
    """Gradient boosting with early stopping; the one base learner every meta-learner shares."""
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=500,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )


def _check(X: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, t, y = np.asarray(X, dtype=float), np.asarray(t), np.asarray(y, dtype=float)
    if X.ndim != 2 or t.shape != (len(X),) or y.shape != (len(X),):
        raise ValueError("X must be (n, p) with t and y of length n")
    if not np.isin(t, (0, 1)).all() or len(np.unique(t)) < 2:
        raise ValueError("treatment must be 0/1 with both arms present")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("covariates and outcomes must be finite")
    return X, t.astype(int), y


@dataclass
class SLearner:
    """One outcome model with the treatment as a feature."""

    regressor: RegressorFactory = default_regressor
    seed: int = 0
    name: str = "s_learner"
    model_: RegressorMixin | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> SLearner:
        """Fit ``y ~ f(x, t)``."""
        X, t, y = _check(X, t, y)
        self.model_ = self.regressor(self.seed).fit(np.column_stack([X, t]), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``f(x, 1) - f(x, 0)``."""
        if self.model_ is None:
            raise RuntimeError("fit the learner first")
        X = np.asarray(X, dtype=float)
        ones, zeros = np.ones((len(X), 1)), np.zeros((len(X), 1))
        return self.model_.predict(np.hstack([X, ones])) - self.model_.predict(np.hstack([X, zeros]))


@dataclass
class TLearner:
    """Separate outcome models for control and treated customers."""

    regressor: RegressorFactory = default_regressor
    seed: int = 0
    name: str = "t_learner"
    models_: tuple[RegressorMixin, RegressorMixin] | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> TLearner:
        """Fit ``mu_0`` on control rows and ``mu_1`` on treated rows."""
        X, t, y = _check(X, t, y)
        self.models_ = (
            self.regressor(self.seed).fit(X[t == 0], y[t == 0]),
            self.regressor(self.seed + 1).fit(X[t == 1], y[t == 1]),
        )
        return self

    def arm_models(self) -> tuple[RegressorMixin, RegressorMixin]:
        """The fitted ``(mu_0, mu_1)`` pair."""
        if self.models_ is None:
            raise RuntimeError("fit the learner first")
        return self.models_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``mu_1(x) - mu_0(x)``."""
        mu0, mu1 = self.arm_models()
        X = np.asarray(X, dtype=float)
        return mu1.predict(X) - mu0.predict(X)


@dataclass
class XLearner:
    """Imputed-effect regressions per arm, blended by the propensity score."""

    regressor: RegressorFactory = default_regressor
    seed: int = 0
    name: str = "x_learner"
    effect_models_: tuple[RegressorMixin, RegressorMixin] | None = field(default=None, init=False, repr=False)
    propensity_: PropensityModel | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> XLearner:
        """Stage 1: arm models; stage 2: regress imputed effects; keep the propensity model for blending."""
        X, t, y = _check(X, t, y)
        mu0, mu1 = TLearner(self.regressor, self.seed).fit(X, t, y).arm_models()
        treated, control = t == 1, t == 0
        imputed_treated = y[treated] - mu0.predict(X[treated])
        imputed_control = mu1.predict(X[control]) - y[control]
        self.effect_models_ = (
            self.regressor(self.seed + 2).fit(X[control], imputed_control),
            self.regressor(self.seed + 3).fit(X[treated], imputed_treated),
        )
        self.propensity_ = PropensityModel.fit(X, t)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``e(x) tau_0(x) + (1 - e(x)) tau_1(x)``: lean on the effect model from the larger arm."""
        if self.effect_models_ is None or self.propensity_ is None:
            raise RuntimeError("fit the learner first")
        X = np.asarray(X, dtype=float)
        g = self.propensity_.predict(X)
        return g * self.effect_models_[0].predict(X) + (1.0 - g) * self.effect_models_[1].predict(X)


@dataclass
class DRLearner:
    """Regression of cross-fitted AIPW pseudo-outcomes on the covariates."""

    regressor: RegressorFactory = default_regressor
    seed: int = 0
    folds: int = 2
    clip: float = 0.01
    name: str = "dr_learner"
    model_: RegressorMixin | None = field(default=None, init=False, repr=False)
    pseudo_outcomes_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> DRLearner:
        """Cross-fit the nuisances, build pseudo-outcomes, then regress them on ``x``."""
        X, t, y = _check(X, t, y)
        phi = np.empty(len(y))
        splitter = KFold(self.folds, shuffle=True, random_state=self.seed)
        for k, (train, test) in enumerate(splitter.split(X)):
            arms = TLearner(self.regressor, self.seed + 10 * (k + 1)).fit(X[train], t[train], y[train])
            e = PropensityModel.fit(X[train], t[train], clip=self.clip).predict(X[test])
            mu0, mu1 = (m.predict(X[test]) for m in arms.arm_models())
            tt, yy = t[test], y[test]
            phi[test] = mu1 - mu0 + tt * (yy - mu1) / e - (1 - tt) * (yy - mu0) / (1.0 - e)
        self.pseudo_outcomes_ = phi
        self.model_ = self.regressor(self.seed + 99).fit(X, phi)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted CATE."""
        if self.model_ is None:
            raise RuntimeError("fit the learner first")
        return self.model_.predict(np.asarray(X, dtype=float))


CATELearner = SLearner | TLearner | XLearner | DRLearner


def make_learner(name: str, seed: int = 0, regressor: RegressorFactory = default_regressor) -> CATELearner:
    """Build a learner by name (one of ``LEARNERS``)."""
    classes: dict[str, type[CATELearner]] = {
        "s_learner": SLearner,
        "t_learner": TLearner,
        "x_learner": XLearner,
        "dr_learner": DRLearner,
    }
    if name not in classes:
        raise ValueError(f"unknown learner {name!r}; choose from {', '.join(LEARNERS)}")
    return classes[name](regressor=regressor, seed=seed)
