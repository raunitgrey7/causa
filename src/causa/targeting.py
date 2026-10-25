"""Uplift targeting service model: train the CATE learners, pick one on a randomised holdout, decide coupons.

A team cannot select a CATE model by PEHE, because individual effects are never observed. It can run a
small randomised holdout and compare rankings by Qini coefficient, which needs only observed outcomes and
assignment. :func:`train_targeting_model` does exactly that: it fits every learner on the confounded
campaign history, scores each on a randomised holdout, keeps the one with the highest Qini coefficient and
records every candidate's score so the choice is auditable. The holdout also yields an IPW estimate of the
incremental revenue at each budget, which is what the dashboard plots.

Decisions follow :func:`causa.uplift.budget_policy`: rank customers by predicted uplift and send coupons to
the top ``budget`` share whose predicted uplift exceeds the cost of a contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from causa import __version__
from causa.data.campaign import CampaignConfig, simulate_campaign
from causa.features import feature_matrix
from causa.learners import LEARNERS, CATELearner, make_learner
from causa.uplift import auuc, budget_policy, ipw_policy_value, qini_coefficient

CURVE_BUDGETS: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
TRAIN_THREADS = 2


@dataclass(frozen=True)
class TargetingDecision:
    """Coupon decisions for a batch of customers."""

    uplift: np.ndarray
    treat: np.ndarray
    budget: float
    cost: float

    @property
    def expected_incremental_revenue(self) -> float:
        """Sum of predicted uplift net of contact cost over the customers selected."""
        return float(np.sum((self.uplift - self.cost) * self.treat))


@dataclass
class TargetingModel:
    """The selected learner plus the validation evidence for choosing it."""

    learner_name: str
    learner: CATELearner
    validation: dict[str, dict[str, float]]
    budget_curve: dict[str, list[float]]
    n_train: int
    n_holdout: int
    seed: int
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    version: str = __version__

    def predict(self, customers: pd.DataFrame) -> np.ndarray:
        """Predicted incremental 30-day revenue of a coupon for each customer (INR)."""
        return self.learner.predict(feature_matrix(customers))

    def decide(self, customers: pd.DataFrame, budget: float, cost: float = 0.0) -> TargetingDecision:
        """Rank by predicted uplift and select within the budget."""
        uplift = self.predict(customers)
        return TargetingDecision(
            uplift=uplift, treat=budget_policy(uplift, budget, cost), budget=budget, cost=cost
        )


def _validate_learners(
    fitted: dict[str, CATELearner], holdout_X: np.ndarray, y: np.ndarray, t: np.ndarray
) -> dict[str, dict[str, float]]:
    scores = {}
    for name, learner in fitted.items():
        pred = learner.predict(holdout_X)
        scores[name] = {"qini": qini_coefficient(pred, y, t), "auuc": auuc(pred, y, t)}
    return scores


def train_targeting_model(
    n_train: int = 50_000, n_holdout: int = 20_000, seed: int = 42, learners: tuple[str, ...] = LEARNERS
) -> TargetingModel:
    """Fit every learner on a simulated campaign history and keep the best by holdout Qini.

    Training is capped at ``TRAIN_THREADS`` OpenMP threads so a server start does not take over the machine.
    """
    train = simulate_campaign(CampaignConfig(n=n_train, seed=seed))
    holdout = simulate_campaign(CampaignConfig(n=n_holdout, seed=seed + 1, randomized=True))
    X, Xh = feature_matrix(train.customers), feature_matrix(holdout.customers)
    with threadpool_limits(TRAIN_THREADS):
        fitted = {
            name: make_learner(name, seed=seed).fit(X, train.treatment, train.revenue) for name in learners
        }
        validation = _validate_learners(fitted, Xh, holdout.revenue, holdout.treatment)
    best = max(validation, key=lambda name: validation[name]["qini"])
    scores = fitted[best].predict(Xh)
    curve = [
        ipw_policy_value(budget_policy(scores, b), holdout.revenue, holdout.treatment, 0.5)
        for b in CURVE_BUDGETS
    ]
    return TargetingModel(
        learner_name=best,
        learner=fitted[best],
        validation=validation,
        budget_curve={"budgets": list(CURVE_BUDGETS), "ipw_value_per_customer": curve},
        n_train=n_train,
        n_holdout=n_holdout,
        seed=seed,
    )


def save_model(model: TargetingModel, path: Path) -> None:
    """Persist with joblib (scikit-learn estimators inside)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> TargetingModel:
    """Load a model written by :func:`save_model`; refuse anything else."""
    model = joblib.load(path)
    if not isinstance(model, TargetingModel):
        raise TypeError(f"{path} does not hold a TargetingModel")
    return model
