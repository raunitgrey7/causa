"""Model features: one numeric encoding of the customer table, plus a polynomial basis for linear models.

Every model in the package (propensity, outcome regressions, CATE learners, the targeting service) reads
customers through :func:`feature_matrix`, so a covariate added to the generator or an uploaded file reaches
all of them the same way. Counts and money are logged because their effects are multiplicative.

:class:`Basis` turns those features into standardised columns, squares and pairwise products of the
continuous ones, plus an intercept. It is generic (it does not know the generator's coefficients), and it
is linear, which is what lets :mod:`causa.linear` refit hundreds of bootstrap replicates in one batched
solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

FEATURE_NAMES: tuple[str, ...] = (
    "log_tenure",
    "log_orders",
    "log_recency",
    "log_sessions",
    "email_open_rate",
    "log_aov",
    "discount_affinity",
    "app_user",
    "loyalty_member",
    "tier_2",
    "tier_3",
)
CONTINUOUS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
REQUIRED_COLUMNS: tuple[str, ...] = (
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


def validate_customers(customers: pd.DataFrame) -> None:
    """Raise ``ValueError`` with a specific message when a customer table cannot be encoded."""
    missing = [c for c in REQUIRED_COLUMNS if c not in customers.columns]
    if missing:
        raise ValueError(f"missing customer columns: {', '.join(missing)}")
    values = customers[list(REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        bad = [c for c in REQUIRED_COLUMNS if values[c].isna().any()]
        raise ValueError(f"non-numeric or missing values in: {', '.join(bad)}")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("customer columns must be finite")
    checks = {
        "tenure_months": values["tenure_months"] >= 0,
        "orders_12m": values["orders_12m"] >= 0,
        "recency_days": values["recency_days"] > 0,
        "sessions_30d": values["sessions_30d"] >= 0,
        "email_open_rate": values["email_open_rate"].between(0, 1),
        "avg_order_value": values["avg_order_value"] > 0,
        "discount_affinity": values["discount_affinity"].between(0, 1),
        "app_user": values["app_user"].isin([0, 1]),
        "loyalty_member": values["loyalty_member"].isin([0, 1]),
        "city_tier": values["city_tier"].isin([1, 2, 3]),
    }
    bad = [name for name, ok in checks.items() if not bool(ok.all())]
    if bad:
        raise ValueError(f"out-of-range values in: {', '.join(bad)}")


def feature_matrix(customers: pd.DataFrame) -> np.ndarray:
    """Encode a customer table as the float matrix every model consumes (columns in ``FEATURE_NAMES``)."""
    validate_customers(customers)
    c = customers
    tier = c["city_tier"].to_numpy(int)
    return np.column_stack(
        [
            np.log1p(c["tenure_months"].to_numpy(float)),
            np.log1p(c["orders_12m"].to_numpy(float)),
            np.log(c["recency_days"].to_numpy(float)),
            np.log1p(c["sessions_30d"].to_numpy(float)),
            c["email_open_rate"].to_numpy(float),
            np.log(c["avg_order_value"].to_numpy(float)),
            c["discount_affinity"].to_numpy(float),
            c["app_user"].to_numpy(float),
            c["loyalty_member"].to_numpy(float),
            (tier == 2).astype(float),
            (tier == 3).astype(float),
        ]
    )


@dataclass(frozen=True)
class Basis:
    """Intercept, standardised features, squares and pairwise products of the continuous features."""

    mean: np.ndarray
    scale: np.ndarray
    continuous: tuple[int, ...]

    @classmethod
    def fit(cls, X: np.ndarray, continuous: tuple[int, ...] = CONTINUOUS) -> Basis:
        """Learn the standardisation from training rows only, so held-out folds stay held out."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("Basis.fit needs a 2-D array with at least two rows")
        scale = X.std(axis=0)
        return cls(mean=X.mean(axis=0), scale=np.where(scale > 1e-12, scale, 1.0), continuous=continuous)

    @property
    def n_columns(self) -> int:
        """Width of the transformed matrix."""
        k = len(self.continuous)
        return 1 + len(self.mean) + k + k * (k - 1) // 2

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Expand ``X`` into the basis; column 0 is the intercept."""
        Z = (np.asarray(X, dtype=float) - self.mean) / self.scale
        cont = Z[:, list(self.continuous)]
        products = [cont[:, i] * cont[:, j] for i, j in combinations(range(cont.shape[1]), 2)]
        return np.column_stack([np.ones(len(Z)), Z, cont**2, *products])
