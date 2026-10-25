"""Customer encoding and the polynomial basis."""

import numpy as np
import pandas as pd
import pytest

from causa.data.campaign import simulate_customers
from causa.features import FEATURE_NAMES, Basis, feature_matrix, infer_continuous, validate_customers


@pytest.fixture
def customers() -> pd.DataFrame:
    return simulate_customers(300, np.random.default_rng(0))


def test_feature_matrix_has_named_columns_and_log_transforms(customers):
    X = feature_matrix(customers)
    assert X.shape == (300, len(FEATURE_NAMES))
    np.testing.assert_allclose(X[:, 5], np.log(customers["avg_order_value"]))
    np.testing.assert_allclose(X[:, 9] + X[:, 10], (customers["city_tier"] > 1).astype(float))


def test_basis_width_matches_transform_and_starts_with_an_intercept(customers):
    X = feature_matrix(customers)
    basis = Basis.fit(X)
    Z = basis.transform(X)
    assert Z.shape[1] == basis.n_columns == 1 + 11 + 7 + 21
    np.testing.assert_array_equal(Z[:, 0], 1.0)
    np.testing.assert_allclose(Z[:, 1:12].mean(axis=0), 0.0, atol=1e-12)


def test_continuous_columns_are_inferred_and_binary_ones_are_not_squared(customers):
    assert infer_continuous(feature_matrix(customers)) == (0, 1, 2, 3, 4, 5, 6)


def test_constant_columns_do_not_divide_by_zero():
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    Z = Basis.fit(X).transform(X)
    assert np.isfinite(Z).all()


def test_basis_needs_two_rows():
    with pytest.raises(ValueError):
        Basis.fit(np.ones((1, 3)))


def test_missing_columns_are_named(customers):
    with pytest.raises(ValueError, match="recency_days"):
        validate_customers(customers.drop(columns=["recency_days"]))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("email_open_rate", 1.5),
        ("recency_days", 0),
        ("city_tier", 4),
        ("app_user", 2),
        ("avg_order_value", -1),
    ],
)
def test_out_of_range_values_are_rejected(customers, column, value):
    bad = customers.copy()
    bad.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        feature_matrix(bad)


def test_non_numeric_and_infinite_values_are_rejected(customers):
    bad = customers.astype({"tenure_months": object})
    bad.loc[0, "tenure_months"] = "ten"
    with pytest.raises(ValueError, match="non-numeric"):
        feature_matrix(bad)
    inf = customers.copy()
    inf.loc[0, "avg_order_value"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        feature_matrix(inf)
