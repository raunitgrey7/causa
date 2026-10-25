"""Batched weighted solvers: agreement with scikit-learn and the identities the refit bootstrap relies on."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.linear_model import LogisticRegression, Ridge

from causa import linear
from causa.linear import predict_logistic, row_outer, weighted_logistic, weighted_ridge


def _design(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Z = np.column_stack([np.ones(n), rng.standard_normal((n, d - 1))])
    beta = rng.standard_normal(d)
    y_bin = (rng.random(n) < 1 / (1 + np.exp(-(Z @ beta)))).astype(float)
    y_lin = Z @ beta + rng.standard_normal(n)
    return Z, y_bin, y_lin


def test_logistic_matches_scikit_learn_with_unpenalised_intercept():
    Z, y, _ = _design(800, 6, 0)
    ours = weighted_logistic(Z, y, np.ones((1, len(y))), lam=2.0)[0]
    ref = LogisticRegression(C=0.5, tol=1e-12, max_iter=10_000).fit(Z[:, 1:], y)
    np.testing.assert_allclose(ours[1:], ref.coef_[0], atol=1e-5)
    assert abs(ours[0] - ref.intercept_[0]) < 1e-5


def test_ridge_matches_scikit_learn():
    Z, _, y = _design(500, 5, 1)
    ours = weighted_ridge(Z, y, np.ones((1, len(y))), lam=3.0)[0]
    ref = Ridge(alpha=3.0).fit(Z[:, 1:], y)
    np.testing.assert_allclose(ours[1:], ref.coef_, atol=1e-9)
    assert abs(ours[0] - ref.intercept_) < 1e-9


def test_integer_weights_equal_duplicated_rows():
    Z, y, _ = _design(300, 4, 2)
    counts = np.random.default_rng(3).integers(0, 4, len(y))
    weighted = weighted_logistic(Z, y, counts[None, :].astype(float))[0]
    rows = np.repeat(np.arange(len(y)), counts)
    duplicated = weighted_logistic(Z[rows], y[rows], np.ones((1, len(rows))))[0]
    np.testing.assert_allclose(weighted, duplicated, atol=1e-8)


def test_zero_weight_rows_are_ignored():
    Z, _, y = _design(400, 4, 4)
    w = np.ones(len(y))
    w[:100] = 0.0
    np.testing.assert_allclose(
        weighted_ridge(Z, y, w[None, :])[0], weighted_ridge(Z[100:], y[100:], np.ones((1, 300)))[0], atol=1e-9
    )


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 10_000), batch=st.integers(1, 6))
def test_batched_fit_equals_a_loop_of_single_fits(seed, batch):
    Z, y, _ = _design(200, 4, seed)
    W = np.random.default_rng(seed).integers(1, 3, (batch, len(y))).astype(float)
    batched = weighted_logistic(Z, y, W, ZZ=row_outer(Z))
    looped = np.vstack([weighted_logistic(Z, y, W[b : b + 1])[0] for b in range(batch)])
    np.testing.assert_allclose(batched, looped, atol=1e-7)


def test_chunked_gram_equals_cached_gram(monkeypatch):
    Z, _, _ = _design(1000, 5, 5)
    V = np.random.default_rng(5).random((3, 1000))
    monkeypatch.setattr(linear, "GRAM_CHUNK_ELEMENTS", 100)
    np.testing.assert_allclose(linear._gram(V, Z, None), linear._gram(V, Z, row_outer(Z)), atol=1e-9)


def test_separable_data_still_converges_thanks_to_the_ridge_term():
    Z = np.column_stack([np.ones(100), np.linspace(-1, 1, 100)])
    y = (Z[:, 1] > 0).astype(float)
    beta = weighted_logistic(Z, y, np.ones((1, 100)), lam=1.0)
    assert np.isfinite(beta).all()
    p = predict_logistic(Z, beta)[0]
    assert p[-1] > 0.9 and p[0] < 0.1


def test_non_convergence_is_an_error_not_a_silent_answer():
    Z, y, _ = _design(300, 4, 6)
    with pytest.raises(RuntimeError, match="did not converge"):
        weighted_logistic(Z, y, np.ones((1, 300)), max_iter=1)


@pytest.mark.parametrize(
    ("Z", "y", "W"),
    [
        (np.ones((5, 2)), np.ones(4), np.ones((1, 5))),
        (np.ones((5, 2)), np.ones(5), -np.ones((1, 5))),
        (np.ones((5, 2)), np.ones(5), np.ones((1, 4))),
    ],
)
def test_bad_shapes_and_negative_weights_are_rejected(Z, y, W):
    with pytest.raises(ValueError):
        weighted_ridge(Z, y, W)


def test_logistic_rejects_non_binary_targets():
    with pytest.raises(ValueError, match="0 or 1"):
        weighted_logistic(np.ones((3, 1)), np.array([0.0, 0.5, 1.0]), np.ones((1, 3)))
