"""Weighted ridge and logistic regression, solved for a batch of weight vectors at once.

A nonparametric bootstrap of a propensity-weighted estimator has to refit the propensity model on every
resample. Looping over scikit-learn fits costs milliseconds per refit, which turns a Monte Carlo coverage
study (hundreds of replications with hundreds of resamples each) into hours. A bootstrap resample is just a
vector of multinomial counts over the original rows, so the weighted normal equations of all resamples share
the row-wise outer products z_i z_i^T. Stacking the B weight vectors into a matrix W turns B Gram matrices
into one product ``W @ [vec(z_i z_i^T)]_i`` that BLAS runs in a single call. Newton steps for logistic
regression reuse the trick with weights ``w_i p_i (1 - p_i)``.

Row 0 of ``W`` is usually all ones, so the point estimate and its bootstrap replicates come out of the same
call. The intercept (column 0 of ``Z``) is not penalised; the small ridge term on the other columns keeps
every Hessian invertible when a resample is near-separable or near-collinear.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

MAX_NEWTON_STEP = 5.0


def row_outer(Z: np.ndarray) -> np.ndarray:
    """Row-wise outer products flattened to shape ``(n, d * d)``; reused across Newton steps and folds."""
    n, d = Z.shape
    return (Z[:, :, None] * Z[:, None, :]).reshape(n, d * d)


def _penalty(d: int, lam: float) -> np.ndarray:
    penalty = lam * np.eye(d)
    penalty[0, 0] = 0.0
    return penalty


def _gram(V: np.ndarray, Z: np.ndarray, ZZ: np.ndarray | None) -> np.ndarray:
    """Weighted Gram matrices ``Z^T diag(v_b) Z`` for every row ``v_b`` of ``V``."""
    d = Z.shape[1]
    if ZZ is None:
        return np.einsum("bn,ni,nj->bij", V, Z, Z, optimize=True)
    return (V @ ZZ).reshape(len(V), d, d)


def _check(Z: np.ndarray, y: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    W = np.atleast_2d(np.asarray(W, dtype=float))
    if Z.ndim != 2 or y.shape != (Z.shape[0],) or W.shape[1] != Z.shape[0]:
        raise ValueError("shapes must be Z (n, d), y (n,) and W (B, n)")
    if (W < 0).any():
        raise ValueError("weights must be non-negative")
    return Z, y, W


def weighted_ridge(
    Z: np.ndarray, y: np.ndarray, W: np.ndarray, lam: float = 1.0, ZZ: np.ndarray | None = None
) -> np.ndarray:
    """Coefficients ``(B, d)`` minimising ``sum_i w_bi (y_i - z_i b)^2 + lam * |b[1:]|^2`` for every row b."""
    Z, y, W = _check(Z, y, W)
    gram = _gram(W, Z, ZZ) + _penalty(Z.shape[1], lam)
    rhs = (W * y) @ Z
    return np.linalg.solve(gram, rhs[..., None])[..., 0]


def weighted_logistic(
    Z: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    lam: float = 1.0,
    ZZ: np.ndarray | None = None,
    init: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Penalised logistic regression by batched Newton steps; returns coefficients ``(B, d)``.

    The objective per row is ``sum_i w_bi * loglik_i(b) - lam / 2 * |b[1:]|^2``. Steps are capped at
    ``MAX_NEWTON_STEP`` in max-norm, which keeps early iterations stable on separable resamples without
    changing the fixed point.
    """
    Z, y, W = _check(Z, y, W)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("logistic targets must be 0 or 1")
    penalty = _penalty(Z.shape[1], lam)
    beta = (
        np.zeros((len(W), Z.shape[1]))
        if init is None
        else np.array(np.broadcast_to(init, (len(W), Z.shape[1])))
    )
    for _ in range(max_iter):
        p = expit(beta @ Z.T)
        grad = (W * (y - p)) @ Z - beta @ penalty
        hess = _gram(W * p * (1.0 - p), Z, ZZ) + penalty
        step = np.linalg.solve(hess, grad[..., None])[..., 0]
        largest = np.abs(step).max(axis=1, keepdims=True)
        step *= np.minimum(1.0, MAX_NEWTON_STEP / np.maximum(largest, 1e-300))
        beta += step
        if float(largest.max()) < tol:
            return beta
    raise RuntimeError(f"logistic regression did not converge in {max_iter} Newton steps")


def predict_linear(Z: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Predictions ``(B, n)`` of a batch of linear models."""
    return np.atleast_2d(beta) @ np.asarray(Z, dtype=float).T


def predict_logistic(Z: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Probabilities ``(B, n)`` of a batch of logistic models."""
    return expit(predict_linear(Z, beta))
