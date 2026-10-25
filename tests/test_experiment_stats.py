"""Difference in means and CUPED from sufficient statistics, checked against direct numpy formulas."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causa.experiment.stats import ArmSums, Contrast, cuped, difference_in_means


def arms(y: np.ndarray, x: np.ndarray, t: np.ndarray) -> tuple[ArmSums, ArmSums]:
    return ArmSums.from_masked(y, x, t == 0), ArmSums.from_masked(y, x, t == 1)


def sample(n: int, seed: int, rho: float = 0.6, effect: float = 0.3) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    t = (rng.random(n) < 0.5).astype(int)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n) + effect * t
    return y, x, t


def test_difference_in_means_and_welch_variance_match_numpy():
    y, x, t = sample(1000, 0)
    c = difference_in_means(*arms(y, x, t))
    assert float(c.effect) == pytest.approx(y[t == 1].mean() - y[t == 0].mean())
    welch = y[t == 1].var(ddof=1) / (t == 1).sum() + y[t == 0].var(ddof=1) / (t == 0).sum()
    assert float(c.variance) == pytest.approx(welch)


def test_cuped_matches_the_textbook_formula():
    y, x, t = sample(2000, 1)
    contrast, theta = cuped(*arms(y, x, t))
    slope = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    assert float(theta) == pytest.approx(slope)
    adjusted = y - slope * (x - x.mean())
    assert float(contrast.effect) == pytest.approx(adjusted[t == 1].mean() - adjusted[t == 0].mean())
    expected_var = (
        adjusted[t == 1].var(ddof=1) / (t == 1).sum() + adjusted[t == 0].var(ddof=1) / (t == 0).sum()
    )
    assert float(contrast.variance) == pytest.approx(expected_var, rel=1e-9)


def test_cuped_removes_rho_squared_of_the_variance():
    y, x, t = sample(400_000, 2, rho=0.7)
    raw = difference_in_means(*arms(y, x, t))
    adjusted, _ = cuped(*arms(y, x, t))
    assert 1 - float(adjusted.variance / raw.variance) == pytest.approx(0.49, abs=0.01)


def test_unrelated_covariate_leaves_the_estimate_alone():
    y, x, t = sample(100_000, 3, rho=0.0)
    raw = difference_in_means(*arms(y, x, t))
    adjusted, theta = cuped(*arms(y, x, t))
    assert abs(float(theta)) < 0.02
    assert float(adjusted.effect) == pytest.approx(float(raw.effect), abs=0.01)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), shift=st.floats(-1e3, 1e3), scale=st.floats(0.1, 10.0))
def test_cuped_is_invariant_to_affine_rescaling_of_the_covariate(seed, shift, scale):
    y, x, t = sample(500, seed)
    a, _ = cuped(*arms(y, x, t))
    b, _ = cuped(*arms(y, scale * x + shift, t))
    assert float(a.effect) == pytest.approx(float(b.effect), rel=1e-6, abs=1e-8)


def test_constant_covariate_falls_back_to_difference_in_means():
    y, _, t = sample(300, 4)
    adjusted, theta = cuped(*arms(y, np.full(300, 7.0), t))
    assert float(theta) == 0.0
    assert float(adjusted.effect) == pytest.approx(float(difference_in_means(*arms(y, None, t)).effect))


def test_cumulative_sums_equal_sums_of_prefixes():
    rng = np.random.default_rng(5)
    y = rng.standard_normal((2, 6, 10))
    increments = ArmSums.from_masked(y, None, np.ones_like(y, dtype=bool))
    running = increments.cumulative()
    np.testing.assert_allclose(running.y[:, 3], y[:, :4].sum(axis=(1, 2)))
    np.testing.assert_allclose(running.n[:, -1], 60.0)


def test_relative_effect_uses_the_delta_method():
    c = Contrast(np.array(10.0), np.array(12.0), np.array(0.25), np.array(0.36))
    rel, _, hi = c.relative(0.05)
    var = 0.36 / 100 + 144 * 0.25 / 10_000
    assert float(rel) == pytest.approx(0.2)
    assert float(hi - rel) == pytest.approx(1.959964 * np.sqrt(var), rel=1e-5)
    nan_rel = Contrast(np.array(0.0), np.array(1.0), np.array(0.1), np.array(0.1)).relative(0.05)[0]
    assert np.isnan(nan_rel)


def test_zero_variance_contrasts_give_degenerate_but_valid_p_values():
    same = Contrast(np.array(3.0), np.array(3.0), np.array(0.0), np.array(0.0))
    differ = Contrast(np.array(3.0), np.array(4.0), np.array(0.0), np.array(0.0))
    assert float(same.p_value()) == 1.0 and float(differ.p_value()) == 0.0
