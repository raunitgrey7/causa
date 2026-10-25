"""Mixture SPRT and the sequential SRM test."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import betaln

from causa.experiment.sequential import MixtureSPRT, sequential_srm_p_values


def test_likelihood_ratio_closed_form():
    sprt = MixtureSPRT(alpha=0.05, tau=2.0)
    v, d = 0.5, 1.3
    expected = np.sqrt(v / (v + 4.0)) * np.exp(4.0 * d**2 / (2 * v * (v + 4.0)))
    assert float(np.exp(sprt.log_likelihood_ratio(d, v))) == pytest.approx(expected)


def test_confidence_sequence_boundary_is_where_the_test_rejects():
    sprt = MixtureSPRT(alpha=0.05, tau=1.5)
    v = 0.3
    r = float(sprt.radius(v))
    assert float(sprt.log_likelihood_ratio(r, v)) == pytest.approx(np.log(1 / 0.05))


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), looks=st.integers(2, 30))
def test_p_values_never_increase_and_intervals_never_widen(seed, looks):
    rng = np.random.default_rng(seed)
    effect = rng.standard_normal(looks).cumsum() / np.arange(1, looks + 1)
    variance = 1.0 / np.arange(1, looks + 1)
    sprt = MixtureSPRT(tau=0.5)
    p = sprt.p_values(effect, variance)
    lo, hi = sprt.confidence_sequence(effect, variance)
    assert (np.diff(p) <= 1e-15).all() and ((p > 0) & (p <= 1)).all()
    assert (np.diff(lo) >= 0).all() and (np.diff(hi) <= 0).all()


def test_false_positive_rate_under_continuous_monitoring_is_controlled():
    rng = np.random.default_rng(0)
    reps, looks, per_look = 4000, 50, 20
    steps = rng.standard_normal((reps, looks, per_look)).sum(axis=2)
    n = per_look * np.arange(1, looks + 1)
    effect = steps.cumsum(axis=1) / n
    variance = 1.0 / n
    p = MixtureSPRT(alpha=0.05, tau=0.1).p_values(effect, variance)
    naive = np.abs(effect / np.sqrt(variance)) > 1.959964
    assert (p[:, -1] < 0.05).mean() <= 0.05
    assert naive.any(axis=1).mean() > 0.15


def test_sprt_rejects_settings_without_a_test():
    with pytest.raises(ValueError):
        MixtureSPRT(alpha=0.0)
    with pytest.raises(ValueError):
        MixtureSPRT(tau=0.0)


def test_sequential_srm_bayes_factor_is_the_beta_binomial_ratio():
    counts = np.array([[480.0, 520.0]])
    c = 1000.0
    log_bf = betaln(c / 2 + 480, c / 2 + 520) - betaln(c / 2, c / 2) - 1000 * np.log(0.5)
    p = sequential_srm_p_values(counts, [0.5, 0.5], concentration=c)
    assert float(p[0]) == pytest.approx(min(1.0, np.exp(-log_bf)))


def test_sequential_srm_flags_gross_mismatch_and_not_balance():
    balanced = np.cumsum(np.full((14, 2), 5000.0), axis=0)
    skewed = np.cumsum(np.tile([5200.0, 4800.0], (14, 1)), axis=0)
    assert sequential_srm_p_values(balanced, [0.5, 0.5])[-1] == 1.0
    assert sequential_srm_p_values(skewed, [0.5, 0.5])[-1] < 1e-6


def test_sequential_srm_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        sequential_srm_p_values(np.array([[1.0, 2.0]]), [0.6, 0.6])
    with pytest.raises(ValueError):
        sequential_srm_p_values(np.array([[-1.0, 2.0]]), [0.5, 0.5])
    with pytest.raises(ValueError):
        sequential_srm_p_values(np.array([[1.0, 2.0, 3.0]]), [0.5, 0.5])
