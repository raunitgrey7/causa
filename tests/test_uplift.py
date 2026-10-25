"""Uplift curves, Qini and AUUC, budgeted policies and their value."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causa.uplift import (
    auuc,
    budget_policy,
    ipw_policy_value,
    policy_value,
    qini_coefficient,
    qini_curve,
    true_gain_curve,
    uplift_curve,
)


def randomised_world(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tau = rng.normal(0.0, 1.0, n)
    t = (rng.random(n) < 0.5).astype(float)
    y = 2.0 + t * tau + rng.standard_normal(n)
    return tau, t, y


def test_qini_curve_by_hand():
    score = np.array([3.0, 2.0, 1.0, 0.0])
    t = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([5.0, 1.0, 2.0, 3.0])
    curve = qini_curve(score, y, t)
    # k=1: 5 - 0; k=2: 5 - 1*1/1; k=3: 7 - 1*2/1; k=4: 7 - 4*2/2
    np.testing.assert_allclose(curve.value, np.array([0.0, 5.0, 4.0, 5.0, 3.0]) / 4)
    np.testing.assert_allclose(curve.fraction, [0, 0.25, 0.5, 0.75, 1.0])


def test_curves_start_at_zero_and_end_at_the_total_effect():
    tau, t, y = randomised_world(2000, 0)
    q, u = qini_curve(tau, y, t), uplift_curve(tau, y, t)
    assert q.value[0] == 0 and u.value[0] == 0
    total = y[t == 1].mean() - y[t == 0].mean()
    assert u.value[-1] == pytest.approx(total)


def test_perfect_ranking_matches_the_closed_form_limits():
    # tau ~ N(0, 1) with a 50/50 split: the uplift-curve area tends to E[phi(Z)] = 1 / (2 sqrt(pi)); the Qini
    # curve counts only the treated half of the top k, so its coefficient tends to half of that.
    tau, t, y = randomised_world(200_000, 1)
    assert qini_coefficient(tau, y, t) == pytest.approx(1 / (4 * np.sqrt(np.pi)), abs=0.005)
    assert auuc(tau, y, t) == pytest.approx(1 / (2 * np.sqrt(np.pi)), abs=0.01)


def test_random_ranking_has_no_qini():
    tau, t, y = randomised_world(20_000, 2)
    random_score = np.random.default_rng(2).random(len(tau))
    assert abs(qini_coefficient(random_score, y, t)) < 0.02
    assert auuc(tau, y, t) > auuc(random_score, y, t) + 0.1


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 1000), scale=st.floats(0.01, 100), shift=st.floats(-50, 50))
def test_qini_is_invariant_to_monotone_transforms_of_the_score(seed, scale, shift):
    _, t, y = randomised_world(300, seed)
    score = np.random.default_rng(seed).standard_normal(300)
    assert qini_coefficient(score, y, t) == pytest.approx(
        qini_coefficient(np.exp(score) * scale + shift, y, t)
    )


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 1000), budget=st.floats(0.05, 1.0))
def test_oracle_policy_is_at_least_as_valuable_as_any_other_with_the_same_budget(seed, budget):
    rng = np.random.default_rng(seed)
    tau = rng.normal(0.0, 1.0, 200)
    other = budget_policy(rng.standard_normal(200), budget)
    assert policy_value(budget_policy(tau, budget), tau) >= policy_value(other, tau) - 1e-12


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 1000))
def test_true_gain_area_is_largest_for_the_true_ranking(seed):
    rng = np.random.default_rng(seed)
    tau = rng.normal(0.0, 1.0, 100)
    assert true_gain_curve(tau, tau).area() >= true_gain_curve(rng.standard_normal(100), tau).area() - 1e-12


def test_budget_policy_respects_budget_cost_and_tie_order():
    score = np.array([5.0, 1.0, 1.0, -2.0, 3.0])
    mask = budget_policy(score, 0.6)
    np.testing.assert_array_equal(mask, [True, True, False, False, True])
    np.testing.assert_array_equal(budget_policy(score, 1.0, cost=2.0), [True, False, False, False, True])
    assert budget_policy(score, 0.1).sum() == 0


@pytest.mark.parametrize("budget", [0.0, 1.5, -0.1])
def test_budget_must_be_a_fraction(budget):
    with pytest.raises(ValueError):
        budget_policy(np.ones(4), budget)


def test_ipw_policy_value_is_unbiased_on_a_randomised_holdout():
    tau, t, y = randomised_world(200_000, 3)
    mask = budget_policy(tau, 0.3)
    estimate = ipw_policy_value(mask, y, t, 0.5)
    assert estimate == pytest.approx(policy_value(mask, tau), abs=0.03)


def test_policy_value_nets_out_the_contact_cost():
    tau = np.array([10.0, 5.0, -1.0])
    mask = np.array([True, True, False])
    assert policy_value(mask, tau, cost=2.0) == pytest.approx((8.0 + 3.0) / 3)


def test_invalid_inputs():
    with pytest.raises(ValueError, match="strictly between"):
        ipw_policy_value(np.ones(3, bool), np.ones(3), np.ones(3), 1.0)
    with pytest.raises(ValueError, match="same length"):
        qini_curve(np.ones(3), np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        auuc(np.array([1.0, np.nan]), np.ones(2), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="0 or 1"):
        qini_curve(np.ones(2), np.ones(2), np.array([0.0, 2.0]))


def test_curve_sampling_for_plots():
    tau, t, y = randomised_world(500, 4)
    sampled = qini_curve(tau, y, t).sample(11)
    assert len(sampled["fraction"]) == 11 and sampled["value"][0] == 0.0
