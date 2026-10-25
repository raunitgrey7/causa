"""Targeting model: holdout-based selection, budgeted decisions and persistence."""

import joblib
import numpy as np
import pytest

from causa.data.campaign import simulate_customers
from causa.learners import LEARNERS
from causa.targeting import CURVE_BUDGETS, load_model, save_model


def test_model_keeps_the_learner_with_the_best_holdout_qini(small_model):
    assert set(small_model.validation) == set(LEARNERS)
    best = max(small_model.validation, key=lambda k: small_model.validation[k]["qini"])
    assert small_model.learner_name == best


def test_budget_curve_has_one_estimate_per_budget(small_model):
    curve = small_model.budget_curve
    assert curve["budgets"] == list(CURVE_BUDGETS)
    assert len(curve["ipw_value_per_customer"]) == len(CURVE_BUDGETS)


@pytest.mark.parametrize("budget", [0.05, 0.2, 1.0])
def test_decisions_respect_the_budget(small_model, budget):
    customers = simulate_customers(400, np.random.default_rng(1))
    decision = small_model.decide(customers, budget)
    assert decision.treat.sum() <= int(budget * 400)
    assert (decision.uplift[decision.treat] > 0).all()
    expected = float(np.sum(decision.uplift[decision.treat]))
    assert decision.expected_incremental_revenue == pytest.approx(expected)


def test_contact_cost_shrinks_the_selection(small_model):
    customers = simulate_customers(400, np.random.default_rng(2))
    free = small_model.decide(customers, 1.0, cost=0.0).treat.sum()
    costly = small_model.decide(customers, 1.0, cost=50.0).treat.sum()
    assert costly <= free


def test_save_and_load_round_trip(small_model, tmp_path):
    path = tmp_path / "m.joblib"
    save_model(small_model, path)
    loaded = load_model(path)
    customers = simulate_customers(50, np.random.default_rng(3))
    np.testing.assert_array_equal(loaded.predict(customers), small_model.predict(customers))


def test_load_refuses_files_that_are_not_models(tmp_path):
    path = tmp_path / "not_a_model.joblib"
    joblib.dump({"learner": "fake"}, path)
    with pytest.raises(TypeError):
        load_model(path)
