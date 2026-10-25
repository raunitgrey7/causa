"""CATE meta-learners on a world whose effect is known in closed form."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from causa.learners import LEARNERS, DRLearner, make_learner
from tests.conftest import linear_world


def linear(seed: int) -> LinearRegression:
    return LinearRegression()


@pytest.fixture(scope="module")
def world() -> dict[str, np.ndarray]:
    return linear_world(6000, 21, randomized=False)


@pytest.mark.parametrize("name", ["t_learner", "x_learner", "dr_learner"])
def test_learners_with_linear_base_models_recover_a_linear_cate(world, name):
    learner = make_learner(name, seed=0, regressor=linear).fit(world["X"], world["t"], world["y"])
    pred = learner.predict(world["X"])
    pehe = np.sqrt(np.mean((pred - world["tau"]) ** 2))
    assert pehe < 0.2, pehe


def test_s_learner_with_a_linear_base_model_can_only_find_the_average(world):
    pred = (
        make_learner("s_learner", regressor=linear)
        .fit(world["X"], world["t"], world["y"])
        .predict(world["X"])
    )
    assert np.ptp(pred) < 1e-9
    assert abs(pred.mean() - world["tau"].mean()) < 0.15


@pytest.mark.parametrize("name", LEARNERS)
def test_boosted_learners_rank_customers_by_their_true_effect(world, name):
    pred = make_learner(name, seed=1).fit(world["X"], world["t"], world["y"]).predict(world["X"])
    assert np.corrcoef(pred, world["tau"])[0, 1] > 0.85


def test_dr_pseudo_outcomes_average_to_the_ate(world):
    learner = DRLearner(regressor=linear).fit(world["X"], world["t"], world["y"])
    assert learner.pseudo_outcomes_ is not None
    se = learner.pseudo_outcomes_.std() / np.sqrt(len(world["y"]))
    assert abs(learner.pseudo_outcomes_.mean() - world["tau"].mean()) < 4 * se


@pytest.mark.parametrize("name", LEARNERS)
def test_same_seed_same_predictions(world, name):
    X, t, y = world["X"][:2000], world["t"][:2000], world["y"][:2000]
    a = make_learner(name, seed=3).fit(X, t, y).predict(X)
    b = make_learner(name, seed=3).fit(X, t, y).predict(X)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", LEARNERS)
def test_predict_before_fit_is_an_error(name):
    with pytest.raises(RuntimeError, match="fit"):
        make_learner(name).predict(np.zeros((2, 3)))


def test_unknown_learner_name():
    with pytest.raises(ValueError, match="unknown learner"):
        make_learner("r_learner")


def test_single_arm_and_non_finite_inputs_are_rejected():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError, match="both arms"):
        make_learner("t_learner").fit(X, np.ones(10), np.zeros(10))
    with pytest.raises(ValueError, match="finite"):
        make_learner("t_learner").fit(X, np.r_[np.zeros(5), np.ones(5)], np.r_[np.zeros(9), np.inf])
