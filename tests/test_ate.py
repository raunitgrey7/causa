"""ATE estimators: closed-form identities, recovery on a known world, determinism and hostile inputs."""

import numpy as np
import pytest

from causa.ate import METHODS, ATEConfig, Estimate, _aipw_score, bootstrap_weights, estimate_ate
from causa.data.campaign import CampaignConfig, population_ate, simulate_campaign
from causa.features import feature_matrix
from tests.conftest import linear_world


def test_bootstrap_weights_row_zero_is_the_sample_and_rows_are_resamples():
    W = bootstrap_weights(50, 30, np.random.default_rng(0))
    assert W.shape == (31, 50)
    np.testing.assert_array_equal(W[0], 1.0)
    np.testing.assert_array_equal(W.sum(axis=1), 50.0)
    assert (W >= 0).all() and np.allclose(W, np.round(W))


def test_difference_in_means_point_estimate_is_the_closed_form():
    w = linear_world(1000, 1)
    est = estimate_ate(w["X"], w["t"], w["y"], ATEConfig(n_boot=50))
    expected = w["y"][w["t"] == 1].mean() - w["y"][w["t"] == 0].mean()
    assert est.estimates["difference_in_means"].estimate == pytest.approx(expected)


def test_randomised_linear_world_every_estimator_recovers_the_effect():
    w = linear_world(4000, 2)
    result = estimate_ate(w["X"], w["t"], w["y"], ATEConfig(n_boot=100))
    truth = float(w["tau"].mean())
    for name, est in result.estimates.items():
        assert abs(est.estimate - truth) < 4 * est.se + 0.02, name
        assert est.ci_low < est.estimate < est.ci_high


def test_confounded_world_naive_is_biased_and_doubly_robust_is_not():
    w = linear_world(6000, 3, randomized=False)
    result = estimate_ate(w["X"], w["t"], w["y"], ATEConfig(n_boot=100))
    truth = float(w["tau"].mean())
    naive, aipw = result.estimates["difference_in_means"], result.estimates["aipw"]
    assert naive.estimate - truth > 5 * naive.se
    assert aipw.covers(truth)


def test_aipw_score_with_true_nuisances_is_unbiased():
    rng = np.random.default_rng(4)
    n = 200_000
    x = rng.standard_normal(n)
    e = 1 / (1 + np.exp(-x))
    t = (rng.random(n) < e).astype(float)
    mu0, mu1 = x, x + 2.0
    y = np.where(t == 1, mu1, mu0) + rng.standard_normal(n)
    phi = _aipw_score(t, y, e, mu0, mu1)
    assert abs(phi.mean() - 2.0) < 4 * phi.std() / np.sqrt(n)


def test_campaign_estimate_covers_the_population_ate(campaign):
    truth = population_ate(CampaignConfig(), n_draws=400_000)["ate"]
    result = estimate_ate(feature_matrix(campaign.customers), campaign.treatment, campaign.revenue)
    assert result.estimates["aipw"].covers(truth)
    assert not result.estimates["difference_in_means"].covers(truth)
    assert set(result.estimates) == set(METHODS)


def test_same_seed_same_estimates_and_intervals(campaign):
    X = feature_matrix(campaign.customers)
    a = estimate_ate(X, campaign.treatment, campaign.revenue, ATEConfig(n_boot=40, seed=5))
    b = estimate_ate(X, campaign.treatment, campaign.revenue, ATEConfig(n_boot=40, seed=5))
    assert a.estimates == b.estimates


def test_full_sample_point_estimates_do_not_depend_on_row_order(campaign):
    X = feature_matrix(campaign.customers)
    order = np.random.default_rng(6).permutation(len(X))
    a = estimate_ate(X, campaign.treatment, campaign.revenue, ATEConfig(n_boot=20), diagnostics=False)
    b = estimate_ate(
        X[order], campaign.treatment[order], campaign.revenue[order], ATEConfig(n_boot=20), diagnostics=False
    )
    for name in ("difference_in_means", "outcome_regression", "ipw", "aipw"):
        assert a.estimates[name].estimate == pytest.approx(b.estimates[name].estimate, rel=1e-8, abs=1e-8)


def test_near_separable_assignment_still_gives_finite_answers_and_warns():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((1500, 2))
    t = (X[:, 0] + 0.05 * rng.standard_normal(1500) > 0).astype(int)
    y = X[:, 0] + t + rng.standard_normal(1500)
    result = estimate_ate(X, t, y, ATEConfig(n_boot=30))
    assert all(np.isfinite([e.estimate, e.ci_low, e.ci_high]).all() for e in result.estimates.values())
    assert result.overlap is not None and result.overlap.warnings


def test_huge_outcome_scale_does_not_break_the_solvers():
    w = linear_world(1000, 8)
    result = estimate_ate(w["X"], w["t"], w["y"] * 1e9, ATEConfig(n_boot=30))
    assert result.estimates["aipw"].estimate == pytest.approx(result.estimates["aipw_wald"].estimate)
    assert np.isfinite(result.estimates["aipw"].se)


@pytest.mark.parametrize(
    ("t", "y", "match"),
    [
        (np.r_[np.zeros(50), np.full(50, 2)], np.zeros(100), "0 or 1"),
        (np.r_[np.zeros(95), np.ones(5)], np.zeros(100), "at least"),
        (np.r_[np.zeros(50), np.ones(50)], np.r_[np.zeros(99), np.nan], "finite"),
    ],
)
def test_invalid_inputs_are_rejected_with_specific_messages(t, y, match):
    with pytest.raises(ValueError, match=match):
        estimate_ate(np.random.default_rng(0).standard_normal((100, 2)), t, y)


@pytest.mark.parametrize(
    "kwargs", [{"n_boot": 5}, {"folds": 1}, {"alpha": 1.5}, {"ridge": 0.0}, {"clip": 0.5}]
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        ATEConfig(**kwargs)


def test_estimate_helpers():
    est = Estimate("x", 1.0, 0.5, 0.0, 2.0, "wald")
    assert est.covers(1.9) and not est.covers(2.1)
    assert est.width == 2.0


def test_weak_overlap_campaign_is_flagged():
    data = simulate_campaign(CampaignConfig(n=3000, targeting_strength=1.8, seed=9))
    result = estimate_ate(feature_matrix(data.customers), data.treatment, data.revenue, ATEConfig(n_boot=20))
    assert result.overlap is not None
    assert result.overlap.share_outside_band > 0.1
    assert result.to_dict()["overlap"]["warnings"]
