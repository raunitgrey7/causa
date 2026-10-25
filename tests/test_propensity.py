"""Propensity model quality and overlap diagnostics."""

import numpy as np
import pytest

from causa.data.campaign import CampaignConfig, simulate_campaign
from causa.features import FEATURE_NAMES, feature_matrix
from causa.propensity import (
    PropensityModel,
    effective_sample_size,
    ipw_weights,
    overlap_report,
    standardized_mean_differences,
)


def test_correctly_specified_model_converges_to_the_true_propensity():
    errors = []
    for n in (4000, 20_000):
        data = simulate_campaign(CampaignConfig(n=n, seed=11))
        X = feature_matrix(data.customers)
        e = PropensityModel.fit(X, data.treatment).predict(X)
        assert np.corrcoef(e, data.propensity)[0, 1] > 0.97
        errors.append(float(np.mean(np.abs(e - data.propensity))))
    assert errors[0] < 0.05
    assert errors[1] < 0.6 * errors[0]


def test_predictions_are_clipped():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 2))
    t = (X[:, 0] > 0).astype(int)
    e = PropensityModel.fit(X, t, clip=0.05).predict(X)
    assert e.min() >= 0.05 and e.max() <= 0.95


def test_effective_sample_size_of_equal_and_degenerate_weights():
    assert effective_sample_size(np.ones(50)) == pytest.approx(50.0)
    assert effective_sample_size(np.r_[1e6, np.ones(99)]) == pytest.approx(1.0, abs=1e-3)
    assert effective_sample_size(np.zeros(3)) == 0.0


def test_smd_of_a_known_shift():
    X = np.r_[np.zeros(100), np.ones(100)][:, None] + np.tile([-1.0, 1.0], 100)[:, None]
    t = np.r_[np.zeros(100), np.ones(100)].astype(int)
    assert standardized_mean_differences(X, t)[0] == pytest.approx(1.0 / np.sqrt(np.var(X[:100], ddof=1)))


def test_true_inverse_propensity_weights_balance_covariates():
    data = simulate_campaign(CampaignConfig(n=40_000, seed=3))
    X = feature_matrix(data.customers)
    before = np.abs(standardized_mean_differences(X, data.treatment)).max()
    after = np.abs(
        standardized_mean_differences(X, data.treatment, ipw_weights(data.treatment, data.propensity))
    ).max()
    assert before > 0.5 and after < 0.05


def test_weak_overlap_raises_warnings_and_randomised_data_does_not():
    weak = simulate_campaign(CampaignConfig(n=6000, targeting_strength=2.2, seed=5))
    X = feature_matrix(weak.customers)
    report = overlap_report(
        X, weak.treatment, PropensityModel.fit(X, weak.treatment).predict(X), FEATURE_NAMES
    )
    assert report.share_outside_band > 0.05
    assert any("weak overlap" in w for w in report.warnings)
    rand = simulate_campaign(CampaignConfig(n=6000, randomized=True, seed=5))
    Xr = feature_matrix(rand.customers)
    clean = overlap_report(
        Xr, rand.treatment, PropensityModel.fit(Xr, rand.treatment).predict(Xr), FEATURE_NAMES
    )
    assert clean.warnings == [] and clean.auc < 0.6


def test_report_lists_every_feature_and_serialises(campaign):
    X = feature_matrix(campaign.customers)
    report = overlap_report(X, campaign.treatment, campaign.propensity, FEATURE_NAMES)
    d = report.to_dict()
    assert [row["feature"] for row in d["smd"]] == list(FEATURE_NAMES)
    assert d["n_treated"] + d["n_control"] == len(campaign)


def test_clip_must_leave_a_valid_interval():
    with pytest.raises(ValueError):
        PropensityModel.fit(np.ones((10, 1)), np.r_[np.zeros(5), np.ones(5)], clip=0.5)
