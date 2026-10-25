"""The campaign generator: determinism, closed-form ground truth, and the confounding it is built to have."""

import numpy as np
import pandas as pd
import pytest

from causa.data.campaign import (
    COVARIATES,
    CampaignConfig,
    population_ate,
    potential_outcomes,
    simulate_campaign,
    simulate_customers,
    true_propensity,
)


def test_same_seed_gives_identical_campaigns():
    a, b = simulate_campaign(CampaignConfig(n=500, seed=9)), simulate_campaign(CampaignConfig(n=500, seed=9))
    pd.testing.assert_frame_equal(a.customers, b.customers)
    np.testing.assert_array_equal(a.treatment, b.treatment)
    np.testing.assert_array_equal(a.revenue, b.revenue)


def test_different_seeds_differ():
    a, b = simulate_campaign(CampaignConfig(n=500, seed=1)), simulate_campaign(CampaignConfig(n=500, seed=2))
    assert not np.array_equal(a.revenue, b.revenue)


def test_customer_table_has_documented_columns_and_ranges():
    c = simulate_customers(5000, np.random.default_rng(0))
    assert tuple(c.columns) == COVARIATES
    assert c["email_open_rate"].between(0, 1).all() and c["discount_affinity"].between(0, 1).all()
    assert (c["recency_days"] >= 1).all() and (c["avg_order_value"] > 0).all()
    assert set(c["city_tier"].unique()) <= {1, 2, 3}


def test_cate_is_closed_form_difference_of_potential_outcome_means(campaign):
    truth = campaign.truth
    np.testing.assert_allclose(campaign.cate, truth.p1 * truth.v1 - truth.p0 * truth.v0)
    assert ((truth.p0 > 0) & (truth.p0 < 1) & (truth.p1 > 0) & (truth.p1 < 1)).all()


def test_coupon_cuts_price_and_lifts_basket_by_configured_amounts(campaign):
    cfg = CampaignConfig()
    np.testing.assert_allclose(
        campaign.truth.v1, campaign.truth.v0 * (1 - cfg.discount) * (1 + cfg.basket_lift)
    )


def test_revenue_is_zero_exactly_when_customer_did_not_buy(campaign):
    assert ((campaign.revenue > 0) == (campaign.converted == 1)).all()
    assert (campaign.revenue >= 0).all()


def test_effects_have_both_signs_so_targeting_matters(campaign):
    share_negative = float(np.mean(campaign.cate < 0))
    assert 0.2 < share_negative < 0.7


def test_historical_assignment_is_confounded():
    data = simulate_campaign(CampaignConfig(n=60_000, seed=4))
    naive = data.revenue[data.treatment == 1].mean() - data.revenue[data.treatment == 0].mean()
    assert naive > data.cate.mean() + 100.0


def test_randomized_design_removes_the_confounding():
    data = simulate_campaign(CampaignConfig(n=200_000, seed=4, randomized=True))
    naive = data.revenue[data.treatment == 1].mean() - data.revenue[data.treatment == 0].mean()
    se = np.sqrt(data.revenue[data.treatment == 1].var() / (data.treatment == 1).sum() * 4)
    assert abs(naive - data.cate.mean()) < 4 * se
    assert np.all(data.propensity == 0.5)


def test_zero_targeting_strength_means_coin_flip_assignment():
    c = simulate_customers(1000, np.random.default_rng(1))
    np.testing.assert_allclose(true_propensity(c, strength=0.0), 0.5)


def test_stronger_targeting_pushes_propensities_toward_the_edges():
    c = simulate_customers(20_000, np.random.default_rng(2))
    weak, strong = true_propensity(c, 1.0), true_propensity(c, 1.8)
    outside = lambda e: np.mean((e < 0.05) | (e > 0.95))  # noqa: E731
    assert outside(strong) > 3 * outside(weak)


def test_population_ate_is_precise_and_independent_of_targeting():
    a = population_ate(CampaignConfig(targeting_strength=1.0), n_draws=400_000)
    b = population_ate(CampaignConfig(targeting_strength=1.8), n_draws=400_000)
    assert a["ate"] == b["ate"]
    assert a["mc_se"] < 0.5 and a["draws"] == 400_000


def test_potential_outcomes_do_not_depend_on_the_draw_of_treatment(campaign):
    again = potential_outcomes(campaign.customers, CampaignConfig())
    np.testing.assert_allclose(again.cate, campaign.cate)


@pytest.mark.parametrize(
    "kwargs", [{"n": 0}, {"targeting_strength": -1.0}, {"discount": 1.0}, {"basket_shape": 0.0}]
)
def test_config_rejects_degenerate_worlds(kwargs):
    with pytest.raises(ValueError):
        CampaignConfig(**kwargs)
