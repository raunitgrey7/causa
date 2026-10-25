"""The A/B generator: closed-form effects, the SRM fault and determinism."""

import numpy as np
import pandas as pd
import pytest

from causa.data.experiments import ExperimentConfig, expected_effects, simulate_arrays, simulate_experiment


def test_same_seed_gives_identical_experiments():
    cfg = ExperimentConfig(n_users=2000, seed=3)
    pd.testing.assert_frame_equal(simulate_experiment(cfg), simulate_experiment(cfg))


def test_closed_form_control_mean_and_conversion_match_simulation():
    cfg = ExperimentConfig(n_users=400_000, lift=0.1, seed=1)
    frame = simulate_experiment(cfg)
    truth = expected_effects(cfg)
    control = frame[frame.variant == "control"]
    treated = frame[frame.variant == "treatment"]
    se = control.revenue.std() / np.sqrt(len(control))
    assert abs(control.revenue.mean() - truth["revenue_control_mean"]) < 4 * se
    assert abs(control.converted.mean() - truth["converted_control"]) < 0.004
    effect_se = np.sqrt(control.revenue.var() / len(control) + treated.revenue.var() / len(treated))
    assert abs(treated.revenue.mean() - control.revenue.mean() - truth["revenue"]) < 4 * effect_se


def test_pre_period_revenue_is_correlated_with_the_outcome():
    frame = simulate_experiment(ExperimentConfig(n_users=50_000, seed=2))
    assert np.corrcoef(frame.pre_revenue, frame.revenue)[0, 1] > 0.45


def test_srm_fault_drops_only_non_converting_treatment_users():
    clean = simulate_arrays(ExperimentConfig(n_users=50_000, seed=4), 1, np.random.default_rng(4))
    faulty = simulate_arrays(
        ExperimentConfig(n_users=50_000, srm_drop=0.2, seed=4), 1, np.random.default_rng(4)
    )
    assert clean.logged.all()
    lost = ~faulty.logged[0]
    assert lost.any()
    assert faulty.treated[0][lost].all()
    assert not faulty.converted[0][lost].any()


def test_frame_drops_lost_users_and_is_sorted_by_day():
    frame = simulate_experiment(ExperimentConfig(n_users=10_000, srm_drop=0.3, seed=6))
    assert len(frame) < 10_000
    assert frame["day"].is_monotonic_increasing
    assert set(frame.columns) >= {
        "variant",
        "day",
        "revenue",
        "pre_revenue",
        "support_contact",
        "page_load_ms",
    }


def test_vectorised_arrays_have_one_row_per_replication():
    arrays = simulate_arrays(ExperimentConfig(n_users=300), 7, np.random.default_rng(0))
    assert arrays.revenue.shape == (7, 300) and arrays.treated.dtype == bool


@pytest.mark.parametrize(
    "kwargs", [{"n_users": 1}, {"split": 1.0}, {"lift": -1.0}, {"srm_drop": 1.0}, {"support_harm": 0.99}]
)
def test_config_rejects_impossible_settings(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)
