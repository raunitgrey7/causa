"""Decision reports: each verdict on a world built to produce it, rule ordering and hostile tables."""

import json

import numpy as np
import pandas as pd
import pytest

from causa.data.experiments import ExperimentConfig, expected_effects, simulate_experiment
from causa.experiment.report import (
    EffectSummary,
    ExperimentSpec,
    Guardrail,
    GuardrailResult,
    PowerSummary,
    _decide,
    analyze_experiment,
)

GUARDS = (Guardrail("support_contact", 0.01), Guardrail("page_load_ms", 25.0))


def spec(**kwargs: object) -> ExperimentSpec:
    return ExperimentSpec(name="t", mde=40.0, guardrails=GUARDS, **kwargs)


@pytest.mark.parametrize(
    ("config", "decision"),
    [
        (ExperimentConfig(n_users=60_000, lift=0.08, seed=1), "ship"),
        (ExperimentConfig(n_users=120_000, lift=0.0, seed=2), "stop_no_effect"),
        (ExperimentConfig(n_users=60_000, lift=0.08, srm_drop=0.10, seed=3), "invalid"),
        (ExperimentConfig(n_users=60_000, lift=0.08, support_harm=0.02, seed=4), "do_not_ship"),
        (ExperimentConfig(n_users=60_000, lift=-0.08, seed=5), "do_not_ship"),
        (ExperimentConfig(n_users=3_000, lift=0.02, seed=6), "keep_running"),
    ],
)
def test_each_verdict_on_a_world_built_to_produce_it(config, decision):
    report = analyze_experiment(simulate_experiment(config), spec())
    assert report.decision == decision, report.reasons


def test_cuped_estimate_is_close_to_the_truth_and_removes_variance(experiment_frame):
    report = analyze_experiment(experiment_frame, spec())
    truth = expected_effects(ExperimentConfig(n_users=20_000, lift=0.08))["revenue"]
    assert report.primary.method == "cuped"
    assert abs(report.primary.effect - truth) < 3 * report.primary.se
    assert report.primary.se < report.unadjusted.se
    assert 0.2 < report.power.variance_reduction < 0.5


def test_always_valid_interval_is_wider_than_the_fixed_one(experiment_frame):
    p = analyze_experiment(experiment_frame, spec()).primary
    assert p.always_valid_ci[0] <= p.fixed_ci[0] and p.always_valid_ci[1] >= p.fixed_ci[1]
    assert p.always_valid_p_value >= p.fixed_p_value


def test_looks_follow_the_days_and_end_at_the_full_sample(experiment_frame):
    report = analyze_experiment(experiment_frame, spec())
    days = [look["day"] for look in report.looks]
    assert days == sorted(set(experiment_frame["day"].astype(float)))
    last = report.looks[-1]
    assert last["n_control"] + last["n_treatment"] == len(experiment_frame)


def test_report_is_deterministic(experiment_frame):
    assert (
        analyze_experiment(experiment_frame, spec()).to_dict()
        == analyze_experiment(experiment_frame, spec()).to_dict()
    )


def test_without_a_day_column_the_data_is_one_look(experiment_frame):
    report = analyze_experiment(experiment_frame.drop(columns=["day"]), spec(day_column=None))
    assert len(report.looks) == 1
    assert any("single look" in w for w in report.warnings)


def test_without_a_covariate_the_primary_is_the_difference_in_means(experiment_frame):
    report = analyze_experiment(experiment_frame, spec(covariate=None))
    assert report.primary.method == "difference_in_means"
    assert report.primary.effect == pytest.approx(report.unadjusted.effect)


def test_a_constant_metric_does_not_crash_or_ship():
    frame = pd.DataFrame({"variant": ["control", "treatment"] * 50, "revenue": 5.0, "day": 1})
    report = analyze_experiment(frame, ExperimentSpec(covariate=None, mde=1.0))
    assert report.primary.effect == 0.0 and report.decision != "ship"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda f: f.drop(columns=["revenue"]), "missing columns"),
        (lambda f: f.assign(variant=np.where(f.index % 7 == 0, "holdout", f.variant)), "unexpected variant"),
        (lambda f: f.head(30), "at least"),
        (lambda f: f.assign(revenue=f.revenue.where(f.index != 3)), "missing or non-numeric"),
        (
            lambda f: f.assign(pre_revenue=f.pre_revenue.astype(object).where(f.index != 5, "lots")),
            "non-numeric",
        ),
        (lambda f: f.assign(revenue=f.revenue.where(f.index != 8, np.inf)), "missing or non-numeric"),
    ],
)
def test_hostile_tables_are_rejected_with_specific_messages(experiment_frame, mutate, match):
    with pytest.raises(ValueError, match=match):
        analyze_experiment(mutate(experiment_frame.copy()), spec())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"treatment_share": 1.0},
        {"alpha": 0.6},
        {"mde": -1.0},
        {"control": "treatment"},
        {"mde_relative": 0.0},
    ],
)
def test_spec_validation(kwargs):
    with pytest.raises(ValueError):
        ExperimentSpec(**kwargs)


def test_guardrail_margin_must_be_positive():
    with pytest.raises(ValueError):
        Guardrail("x", 0.0)


def _summary(effect: float, lo: float, hi: float, p: float) -> EffectSummary:
    return EffectSummary("cuped", 100.0, 100.0 + effect, effect, 1.0, effect / 100, (lo, hi), p, (lo, hi), p)


POWER = PowerSummary(10.0, 50.0, 60.0, 0.3, 1000, 1000, 0.5, 4000, 12.0)


def test_rule_order_srm_beats_everything_and_guardrail_failure_beats_a_win():
    win = _summary(20.0, 5.0, 35.0, 0.001)
    failing = [GuardrailResult("g", 0.01, 0.05, (0.02, 0.08), "fail")]
    assert _decide(win, POWER, 1e-6, failing, ExperimentSpec())[0] == "invalid"
    assert _decide(win, POWER, 0.5, failing, ExperimentSpec())[0] == "do_not_ship"
    unclear = [GuardrailResult("g", 0.01, 0.0, (-0.01, 0.02), "inconclusive")]
    assert _decide(win, POWER, 0.5, unclear, ExperimentSpec())[0] == "keep_running"
    assert _decide(win, POWER, 0.5, [], ExperimentSpec())[0] == "ship"


def test_futility_needs_the_interval_inside_the_mde_band():
    inside = _summary(1.0, -8.0, 9.0, 0.6)
    straddling = _summary(1.0, -8.0, 12.0, 0.6)
    assert _decide(inside, POWER, 0.5, [], ExperimentSpec())[0] == "stop_no_effect"
    decision, reasons = _decide(straddling, POWER, 0.5, [], ExperimentSpec())
    assert decision == "keep_running" and "3000 more users" in reasons[0]
    reached = PowerSummary(10.0, 50.0, 60.0, 0.3, 5000, 5000, 0.9, 4000, 8.0)
    assert "planned sample reached" in _decide(straddling, reached, 0.5, [], ExperimentSpec())[1][0]


def test_a_borderline_logging_fault_is_caught_by_the_combined_srm_rule():
    caught = 0
    for seed in (3, 13, 23):
        frame = simulate_experiment(ExperimentConfig(n_users=100_000, lift=0.08, srm_drop=0.05, seed=seed))
        caught += analyze_experiment(frame, spec()).decision == "invalid"
    assert caught == 3


def test_report_serialises_to_plain_json(experiment_frame):
    payload = json.dumps(analyze_experiment(experiment_frame, spec()).to_dict(), allow_nan=False)
    assert '"decision"' in payload
