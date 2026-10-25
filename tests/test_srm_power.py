"""Chi-square SRM test and power calculations against textbook values."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import chisquare

from causa.experiment.power import minimum_detectable_effect, power_two_sample, sample_size_per_arm
from causa.experiment.srm import srm_test


def test_chi_square_matches_scipy():
    result = srm_test([50_400, 49_600], [0.5, 0.5])
    ref = chisquare([50_400, 49_600], [50_000, 50_000])
    assert result.statistic == pytest.approx(ref.statistic)
    assert result.p_value == pytest.approx(ref.pvalue)


def test_mismatch_is_flagged_only_when_the_split_is_off():
    assert srm_test([50_900, 49_100], [0.5, 0.5]).mismatch
    assert not srm_test([50_050, 49_950], [0.5, 0.5]).mismatch
    assert not srm_test([30_100, 69_900], [0.3, 0.7]).mismatch


@pytest.mark.parametrize(
    ("counts", "expected"), [([1, 2], [0.5, 0.6]), ([0, 0], [0.5, 0.5]), ([1], [1.0]), ([-1, 5], [0.5, 0.5])]
)
def test_srm_rejects_invalid_inputs(counts, expected):
    with pytest.raises(ValueError):
        srm_test(counts, expected)


def test_sample_size_matches_cohen_for_a_small_effect():
    # d = 0.2 at 80% power, two-sided alpha 0.05: the normal approximation gives 393 per arm.
    assert sample_size_per_arm(0.2, 1.0, 0.8, 0.05) == 393


def test_power_at_the_required_sample_size_reaches_the_target():
    n = sample_size_per_arm(3.0, 40.0, 0.9, 0.05)
    assert power_two_sample(3.0, 40.0, n, n) >= 0.9
    assert power_two_sample(3.0, 40.0, n - 50, n - 50) < 0.9


def test_mde_inverts_power():
    mde = minimum_detectable_effect(25.0, 4000, 4000, 0.8, 0.05)
    assert power_two_sample(mde, 25.0, 4000, 4000) == pytest.approx(0.8, abs=0.001)


@settings(max_examples=30, deadline=None)
@given(n=st.integers(10, 100_000), effect=st.floats(0.001, 10.0), sd=st.floats(0.1, 100.0))
def test_power_is_a_probability_that_grows_with_sample_size(n, effect, sd):
    small, large = power_two_sample(effect, sd, n, n), power_two_sample(effect, sd, 2 * n, 2 * n)
    assert 0.0 <= small <= large <= 1.0


def test_zero_effect_has_power_equal_to_alpha():
    assert power_two_sample(0.0, 1.0, 1000, 1000, alpha=0.05) == pytest.approx(0.05)


@pytest.mark.parametrize(
    "call",
    [
        lambda: power_two_sample(1.0, 0.0, 10, 10),
        lambda: power_two_sample(1.0, 1.0, 0, 10),
        lambda: sample_size_per_arm(0.0, 1.0),
        lambda: sample_size_per_arm(1.0, 1.0, power=1.0),
        lambda: minimum_detectable_effect(1.0, 10, 10, alpha=0.0),
    ],
)
def test_power_functions_reject_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_mde_shrinks_with_the_square_root_of_sample_size():
    a = minimum_detectable_effect(10.0, 1000, 1000)
    b = minimum_detectable_effect(10.0, 4000, 4000)
    assert a / b == pytest.approx(2.0)
    assert np.isfinite(a)
