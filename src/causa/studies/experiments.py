"""Monte Carlo studies of the experiment tools: CUPED, peeking against the mixture SPRT, and SRM checks.

All three draw users from :func:`causa.data.experiments.simulate_arrays` in blocks of replications and
reduce each block to per-arm sufficient statistics, the same :class:`causa.experiment.stats.ArmSums` the
decision report uses, so the studies measure the shipped code path.

* CUPED: variance of the CUPED and plain difference-in-means estimates across replications, their bias and
  interval coverage against the closed-form effect, and empirical power next to the power the analytic
  formula predicts.
* Peeking: under A/A (no effect), how often a z-test checked at every look, a fixed-horizon z-test and the
  mixture SPRT (with and without CUPED) declare a winner; under A/B, their power and the number of users
  at which the sequential test stops.
* SRM: how often the chi-square test (checked once or daily) and the sequential SRM test raise an alarm,
  with no fault and with a logging fault that drops a share of non-converting treatment users, plus the
  bias that fault puts into the revenue effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causa.data.experiments import ExperimentConfig, expected_effects, simulate_arrays
from causa.experiment.power import power_two_sample
from causa.experiment.sequential import MixtureSPRT, sequential_srm_p_values
from causa.experiment.srm import SRM_ALPHA, chi_square_statistic
from causa.experiment.stats import ArmSums, Contrast, cuped, difference_in_means

BLOCK = 100


def _rate(hits: np.ndarray) -> dict[str, float]:
    rate = float(np.mean(hits))
    return {"rate": rate, "mcse": float(np.sqrt(rate * (1.0 - rate) / len(hits)))}


def _arms(y: np.ndarray, x: np.ndarray, treated: np.ndarray, logged: np.ndarray) -> tuple[ArmSums, ArmSums]:
    return ArmSums.from_masked(y, x, ~treated & logged), ArmSums.from_masked(y, x, treated & logged)


@dataclass(frozen=True)
class CupedStudyConfig:
    """Replications, users per replication and the true relative lift."""

    reps: int = 2000
    n_users: int = 20_000
    lift: float = 0.08
    alpha: float = 0.05
    seed: int = 42


def _contrast_metrics(c: Contrast, truth: float, alpha: float) -> dict[str, float]:
    lo, hi = c.interval(alpha)
    effect = c.effect
    return {
        "bias": float(effect.mean() - truth),
        "sd": float(effect.std(ddof=1)),
        "mean_se": float(c.se.mean()),
        "coverage": float(np.mean((lo <= truth) & (truth <= hi))),
        "power": float(np.mean(c.p_value() < alpha)),
    }


def cuped_study(cfg: CupedStudyConfig) -> dict[str, object]:
    """Variance reduction, unbiasedness, coverage and power of CUPED against plain difference in means."""
    exp = ExperimentConfig(n_users=cfg.n_users, lift=cfg.lift)
    truth = expected_effects(exp)["revenue"]
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 11]))
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("dim_m", "dim_v", "cup_m", "cup_v", "sd", "rho2")}
    for start in range(0, cfg.reps, BLOCK):
        a = simulate_arrays(exp, min(BLOCK, cfg.reps - start), rng)
        control, treatment = _arms(a.revenue, a.pre_revenue, a.treated, a.logged)
        dim, (cup, _) = difference_in_means(control, treatment), cuped(control, treatment)
        parts["dim_m"].append(np.stack([dim.control_mean, dim.treatment_mean]))
        parts["dim_v"].append(np.stack([dim.var_control_mean, dim.var_treatment_mean]))
        parts["cup_m"].append(np.stack([cup.control_mean, cup.treatment_mean]))
        parts["cup_v"].append(np.stack([cup.var_control_mean, cup.var_treatment_mean]))
        parts["sd"].append(a.revenue.std(axis=1, ddof=1))
        rho = [np.corrcoef(a.pre_revenue[i], a.revenue[i])[0, 1] for i in range(len(a.revenue))]
        parts["rho2"].append(np.square(rho))
    cat = {k: np.concatenate(v, axis=-1) for k, v in parts.items()}
    dim = Contrast(cat["dim_m"][0], cat["dim_m"][1], cat["dim_v"][0], cat["dim_v"][1])
    cup = Contrast(cat["cup_m"][0], cat["cup_m"][1], cat["cup_v"][0], cat["cup_v"][1])
    sd, rho2, n_arm = float(cat["sd"].mean()), float(cat["rho2"].mean()), cfg.n_users / 2.0
    predicted = {
        "difference_in_means": power_two_sample(truth, sd, n_arm, n_arm, cfg.alpha),
        "cuped": power_two_sample(truth, sd * np.sqrt(1.0 - rho2), n_arm, n_arm, cfg.alpha),
    }
    metrics = {
        "difference_in_means": _contrast_metrics(dim, truth, cfg.alpha),
        "cuped": _contrast_metrics(cup, truth, cfg.alpha),
    }
    return {
        "reps": cfg.reps,
        "n_users": cfg.n_users,
        "true_effect": truth,
        "rho2": rho2,
        "variance_reduction": float(1.0 - cup.effect.var(ddof=1) / dim.effect.var(ddof=1)),
        **metrics,
        "predicted_power": predicted,
        "power_gap": {k: metrics[k]["power"] - predicted[k] for k in predicted},
    }


@dataclass(frozen=True)
class SequentialStudyConfig:
    """A/A and A/B streams of ``n_users`` users analysed at ``looks`` evenly spaced looks."""

    reps: int = 2000
    n_users: int = 20_000
    looks: int = 40
    lift: float = 0.08
    alpha: float = 0.05
    seed: int = 42


def _looked(a: np.ndarray, looks: int) -> np.ndarray:
    return a.reshape(a.shape[0], looks, -1)


def _stream_block(
    exp: ExperimentConfig, reps: int, looks: int, rng: np.random.Generator
) -> tuple[Contrast, Contrast]:
    a = simulate_arrays(exp, reps, rng)
    y, x = _looked(a.revenue, looks), _looked(a.pre_revenue, looks)
    t, logged = _looked(a.treated, looks), _looked(a.logged, looks)
    control = ArmSums.from_masked(y, x, ~t & logged).cumulative()
    treatment = ArmSums.from_masked(y, x, t & logged).cumulative()
    return difference_in_means(control, treatment), cuped(control, treatment)[0]


def _stream_scenario(cfg: SequentialStudyConfig, lift: float, tau: float, salt: int) -> dict[str, object]:
    exp = ExperimentConfig(n_users=cfg.n_users, lift=lift)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, salt]))
    sprt = MixtureSPRT(alpha=cfg.alpha, tau=tau)
    hits: dict[str, list[np.ndarray]] = {k: [] for k in ("naive", "fixed", "msprt", "msprt_cuped")}
    stop_look: list[np.ndarray] = []
    for start in range(0, cfg.reps, BLOCK):
        dim, cup = _stream_block(exp, min(BLOCK, cfg.reps - start), cfg.looks, rng)
        p_fixed = dim.p_value()
        hits["naive"].append((p_fixed < cfg.alpha).any(axis=1))
        hits["fixed"].append(p_fixed[:, -1] < cfg.alpha)
        p_seq = sprt.p_values(dim.effect, dim.variance)
        p_cup = sprt.p_values(cup.effect, cup.variance)
        hits["msprt"].append(p_seq[:, -1] < cfg.alpha)
        hits["msprt_cuped"].append(p_cup[:, -1] < cfg.alpha)
        crossed = p_cup < cfg.alpha
        stop_look.append(np.where(crossed.any(axis=1), crossed.argmax(axis=1), cfg.looks - 1))
    per_look = cfg.n_users / cfg.looks
    users_at_stop = (np.concatenate(stop_look) + 1) * per_look
    return {
        "true_effect": expected_effects(exp)["revenue"],
        "reject": {k: _rate(np.concatenate(v)) for k, v in hits.items()},
        "msprt_cuped_users_at_decision_mean": float(users_at_stop.mean()),
        "msprt_cuped_users_at_decision_median": float(np.median(users_at_stop)),
    }


def sequential_study(cfg: SequentialStudyConfig) -> dict[str, object]:
    """Peeking false positive rates under A/A and power under A/B for every testing strategy."""
    tau = expected_effects(ExperimentConfig(n_users=cfg.n_users, lift=cfg.lift))["revenue"]
    return {
        "reps": cfg.reps,
        "n_users": cfg.n_users,
        "looks": cfg.looks,
        "alpha": cfg.alpha,
        "tau": tau,
        "aa": _stream_scenario(cfg, 0.0, tau, 21),
        "ab": _stream_scenario(cfg, cfg.lift, tau, 22),
    }


@dataclass(frozen=True)
class SRMStudyConfig:
    """Count-level SRM simulation with daily looks, plus a user-level run for the fault's bias."""

    reps: int = 10_000
    n_users: int = 100_000
    days: int = 14
    drops: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05)
    bias_reps: int = 200
    seed: int = 42


def _daily_counts(cfg: SRMStudyConfig, drop: float, p_buy: float, rng: np.random.Generator) -> np.ndarray:
    """Cumulative (control, treatment) logged counts per day: shape (reps, days, 2)."""
    per_day = rng.multinomial(cfg.n_users, np.full(cfg.days, 1.0 / cfg.days), size=cfg.reps)
    treated = rng.binomial(per_day, 0.5)
    lost = rng.binomial(rng.binomial(treated, 1.0 - p_buy), drop)
    daily = np.stack([per_day - treated, treated - lost], axis=-1)
    return np.cumsum(daily, axis=1)


def _fault_bias(cfg: SRMStudyConfig, drop: float, rng: np.random.Generator) -> float:
    exp = ExperimentConfig(n_users=cfg.n_users, srm_drop=drop)
    effects = []
    for start in range(0, cfg.bias_reps, 20):
        a = simulate_arrays(exp, min(20, cfg.bias_reps - start), rng)
        control, treatment = _arms(a.revenue, a.pre_revenue, a.treated, a.logged)
        effects.append(difference_in_means(control, treatment).effect)
    return float(np.concatenate(effects).mean())


def srm_study(cfg: SRMStudyConfig) -> dict[str, object]:
    """Alarm rates of the SRM checks by fault size, and the effect bias the largest fault causes."""
    exp = ExperimentConfig(n_users=cfg.n_users)
    effects = expected_effects(exp)
    p_buy = effects["converted"] + effects["converted_control"]
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 31]))
    expected = np.array([0.5, 0.5])
    rows = {}
    for drop in cfg.drops:
        counts = _daily_counts(cfg, drop, p_buy, rng)
        _, p_daily = chi_square_statistic(counts, expected)
        seq = sequential_srm_p_values(counts, expected)
        rows[f"drop_{drop:.3f}".replace(".", "_")] = {
            "drop": drop,
            "chi_square_final": _rate(p_daily[:, -1] < SRM_ALPHA),
            "chi_square_daily": _rate((p_daily < SRM_ALPHA).any(axis=1)),
            "sequential": _rate(seq[:, -1] < SRM_ALPHA),
        }
    worst = max(cfg.drops)
    return {
        "reps": cfg.reps,
        "n_users": cfg.n_users,
        "days": cfg.days,
        "alpha": SRM_ALPHA,
        "by_drop": rows,
        "fault_bias": {
            "drop": worst,
            "true_effect": effects["revenue"],
            "mean_estimated_effect": _fault_bias(cfg, worst, rng),
            "reps": cfg.bias_reps,
        },
    }
