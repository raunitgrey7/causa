"""Decision report for an uploaded experiment: effect, interval, power, guardrails, SRM and a verdict.

The report assumes the team has been looking at the experiment every day, because they have. When the
table has a day column, every statistic is computed at each daily look on the users enrolled so far, and
the verdict rests on the always-valid p-value and confidence sequence of :mod:`causa.experiment.sequential`
rather than on a fixed-horizon test that peeking has already invalidated. The fixed-horizon numbers are
still reported, labelled as such.

Decision rule, applied in order:

1. ``invalid``: the sample ratio test fails (sequential SRM p-value below ``srm_alpha`` when there are
   daily looks, chi-square otherwise). Nothing else is trusted.
2. ``do_not_ship``: a guardrail fails, or the primary metric is significantly worse.
3. ``ship``: the primary metric is significantly better and every guardrail passes.
4. ``keep_running``: the primary metric is significantly better but a guardrail is not yet cleared.
5. ``stop_no_effect``: the confidence sequence excludes the minimum detectable effect in both directions,
   so any remaining effect is too small to matter. This is the futility stop.
6. ``keep_running``: none of the above; the report says how many users per arm the planned power needs.

Guardrails are non-inferiority checks with a tolerated harm ``max_harm`` in metric units: ``pass`` when the
upper end of the harm interval is within the margin, ``fail`` when the harm is significant and the margin
cannot be confirmed, ``inconclusive`` otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from causa.experiment.power import minimum_detectable_effect, power_two_sample, sample_size_per_arm
from causa.experiment.sequential import MixtureSPRT, sequential_srm_p_values
from causa.experiment.srm import SRMResult, srm_test
from causa.experiment.stats import ArmSums, Contrast, cuped, difference_in_means

MIN_USERS_PER_ARM = 20
DECISIONS: tuple[str, ...] = ("invalid", "do_not_ship", "ship", "keep_running", "stop_no_effect")


@dataclass(frozen=True)
class Guardrail:
    """A metric that must not get worse by more than ``max_harm`` (in its own units)."""

    metric: str
    max_harm: float
    higher_is_worse: bool = True

    def __post_init__(self) -> None:
        """A margin must be positive to be testable."""
        if not self.max_harm > 0:
            raise ValueError(f"guardrail {self.metric}: max_harm must be positive")


@dataclass(frozen=True)
class ExperimentSpec:
    """How to read an experiment table and what the team planned for it."""

    name: str = "experiment"
    metric: str = "revenue"
    covariate: str | None = "pre_revenue"
    variant_column: str = "variant"
    control: str = "control"
    treatment: str = "treatment"
    treatment_share: float = 0.5
    day_column: str | None = "day"
    alpha: float = 0.05
    mde: float | None = None
    mde_relative: float = 0.03
    power: float = 0.8
    srm_alpha: float = 0.001
    guardrails: tuple[Guardrail, ...] = ()

    def __post_init__(self) -> None:
        """Reject specs that cannot produce a valid analysis."""
        if not 0 < self.treatment_share < 1:
            raise ValueError("treatment_share must be in (0, 1)")
        if not 0 < self.alpha < 0.5 or not 0 < self.srm_alpha < 0.5 or not 0 < self.power < 1:
            raise ValueError("alpha and srm_alpha must be in (0, 0.5) and power in (0, 1)")
        if self.mde is not None and not self.mde > 0:
            raise ValueError("mde must be positive")
        if not self.mde_relative > 0:
            raise ValueError("mde_relative must be positive")
        if self.control == self.treatment:
            raise ValueError("control and treatment labels must differ")


@dataclass(frozen=True)
class EffectSummary:
    """An effect with its fixed-horizon and always-valid inference at the latest look."""

    method: str
    control_mean: float
    treatment_mean: float
    effect: float
    se: float
    relative_effect: float
    fixed_ci: tuple[float, float]
    fixed_p_value: float
    always_valid_ci: tuple[float, float]
    always_valid_p_value: float


@dataclass(frozen=True)
class GuardrailResult:
    """Verdict on one guardrail metric."""

    metric: str
    max_harm: float
    harm: float
    harm_ci: tuple[float, float]
    status: str


@dataclass(frozen=True)
class PowerSummary:
    """Planning numbers at the minimum detectable effect."""

    mde: float
    sd: float
    sd_unadjusted: float
    variance_reduction: float
    n_control: int
    n_treatment: int
    power_at_mde: float
    required_n_per_arm: int
    mde_at_current_n: float


@dataclass(frozen=True)
class DecisionReport:
    """Everything the experiment owner needs to decide, plus the per-look history for plots."""

    experiment: str
    metric: str
    decision: str
    reasons: list[str]
    primary: EffectSummary
    unadjusted: EffectSummary
    power: PowerSummary
    srm: SRMResult
    srm_sequential_p_value: float
    guardrails: list[GuardrailResult]
    looks: list[dict[str, float]]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Plain-JSON form."""
        return asdict(self)


@dataclass(frozen=True)
class _Table:
    treated: np.ndarray
    look: np.ndarray
    days: np.ndarray
    columns: dict[str, np.ndarray]


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    bad = int((~np.isfinite(values)).sum())
    if bad:
        raise ValueError(f"column {column!r} has {bad} missing or non-numeric values")
    return values


def _prepare(frame: pd.DataFrame, spec: ExperimentSpec) -> _Table:
    """Validate the table and encode arms, looks and the metric columns the spec names."""
    needed = [spec.variant_column, spec.metric, *(g.metric for g in spec.guardrails)]
    needed += [c for c in (spec.covariate, spec.day_column) if c]
    missing = [c for c in dict.fromkeys(needed) if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    labels = frame[spec.variant_column].astype(str)
    unexpected = sorted(set(labels) - {spec.control, spec.treatment})
    if unexpected:
        raise ValueError(f"unexpected variant labels: {', '.join(unexpected[:5])}")
    treated = (labels == spec.treatment).to_numpy()
    if min(int(treated.sum()), int((~treated).sum())) < MIN_USERS_PER_ARM:
        raise ValueError(f"each arm needs at least {MIN_USERS_PER_ARM} users")
    if spec.day_column:
        day = _numeric(frame, spec.day_column)
        days, look = np.unique(day, return_inverse=True)
    else:
        days, look = np.array([1.0]), np.zeros(len(frame), dtype=int)
    names = [spec.metric, *(g.metric for g in spec.guardrails)] + ([spec.covariate] if spec.covariate else [])
    columns = {name: _numeric(frame, name) for name in dict.fromkeys(names)}
    return _Table(treated=treated, look=look, days=days, columns=columns)


def _look_sums(table: _Table, y: np.ndarray, x: np.ndarray | None, arm: bool) -> ArmSums:
    """Cumulative sums per look for one arm, via bincount over the look index."""
    k = len(table.days)
    m = (table.treated == arm).astype(float)
    xv = np.zeros_like(y) if x is None else x

    def per_look(w: np.ndarray) -> np.ndarray:
        return np.bincount(table.look, weights=w * m, minlength=k)

    return ArmSums(
        n=per_look(np.ones_like(y)),
        y=per_look(y),
        yy=per_look(y * y),
        x=per_look(xv),
        xx=per_look(xv * xv),
        xy=per_look(xv * y),
    ).cumulative()


def _summarise(method: str, contrast: Contrast, sprt: MixtureSPRT, alpha: float) -> EffectSummary:
    """Latest-look summary: fixed-horizon numbers plus the always-valid ones over all looks."""
    p_av = sprt.p_values(contrast.effect, contrast.variance)
    lo_av, hi_av = sprt.confidence_sequence(contrast.effect, contrast.variance)
    lo, hi = contrast.interval(alpha)
    rel = contrast.relative(alpha)[0]
    return EffectSummary(
        method=method,
        control_mean=float(contrast.control_mean[-1]),
        treatment_mean=float(contrast.treatment_mean[-1]),
        effect=float(contrast.effect[-1]),
        se=float(contrast.se[-1]),
        relative_effect=float(rel[-1]),
        fixed_ci=(float(lo[-1]), float(hi[-1])),
        fixed_p_value=float(contrast.p_value()[-1]),
        always_valid_ci=(float(lo_av[-1]), float(hi_av[-1])),
        always_valid_p_value=float(p_av[-1]),
    )


def _guardrail(table: _Table, guard: Guardrail, alpha: float) -> GuardrailResult:
    y = table.columns[guard.metric]
    contrast = difference_in_means(_look_sums(table, y, None, False), _look_sums(table, y, None, True))
    sign = 1.0 if guard.higher_is_worse else -1.0
    lo, hi = MixtureSPRT(alpha=alpha, tau=guard.max_harm).confidence_sequence(
        sign * contrast.effect, contrast.variance
    )
    harm_lo, harm_hi = float(lo[-1]), float(hi[-1])
    if harm_hi <= guard.max_harm:
        status = "pass"
    elif harm_lo > 0:
        status = "fail"
    else:
        status = "inconclusive"
    return GuardrailResult(
        guard.metric, guard.max_harm, float(sign * contrast.effect[-1]), (harm_lo, harm_hi), status
    )


def _power(
    contrast: Contrast, raw: Contrast, n_c: float, n_t: float, mde: float, spec: ExperimentSpec
) -> PowerSummary:
    """Planning numbers from the latest look; the adjusted SD reflects CUPED when a covariate is used."""
    sd = float(np.sqrt((contrast.var_control_mean[-1] * n_c + contrast.var_treatment_mean[-1] * n_t) / 2.0))
    sd_raw = float(np.sqrt((raw.var_control_mean[-1] * n_c + raw.var_treatment_mean[-1] * n_t) / 2.0))
    sd, sd_raw = max(sd, 1e-12), max(sd_raw, 1e-12)
    return PowerSummary(
        mde=mde,
        sd=sd,
        sd_unadjusted=sd_raw,
        variance_reduction=float(1.0 - (sd / sd_raw) ** 2),
        n_control=int(n_c),
        n_treatment=int(n_t),
        power_at_mde=power_two_sample(mde, sd, n_c, n_t, spec.alpha),
        required_n_per_arm=sample_size_per_arm(mde, sd, spec.power, spec.alpha),
        mde_at_current_n=minimum_detectable_effect(sd, n_c, n_t, spec.power, spec.alpha),
    )


def _decide(
    primary: EffectSummary,
    power: PowerSummary,
    srm_p: float,
    guardrails: list[GuardrailResult],
    spec: ExperimentSpec,
) -> tuple[str, list[str]]:
    """Apply the documented rule in order and say why."""
    lo, hi = primary.always_valid_ci
    interval = f"always-valid interval {lo:.4g} to {hi:.4g}"
    failed = [g.metric for g in guardrails if g.status == "fail"]
    unclear = [g.metric for g in guardrails if g.status == "inconclusive"]
    significant = primary.always_valid_p_value < spec.alpha
    need = max(power.required_n_per_arm - min(power.n_control, power.n_treatment), 0)
    rules = [
        (
            srm_p < spec.srm_alpha,
            "invalid",
            f"sample ratio mismatch: SRM p-value {srm_p:.2e} < {spec.srm_alpha}",
        ),
        (bool(failed), "do_not_ship", f"guardrail failed: {', '.join(failed)}"),
        (
            significant and primary.effect < 0,
            "do_not_ship",
            f"{spec.metric} is significantly worse ({interval})",
        ),
        (significant and not unclear, "ship", f"{spec.metric} is significantly better ({interval})"),
        (
            significant,
            "keep_running",
            f"{spec.metric} is better but guardrails are not cleared: {', '.join(unclear)}",
        ),
        (
            hi < power.mde and lo > -power.mde,
            "stop_no_effect",
            f"the {interval} excludes the minimum detectable effect {power.mde:.4g}",
        ),
    ]
    for hit, decision, reason in rules:
        if hit:
            return decision, [reason]
    return "keep_running", [
        f"no decision yet: about {need} more users per arm give {spec.power:.0%} power at MDE {power.mde:.4g}"
    ]


def _looks(
    table: _Table, contrast: Contrast, sprt: MixtureSPRT, n_c: np.ndarray, n_t: np.ndarray
) -> list[dict[str, float]]:
    lo, hi = sprt.confidence_sequence(contrast.effect, contrast.variance)
    p = sprt.p_values(contrast.effect, contrast.variance)
    return [
        {
            "day": float(day),
            "n_control": float(n_c[k]),
            "n_treatment": float(n_t[k]),
            "effect": float(contrast.effect[k]),
            "ci_low": float(lo[k]),
            "ci_high": float(hi[k]),
            "p_value": float(p[k]),
        }
        for k, day in enumerate(table.days)
    ]


def analyze_experiment(frame: pd.DataFrame, spec: ExperimentSpec | None = None) -> DecisionReport:
    """Validate an experiment table and build its decision report."""
    spec = spec or ExperimentSpec()
    table = _prepare(frame, spec)
    y = table.columns[spec.metric]
    x = table.columns[spec.covariate] if spec.covariate else None
    control, treatment = _look_sums(table, y, x, False), _look_sums(table, y, x, True)
    raw = difference_in_means(control, treatment)
    adjusted, method = (
        (cuped(control, treatment)[0], "cuped") if x is not None else (raw, "difference_in_means")
    )
    mde = (
        spec.mde if spec.mde is not None else spec.mde_relative * max(abs(float(raw.control_mean[-1])), 1e-9)
    )
    sprt = MixtureSPRT(alpha=spec.alpha, tau=mde)
    primary = _summarise(method, adjusted, sprt, spec.alpha)
    unadjusted = _summarise("difference_in_means", raw, sprt, spec.alpha)
    n_c, n_t = control.n, treatment.n
    power = _power(adjusted, raw, float(n_c[-1]), float(n_t[-1]), mde, spec)
    srm = srm_test(
        [int(n_c[-1]), int(n_t[-1])], [1.0 - spec.treatment_share, spec.treatment_share], spec.srm_alpha
    )
    counts = np.column_stack([n_c, n_t])
    srm_seq = float(sequential_srm_p_values(counts, [1.0 - spec.treatment_share, spec.treatment_share])[-1])
    guardrails = [_guardrail(table, g, spec.alpha) for g in spec.guardrails]
    decision_srm = srm_seq if len(table.days) > 1 else srm.p_value
    decision, reasons = _decide(primary, power, decision_srm, guardrails, spec)
    warnings = []
    if len(table.days) == 1:
        warnings.append("no day column: the analysis treats the data as a single look")
    if primary.always_valid_ci[0] > primary.always_valid_ci[1]:
        warnings.append("the confidence sequence is empty, which happens with probability below alpha")
    return DecisionReport(
        experiment=spec.name,
        metric=spec.metric,
        decision=decision,
        reasons=reasons,
        primary=primary,
        unadjusted=unadjusted,
        power=power,
        srm=srm,
        srm_sequential_p_value=srm_seq,
        guardrails=guardrails,
        looks=_looks(table, adjusted, sprt, n_c, n_t),
        warnings=warnings,
    )
