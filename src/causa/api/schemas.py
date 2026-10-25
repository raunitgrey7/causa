"""Typed request and response schemas.

Requests are validated at the edge: bounded sizes, finite numbers, known enums, so the analysis code never
sees an input it was not written for. Responses reuse the analysis dataclasses (decision reports, ATE
results) directly, which keeps one definition of each result and still gives a typed OpenAPI schema.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causa.ate import ATEResult
from causa.experiment.report import DecisionReport, ExperimentSpec, Guardrail

ColumnName = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")]
MAX_COLUMNS = 64


class GuardrailIn(BaseModel):
    """A guardrail metric and the harm it may tolerate."""

    model_config = ConfigDict(extra="forbid")
    metric: ColumnName
    max_harm: float = Field(gt=0, allow_inf_nan=False)
    higher_is_worse: bool = True


class ExperimentSpecIn(BaseModel):
    """How to read the uploaded table; mirrors :class:`causa.experiment.report.ExperimentSpec`."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="experiment", min_length=1, max_length=120)
    metric: str = Field(default="revenue", min_length=1, max_length=64)
    covariate: str | None = Field(default="pre_revenue", max_length=64)
    variant_column: str = Field(default="variant", min_length=1, max_length=64)
    control: str = Field(default="control", min_length=1, max_length=64)
    treatment: str = Field(default="treatment", min_length=1, max_length=64)
    treatment_share: float = Field(default=0.5, gt=0, lt=1)
    day_column: str | None = Field(default="day", max_length=64)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    mde: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    mde_relative: float = Field(default=0.03, gt=0, le=10, allow_inf_nan=False)
    power: float = Field(default=0.8, gt=0, lt=1)
    srm_alpha: float = Field(default=0.001, gt=0, lt=0.5)
    guardrails: list[GuardrailIn] = Field(default_factory=list, max_length=10)

    def to_spec(self) -> ExperimentSpec:
        """The analysis-layer spec."""
        data = self.model_dump(exclude={"guardrails"})
        return ExperimentSpec(**data, guardrails=tuple(Guardrail(**g.model_dump()) for g in self.guardrails))


Column = list[float | int | str | None]


def _check_columns(data: dict[str, Column]) -> None:
    if not data or len(data) > MAX_COLUMNS:
        raise ValueError(f"data must have between 1 and {MAX_COLUMNS} columns")
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1:
        raise ValueError("all columns must have the same length")
    if lengths.pop() == 0:
        raise ValueError("data has no rows")


class ExperimentUpload(BaseModel):
    """An experiment table in columnar JSON: ``{"column": [values, ...]}``."""

    model_config = ConfigDict(extra="forbid")
    spec: ExperimentSpecIn = Field(default_factory=ExperimentSpecIn)
    data: dict[str, Column]

    @model_validator(mode="after")
    def _columns(self) -> Self:
        _check_columns(self.data)
        return self


def default_guardrails() -> list[GuardrailIn]:
    """Guardrails of the bundled demo experiment."""
    return [
        GuardrailIn(metric="support_contact", max_harm=0.01),
        GuardrailIn(metric="page_load_ms", max_harm=25.0),
    ]


class SimulateRequest(BaseModel):
    """Generate a synthetic experiment and analyse it (the dashboard's demo)."""

    model_config = ConfigDict(extra="forbid")
    n_users: int = Field(default=40_000, ge=200, le=200_000)
    lift: float = Field(default=0.05, gt=-0.5, lt=1.0)
    srm_drop: float = Field(default=0.0, ge=0.0, lt=0.5)
    support_harm: float = Field(default=0.0, ge=0.0, le=0.2)
    latency_shift_ms: float = Field(default=0.0, ge=-500.0, le=1000.0)
    seed: int = Field(default=7, ge=0, le=2**31 - 1)
    spec: ExperimentSpecIn = Field(default_factory=lambda: ExperimentSpecIn(guardrails=default_guardrails()))


class ExperimentRecord(BaseModel):
    """A stored decision report."""

    id: str
    created_at: str
    name: str
    decision: str
    report: DecisionReport


class ExperimentSummary(BaseModel):
    """One row of the experiment list."""

    id: str
    created_at: str
    name: str
    decision: str


class CustomerIn(BaseModel):
    """One customer's CRM features, with the same bounds the generator respects."""

    model_config = ConfigDict(extra="forbid")
    customer_id: str | None = Field(default=None, max_length=64)
    tenure_months: float = Field(ge=0, le=600, allow_inf_nan=False)
    orders_12m: int = Field(ge=0, le=10_000)
    recency_days: float = Field(gt=0, le=10_000, allow_inf_nan=False)
    sessions_30d: int = Field(ge=0, le=100_000)
    email_open_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    avg_order_value: float = Field(gt=0, le=10_000_000, allow_inf_nan=False)
    discount_affinity: float = Field(ge=0, le=1, allow_inf_nan=False)
    app_user: int = Field(ge=0, le=1)
    loyalty_member: int = Field(ge=0, le=1)
    city_tier: int = Field(ge=1, le=3)


class TargetingRequest(BaseModel):
    """Customers to rank and the coupon budget."""

    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerIn] = Field(min_length=1, max_length=50_000)
    budget: float = Field(default=0.2, gt=0, le=1)
    cost_per_contact: float = Field(default=0.0, ge=0, le=100_000, allow_inf_nan=False)


class TargetingResult(BaseModel):
    """Decision for one customer, in input order."""

    customer_id: str | None
    uplift: float
    treat: bool
    rank: int


class TargetingResponse(BaseModel):
    """Decisions plus the expected incremental revenue of the selection."""

    model: str
    budget: float
    cost_per_contact: float
    n_customers: int
    n_treated: int
    expected_incremental_revenue: float
    expected_incremental_revenue_per_customer: float
    results: list[TargetingResult]


class ModelInfo(BaseModel):
    """The serving model and the holdout evidence for choosing it."""

    learner: str
    trained_at: str
    n_train: int
    n_holdout: int
    seed: int
    version: str
    validation: dict[str, dict[str, float]]
    budget_curve: dict[str, list[float]]


class ObservationalUpload(BaseModel):
    """An observational table: covariates, a binary treatment and an outcome, in columnar JSON."""

    model_config = ConfigDict(extra="forbid")
    covariates: list[ColumnName] = Field(min_length=1, max_length=MAX_COLUMNS)
    treatment: ColumnName
    outcome: ColumnName
    data: dict[str, Column]
    n_boot: int = Field(default=100, ge=20, le=300)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def _columns(self) -> Self:
        _check_columns(self.data)
        names = [*self.covariates, self.treatment, self.outcome]
        if len(set(names)) != len(names):
            raise ValueError("covariates, treatment and outcome must be distinct columns")
        missing = [c for c in names if c not in self.data]
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")
        return self


class ObservationalResponse(BaseModel):
    """ATE estimates with intervals and overlap diagnostics."""

    result: ATEResult
    recommended: str


class HealthResponse(BaseModel):
    """Liveness and what the process has loaded."""

    status: str
    version: str
    model_loaded: bool
    model_learner: str | None
    experiments_stored: int
