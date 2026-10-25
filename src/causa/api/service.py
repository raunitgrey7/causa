"""Service layer between the HTTP routes and the analysis code.

Routes stay thin: they parse, call one method here and return its result. This layer enforces the
configured size limits, converts uploaded tables into the arrays the analysis expects, maps analysis
errors to HTTP errors with specific messages, stores decision reports and counts what it did in metrics.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd
from pydantic import ValidationError

from causa import __version__
from causa.api.middleware import Metrics
from causa.api.schemas import (
    CustomerIn,
    ExperimentRecord,
    ExperimentSpecIn,
    ExperimentSummary,
    ExperimentUpload,
    HealthResponse,
    ModelInfo,
    ObservationalResponse,
    ObservationalUpload,
    SimulateRequest,
    TargetingRequest,
    TargetingResponse,
    TargetingResult,
)
from causa.api.store import ExperimentStore
from causa.ate import ATEConfig, estimate_ate
from causa.config import Settings
from causa.data.campaign import simulate_customers
from causa.data.experiments import ExperimentConfig, simulate_experiment
from causa.experiment.report import DecisionReport, analyze_experiment
from causa.targeting import TargetingModel, load_model, save_model, train_targeting_model

MAX_CSV_BYTES = 64 * 1024 * 1024


class ServiceError(Exception):
    """An error with an HTTP status and a message safe to show the client."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class Service:
    """Holds the targeting model, the experiment store and the metrics for one app instance."""

    def __init__(
        self, settings: Settings, store: ExperimentStore, metrics: Metrics, logger: logging.Logger
    ) -> None:
        self.settings = settings
        self.store = store
        self.metrics = metrics
        self.logger = logger
        self.model: TargetingModel | None = None

    def load_model(self, train_if_missing: bool) -> bool:
        """Load the targeting model from disk, training and saving one first if asked and absent."""
        path = self.settings.model_path
        if not path.exists() and train_if_missing:
            self.logger.info("training targeting model", extra={"rows": self.settings.train_rows})
            save_model(train_targeting_model(n_train=self.settings.train_rows, seed=self.settings.seed), path)
        if path.exists():
            self.model = load_model(path)
        self.metrics.model_loaded.set(1.0 if self.model else 0.0)
        return self.model is not None

    def health(self) -> HealthResponse:
        """Liveness plus what is loaded."""
        return HealthResponse(
            status="ok",
            version=__version__,
            model_loaded=self.model is not None,
            model_learner=self.model.learner_name if self.model else None,
            experiments_stored=self.store.count(),
        )

    def _record(self, report: DecisionReport) -> ExperimentRecord:
        record_id, created = self.store.save(report.experiment, report.decision, asdict(report))
        self.metrics.decisions.labels(report.decision).inc()
        self.logger.info(
            "experiment analysed", extra={"experiment_id": record_id, "decision": report.decision}
        )
        return ExperimentRecord(
            id=record_id, created_at=created, name=report.experiment, decision=report.decision, report=report
        )

    def analyze_frame(self, frame: pd.DataFrame, spec: ExperimentSpecIn) -> ExperimentRecord:
        """Analyse a parsed table under the row limit."""
        if len(frame) > self.settings.max_rows:
            raise ServiceError(413, f"{len(frame)} rows exceed the limit of {self.settings.max_rows}")
        try:
            report = analyze_experiment(frame, spec.to_spec())
        except ValueError as exc:
            raise ServiceError(422, str(exc)) from exc
        return self._record(report)

    def analyze_upload(self, upload: ExperimentUpload) -> ExperimentRecord:
        """Analyse a columnar JSON upload."""
        return self.analyze_frame(pd.DataFrame(upload.data), upload.spec)

    def analyze_csv(self, content: bytes, spec_json: str | None) -> ExperimentRecord:
        """Analyse an uploaded CSV file; ``spec_json`` is an optional JSON-encoded spec."""
        if len(content) > MAX_CSV_BYTES:
            raise ServiceError(413, f"file exceeds {MAX_CSV_BYTES // (1024 * 1024)} MB")
        try:
            spec = ExperimentSpecIn.model_validate(json.loads(spec_json)) if spec_json else ExperimentSpecIn()
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ServiceError(422, f"invalid spec: {exc}") from exc
        try:
            frame = pd.read_csv(io.BytesIO(content))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise ServiceError(422, f"could not parse CSV: {exc}") from exc
        return self.analyze_frame(frame, spec)

    def simulate(self, req: SimulateRequest) -> ExperimentRecord:
        """Generate a synthetic experiment with the requested faults and analyse it."""
        config = ExperimentConfig(
            n_users=req.n_users,
            lift=req.lift,
            srm_drop=req.srm_drop,
            support_harm=req.support_harm,
            latency_shift_ms=req.latency_shift_ms,
            seed=req.seed,
        )
        return self.analyze_frame(simulate_experiment(config), req.spec)

    def get_experiment(self, record_id: str) -> ExperimentRecord:
        """A stored report by id."""
        row = self.store.get(record_id)
        if row is None:
            raise ServiceError(404, f"no experiment {record_id}")
        return ExperimentRecord.model_validate(row)

    def list_experiments(self, limit: int) -> list[ExperimentSummary]:
        """Most recent reports first."""
        return [ExperimentSummary(**row) for row in self.store.list(limit)]

    def _require_model(self) -> TargetingModel:
        if self.model is None:
            raise ServiceError(
                503, "no targeting model loaded; run `causa train` or start with --train-if-missing"
            )
        return self.model

    def target(self, req: TargetingRequest) -> TargetingResponse:
        """Score customers and select coupon recipients within the budget."""
        model = self._require_model()
        frame = pd.DataFrame([c.model_dump(exclude={"customer_id"}) for c in req.customers])
        decision = model.decide(frame, req.budget, req.cost_per_contact)
        ranks = np.empty(len(frame), dtype=int)
        ranks[np.argsort(-decision.uplift, kind="stable")] = np.arange(1, len(frame) + 1)
        self.metrics.customers_scored.inc(len(frame))
        results = [
            TargetingResult(customer_id=c.customer_id, uplift=float(u), treat=bool(t), rank=int(r))
            for c, u, t, r in zip(req.customers, decision.uplift, decision.treat, ranks, strict=True)
        ]
        total = decision.expected_incremental_revenue
        return TargetingResponse(
            model=model.learner_name,
            budget=req.budget,
            cost_per_contact=req.cost_per_contact,
            n_customers=len(frame),
            n_treated=int(decision.treat.sum()),
            expected_incremental_revenue=total,
            expected_incremental_revenue_per_customer=total / len(frame),
            results=results,
        )

    def sample_customers(self, n: int, seed: int) -> list[CustomerIn]:
        """Synthetic customers from the generator's population, for demos and smoke tests."""
        frame = simulate_customers(n, np.random.default_rng(seed))
        return [
            CustomerIn(customer_id=f"c{seed}-{i}", **row) for i, row in enumerate(frame.to_dict("records"))
        ]

    def model_info(self) -> ModelInfo:
        """Metadata and validation evidence of the serving model."""
        m = self._require_model()
        return ModelInfo(
            learner=m.learner_name,
            trained_at=m.trained_at,
            n_train=m.n_train,
            n_holdout=m.n_holdout,
            seed=m.seed,
            version=m.version,
            validation=m.validation,
            budget_curve=m.budget_curve,
        )

    def _observational_arrays(self, up: ObservationalUpload) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(up.data[up.outcome])
        if n > self.settings.max_ate_rows:
            raise ServiceError(413, f"{n} rows exceed the limit of {self.settings.max_ate_rows}")
        if len(up.covariates) > self.settings.max_covariates:
            raise ServiceError(422, f"at most {self.settings.max_covariates} covariates are supported")
        frame = pd.DataFrame({c: up.data[c] for c in [*up.covariates, up.treatment, up.outcome]})
        values = frame.apply(pd.to_numeric, errors="coerce")
        bad = [c for c in values.columns if not np.isfinite(values[c].to_numpy(float)).all()]
        if bad:
            raise ServiceError(422, f"non-numeric, missing or infinite values in: {', '.join(bad)}")
        return (
            values[up.covariates].to_numpy(float),
            values[up.treatment].to_numpy(float),
            values[up.outcome].to_numpy(float),
        )

    def observational(self, up: ObservationalUpload) -> ObservationalResponse:
        """Overlap diagnostics and ATE estimates with refit bootstrap intervals."""
        X, t, y = self._observational_arrays(up)
        try:
            result = estimate_ate(
                X, t, y, ATEConfig(n_boot=up.n_boot, seed=up.seed), feature_names=up.covariates
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            raise ServiceError(422, str(exc)) from exc
        self.metrics.ate_analyses.inc()
        return ObservationalResponse(result=result, recommended="aipw")
