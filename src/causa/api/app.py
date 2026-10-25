"""FastAPI application factory.

``create_app`` wires settings, the experiment store, metrics and the service, then declares the routes.
Everything is injected, so tests build an app on a temporary directory with a small model and never touch
the real data or artifacts.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from causa import __version__
from causa.api.middleware import Metrics, ObservabilityMiddleware
from causa.api.schemas import (
    CustomerIn,
    ExperimentRecord,
    ExperimentSummary,
    ExperimentUpload,
    HealthResponse,
    ModelInfo,
    ObservationalResponse,
    ObservationalUpload,
    SimulateRequest,
    TargetingRequest,
    TargetingResponse,
)
from causa.api.security import require_api_key
from causa.api.service import Service, ServiceError
from causa.api.store import ExperimentStore
from causa.config import Settings, get_settings
from causa.jsonlog import configure_logging
from causa.targeting import TargetingModel

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    settings: Settings | None = None, model: TargetingModel | None = None, train_if_missing: bool = False
) -> FastAPI:
    """Build the application; a ``model`` passed in skips loading from disk."""
    settings = settings or get_settings()
    logger = configure_logging(settings.log_level)
    metrics = Metrics()
    service = Service(settings, ExperimentStore(settings.store_path), metrics, logger)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if model is not None:
            service.model = model
            metrics.model_loaded.set(1.0)
        else:
            service.load_model(train_if_missing)
        logger.info("startup", extra={"model": service.model.learner_name if service.model else None})
        yield

    app = FastAPI(
        title="causa",
        version=__version__,
        description="Experiment decision reports, uplift targeting and observational ATE estimation",
        lifespan=lifespan,
        dependencies=[Depends(require_api_key)],
    )
    app.state.settings = settings
    app.state.service = service
    app.add_middleware(ObservabilityMiddleware, metrics=metrics, logger=logger)
    _register_handlers(app)
    _register_ops_routes(app, service, metrics, settings)
    _register_experiment_routes(app, service)
    _register_targeting_routes(app, service)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _register_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail, "request_id": request_id})


def _register_ops_routes(app: FastAPI, service: Service, metrics: Metrics, settings: Settings) -> None:
    """Health, metrics, dashboard and benchmark results."""

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return service.health()

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    def prometheus_metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/v1/benchmark", tags=["ops"])
    def benchmark() -> JSONResponse:
        path = settings.benchmark_path
        if not path.exists():
            raise HTTPException(status_code=404, detail="no benchmark results; run `make bench`")
        return JSONResponse(json.loads(path.read_text()))


def _register_experiment_routes(app: FastAPI, service: Service) -> None:
    """Upload an experiment (JSON or CSV), simulate one, and read stored reports."""

    @app.post("/v1/experiments", response_model=ExperimentRecord, status_code=201, tags=["experiments"])
    def analyze(body: ExperimentUpload) -> ExperimentRecord:
        return service.analyze_upload(body)

    @app.post("/v1/experiments/csv", response_model=ExperimentRecord, status_code=201, tags=["experiments"])
    async def analyze_csv(
        file: Annotated[UploadFile, File()], spec: Annotated[str | None, Form()] = None
    ) -> ExperimentRecord:
        content = await file.read()
        return await run_in_threadpool(service.analyze_csv, content, spec)

    @app.post(
        "/v1/experiments/simulate", response_model=ExperimentRecord, status_code=201, tags=["experiments"]
    )
    def simulate(body: SimulateRequest) -> ExperimentRecord:
        return service.simulate(body)

    @app.get("/v1/experiments", response_model=list[ExperimentSummary], tags=["experiments"])
    def list_experiments(limit: int = Query(default=50, ge=1, le=500)) -> list[ExperimentSummary]:
        return service.list_experiments(limit)

    @app.get("/v1/experiments/{record_id}", response_model=ExperimentRecord, tags=["experiments"])
    def get_experiment(record_id: str) -> ExperimentRecord:
        return service.get_experiment(record_id)


def _register_targeting_routes(app: FastAPI, service: Service) -> None:
    """Uplift targeting and observational ATE."""

    @app.post("/v1/targeting", response_model=TargetingResponse, tags=["targeting"])
    def target(body: TargetingRequest) -> TargetingResponse:
        return service.target(body)

    @app.get("/v1/customers/sample", response_model=list[CustomerIn], tags=["targeting"])
    def sample_customers(
        n: int = Query(default=200, ge=1, le=5_000), seed: int = Query(default=0, ge=0, le=2**31 - 1)
    ) -> list[CustomerIn]:
        return service.sample_customers(n, seed)

    @app.get("/v1/targeting/model", response_model=ModelInfo, tags=["targeting"])
    def model_info() -> ModelInfo:
        return service.model_info()

    @app.post("/v1/observational/ate", response_model=ObservationalResponse, tags=["observational"])
    def observational(body: ObservationalUpload) -> ObservationalResponse:
        return service.observational(body)
