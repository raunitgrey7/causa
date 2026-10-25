"""Request ids, JSON access logs and Prometheus metrics in one middleware.

Every response carries ``X-Request-ID`` (echoed from the request when the client sent a sane one) so a log
line, a metric sample and a client's error report can be joined. Metrics are labelled by route template,
not raw path, so experiment ids in URLs cannot blow up label cardinality.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class Metrics:
    """Prometheus collectors bound to one registry (one per app instance, so tests stay isolated)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "causa_requests_total", "HTTP requests", ["method", "path", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "causa_request_latency_seconds",
            "Request latency",
            ["path"],
            registry=self.registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        )
        self.decisions = Counter(
            "causa_experiment_decisions_total",
            "Decision reports by verdict",
            ["decision"],
            registry=self.registry,
        )
        self.customers_scored = Counter(
            "causa_customers_scored_total", "Customers scored by the targeting model", registry=self.registry
        )
        self.ate_analyses = Counter(
            "causa_observational_analyses_total", "Observational ATE analyses", registry=self.registry
        )
        self.model_loaded = Gauge(
            "causa_targeting_model_loaded", "1 when a targeting model is in memory", registry=self.registry
        )


def route_template(request: Request) -> str:
    """Path template of the matched route, or the raw path for unmatched requests."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else "unmatched"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Attach a request id, time the request, log it as JSON and record metrics."""

    def __init__(self, app: ASGIApp, metrics: Metrics, logger: logging.Logger) -> None:
        super().__init__(app)
        self.metrics = metrics
        self.logger = logger

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Wrap one request."""
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if REQUEST_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            self.logger.exception(
                "request failed", extra={"request_id": request_id, "path": request.url.path}
            )
            raise
        elapsed = time.perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        template = route_template(request)
        self.metrics.requests.labels(request.method, template, str(response.status_code)).inc()
        self.metrics.latency.labels(template).observe(elapsed)
        self.logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(elapsed * 1000.0, 2),
            },
        )
        return response
