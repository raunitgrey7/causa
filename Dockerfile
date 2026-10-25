# Multi-stage build: dependencies are resolved from the lock file in a builder stage, then only the
# virtual environment and the benchmark results are copied into a slim runtime image that runs as an
# unprivileged user.
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim AS runtime
RUN groupadd --system causa && useradd --system --gid causa --home /app --shell /usr/sbin/nologin causa
WORKDIR /app
COPY --from=builder --chown=causa:causa /app/.venv /app/.venv
COPY --chown=causa:causa benchmarks ./benchmarks
RUN mkdir -p /app/data /app/artifacts && chown -R causa:causa /app
ENV PATH="/app/.venv/bin:$PATH" \
    CAUSA_DATA_DIR=/app/data \
    CAUSA_ARTIFACTS_DIR=/app/artifacts \
    CAUSA_BENCHMARK_PATH=/app/benchmarks/results.json \
    OMP_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1
USER causa
EXPOSE 8773
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8773/health', timeout=4).status == 200 else 1)"
# The first start trains the targeting model on a synthetic campaign history (well under a minute) and
# caches it in /app/artifacts; mount a volume there to keep it across restarts.
CMD ["causa", "serve", "--host", "0.0.0.0", "--port", "8773", "--train-if-missing"]
