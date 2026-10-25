"""Structured JSON logging.

Each record becomes one JSON object per line with a timestamp, level, logger, message and any ``extra``
fields (request ids, latencies, decisions), so log pipelines parse it without regular expressions.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_RESERVED = set(logging.LogRecord("x", 0, "x", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render a record as a single JSON line, keeping custom ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record``."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Install the JSON formatter on the ``causa`` logger once and return it."""
    logger = logging.getLogger("causa")
    logger.setLevel(level.upper())
    logger.propagate = False
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger
