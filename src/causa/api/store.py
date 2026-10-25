"""Sqlite store of decision reports.

An uploaded experiment's report is kept so the dashboard and other services can fetch it by id later and
so a decision has a record. Raw rows are not stored: they can be large and may hold personal data, and the
report carries every number the decision used. Each call opens its own connection, which keeps the store
safe under FastAPI's worker threads without a pool.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    decision TEXT NOT NULL,
    report TEXT NOT NULL
)
"""


class ExperimentStore:
    """Insert, fetch and list decision reports."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10.0)

    def save(self, name: str, decision: str, report: dict[str, Any]) -> tuple[str, str]:
        """Store a report and return its ``(id, created_at)``."""
        record_id = uuid.uuid4().hex[:16]
        created = datetime.now(UTC).isoformat(timespec="seconds")
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO reports (id, created_at, name, decision, report) VALUES (?, ?, ?, ?, ?)",
                (record_id, created, name, decision, json.dumps(report)),
            )
        return record_id, created

    def get(self, record_id: str) -> dict[str, Any] | None:
        """One stored record, or None."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, created_at, name, decision, report FROM reports WHERE id = ?", (record_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "name": row[2],
            "decision": row[3],
            "report": json.loads(row[4]),
        }

    def list(self, limit: int = 50) -> list[dict[str, str]]:
        """Most recent records first, without the report bodies."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, created_at, name, decision FROM reports "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"id": r[0], "created_at": r[1], "name": r[2], "decision": r[3]} for r in rows]

    def count(self) -> int:
        """Number of stored reports."""
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0])
