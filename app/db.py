from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import RunRecord, RunSummary, StepRecord

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "atlas.db"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                workspace_path TEXT NOT NULL,
                context TEXT,
                status TEXT NOT NULL,
                final_report TEXT,
                error_message TEXT,
                max_revision_cycles INTEGER NOT NULL,
                current_revision INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                name TEXT NOT NULL,
                agent TEXT NOT NULL,
                status TEXT NOT NULL,
                input_text TEXT,
                output_text TEXT,
                metadata TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
            )
            """
        )


def create_run(task: str, workspace_path: str, context: str | None, max_revision_cycles: int) -> str:
    run_id = str(uuid.uuid4())
    now = utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO runs (
                id, task, workspace_path, context, status,
                max_revision_cycles, current_revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, task, workspace_path, context, "queued", max_revision_cycles, 0, now, now),
        )
    return run_id


def update_run(
    run_id: str,
    status: str | None = None,
    final_report: str | None = None,
    error_message: str | None = None,
    current_revision: int | None = None,
) -> None:
    assignments = ["updated_at = ?"]
    values: list[Any] = [utcnow().isoformat()]

    if status is not None:
        assignments.append("status = ?")
        values.append(status)
    if final_report is not None:
        assignments.append("final_report = ?")
        values.append(final_report)
    if error_message is not None:
        assignments.append("error_message = ?")
        values.append(error_message)
    if current_revision is not None:
        assignments.append("current_revision = ?")
        values.append(current_revision)

    values.append(run_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE runs SET {', '.join(assignments)} WHERE id = ?", values)


def create_step(
    run_id: str,
    name: str,
    agent: str,
    status: str = "running",
    input_text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    with get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO steps (
                run_id, name, agent, status, input_text, metadata, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                name,
                agent,
                status,
                input_text,
                json.dumps(metadata or {}),
                utcnow().isoformat(),
            ),
        )
        return int(cursor.lastrowid)


def update_step(
    step_id: int,
    status: str | None = None,
    output_text: str | None = None,
    metadata: dict[str, Any] | None = None,
    mark_started: bool = False,
    mark_finished: bool = False,
) -> None:
    assignments: list[str] = []
    values: list[Any] = []

    if status is not None:
        assignments.append("status = ?")
        values.append(status)
    if output_text is not None:
        assignments.append("output_text = ?")
        values.append(output_text)
    if metadata is not None:
        assignments.append("metadata = ?")
        values.append(json.dumps(metadata))
    if mark_started:
        assignments.append("started_at = ?")
        values.append(utcnow().isoformat())
    if mark_finished:
        assignments.append("finished_at = ?")
        values.append(utcnow().isoformat())

    if not assignments:
        return

    values.append(step_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE steps SET {', '.join(assignments)} WHERE id = ?", values)


def _row_to_run_summary(row: sqlite3.Row) -> RunSummary:
    return RunSummary(
        id=row["id"],
        task=row["task"],
        workspace_path=row["workspace_path"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_step(row: sqlite3.Row) -> StepRecord:
    return StepRecord(
        id=row["id"],
        run_id=row["run_id"],
        name=row["name"],
        agent=row["agent"],
        status=row["status"],
        input_text=row["input_text"],
        output_text=row["output_text"],
        metadata=json.loads(row["metadata"] or "{}"),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
    )


def get_run(run_id: str) -> RunRecord | None:
    with get_conn() as conn:
        run_row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run_row is None:
            return None

        step_rows = conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()

    return RunRecord(
        id=run_row["id"],
        task=run_row["task"],
        workspace_path=run_row["workspace_path"],
        context=run_row["context"],
        status=run_row["status"],
        final_report=run_row["final_report"],
        error_message=run_row["error_message"],
        max_revision_cycles=run_row["max_revision_cycles"],
        current_revision=run_row["current_revision"],
        created_at=datetime.fromisoformat(run_row["created_at"]),
        updated_at=datetime.fromisoformat(run_row["updated_at"]),
        steps=[_row_to_step(row) for row in step_rows],
    )


def list_runs(limit: int = 25) -> list[RunSummary]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_run_summary(row) for row in rows]


def reset_run(run_id: str) -> None:
    now = utcnow().isoformat()
    with get_conn() as conn:
        conn.execute("DELETE FROM steps WHERE run_id = ?", (run_id,))
        conn.execute(
            """
            UPDATE runs
            SET status = ?, final_report = NULL, error_message = NULL,
                current_revision = 0, updated_at = ?
            WHERE id = ?
            """,
            ("queued", now, run_id),
        )
