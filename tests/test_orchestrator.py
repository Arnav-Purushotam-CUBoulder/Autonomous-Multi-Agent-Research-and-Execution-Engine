from __future__ import annotations

import asyncio
from pathlib import Path

from app import db
from app.orchestrator import RunOrchestrator


class RecordingPublisher:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def publish(self, event_type: str, payload: dict[str, object]) -> bool:
        self.events.append((event_type, payload))
        return True


def test_execute_run_emits_completion_events(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "atlas.db")
    db.init_db()
    run_id = db.create_run("Produce a grounded workspace summary", str(tmp_path), None, 1)
    publisher = RecordingPublisher()
    orchestrator = RunOrchestrator(publisher=publisher)

    async def fake_plan(_: str, __: str, ___: str | None) -> str:
        return '{"subtasks":[{"id":"S1"}]}'

    async def fake_research(_: str, __: str, ___: str | None, ____: str, _____: str | None) -> str:
        return "# Research Summary\n\n- Evidence captured."

    async def fake_execute(_: str, __: str, ___: str, ____: str, _____: str | None) -> str:
        return '{"report_markdown":"# Final Report\\n\\nConcrete evidence and next actions."}'

    async def fake_critic(_: str, __: str, ___: str, ____: str) -> str:
        return '{"approved": true, "revision_request": ""}'

    monkeypatch.setattr(orchestrator, "_call_planner", fake_plan)
    monkeypatch.setattr(orchestrator, "_call_researcher", fake_research)
    monkeypatch.setattr(orchestrator, "_call_executor", fake_execute)
    monkeypatch.setattr(orchestrator, "_call_critic", fake_critic)

    asyncio.run(orchestrator.execute_run(run_id))

    run = db.get_run(run_id)
    assert run is not None
    assert run.status == "completed"
    assert "Final Report" in (run.final_report or "")

    event_names = [name for name, _ in publisher.events]
    assert "run.started" in event_names
    assert "stage.completed" in event_names
    assert "run.completed" in event_names


def test_execute_run_emits_failure_events(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "atlas.db")
    db.init_db()
    run_id = db.create_run("Trigger a failure path", str(tmp_path), None, 1)
    publisher = RecordingPublisher()
    orchestrator = RunOrchestrator(publisher=publisher)

    async def fake_plan(_: str, __: str, ___: str | None) -> str:
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(orchestrator, "_call_planner", fake_plan)

    try:
        asyncio.run(orchestrator.execute_run(run_id))
    except RuntimeError as exc:
        assert str(exc) == "planner exploded"
    else:
        raise AssertionError("execute_run should have raised")

    run = db.get_run(run_id)
    assert run is not None
    assert run.status == "failed"

    event_names = [name for name, _ in publisher.events]
    assert "stage.failed" in event_names
    assert "run.failed" in event_names
