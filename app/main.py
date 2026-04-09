from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from . import db
from .models import ResumeRunRequest, RunCreateRequest, RunRecord, RunSummary
from .orchestrator import RunOrchestrator

app = FastAPI(
    title="Atlas: Multi-Agent Research and Execution Platform",
    version="0.1.0",
    description="Backend-only multi-agent orchestration service built with FastAPI, OpenAI Agents SDK, and SQLite.",
)

orchestrator = RunOrchestrator()
active_tasks: dict[str, asyncio.Task[Any]] = {}


@app.on_event("startup")
async def on_startup() -> None:
    db.init_db()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs", response_model=list[RunSummary])
async def list_runs() -> list[RunSummary]:
    return db.list_runs()


@app.post("/runs", response_model=RunRecord)
async def create_run(request: RunCreateRequest) -> RunRecord:
    run_id = db.create_run(
        task=request.task,
        workspace_path=request.workspace_path,
        context=request.context,
        max_revision_cycles=request.max_revision_cycles,
    )
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="Failed to create run")
    if request.auto_start:
        _launch_run(run_id)
    return db.get_run(run_id) or run


@app.post("/runs/{run_id}/resume", response_model=RunRecord)
async def resume_run(run_id: str, request: ResumeRunRequest) -> RunRecord:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if request.force_restart:
        db.reset_run(run_id)
    _launch_run(run_id, force=request.force_restart)
    refreshed = db.get_run(run_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Failed to refresh run")
    return refreshed


@app.get("/runs/{run_id}", response_model=RunRecord)
async def get_run(run_id: str) -> RunRecord:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/runs/{run_id}/report", response_class=PlainTextResponse)
async def get_run_report(run_id: str) -> str:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if not run.final_report:
        raise HTTPException(status_code=404, detail="Final report not available yet")
    return run.final_report


def _launch_run(run_id: str, force: bool = False) -> None:
    existing = active_tasks.get(run_id)
    if existing and not existing.done():
        if force:
            existing.cancel()
        else:
            return
    task = asyncio.create_task(_run_and_cleanup(run_id))
    active_tasks[run_id] = task


async def _run_and_cleanup(run_id: str) -> None:
    try:
        await orchestrator.execute_run(run_id)
    finally:
        task = active_tasks.get(run_id)
        if task and task.done():
            active_tasks.pop(run_id, None)
