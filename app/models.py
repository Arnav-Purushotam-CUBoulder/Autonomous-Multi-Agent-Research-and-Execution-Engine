from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

RunStatus = Literal["queued", "running", "completed", "failed"]
StepStatus = Literal["queued", "running", "completed", "failed"]


class RunCreateRequest(BaseModel):
    task: str = Field(..., description="High-level goal for the multi-agent run.")
    workspace_path: str = Field(
        default="sample_data",
        description="Local workspace or document folder the agents may inspect.",
    )
    context: str | None = Field(
        default=None,
        description="Optional extra user context, constraints, or success criteria.",
    )
    auto_start: bool = Field(
        default=True,
        description="Whether to launch the orchestration immediately.",
    )
    max_revision_cycles: int = Field(
        default=1,
        ge=0,
        le=3,
        description="How many critic-triggered revision cycles to allow.",
    )


class RunSummary(BaseModel):
    id: str
    task: str
    workspace_path: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime


class StepRecord(BaseModel):
    id: int
    run_id: str
    name: str
    agent: str
    status: StepStatus
    input_text: str | None = None
    output_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RunRecord(RunSummary):
    context: str | None = None
    final_report: str | None = None
    error_message: str | None = None
    max_revision_cycles: int = 1
    current_revision: int = 0
    steps: list[StepRecord] = Field(default_factory=list)


class ResumeRunRequest(BaseModel):
    force_restart: bool = Field(
        default=False,
        description="Restart even if the run is currently queued or running.",
    )


class EvaluationTask(BaseModel):
    name: str
    task: str
    workspace_path: str
    expected_keywords: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    name: str
    status: RunStatus
    keyword_hits: int
    total_keywords: int
    final_report_preview: str
