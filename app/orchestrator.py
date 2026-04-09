from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from agents import Runner
except ImportError:
    Runner = None

from . import db
from .agents import (
    build_critic_prompt,
    build_executor_prompt,
    build_planner_prompt,
    build_research_prompt,
    critic_agent,
    executor_agent,
    planner_agent,
    researcher_agent,
)
from .tools import (
    raw_list_workspace_files,
    raw_save_artifact,
    raw_search_workspace,
    raw_workspace_facts,
)

BASE_DIR = Path(__file__).resolve().parent.parent


class RunOrchestrator:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, run_id: str) -> asyncio.Lock:
        return self._locks.setdefault(run_id, asyncio.Lock())

    async def execute_run(self, run_id: str) -> None:
        lock = self._get_lock(run_id)
        async with lock:
            run = db.get_run(run_id)
            if run is None:
                raise ValueError(f"Run not found: {run_id}")

            revision = run.current_revision
            db.update_run(run_id, status="running", error_message="")

            try:
                plan_json = await self._run_planning_stage(run_id)
                research_markdown = await self._run_research_stage(run_id, plan_json)
                execution_output = await self._run_execution_stage(run_id, plan_json, research_markdown)
                critic_output = await self._run_critic_stage(run_id, plan_json, research_markdown, execution_output)
                critic_payload = self._extract_json_dict(critic_output)
                revision_request = critic_payload.get("revision_request")

                while not critic_payload.get("approved", False) and revision < run.max_revision_cycles:
                    revision += 1
                    db.update_run(run_id, status="running", current_revision=revision)
                    research_markdown = await self._run_research_stage(run_id, plan_json, revision_request)
                    execution_output = await self._run_execution_stage(run_id, plan_json, research_markdown, revision_request)
                    critic_output = await self._run_critic_stage(run_id, plan_json, research_markdown, execution_output)
                    critic_payload = self._extract_json_dict(critic_output)
                    revision_request = critic_payload.get("revision_request")

                final_report = self._extract_final_report(execution_output)
                db.update_run(
                    run_id,
                    status="completed",
                    final_report=final_report,
                    error_message="",
                    current_revision=revision,
                )
            except Exception as exc:
                db.update_run(run_id, status="failed", error_message=str(exc), current_revision=revision)
                raise

    async def _run_planning_stage(self, run_id: str) -> str:
        run = self._require_run(run_id)
        input_text = build_planner_prompt(run.task, run.workspace_path, run.context)
        return await self._run_stage(
            run_id,
            "planning",
            "Planner",
            input_text,
            lambda: self._call_planner(run.task, run.workspace_path, run.context),
        )

    async def _run_research_stage(self, run_id: str, plan_json: str, revision_request: str | None = None) -> str:
        run = self._require_run(run_id)
        stage_name = "research" if run.current_revision == 0 else f"research_revision_{run.current_revision}"
        input_text = build_research_prompt(run.task, run.workspace_path, run.context, plan_json, revision_request)
        return await self._run_stage(
            run_id,
            stage_name,
            "Researcher",
            input_text,
            lambda: self._call_researcher(run.task, run.workspace_path, run.context, plan_json, revision_request),
        )

    async def _run_execution_stage(
        self,
        run_id: str,
        plan_json: str,
        research_markdown: str,
        revision_request: str | None = None,
    ) -> str:
        run = self._require_run(run_id)
        stage_name = "execution" if run.current_revision == 0 else f"execution_revision_{run.current_revision}"
        input_text = build_executor_prompt(run_id, run.task, plan_json, research_markdown, revision_request)
        return await self._run_stage(
            run_id,
            stage_name,
            "Executor",
            input_text,
            lambda: self._call_executor(run_id, run.task, plan_json, research_markdown, revision_request),
        )

    async def _run_critic_stage(
        self,
        run_id: str,
        plan_json: str,
        research_markdown: str,
        execution_output: str,
    ) -> str:
        run = self._require_run(run_id)
        stage_name = "critique" if run.current_revision == 0 else f"critique_revision_{run.current_revision}"
        input_text = build_critic_prompt(run.task, plan_json, research_markdown, execution_output)
        return await self._run_stage(
            run_id,
            stage_name,
            "Critic",
            input_text,
            lambda: self._call_critic(run.task, plan_json, research_markdown, execution_output),
        )

    async def _run_stage(
        self,
        run_id: str,
        stage_name: str,
        agent_name: str,
        input_text: str,
        run_callable,
    ) -> str:
        step_id = db.create_step(
            run_id=run_id,
            name=stage_name,
            agent=agent_name,
            status="running",
            input_text=input_text,
            metadata={"agent": agent_name},
        )
        try:
            result = await run_callable()
            output_text = self._stringify_output(result)
            metadata = {
                "last_agent": agent_name,
                "final_output": output_text[:2000],
            }
            db.update_step(
                step_id,
                status="completed",
                output_text=output_text,
                metadata=metadata,
                mark_finished=True,
            )
            return output_text
        except Exception as exc:
            db.update_step(
                step_id,
                status="failed",
                output_text=str(exc),
                metadata={"error": str(exc)},
                mark_finished=True,
            )
            raise

    async def _call_planner(self, task: str, workspace_path: str, context: str | None) -> Any:
        prompt = build_planner_prompt(task, workspace_path, context)
        if Runner and planner_agent and os.getenv("OPENAI_API_KEY"):
            return await Runner.run(planner_agent, prompt)
        return self._offline_plan(task, workspace_path, context)

    async def _call_researcher(
        self,
        task: str,
        workspace_path: str,
        context: str | None,
        plan_json: str,
        revision_request: str | None,
    ) -> Any:
        prompt = build_research_prompt(task, workspace_path, context, plan_json, revision_request)
        if Runner and researcher_agent and os.getenv("OPENAI_API_KEY"):
            return await Runner.run(researcher_agent, prompt)
        return self._offline_research(task, workspace_path, context, plan_json, revision_request)

    async def _call_executor(
        self,
        run_id: str,
        task: str,
        plan_json: str,
        research_markdown: str,
        revision_request: str | None,
    ) -> Any:
        prompt = build_executor_prompt(run_id, task, plan_json, research_markdown, revision_request)
        if Runner and executor_agent and os.getenv("OPENAI_API_KEY"):
            return await Runner.run(executor_agent, prompt)
        return self._offline_execute(run_id, task, plan_json, research_markdown, revision_request)

    async def _call_critic(
        self,
        task: str,
        plan_json: str,
        research_markdown: str,
        execution_output: str,
    ) -> Any:
        prompt = build_critic_prompt(task, plan_json, research_markdown, execution_output)
        if Runner and critic_agent and os.getenv("OPENAI_API_KEY"):
            return await Runner.run(critic_agent, prompt)
        return self._offline_critic(task, plan_json, research_markdown, execution_output)

    def _offline_plan(self, task: str, workspace_path: str, context: str | None) -> str:
        facts = self._extract_json_dict(raw_workspace_facts(workspace_path))
        files = self._extract_json_dict(raw_list_workspace_files(workspace_path, max_files=8))
        file_count = facts.get("files", 0)
        suffix_counts = facts.get("suffix_counts", {})
        visible_files = [item.get("path") for item in files.get("files", [])[:4]]

        payload = {
            "goal": task,
            "workspace_summary": f"Workspace `{workspace_path}` contains {file_count} files with leading suffixes {suffix_counts}. Visible files: {visible_files}.",
            "subtasks": [
                {
                    "id": "S1",
                    "title": "Inspect relevant workspace files",
                    "why": "Ground the execution in local evidence before proposing deliverables.",
                    "deliverable": "Workspace fact pattern and high-signal file list.",
                },
                {
                    "id": "S2",
                    "title": "Synthesize research findings",
                    "why": "Turn raw file evidence into concise conclusions and risks.",
                    "deliverable": "Markdown research summary tied to the task.",
                },
                {
                    "id": "S3",
                    "title": "Generate the final report artifact",
                    "why": "Produce a shareable deliverable with clear next actions.",
                    "deliverable": "Final markdown artifact persisted to local storage and optionally GCS.",
                },
            ],
            "definition_of_done": [
                "The plan is grounded in workspace evidence.",
                "The final report is saved as an artifact.",
                "The output includes actionable next steps.",
            ],
            "risks": [
                "Important files may be missing from the workspace snapshot.",
                "The task may need external context beyond the local workspace.",
                context or "No additional user context was provided.",
            ],
        }
        return json.dumps(payload, indent=2)

    def _offline_research(
        self,
        task: str,
        workspace_path: str,
        context: str | None,
        plan_json: str,
        revision_request: str | None,
    ) -> str:
        facts = self._extract_json_dict(raw_workspace_facts(workspace_path))
        files = self._extract_json_dict(raw_list_workspace_files(workspace_path, max_files=10))
        keywords = self._extract_keywords(task)

        evidence_lines: list[str] = []
        for keyword in keywords[:3]:
            matches = self._extract_json_dict(raw_search_workspace(workspace_path, keyword, max_hits=3))
            for match in matches.get("matches", []):
                evidence_lines.append(
                    f"- `{match.get('path')}` line {match.get('line')}: {match.get('snippet')}"
                )

        if not evidence_lines:
            evidence_lines = [
                f"- Visible file: `{item.get('path')}`"
                for item in files.get("files", [])[:5]
            ]

        return "\n".join(
            [
                "# Research Summary",
                "",
                f"- Task: {task}",
                f"- Workspace: `{workspace_path}`",
                f"- Context: {context or 'None provided.'}",
                f"- File count: {facts.get('files', 0)}",
                f"- Revision request: {revision_request or 'None'}",
                "",
                "## Evidence",
                *evidence_lines,
                "",
                "## Interpretation",
                "- The workspace contains enough structure to support a grounded first draft.",
                "- The strongest findings come from local file names and targeted keyword matches.",
                "- The final deliverable should cite local evidence and keep next steps concrete.",
                "",
                "## Approved plan snapshot",
                plan_json,
            ]
        )

    def _offline_execute(
        self,
        run_id: str,
        task: str,
        plan_json: str,
        research_markdown: str,
        revision_request: str | None,
    ) -> str:
        report_markdown = "\n".join(
            [
                "# Final Report",
                "",
                f"## Goal\n{task}",
                "",
                "## Execution Summary",
                "- Atlas completed planning, research, execution, and critique stages.",
                "- The deliverable is grounded in local workspace inspection.",
                f"- Revision request applied: {revision_request or 'None'}",
                "",
                "## Plan",
                plan_json,
                "",
                "## Research Findings",
                research_markdown,
                "",
                "## Next Actions",
                "- Review the artifact and confirm whether more external context is needed.",
                "- Promote the strongest findings into a productized workflow or demo.",
                "- Re-run with a larger workspace or additional constraints if deeper coverage is required.",
            ]
        )
        artifact_payload = self._extract_json_dict(raw_save_artifact(run_id, "final_report", report_markdown))
        execution_payload = {
            "artifact_path": artifact_payload.get("artifact_path"),
            "gcs_uri": artifact_payload.get("gcs_uri"),
            "report_markdown": report_markdown,
        }
        return json.dumps(execution_payload, indent=2)

    def _offline_critic(
        self,
        task: str,
        plan_json: str,
        research_markdown: str,
        execution_output: str,
    ) -> str:
        final_report = self._extract_final_report(execution_output)
        required_tokens = self._extract_keywords(task)[:4]
        lowered = final_report.lower()
        keyword_hits = sum(1 for token in required_tokens if token in lowered)
        approved = len(final_report) > 400 and keyword_hits >= max(1, min(2, len(required_tokens)))

        payload = {
            "approved": approved,
            "revision_request": "" if approved else "Strengthen the report with more task-specific evidence and explicit next actions.",
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def _stringify_output(output: Any) -> str:
        if isinstance(output, str):
            return output
        if hasattr(output, "final_output"):
            final_output = getattr(output, "final_output")
            if isinstance(final_output, str):
                return final_output
            output = final_output
        if hasattr(output, "model_dump_json"):
            return output.model_dump_json(indent=2)
        if hasattr(output, "model_dump"):
            return json.dumps(output.model_dump(), indent=2)
        return json.dumps(output, indent=2, default=str)

    @staticmethod
    def _extract_json_dict(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def _extract_final_report(execution_output: str) -> str:
        payload = RunOrchestrator._extract_json_dict(execution_output)
        candidate = payload.get("report_markdown")
        if isinstance(candidate, str) and candidate.strip():
            return candidate

        artifact_path = payload.get("artifact_path")
        if isinstance(artifact_path, str):
            resolved = (BASE_DIR / artifact_path).resolve()
            if resolved.exists():
                return resolved.read_text(encoding="utf-8")

        return execution_output

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
        seen: set[str] = set()
        keywords: list[str] = []
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
        return keywords

    @staticmethod
    def _require_run(run_id: str):
        run = db.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        return run
