from __future__ import annotations

import os
from textwrap import dedent

try:
    from agents import Agent
except ImportError:
    Agent = None

from .tools import (
    fetch_url_text,
    list_workspace_files,
    read_text_file,
    save_artifact,
    search_workspace,
    workspace_facts,
)

MODEL_NAME = os.getenv("ATLAS_MODEL", "gpt-4.1-mini")


def _agent_kwargs() -> dict[str, str]:
    return {"model": MODEL_NAME}


PLANNER_INSTRUCTIONS = dedent(
    """
    You are the planning specialist in a multi-agent research and execution system.
    Your job is to convert a high-level task into a concrete, ordered workflow.

    Always do the following:
    1. Inspect the workspace before proposing a plan.
    2. Prefer concise, executable subtasks.
    3. Include a definition of done.
    4. Return JSON only.

    Return exactly this JSON schema:
    {
      "goal": "...",
      "workspace_summary": "...",
      "subtasks": [
        {"id": "S1", "title": "...", "why": "...", "deliverable": "..."}
      ],
      "definition_of_done": ["..."],
      "risks": ["..."]
    }
    """
).strip()

RESEARCHER_INSTRUCTIONS = dedent(
    """
    You are the research specialist in a multi-agent research and execution system.
    Use the available tools aggressively to inspect local files and gather evidence.
    Prefer grounded evidence over speculation.
    Return concise markdown with a workspace summary, key findings, and concrete evidence.
    """
).strip()

EXECUTOR_INSTRUCTIONS = dedent(
    """
    You are the execution specialist in a multi-agent research and execution system.
    Synthesize the plan and research into a polished deliverable.
    Save the final markdown report using save_artifact with the artifact name final_report.
    Return JSON containing artifact_path and report_markdown.
    """
).strip()

CRITIC_INSTRUCTIONS = dedent(
    """
    You are the critic specialist in a multi-agent research and execution system.
    Review the candidate output for completeness, grounding, and task alignment.
    Return JSON with:
    {
      "approved": true|false,
      "revision_request": "..."
    }
    """
).strip()

planner_agent = (
    Agent(
        name="Planner",
        instructions=PLANNER_INSTRUCTIONS,
        tools=[list_workspace_files, workspace_facts, search_workspace],
        **_agent_kwargs(),
    )
    if Agent
    else None
)

researcher_agent = (
    Agent(
        name="Researcher",
        instructions=RESEARCHER_INSTRUCTIONS,
        tools=[list_workspace_files, read_text_file, search_workspace, fetch_url_text, workspace_facts],
        **_agent_kwargs(),
    )
    if Agent
    else None
)

executor_agent = (
    Agent(
        name="Executor",
        instructions=EXECUTOR_INSTRUCTIONS,
        tools=[read_text_file, save_artifact],
        **_agent_kwargs(),
    )
    if Agent
    else None
)

critic_agent = (
    Agent(
        name="Critic",
        instructions=CRITIC_INSTRUCTIONS,
        tools=[read_text_file],
        **_agent_kwargs(),
    )
    if Agent
    else None
)


def build_planner_prompt(task: str, workspace_path: str, context: str | None) -> str:
    return dedent(
        f"""
        Task:
        {task}

        Workspace path:
        {workspace_path}

        Optional context:
        {context or "None provided."}
        """
    ).strip()


def build_research_prompt(
    task: str,
    workspace_path: str,
    context: str | None,
    plan_json: str,
    revision_request: str | None = None,
) -> str:
    extra = f"Critic revision request:\n{revision_request}\n\n" if revision_request else ""
    return dedent(
        f"""
        Original task:
        {task}

        Workspace path:
        {workspace_path}

        Optional context:
        {context or "None provided."}

        Approved plan JSON:
        {plan_json}

        {extra}Gather the evidence needed to complete the task. Prefer local workspace evidence.
        """
    ).strip()


def build_executor_prompt(
    run_id: str,
    task: str,
    plan_json: str,
    research_markdown: str,
    revision_request: str | None = None,
) -> str:
    extra = f"Critic revision request:\n{revision_request}\n\n" if revision_request else ""
    return dedent(
        f"""
        Run ID:
        {run_id}

        Task:
        {task}

        Plan JSON:
        {plan_json}

        Research summary:
        {research_markdown}

        {extra}Generate the final markdown report and save it using save_artifact.
        Use the artifact name final_report.
        """
    ).strip()


def build_critic_prompt(
    task: str,
    plan_json: str,
    research_markdown: str,
    execution_output: str,
) -> str:
    return dedent(
        f"""
        Original task:
        {task}

        Plan JSON:
        {plan_json}

        Research output:
        {research_markdown}

        Candidate execution output:
        {execution_output}
        """
    ).strip()
