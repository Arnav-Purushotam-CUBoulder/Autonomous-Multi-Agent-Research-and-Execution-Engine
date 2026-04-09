# Atlas: Multi-Agent Research and Execution Platform

Atlas is a backend-only orchestration service for long-running research and execution tasks. It uses FastAPI, SQLite, and the OpenAI Agents SDK to coordinate planner, researcher, executor, and critic roles, while keeping every stage resumable and traceable. Artifacts are stored locally by default and can also be pushed to Google Cloud Storage for Cloud Run deployments.

## Highlights

- FastAPI orchestration API with resumable runs and step-level persistence
- Planner, researcher, executor, and critic agent stages with deterministic offline fallback
- SQLite storage for runs, revisions, and step outputs
- Artifact persistence to local disk with optional Google Cloud Storage upload
- Dockerfile and Cloud Run service manifest for GCP deployment

## Quick start

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Optional environment variables

```bash
export OPENAI_API_KEY=sk-...
export ATLAS_MODEL=gpt-4.1-mini
export GCP_GCS_BUCKET=atlas-agent-artifacts
export GCP_GCS_PREFIX=atlas-runs
```

If `OPENAI_API_KEY` is not set, Atlas still runs using deterministic local planning, research, execution, and critique fallbacks so the pipeline remains demoable offline.

## API endpoints

- `GET /health`
- `GET /runs`
- `POST /runs`
- `POST /runs/{run_id}/resume`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/report`

## GCP deployment

Atlas includes a `Dockerfile` and `cloudrun.yaml` so the FastAPI service can be deployed to Google Cloud Run while writing execution artifacts to Google Cloud Storage.
