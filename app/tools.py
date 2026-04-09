from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Annotated
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from agents import function_tool
except ImportError:
    def function_tool(func):  # type: ignore[misc]
        return func

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    gcs_storage = None

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".java",
}


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def raw_list_workspace_files(workspace_path: str, max_files: int = 40) -> str:
    root = _resolve_path(workspace_path)
    if not root.exists():
        return json.dumps({"error": f"Workspace does not exist: {root}"}, indent=2)

    results: list[dict[str, object]] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in TEXT_SUFFIXES and file_path.suffix:
            continue
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        results.append(
            {
                "path": str(file_path.relative_to(BASE_DIR)),
                "size_bytes": size,
            }
        )
        if len(results) >= max_files:
            break

    return json.dumps({"root": str(root.relative_to(BASE_DIR)), "files": results}, indent=2)


@function_tool
def list_workspace_files(
    workspace_path: Annotated[str, "List text-like files available in a workspace."],
    max_files: int = 40,
) -> str:
    return raw_list_workspace_files(workspace_path, max_files)


def raw_read_text_file(path: str, max_chars: int = 5000) -> str:
    file_path = _resolve_path(path)
    if not file_path.exists() or not file_path.is_file():
        return f"File not found: {file_path}"

    text = file_path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = f"{text[:max_chars]}\n\n[truncated]"
    return text


@function_tool
def read_text_file(
    path: Annotated[str, "Read a local text file and return its contents."],
    max_chars: int = 5000,
) -> str:
    return raw_read_text_file(path, max_chars)


def raw_search_workspace(workspace_path: str, query: str, max_hits: int = 10) -> str:
    root = _resolve_path(workspace_path)
    if not root.exists():
        return json.dumps({"error": f"Workspace does not exist: {root}"}, indent=2)

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    hits: list[dict[str, object]] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in TEXT_SUFFIXES and file_path.suffix:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append(
                    {
                        "path": str(file_path.relative_to(BASE_DIR)),
                        "line": line_no,
                        "snippet": line.strip(),
                    }
                )
                if len(hits) >= max_hits:
                    return json.dumps({"matches": hits}, indent=2)
    return json.dumps({"matches": hits}, indent=2)


@function_tool
def search_workspace(
    workspace_path: Annotated[str, "Search text files in the workspace for a query and return snippets."],
    query: str,
    max_hits: int = 10,
) -> str:
    return raw_search_workspace(workspace_path, query, max_hits)


def raw_fetch_url_text(url: str, max_chars: int = 6000) -> str:
    req = Request(url, headers={"User-Agent": "Atlas-Agent/1.0"})
    try:
        with urlopen(req) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except URLError as exc:
        return f"Failed to fetch URL {url}: {exc}"

    text = re.sub(r"<script.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


@function_tool
def fetch_url_text(
    url: Annotated[str, "Fetch plain text from a web page for lightweight external research."],
    max_chars: int = 6000,
) -> str:
    return raw_fetch_url_text(url, max_chars)


def _upload_to_gcs(run_id: str, safe_name: str, content: str) -> str | None:
    bucket_name = os.getenv("GCP_GCS_BUCKET")
    if not bucket_name or gcs_storage is None:
        return None

    prefix = os.getenv("GCP_GCS_PREFIX", "atlas-runs").strip("/")
    blob_name = f"{prefix}/{run_id}/{safe_name}.md"
    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content, content_type="text/markdown")
    return f"gs://{bucket_name}/{blob_name}"


def raw_save_artifact(run_id: str, name: str, content: str) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip() or "artifact")
    run_dir = ARTIFACT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = run_dir / f"{safe_name}.md"
    artifact_path.write_text(content, encoding="utf-8")
    gcs_uri = _upload_to_gcs(run_id, safe_name, content)

    return json.dumps(
        {
            "artifact_path": str(artifact_path.relative_to(BASE_DIR)),
            "gcs_uri": gcs_uri,
        },
        indent=2,
    )


@function_tool
def save_artifact(
    run_id: Annotated[str, "Persist an artifact generated during execution and return its path."],
    name: str,
    content: str,
) -> str:
    return raw_save_artifact(run_id, name, content)


def raw_workspace_facts(workspace_path: str) -> str:
    root = _resolve_path(workspace_path)
    if not root.exists():
        return json.dumps({"error": f"Workspace does not exist: {root}"}, indent=2)

    files = 0
    total_bytes = 0
    suffix_counts: dict[str, int] = {}
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        files += 1
        try:
            total_bytes += file_path.stat().st_size
        except OSError:
            pass
        suffix = file_path.suffix.lower() or "<no_extension>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

    return json.dumps(
        {
            "root": str(root.relative_to(BASE_DIR)),
            "files": files,
            "total_bytes": total_bytes,
            "suffix_counts": suffix_counts,
        },
        indent=2,
    )


@function_tool
def workspace_facts(
    workspace_path: Annotated[str, "Return simple statistics about a workspace."],
) -> str:
    return raw_workspace_facts(workspace_path)
