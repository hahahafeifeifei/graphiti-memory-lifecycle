#!/usr/bin/env python3
"""
Add any file to Graphiti as an unstructured text episode.
Type: resource

Usage:
    python3 graphiti-add-resource.py --file path/to/file.md
    python3 graphiti-add-resource.py --file path/to/file.md --name "My Resource"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conf import add_episode_via_server, send_to_server

AddStatus = Literal["added", "skipped", "failed"]


def episode_exists(source_description: str, episode_name: str) -> tuple[bool, str]:
    response = send_to_server(
        {
            "action": "episode_exists",
            "source_description": source_description,
            "name": episode_name,
        }
    )
    if not response.get("ok"):
        return False, response.get("error", "episode_exists failed")
    return bool(response.get("exists")), ""


def add_text_episode(
    episode_name: str,
    body: str,
    source_description: str,
    *,
    force: bool = False,
) -> tuple[AddStatus, str]:
    if not force:
        exists, err = episode_exists(source_description, episode_name)
        if err:
            return "failed", err
        if exists:
            return "skipped", "already indexed"

    success, msg = add_episode_via_server(
        name=episode_name,
        body=body,
        source="text",
        source_description=source_description,
    )
    if success:
        return "added", msg

    # Server-side ingest may finish after client timeout.
    if "timed out" in msg.lower():
        exists, err = episode_exists(source_description, episode_name)
        if not err and exists:
            return "added", "completed after client timeout"

    return "failed", msg


# ---------------------------------------------------------------------------
# Add resource
# ---------------------------------------------------------------------------

def add_resource(file_path: str, name: str | None = None, force: bool = False) -> tuple[str, str]:
    """Add a file to Graphiti as a resource."""
    file = Path(file_path).expanduser().resolve()

    if not file.exists():
        return "failed", f"File not found: {file}"
    if not file.is_file():
        return "failed", f"Not a file: {file}"

    # Read file content
    try:
        with file.open("r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return "failed", f"Cannot read file: {e}"

    # Determine episode name
    if name:
        episode_name = f"Resource: {name}"
    else:
        episode_name = f"Resource: {file.name}"

    source_description = f"resource:{file}"
    status, msg = add_text_episode(
        episode_name=episode_name,
        body=content,
        source_description=source_description,
        force=force,
    )
    if status == "added":
        print(f"[ADDED]  {episode_name} — {msg}")
    elif status == "skipped":
        print(f"[SKIP]   {episode_name} — {msg}")
    else:
        print(f"[ERROR]  {episode_name} — {msg}")
    return status, msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add any file to Graphiti as a resource")
    parser.add_argument("--file", required=True, help="Path to file to add")
    parser.add_argument("--name", help="Optional name for the resource (default: filename)")
    parser.add_argument("--force", action="store_true", help="Force re-index even if resource already exists")
    args = parser.parse_args()

    status, err = add_resource(args.file, name=args.name, force=args.force)
    if status == "failed":
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
