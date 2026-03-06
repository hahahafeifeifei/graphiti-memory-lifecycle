#!/usr/bin/env python3
"""
Add a skill's SKILL.md to Graphiti as an unstructured text episode.
Type: skills

Usage:
    python3 graphiti-add-skill.py --skill-path skills/weather
"""

import argparse
import os
import re
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
# Add skill
# ---------------------------------------------------------------------------

def add_skill(skill_path: str, force: bool = False) -> tuple[str, str]:
    """Add a skill's SKILL.md to Graphiti."""
    skill_dir = Path(skill_path)
    skill_md = skill_dir / "SKILL.md"

    if not skill_dir.exists():
        return "failed", f"Skill directory not found: {skill_dir}"
    if not skill_md.exists():
        return "failed", f"SKILL.md not found in: {skill_dir}"

    # Read SKILL.md
    with open(skill_md) as f:
        content = f.read()

    # Extract skill name from frontmatter if possible
    skill_name = skill_dir.name
    frontmatter_match = re.search(r'^---\s*name:\s*([^\n]+)\s*', content, re.MULTILINE)
    if frontmatter_match:
        skill_name = frontmatter_match.group(1).strip()

    episode_name = f"Skill: {skill_name}"
    source_description = f"skill:{skill_dir.name}"

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
# Scan and add all skills
# ---------------------------------------------------------------------------

def scan_skills(skills_root: str = "skills") -> list[Path]:
    """Scan for all skill directories with SKILL.md."""
    skills = []
    root = Path(skills_root)
    if not root.exists():
        return skills
    for item in root.iterdir():
        if item.is_dir() and (item / "SKILL.md").exists():
            skills.append(item)
    return skills


def add_all_skills(skills_root: str = "skills", force: bool = False) -> tuple[int, int, int]:
    """Add all skills in the skills directory."""
    skills = scan_skills(skills_root)
    if not skills:
        print(f"No skills found in {skills_root}/")
        return (0, 0, 0)

    print(f"Found {len(skills)} skills\n")
    added = skipped = failed = 0
    for skill_dir in skills:
        status, err = add_skill(str(skill_dir), force=force)
        if status == "added":
            added += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
            if err:
                print(f"[ERROR]  {skill_dir.name}: {err}", file=sys.stderr)
    print(f"\nDone: {added} added, {skipped} skipped, {failed} failed")
    return (added, skipped, failed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add skills to Graphiti")
    parser.add_argument("--skill-path", help="Path to single skill directory")
    parser.add_argument("--all", action="store_true", help="Add all skills in skills/ directory")
    parser.add_argument("--force", action="store_true", help="Force re-index even if skill already exists")
    parser.add_argument("--skills-root", default="skills", help="Root directory for skills (default: skills)")
    args = parser.parse_args()

    if args.skill_path:
        status, err = add_skill(args.skill_path, force=args.force)
        if status == "failed":
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)
    elif args.all:
        _, _, failed = add_all_skills(args.skills_root, force=args.force)
        if failed > 0:
            sys.exit(1)
    else:
        parser.print_help()
        print("\nExample:")
        print("  python3 graphiti-add-skill.py --skill-path skills/weather")
        print("  python3 graphiti-add-skill.py --all")


if __name__ == "__main__":
    main()
