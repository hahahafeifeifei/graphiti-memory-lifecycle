#!/usr/bin/env python3
"""
Combined daily sync:
1) Distill raw hook memories and import distilled files
2) Sync skills to graph
3) Cold-archive stale nodes
"""

from __future__ import annotations

import subprocess
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from conf import GRAPHITI_ARCHIVE_MAX_QUERY_RATE, GRAPHITI_ARCHIVE_MIN_AGE_DAYS

SCRIPT_DIR = Path(__file__).parent


def run_command(cmd: list[str]) -> int:
    cmd_str = " ".join(cmd)
    print(f"[RUN] {cmd_str}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Graphiti daily sync pipeline")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    parser.add_argument("--date", default=yesterday, help="Date for add-memory step (YYYY-MM-DD)")
    parser.add_argument(
        "--skills-root",
        default=str(SCRIPT_DIR.parent.parent),
        help="Root directory scanned by graphiti-add-skill.py --all",
    )
    parser.add_argument(
        "--archive-min-age",
        type=int,
        default=GRAPHITI_ARCHIVE_MIN_AGE_DAYS,
        help=f"Cold-archive min age in days (default: {GRAPHITI_ARCHIVE_MIN_AGE_DAYS})",
    )
    parser.add_argument(
        "--archive-max-query-rate",
        type=float,
        default=GRAPHITI_ARCHIVE_MAX_QUERY_RATE,
        help=f"Cold-archive max avg monthly query rate (default: {GRAPHITI_ARCHIVE_MAX_QUERY_RATE})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Graphiti Daily Sync")
    print("=" * 60)

    process_date = args.date
    print(f"\nStep 1: Distill raw memories for {process_date}")
    rc1 = run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "graphiti-add-memory.py"),
            "--date",
            process_date,
        ]
    )

    print("\nStep 2: Skills sync")
    rc2 = run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "graphiti-add-skill.py"),
            "--all",
            "--skills-root",
            args.skills_root,
        ]
    )

    print("\nStep 3: Cold-archive stale nodes")
    rc3 = run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "graphiti-cold-archive.py"),
            "--min-age",
            str(args.archive_min_age),
            "--max-query-rate",
            str(args.archive_max_query_rate),
        ]
    )

    print("\n" + "=" * 60)
    if rc1 == 0 and rc2 == 0 and rc3 == 0:
        print("Sync complete")
    else:
        print(f"Errors (distill={rc1}, skills={rc2}, archive={rc3})")
        sys.exit(1)


if __name__ == "__main__":
    main()
