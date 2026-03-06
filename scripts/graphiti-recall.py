#!/usr/bin/env python3
"""
Direct Graphiti recall search.

Input one query text and print the retrieved recall text directly.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conf import GRAPHITI_RECALL_LIMIT, send_to_server


def run_recall(query: str, limit: int) -> tuple[bool, str]:
    response = send_to_server(
        {
            "action": "recall",
            "query": query,
            "limit": limit,
        }
    )
    if not response.get("ok"):
        return False, response.get("error", "Unknown error")
    return True, response.get("result", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Graphiti and print recall text")
    parser.add_argument("query", nargs="*", help="Recall query text")
    parser.add_argument("--limit", type=int, default=GRAPHITI_RECALL_LIMIT, help="Recall limit per query")
    args = parser.parse_args()

    query_text = " ".join(args.query).strip()
    if not query_text and not sys.stdin.isatty():
        query_text = sys.stdin.read().strip()
    if not query_text:
        print("No query text provided.", file=sys.stderr)
        sys.exit(1)

    ping = send_to_server({"action": "ping"})
    if not ping.get("ok"):
        print("Server not responding. Start with: python3 graphiti-server.py --daemon", file=sys.stderr)
        sys.exit(1)

    ok, payload = run_recall(query_text, args.limit)
    if not ok:
        print(f"ERROR: {payload}", file=sys.stderr)
        sys.exit(1)
    print(payload.strip())


if __name__ == "__main__":
    main()
