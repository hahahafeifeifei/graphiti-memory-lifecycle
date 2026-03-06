#!/usr/bin/env python3
"""
graphiti-cold-archive.py — Cold archive for Graphiti memory.

Simplified workflow:
  1. Archive and delete stale Episodic nodes (only these are archived)
  2. Delete stale Entity nodes (no archive)
  3. Delete stale Edges/facts (no archive)
  4. Cleanup: only delete edges where both endpoints are already deleted

Note:
  - Dangling nodes are kept (they might still be useful)
  - Only Episodic nodes with source content are archived
  - Everything else can be reconstructed

Criteria for stale:
  - Age > 30 days
  - Avg monthly queries < 3 (query_count / age_months < 3)

Usage:
    python3 graphiti-cold-archive.py [--min-age 30] [--max-query-rate 3]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from conf import (
    DEFAULT_SOCKET,
    GRAPHITI_ARCHIVE_MAX_QUERY_RATE,
    GRAPHITI_ARCHIVE_MIN_AGE_DAYS,
    MEMORY_DIR,
    send_to_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cold-archive] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cold-archive")

MEMORY_TYPES = [
    "Execute error", "User profile", "Learned knowledge",
    "Project progress", "Key decision", "Other memory",
]

TYPE_TO_FILE = {
    "Execute error": "Error", "User profile": "Profile",
    "Learned knowledge": "Learning", "Project progress": "Project",
    "Key decision": "Decision", "Other memory": "Others",
}


def archive_episodic_to_file(nodes: list[dict], archive_dir: Path) -> dict[str, list[str]]:
    """Archive Episodic nodes to cold storage files by type."""
    archived_by_type = {}
    
    for node in nodes:
        node_type = "Other memory"
        name = node.get("name", "Unknown")
        for mt in MEMORY_TYPES:
            if name.lower().startswith(mt.lower().split()[0]):
                node_type = mt
                break
        
        type_file = TYPE_TO_FILE.get(node_type, "Others")
        cold_path = archive_dir / f"{type_file}.cold.md"
        
        content = node.get("content", name)
        entry = (
            f"\n## {name}\n\n"
            f"### Summary\n{content}\n\n"
            f"### Metadata\n"
            f"- Type: Episodic\n"
            f"- Archived: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"- Created: {node.get('created_at', 'unknown')[:10] if node.get('created_at') else 'unknown'}\n"
            f"- Query count: {node.get('query_count', 0)}\n"
            f"- Avg monthly: {node.get('avg_monthly_queries', 0):.2f}\n"
        )
        
        with open(cold_path, "a", encoding="utf-8") as f:
            f.write(entry)
        
        if type_file not in archived_by_type:
            archived_by_type[type_file] = []
        archived_by_type[type_file].append(node["uuid"])
        
        logger.info(f"Archived: {name[:60]}...")
    
    return archived_by_type


def send_request(action: str, params: dict, socket_path: str) -> dict:
    """Send request to server."""
    return send_to_server({"action": action, **params}, socket_path)


def main():
    parser = argparse.ArgumentParser(description="Cold archive stale Graphiti nodes")
    parser.add_argument(
        "--min-age",
        type=int,
        default=GRAPHITI_ARCHIVE_MIN_AGE_DAYS,
        help=f"Minimum age in days (default: {GRAPHITI_ARCHIVE_MIN_AGE_DAYS})",
    )
    parser.add_argument(
        "--max-query-rate",
        type=float,
        default=GRAPHITI_ARCHIVE_MAX_QUERY_RATE,
        help=f"Max avg monthly queries (default: {GRAPHITI_ARCHIVE_MAX_QUERY_RATE})",
    )
    parser.add_argument("--socket", default=DEFAULT_SOCKET, help="Server socket path")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧊 Graphiti Cold Archive (Simplified)")
    print("=" * 60)
    print()
    
    ping = send_request("ping", {}, args.socket)
    if not ping.get("ok"):
        print("❌ Server not responding. Start it with:")
        print("   python3 graphiti-server.py --daemon")
        sys.exit(1)
    
    archive_dir = Path(MEMORY_DIR) / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    total_archived = 0
    total_deleted = {"episodic": 0, "entity": 0, "edge": 0}
    
    # =========================================================================
    # PHASE 1: Archive + Delete stale Episodic nodes
    # =========================================================================
    print(f"\n📦 PHASE 1: Stale Episodic Nodes (Archive + Delete)")
    print(f"   Criteria: > {args.min_age} days, < {args.max_query_rate} avg monthly queries")
    
    stale_episodic = send_request("get_stale_nodes", {
        "min_age_days": args.min_age,
        "max_query_rate": args.max_query_rate
    }, args.socket)
    
    if stale_episodic.get("ok"):
        episodic_nodes = stale_episodic.get("nodes", [])
        print(f"   Found {len(episodic_nodes)} stale Episodic nodes")
        
        if episodic_nodes:
            archived = archive_episodic_to_file(episodic_nodes, archive_dir)
            for type_file, uuids in archived.items():
                print(f"   Archived to {type_file}.cold.md: {len(uuids)} items")
            total_archived += len(episodic_nodes)

            uuids = [n["uuid"] for n in episodic_nodes]
            result = send_request("archive_nodes", {"uuids": uuids}, args.socket)
            if result.get("ok"):
                deleted = result.get("deleted", 0)
                print(f"   Deleted {deleted} Episodic nodes")
                total_deleted["episodic"] += deleted
            else:
                print(f"   Error: {result.get('error')}")
    else:
        print(f"   Error: {stale_episodic.get('error')}")
    
    # =========================================================================
    # PHASE 2: Delete stale Entity nodes (NO ARCHIVE)
    # =========================================================================
    print(f"\n🗑️  PHASE 2: Stale Entity Nodes (Delete Only)")
    
    stale_entities = send_request("get_stale_entities", {
        "min_age_days": args.min_age,
        "max_query_rate": args.max_query_rate
    }, args.socket)
    
    if stale_entities.get("ok"):
        entity_nodes = stale_entities.get("nodes", [])
        print(f"   Found {len(entity_nodes)} stale Entity nodes")
        
        if entity_nodes:
            uuids = [n["uuid"] for n in entity_nodes]
            result = send_request("delete_entities", {"uuids": uuids}, args.socket)
            if result.get("ok"):
                deleted_entities = result.get("deleted", 0)
                deleted_edges = result.get("edges_deleted", 0)
                print(f"   Deleted {deleted_entities} Entities + {deleted_edges} connected Edges")
                total_deleted["entity"] += deleted_entities
                total_deleted["edge"] += deleted_edges
            else:
                print(f"   Error: {result.get('error')}")
    else:
        print(f"   Error: {stale_entities.get('error')}")
    
    # =========================================================================
    # PHASE 3: Delete stale Edges (NO ARCHIVE)
    # =========================================================================
    print(f"\n🗑️  PHASE 3: Stale Edges (Delete Only)")
    
    stale_edges = send_request("get_stale_edges", {
        "min_age_days": args.min_age,
        "max_query_rate": args.max_query_rate
    }, args.socket)
    
    if stale_edges.get("ok"):
        edge_items = stale_edges.get("edges", [])
        print(f"   Found {len(edge_items)} stale Edges")
        
        if edge_items:
            uuids = [e["uuid"] for e in edge_items]
            result = send_request("delete_edges", {"uuids": uuids}, args.socket)
            if result.get("ok"):
                deleted = result.get("deleted", 0)
                print(f"   Deleted {deleted} Edges")
                total_deleted["edge"] += deleted
            else:
                print(f"   Error: {result.get('error')}")
    else:
        print(f"   Error: {stale_edges.get('error')}")

    # =========================================================================
    # PHASE 4: Delete dangling edge records (both source/target entities missing)
    # =========================================================================
    print(f"\n🧹 PHASE 4: Dangling Edge Records Cleanup")

    dangling_edges = send_request("get_dangling_edges", {}, args.socket)
    if dangling_edges.get("ok"):
        edge_items = dangling_edges.get("edges", [])
        print(f"   Found {len(edge_items)} dangling edge records")
        if edge_items:
            uuids = [e["uuid"] for e in edge_items]
            result = send_request("delete_edges", {"uuids": uuids}, args.socket)
            if result.get("ok"):
                deleted = int(result.get("deleted", 0) or 0)
                print(f"   Deleted {deleted} dangling edge records")
                total_deleted["edge"] += deleted
            else:
                print(f"   Error: {result.get('error')}")
    else:
        print(f"   Error: {dangling_edges.get('error')}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ Cold archive complete")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   Archived: {total_archived} Episodic nodes")
    print(f"   Deleted:  {total_deleted['episodic']} Episodic, {total_deleted['entity']} Entities, {total_deleted['edge']} Edges")


if __name__ == "__main__":
    main()
