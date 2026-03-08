#!/usr/bin/env python3
"""
graphiti-server.py — Long-running Graphiti recall daemon via Unix socket.

Keeps Neo4j-backed Graphiti open permanently. Accepts JSON-line requests, returns JSON-line responses.
Used by the OpenClaw plugin (index.js) instead of spawning python3 per request.

Protocol (newline-delimited JSON over Unix socket):
  Request:  {"query": "search text", "limit": 5}
  Response: {"ok": true, "result": "<FACTS>...</FACTS> ... <ENTITIES>...</ENTITIES>"}
  Error:    {"ok": false, "error": "message"}

Lifecycle:
  - Start:  python3 graphiti-server.py
  - Stop:   kill $(cat runtime/graphiti-recall.pid)  OR  send {"action": "shutdown"}
  - Status: echo '{"action":"ping"}' | socat - UNIX-CONNECT:runtime/graphiti-recall.sock

Usage:
    python3 graphiti-server.py                          # foreground
    python3 graphiti-server.py --daemon                 # background (writes PID file)
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conf import (
    GRAPHITI_RECALL_LIMIT,
    GROUP_ID,
    create_graphiti,
)
from graphiti_hook_capture import capture_from_hook

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
RUNTIME_DIR = SKILL_DIR / "runtime"
DEFAULT_SOCKET = str(RUNTIME_DIR / "graphiti-recall.sock")
DEFAULT_PID = str(RUNTIME_DIR / "graphiti-recall.pid")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [graphiti-server] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("graphiti-server")
RRF_RESULT_LIMIT = max(1, GRAPHITI_RECALL_LIMIT)
RRF_MIN_SCORE = 0.5

# ---------------------------------------------------------------------------
# Query result adapter (Neo4j)
# ---------------------------------------------------------------------------

def _rows_from_query_result(result) -> list[dict]:
    """Normalize Graphiti driver query outputs into list[dict]."""
    if result is None:
        return []

    if hasattr(result, "records"):
        records = getattr(result, "records", []) or []
    else:
        records = result

    rows: list[dict] = []
    for record in records or []:
        if isinstance(record, dict):
            rows.append(record)
            continue
        if hasattr(record, "data"):
            try:
                rows.append(record.data())
                continue
            except Exception:
                pass
        if hasattr(record, "items"):
            try:
                rows.append(dict(record.items()))
                continue
            except Exception:
                pass
    return rows

# ---------------------------------------------------------------------------
# Recall logic (reuses singleton from conf.py — DB stays open)
# ---------------------------------------------------------------------------

async def do_recall(query: str, limit: int = 5) -> str:
    """Search Graphiti and return globally ranked mixed recall results.

    Current desired behavior:
    - retrieve both edges (facts) and nodes (entities)
    - keep their original relative search order from Graphiti
    - merge them into one candidate pool
    - return only the global top-N mixed results
    - do not apply temporal expiration filtering
    """
    from datetime import datetime, timezone
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
    from graphiti_core.search.search_filters import SearchFilters

    graphiti = create_graphiti()

    search_limit = max(1, min(int(limit or RRF_RESULT_LIMIT), RRF_RESULT_LIMIT))

    cfg = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
    cfg.limit = search_limit
    cfg.reranker_min_score = RRF_MIN_SCORE

    results = await graphiti._search(
        query=query,
        config=cfg,
        group_ids=[GROUP_ID],
        search_filter=SearchFilters(),
    )

    if not results.edges and not results.nodes:
        return ""

    now = datetime.now(timezone.utc).isoformat()

    candidates: list[dict] = []
    seen_facts: set[str] = set()
    seen_entity_names: set[str] = set()

    # Preserve Graphiti's returned order within each bucket, then rank globally
    # by best available original position. This avoids separate post-truncation
    # for facts/entities and gives a true global top-N mixed result set.
    for idx, edge in enumerate(results.edges or []):
        fact = getattr(edge, 'fact', '').strip()
        if not fact or fact in seen_facts:
            continue
        seen_facts.add(fact)

        valid_at = getattr(edge, 'valid_at', None)
        invalid_at = getattr(edge, 'invalid_at', None)
        date_range = ""
        if valid_at:
            valid_str = valid_at.strftime("%Y-%m-%d") if hasattr(valid_at, 'strftime') else str(valid_at)[:10]
            if invalid_at:
                invalid_str = invalid_at.strftime("%Y-%m-%d") if hasattr(invalid_at, 'strftime') else str(invalid_at)[:10]
                date_range = f" (Date range: {valid_str} - {invalid_str})"
            else:
                date_range = f" (Date range: {valid_str} - present)"

        candidates.append({
            'kind': 'fact',
            'uuid': getattr(edge, 'uuid', None),
            'text': f"- {fact}{date_range}",
            'bucket_rank': idx + 1,
        })

    for idx, node in enumerate(results.nodes or []):
        name = getattr(node, 'name', '').strip()
        if not name or name in seen_entity_names:
            continue
        seen_entity_names.add(name)

        summary = getattr(node, 'summary', '').strip()
        text = f"- {name}"
        if summary:
            text += f": {summary}"

        candidates.append({
            'kind': 'entity',
            'uuid': getattr(node, 'uuid', None),
            'text': text,
            'bucket_rank': idx + 1,
        })

    if not candidates:
        return ""

    # Global ranking: mix nodes and edges together by their original position in
    # the Graphiti result buckets. Ties prefer facts first, then entities.
    candidates.sort(key=lambda item: (item['bucket_rank'], 0 if item['kind'] == 'fact' else 1))
    top_items = candidates[:search_limit]

    facts = [item for item in top_items if item['kind'] == 'fact']
    entities = [item for item in top_items if item['kind'] == 'entity']

    logger.info(
        "Recall mixed top: %s",
        [
            {
                'kind': item['kind'],
                'bucket_rank': item['bucket_rank'],
                'text': item['text'][:80],
            }
            for item in top_items
        ],
    )

    for item in top_items:
        if item.get('uuid'):
            await _update_query_stats(str(item['uuid']), now)

    output_parts = []
    output_parts.append("FACTS and ENTITIES represent relevant context to the current conversation.")
    output_parts.append("These are the most relevant recalled items for the current conversation.")
    output_parts.append(f"Showing the global top {len(top_items)} mixed recall results.")

    if facts:
        output_parts.append("")
        output_parts.append("<FACTS>")
        for item in facts:
            output_parts.append(item['text'])
        output_parts.append("</FACTS>")

    if entities:
        output_parts.append("")
        output_parts.append("These are the most relevant entities")
        output_parts.append("ENTITY_NAME: entity summary")
        output_parts.append("<ENTITIES>")
        for item in entities:
            output_parts.append(item['text'])
        output_parts.append("</ENTITIES>")

    return "\n".join(output_parts)

async def _update_query_stats(uuid: str, timestamp: str):
    """Update query_count and last_query_time on matching nodes/relationships."""
    try:
        from conf import get_graphiti_driver
        driver = get_graphiti_driver()
        if driver is None:
            return
        
        for table in ("Episodic", "Entity"):
            try:
                await driver.execute_query(
                    f"MATCH (e:{table} {{uuid: $uuid}}) "
                    f"SET e.query_count = coalesce(e.query_count, 0) + 1, "
                    f"    e.last_query_time = $time",
                    uuid=uuid,
                    time=timestamp,
                )
            except Exception:
                pass

        try:
            await driver.execute_query(
                "MATCH ()-[r:RELATES_TO {uuid: $uuid}]-() "
                "SET r.query_count = coalesce(r.query_count, 0) + 1, "
                "    r.last_query_time = $time",
                uuid=uuid,
                time=timestamp,
            )
        except Exception:
            pass
            
    except Exception as e:
        logger.debug(f"Failed to update query stats for {uuid[:12]}: {e}")


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

_request_count = 0

async def handle_request(raw: str) -> str:
    """Process one JSON request, return JSON response."""
    global _request_count

    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        return json.dumps({"ok": False, "error": f"Invalid JSON: {e}"})

    action = req.get("action", "recall")

    if action == "ping":
        return json.dumps({"ok": True, "status": "alive", "requests": _request_count})

    if action == "shutdown":
        return json.dumps({"ok": True, "status": "shutting_down"})

    if action == "recall":
        query = req.get("query", "").strip()
        if not query:
            return json.dumps({"ok": False, "error": "Empty query"})

        limit = req.get("limit", RRF_RESULT_LIMIT)
        try:
            _request_count += 1
            result = await do_recall(query, limit)
            return json.dumps({"ok": True, "result": result})
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return json.dumps({"ok": False, "error": str(e)})

    if action == "add_episode":
        # Add new episode via server (so scripts don't need direct DB access)
        import time
        import gc
        start_time = time.time()
        
        name = req.get("name", "").strip()
        body = req.get("body", "").strip()
        source = req.get("source", "text")
        source_desc = req.get("source_description", "")
        
        if not name or not body:
            return json.dumps({"ok": False, "error": "Missing name or body"})
        
        logger.info(f"Adding episode: {name[:60]}...")
        
        try:
            from graphiti_core.nodes import EpisodeType
            from datetime import datetime, timezone
            
            graphiti = create_graphiti()
            result = await graphiti.add_episode(
                name=name,
                episode_body=body,
                source=EpisodeType.text if source == "text" else EpisodeType.json,
                source_description=source_desc,
                reference_time=datetime.now(timezone.utc),
                group_id=GROUP_ID,
            )
            
            # Extract result data before cleanup
            nodes_count = len(result.nodes)
            edges_count = len(result.edges)
            
            # Force GC to release large temporary objects after ingestion.
            del result
            gc.collect()
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Completed add_episode in {elapsed:.1f} ms")
            
            return json.dumps({
                "ok": True, 
                "nodes": nodes_count, 
                "edges": edges_count
            })
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"Add episode failed after {elapsed:.1f} ms: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Force GC even on error
            gc.collect()
            return json.dumps({"ok": False, "error": str(e)})

    if action == "episode_exists":
        # Check if an Episodic node exists by source_description (preferred) or name
        source_desc = str(req.get("source_description", "")).strip()
        name = str(req.get("name", "")).strip()

        if not source_desc and not name:
            return json.dumps({"ok": False, "error": "Missing source_description or name"})

        try:
            graphiti = create_graphiti()
            driver = graphiti.driver

            if source_desc:
                query_result = await driver.execute_query(
                    "MATCH (e:Episodic {source_description: $sd}) RETURN count(e) AS c",
                    sd=source_desc,
                )
            else:
                query_result = await driver.execute_query(
                    "MATCH (e:Episodic {name: $name}) RETURN count(e) AS c",
                    name=name,
                )

            result_list = _rows_from_query_result(query_result)
            count = 0
            if result_list:
                count = int(result_list[0].get("c", 0) or 0)

            return json.dumps({"ok": True, "exists": count > 0, "count": count})
        except Exception as e:
            logger.error(f"episode_exists failed: {e}")
            return json.dumps({"ok": False, "error": str(e)})

    if action == "capture_from_hook":
        trigger = req.get("trigger", "unknown")
        session_file = req.get("session_file")
        messages = req.get("messages")
        session_id = req.get("session_id")
        now_file_path = req.get("now_file_path")

        try:
            result = await capture_from_hook(
                trigger=trigger,
                session_file=session_file,
                messages=messages,
                session_id=session_id,
                now_file_path=now_file_path,
            )
            return json.dumps({"ok": True, **result})
        except Exception as e:
            logger.error(f"capture_from_hook failed: {e}")
            return json.dumps({"ok": False, "error": str(e)})

    if action == "get_stale_nodes":
        # Get nodes older than N days with low query count
        min_age_days = req.get("min_age_days", 30)
        max_query_rate = req.get("max_query_rate", 3)  # avg monthly queries
        
        try:
            from datetime import datetime, timezone, timedelta
            
            graphiti = create_graphiti()
            driver = graphiti.driver
            
            # Calculate cutoff date
            cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
            
            # Query stale Episodic nodes
            query = """
                MATCH (e:Episodic)
                WHERE e.created_at < datetime($cutoff)
                RETURN e.uuid AS uuid, e.name AS name, e.created_at AS created_at,
                       e.query_count AS query_count, e.content AS content
            """
            query_result = await driver.execute_query(query, cutoff=cutoff)
            result_list = _rows_from_query_result(query_result)
            
            nodes = []
            # Result is a list of dicts
            for row in result_list:
                if not isinstance(row, dict):
                    continue
                uuid = row.get('uuid')
                name = row.get('name')
                created_at = row.get('created_at')
                query_count = row.get('query_count', 0)
                content = row.get('content')
                
                if not created_at:
                    continue
                try:
                    age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(str(created_at))).days
                    age_months = max(1, age_days / 30)
                    avg_monthly = (query_count or 0) / age_months
                    
                    if avg_monthly < max_query_rate:
                        nodes.append({
                            "uuid": str(uuid),
                            "name": str(name or ""),
                            "created_at": str(created_at),
                            "query_count": query_count or 0,
                            "content": str(content or ""),
                            "avg_monthly_queries": avg_monthly
                        })
                except Exception as e:
                    logger.debug(f"Skipping node {uuid}: {e}")
                    continue
            
            return json.dumps({"ok": True, "nodes": nodes, "count": len(nodes)})
        except Exception as e:
            logger.error(f"Get stale nodes failed: {e}")
            return json.dumps({"ok": False, "error": str(e)})

    if action == "archive_nodes":
        # Delete nodes and return their content for archiving
        uuids = req.get("uuids", [])
        
        if not uuids:
            return json.dumps({"ok": True, "deleted": 0})
        
        try:
            graphiti = create_graphiti()
            driver = graphiti.driver
            
            # Get content before deletion
            content_map = {}
            for uuid in uuids:
                query_result = await driver.execute_query(
                    "MATCH (e:Episodic {uuid: $u}) RETURN e.content AS content",
                    u=uuid
                )
                result_list = _rows_from_query_result(query_result)
                if result_list:
                    row = result_list[0]
                    if row.get('content'):
                        content_map[uuid] = str(row['content'])
            
            # Delete nodes
            deleted = 0
            for uuid in uuids:
                try:
                    await driver.execute_query(
                        "MATCH (e:Episodic {uuid: $u}) DETACH DELETE e",
                        u=uuid
                    )
                    deleted += 1
                except:
                    pass
            
            return json.dumps({"ok": True, "deleted": deleted, "content_map": content_map})
        except Exception as e:
            logger.error(f"Archive nodes failed: {e}")
            return json.dumps({"ok": False, "error": str(e)})

    if action == "get_stale_entities":
        # Get stale Entity nodes based on query statistics
        min_age_days = req.get("min_age_days", 30)
        max_query_rate = req.get("max_query_rate", 3)
        
        try:
            from datetime import datetime, timezone, timedelta
            
            graphiti = create_graphiti()
            driver = graphiti.driver
            
            cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
            
            query = """
                MATCH (e:Entity)
                WHERE e.created_at < datetime($cutoff)
                RETURN e.uuid AS uuid, e.name AS name, e.created_at AS created_at,
                       e.query_count AS query_count, e.summary AS summary
            """
            query_result = await driver.execute_query(query, cutoff=cutoff)
            result_list = _rows_from_query_result(query_result)
            
            nodes = []
            for row in result_list:
                if not isinstance(row, dict):
                    continue
                uuid = row.get('uuid')
                created_at = row.get('created_at')
                query_count = row.get('query_count', 0)
                
                if not created_at:
                    continue
                try:
                    age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(str(created_at))).days
                    age_months = max(1, age_days / 30)
                    avg_monthly = (query_count or 0) / age_months
                    
                    if avg_monthly < max_query_rate:
                        nodes.append({
                            "uuid": str(uuid),
                            "name": str(row.get('name', '')),
                            "created_at": str(created_at),
                            "query_count": query_count or 0,
                            "summary": str(row.get('summary', '')),
                            "avg_monthly_queries": avg_monthly
                        })
                except Exception as e:
                    logger.debug(f"Skipping entity {uuid}: {e}")
                    continue
            
            return json.dumps({"ok": True, "nodes": nodes, "count": len(nodes)})
        except Exception as e:
            logger.error(f"Get stale entities failed: {e}")
            return json.dumps({"ok": False, "error": str(e)})

    if action == "get_stale_edges":
        # Get stale RELATES_TO facts based on query statistics
        min_age_days = req.get("min_age_days", 30)
        max_query_rate = req.get("max_query_rate", 3)
        
        try:
            from datetime import datetime, timezone, timedelta
            
            graphiti = create_graphiti()
            driver = graphiti.driver
            
            cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
            
            query = """
                MATCH (n:Entity)-[r:RELATES_TO]->(m:Entity)
                WHERE r.valid_at < datetime($cutoff)
                RETURN r.uuid AS uuid, r.fact AS fact, r.valid_at AS valid_at,
                       r.query_count AS query_count, r.last_query_time AS last_query_time,
                       n.name AS from_entity, m.name AS to_entity
            """
            query_result = await driver.execute_query(query, cutoff=cutoff)
            result_list = _rows_from_query_result(query_result)
            
            edges = []
            for row in result_list:
                if not isinstance(row, dict):
                    continue
                uuid = row.get('uuid')
                valid_at = row.get('valid_at')
                query_count = row.get('query_count', 0)
                
                if not valid_at:
                    continue
                try:
                    age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(str(valid_at))).days
                    age_months = max(1, age_days / 30)
                    avg_monthly = (query_count or 0) / age_months
                    
                    if avg_monthly < max_query_rate:
                        edges.append({
                            "uuid": str(uuid),
                            "fact": str(row.get('fact', '')),
                            "valid_at": str(valid_at),
                            "query_count": query_count or 0,
                            "from_entity": str(row.get('from_entity', '')),
                            "to_entity": str(row.get('to_entity', '')),
                            "avg_monthly_queries": avg_monthly
                        })
                except Exception as e:
                    logger.debug(f"Skipping edge {uuid}: {e}")
                    continue
            
            return json.dumps({"ok": True, "edges": edges, "count": len(edges)})
        except Exception as e:
            logger.error(f"Get stale edges failed: {e}")
            return json.dumps({"ok": False, "error": str(e)})

    if action == "delete_entities":
        # Delete entities and count connected RELATES_TO relationships
        uuids = req.get("uuids", [])
        
        if not uuids:
            return json.dumps({"ok": True, "deleted": 0, "edges_deleted": 0})
        
        try:
            graphiti = create_graphiti()
            driver = graphiti.driver
            
            deleted_entities = 0
            deleted_edges = 0
            
            for uuid in uuids:
                try:
                    logger.info(f"Attempting to delete entity {uuid[:12]}...")
                    
                    # First check if entity exists
                    query_result = await driver.execute_query(
                        "MATCH (e:Entity {uuid: $uuid}) RETURN e.uuid AS found",
                        uuid=uuid
                    )
                    check_rows = _rows_from_query_result(query_result)
                    
                    if not check_rows:
                        logger.warning(f"Entity {uuid[:12]} not found, skipping")
                        continue
                    
                    # Count connected edges first
                    count_result = await driver.execute_query(
                        "MATCH (n:Entity {uuid: $uuid})-[r:RELATES_TO]-() RETURN count(r) AS edge_count",
                        uuid=uuid
                    )
                    edge_count = 0
                    count_rows = _rows_from_query_result(count_result)
                    if count_rows:
                        edge_count = int(count_rows[0].get("edge_count", 0) or 0)
                    
                    logger.info(f"Entity {uuid[:12]} has {edge_count} connected edges")
                    
                    logger.info(f"Deleting Entity node {uuid[:12]}...")
                    await driver.execute_query(
                        "MATCH (e:Entity {uuid: $uuid}) DETACH DELETE e",
                        uuid=uuid
                    )
                    logger.info(f"Entity node deleted: {uuid[:12]}")
                    
                    deleted_entities += 1
                    deleted_edges += edge_count
                    logger.info(f"Successfully deleted entity {uuid[:12]}... + {edge_count} edges")
                except Exception as e:
                    logger.error(f"Failed to delete entity {uuid[:12]}: {e}")
            
            return json.dumps({"ok": True, "deleted": deleted_entities, "edges_deleted": deleted_edges})
        except Exception as e:
            logger.error(f"Delete entities failed: {e}")
            return json.dumps({"ok": False, "error": str(e)})

    if action == "delete_edges":
        # Delete specific edge records by uuid.
        # Supports both RELATES_TO relationships and EntityEdge nodes.
        uuids = req.get("uuids", [])
        
        if not uuids:
            return json.dumps({"ok": True, "deleted": 0, "deleted_relationships": 0, "deleted_entity_edges": 0})
        
        try:
            graphiti = create_graphiti()
            driver = graphiti.driver
            
            deleted_relationships = 0
            deleted_entity_edges = 0
            for uuid in uuids:
                try:
                    rel_count_result = await driver.execute_query(
                        "MATCH ()-[r:RELATES_TO {uuid: $uuid}]-() RETURN count(r) AS c",
                        uuid=uuid
                    )
                    rel_rows = _rows_from_query_result(rel_count_result)
                    rel_count = int(rel_rows[0].get("c", 0) or 0) if rel_rows else 0
                    if rel_count > 0:
                        await driver.execute_query(
                            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]-() DELETE r",
                            uuid=uuid
                        )
                        deleted_relationships += rel_count

                    node_count_result = await driver.execute_query(
                        "MATCH (e:EntityEdge {uuid: $uuid}) RETURN count(e) AS c",
                        uuid=uuid
                    )
                    node_rows = _rows_from_query_result(node_count_result)
                    node_count = int(node_rows[0].get("c", 0) or 0) if node_rows else 0
                    if node_count > 0:
                        await driver.execute_query(
                            "MATCH (e:EntityEdge {uuid: $uuid}) DETACH DELETE e",
                            uuid=uuid
                        )
                        deleted_entity_edges += node_count

                    if rel_count or node_count:
                        logger.debug(
                            "Deleted edge uuid=%s rels=%s entity_edges=%s",
                            uuid[:12],
                            rel_count,
                            node_count,
                        )
                    else:
                        logger.debug("Edge uuid not found: %s", uuid[:12])
                except Exception as e:
                    logger.warning(f"Failed to delete edge {uuid[:12]}: {e}")
            
            total_deleted = deleted_relationships + deleted_entity_edges
            return json.dumps(
                {
                    "ok": True,
                    "deleted": total_deleted,
                    "deleted_relationships": deleted_relationships,
                    "deleted_entity_edges": deleted_entity_edges,
                }
            )
        except Exception as e:
            logger.error(f"Delete edges failed: {e}")
            return json.dumps({"ok": False, "error": str(e)})

    if action == "get_dangling_edges":
        # Find EntityEdge nodes whose source/target entities are both missing.
        # RELATES_TO relationships with missing endpoints cannot exist in Neo4j.
        try:
            graphiti = create_graphiti()
            driver = graphiti.driver

            query = """
                MATCH (e:EntityEdge)
                WITH e,
                     coalesce(e.source_node_uuid, e.source_uuid, e.source_id, '') AS src_uuid,
                     coalesce(e.target_node_uuid, e.target_uuid, e.target_id, '') AS dst_uuid
                WHERE src_uuid <> '' AND dst_uuid <> ''
                  AND NOT EXISTS { MATCH (:Entity {uuid: src_uuid}) }
                  AND NOT EXISTS { MATCH (:Entity {uuid: dst_uuid}) }
                RETURN e.uuid AS uuid,
                       e.fact AS fact,
                       src_uuid AS source_uuid,
                       dst_uuid AS target_uuid
            """
            query_result = await driver.execute_query(query)
            rows = _rows_from_query_result(query_result)

            edges: list[dict] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                uuid = str(row.get("uuid", "") or "").strip()
                if not uuid:
                    continue
                edges.append(
                    {
                        "uuid": uuid,
                        "fact": str(row.get("fact", "") or ""),
                        "source_uuid": str(row.get("source_uuid", "") or ""),
                        "target_uuid": str(row.get("target_uuid", "") or ""),
                    }
                )

            return json.dumps({"ok": True, "edges": edges, "count": len(edges)})
        except Exception as e:
            logger.error(f"Get dangling edges failed: {e}")
            return json.dumps({"ok": False, "error": str(e)})

    return json.dumps({"ok": False, "error": f"Unknown action: {action}"})


# ---------------------------------------------------------------------------
# Unix socket server
# ---------------------------------------------------------------------------

class Server:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.server = None
        self._shutdown = False

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle one client connection (supports multiple requests per connection)."""
        addr = writer.get_extra_info("peername") or "unix"
        try:
            while not self._shutdown:
                line = await asyncio.wait_for(reader.readline(), timeout=30.0)
                if not line:
                    break  # client disconnected

                raw = line.decode("utf-8").strip()
                if not raw:
                    continue

                response = await handle_request(raw)

                # Check for shutdown request
                try:
                    req = json.loads(raw)
                    if req.get("action") == "shutdown":
                        writer.write((response + "\n").encode("utf-8"))
                        await writer.drain()
                        self._shutdown = True
                        break
                except Exception:
                    pass

                writer.write((response + "\n").encode("utf-8"))
                await writer.drain()

        except asyncio.TimeoutError:
            pass  # idle client, disconnect
        except ConnectionResetError:
            pass
        except Exception as e:
            logger.warning(f"Client error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self):
        """Start the Unix socket server."""
        # Clean up stale socket
        sock_path = Path(self.socket_path)
        sock_path.parent.mkdir(parents=True, exist_ok=True)
        if sock_path.exists():
            sock_path.unlink()

        self.server = await asyncio.start_unix_server(
            self._handle_client, path=self.socket_path
        )
        # Make socket world-accessible (same user anyway)
        os.chmod(self.socket_path, 0o770)

        logger.info(f"Listening on {self.socket_path}")

        # Warm up: create the Graphiti singleton + init schema
        logger.info("Warming up Graphiti (Neo4j + indexes)...")
        try:
            from conf import init_graphiti_schema
            _ = create_graphiti()
            await init_graphiti_schema()
            logger.info("Graphiti ready")
        except Exception as e:
            logger.error(f"Graphiti init failed: {e}")

    async def serve_forever(self):
        """Run until shutdown signal."""
        await self.start()
        try:
            while not self._shutdown:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        """Clean shutdown."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        sock_path = Path(self.socket_path)
        if sock_path.exists():
            sock_path.unlink()
        logger.info("Server stopped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_pid(pid_path: str, pid: int | None = None):
    Path(pid_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pid_path, "w") as f:
        f.write(str(pid if pid is not None else os.getpid()))

def remove_pid(pid_path: str):
    try:
        os.unlink(pid_path)
    except OSError:
        pass

def main():
    parser = argparse.ArgumentParser(description="Graphiti recall daemon (Unix socket)")
    parser.add_argument("--daemon", action="store_true", help="Daemonize (fork to background)")
    args = parser.parse_args()

    if args.daemon:
        pid = os.fork()
        if pid > 0:
            # Parent — write child PID and exit
            write_pid(DEFAULT_PID, pid)
            print(f"Daemon started (PID {pid}, socket {DEFAULT_SOCKET})")
            sys.exit(0)
        # Child — detach
        os.setsid()
        sys.stdin = open(os.devnull, "r")
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    else:
        write_pid(DEFAULT_PID)

    server = Server(DEFAULT_SOCKET)

    loop = asyncio.new_event_loop()

    def _signal_handler():
        server._shutdown = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(server.serve_forever())
    finally:
        remove_pid(DEFAULT_PID)
        loop.close()


if __name__ == "__main__":
    main()
