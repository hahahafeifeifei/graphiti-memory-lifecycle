---
name: graphiti-memory-lifecycle
description: >
  Use for the full OpenClaw Graphiti memory lifecycle: memory generation from
  hooks/transcripts, memory distillation and storage (typed files + knowledge graph +
  workspace), and runtime memory recall/injection.
---

# Graphiti Memory Lifecycle

## When To Use

- Hook capture behavior (`before_compaction`, `before_reset`)
- Recall injection behavior (`before_agent_start`)
- Daily pipeline behavior (`graphiti-daily-sync.py`)
- Graphiti server/socket/knowledge-graph troubleshooting

## Non-Negotiable Contract

- Backend: knowledge graph only (Neo4j driver)
- Runtime config file: `scripts/config-full.env` only
  - Create it from `scripts/config-full.templete.env`
  - Keep `config-full.env` local; do not commit
- Embedding mode: auto
  - `EMBED_MODEL_PATH` exists -> local llama.cpp embedding
  - otherwise -> API embedding
- Inject order: `NOW.md` first, Graphiti recall second
- Daily order:
  1. `graphiti-add-memory.py`
  2. `graphiti-add-skill.py --all`
  3. `graphiti-cold-archive.py`

## Script Roles (Quick Map)

- `index.js`: OpenClaw hook integration; capture on reset/compaction and inject recall on `before_agent_start`.
- `openclaw.plugin.json`: plugin hook list and runtime config schema.
- `scripts/conf.py`: loads `config-full.env`, builds Graphiti singleton, exposes socket client helpers.
- `scripts/graphiti-server.py`: long-running Unix socket server for recall/ingest/archive actions.
- `scripts/graphiti_hook_capture.py`: transcript -> structured memory extraction; writes raw daily memory and `NOW.md`.
- `scripts/graphiti-add-memory.py`: distill raw memories by type, validate, ingest to graph, promote workspace bullets.
- `scripts/graphiti-add-skill.py`: idempotent skill ingestion (`--all` or one skill path).
- `scripts/graphiti-add-resource.py`: idempotent resource/file ingestion.
- `scripts/graphiti-cold-archive.py`: archive low-value stale graph memory through server actions.
- `scripts/graphiti-daily-sync.py`: fixed daily orchestrator (add-memory -> add-skill -> cold-archive).
- `scripts/graphiti-recall.py`: CLI utility to query recall text from server.
- `scripts/gguf_local.py`: local llama.cpp embedding HTTP client used in embedding auto mode.

## Socket Is Required

`runtime/graphiti-recall.sock` is required in current architecture.

Why:
- `index.js` sends recall/capture requests to `graphiti-server.py` via Unix socket
- ingest/archive scripts also call server actions via socket client in `conf.py`

So this socket is not optional unless client/server architecture is rewritten.

## Minimum Context Before Edits

1. Runtime health
- Knowledge graph reachable (`NEO4J_URI`)
- server running: `python3 scripts/graphiti-server.py --daemon`
- socket exists: `runtime/graphiti-recall.sock`
- plugin installed and gateway restarted after latest changes

2. Required config keys (`scripts/config-full.env`)
- `GRAPHITI_GROUP_ID`, `MEMORY_DIR`, `WORKSPACE_DIR`
- Knowledge-graph connection keys: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_AUTH_ENABLED`, `NEO4J_PASSWORD`
- `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `OPENAI_API_KEY`
- `GRAPHITI_RECALL_LIMIT`, `GRAPHITI_ARCHIVE_MIN_AGE_DAYS`, `GRAPHITI_ARCHIVE_MAX_QUERY_RATE`
- `EMBED_MODEL_PATH`, `EMBED_DIMS`, `EMBED_BASE_URL`, `EMBED_ENDPOINT`, `EMBED_MODEL`, `EMBED_API_KEY`, `EMBED_TIMEOUT`

## Bootstrap (If Environment Is New)

```bash
baseDir=~/.openclaw/workspace/skills/graphiti-memory-lifecycle

sudo apt-get update
sudo apt-get install -y neo4j
sudo systemctl enable neo4j
sudo systemctl start neo4j

pip install graphiti-core[neo4j] openai pydantic

# create local runtime config from template
cd "$baseDir/scripts"
cp config-full.templete.env config-full.env
# edit config-full.env (local only; do not commit)

openclaw plugins install --link "$baseDir"
openclaw gateway restart
python3 "$baseDir/scripts/graphiti-server.py" --daemon

openclaw cron add --name graphiti-sync --schedule "0 3 * * *" \
  --command "python3 $baseDir/scripts/graphiti-daily-sync.py"
```

## Minimal Validation

```bash
python3 -m py_compile scripts/conf.py scripts/graphiti_hook_capture.py scripts/graphiti-add-memory.py scripts/graphiti-server.py
node --check index.js
python3 scripts/graphiti-recall.py "health check"
```

## Load On Demand

- `README.md` (full workflow and operations)
- `PLAN.md` (locked sequencing decisions)
- `references/` (memory templates and classifications)
