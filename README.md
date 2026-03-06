# Graphiti Memory Lifecycle

Use this skill for the full OpenClaw Graphiti memory lifecycle: memory generation from hooks/transcripts, memory distillation and storage (typed files + knowledge graph + workspace), and runtime memory recall/injection.

## Setup Tutorial

```bash
baseDir=~/.openclaw/workspace/skills/graphiti-memory-lifecycle

# 1) Install and start Neo4j (example: Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y neo4j
sudo systemctl enable neo4j
sudo systemctl start neo4j

# 2) Install dependencies
pip install graphiti-core[neo4j] openai pydantic

# 3) Configure environment
cd "$baseDir/scripts"
cp config-full.templete.env config-full.env
# edit config-full.env (local secrets file, do not commit)

# 4) Install plugin and restart gateway
openclaw plugins install --link "$baseDir"
openclaw gateway restart

# 5) Start graph memory server
python3 graphiti-server.py --daemon

# 6) Configure daily cron job (03:00 every day)
openclaw cron add --name graphiti-sync --schedule "0 3 * * *" \
  --command "python3 $baseDir/scripts/graphiti-daily-sync.py"
```

## Lifecycle (Memory Flow)

```text
┌──────────────────────────────────────────────────────────────────────┐
│ [0] CONVERSION                                                      │
│ input: hook sessionFile + event.messages                            │
│ output: normalized USER/ASSISTANT transcript                        │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ [1] CAPTURE (hook time)                                             │
│ process: hooks -> graphiti_hook_capture.py                          │
│ outputs:                                                            │
│ - memory/YYYY-MM-DD.md: raw daily memory ledger (append-only)       │
│ - NOW.md: short active context for immediate next turns             │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ [2] DISTILL + INGEST (daily batch)                                  │
│ process: graphiti-daily-sync.py -> add-memory/add-skill/cold-archive│
│ outputs:                                                            │
│ - memory/Error|Profile|Learning|Project|Decision|Others/*.md        │
│ - Neo4j + Graphiti updated (entities/facts/episodes)                │
│ - workspace prompts updated (one-sentence bullets, <1KB per file)   │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ [3] RECALL + INJECT (before each turn)                              │
│ process: before_agent_start -> index.js -> graphiti-server.py       │
│ outputs:                                                            │
│ - FACTS/ENTITIES recall text from graph                             │
│ - recall context injected into current request                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Step-by-step Details

1. Capture raw memory (extraction stage)
- The hook event provides conversation context from `sessions/*.jsonl`.
- `scripts/graphiti_hook_capture.py` extracts memory candidates and appends them into `memory/YYYY-MM-DD.md` by date.
- The same step also regenerates `NOW.md` from the built-in default template.

2. Distill and ingest (daily stage)
- `scripts/graphiti-daily-sync.py` orchestrates three actions in order.
- `scripts/graphiti-add-memory.py` reads raw daily memory, distills it into typed files, ingests structured episodes/facts/entities into Neo4j, and promotes workspace prompts from the current day's typed distilled files.
- `scripts/graphiti-add-skill.py --all` ingests skill documentation and updates skill-related graph memory.
- `scripts/graphiti-cold-archive.py` archives old low-value graph memory based on age and query-frequency thresholds.

3. Recall for each turn (runtime stage)
- At `before_agent_start`, `index.js` requests recall from `scripts/graphiti-server.py`.
- The server queries Neo4j/Graphiti and returns formatted FACTS/ENTITIES text.
- `index.js` injects this recall text into the current request context.

## Memory Types

| Type | Use when |
|---|---|
| Execute error | command failure, exception, timeout, unexpected output |
| User profile | stable user preference, communication style, habits |
| Learned knowledge | reusable methods or principles |
| Project progress | milestone, status, next step |
| Key decision | important decision and rationale |
| Other memory | useful information that does not fit other types |

## Workspace Files

| Learning Type | Promote To |
|---|---|
| Self perception | `SOUL.md` |
| Behavioral patterns | `IDENTITY.md` |
| Tool gotchas and workflow improvements | `TOOLS.md` |
| User preferences | `USER.md` |
| Project progress | `PROJECT.md` |
| Key decisions and other memory | `MEMORY.md` |

## Configuration

### Required environment variables

| Variable | Notes |
|---|---|
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_AUTH_ENABLED`, `NEO4J_PASSWORD` | Neo4j connection |
| `GRAPHITI_GROUP_ID` | Graphiti group scope |
| `MEMORY_DIR`, `WORKSPACE_DIR` | Memory and workspace roots |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Extraction / distill LLM |
| `OPENAI_API_KEY` | Compatibility for graphiti-core internal client |

### Config file bootstrap

Create local runtime config from the template:

```bash
cd ~/.openclaw/workspace/skills/graphiti-memory-lifecycle/scripts
cp config-full.templete.env config-full.env
```

`config-full.env` is local runtime config and should not be committed.

### Embedding auto-mode keys

`conf.py` uses unified embedding keys and auto-detects mode:
- Local mode when `EMBED_MODEL_PATH` exists
- API mode otherwise

Keys:
- `EMBED_MODEL_PATH`, `EMBED_DIMS`, `EMBED_BASE_URL`, `EMBED_ENDPOINT`
- `EMBED_MODEL`, `EMBED_API_KEY`, `EMBED_TIMEOUT`

### Tunable env variables (kept intentionally small)

| Variable | Default | Scope |
|---|---|---|
| `GRAPHITI_RECALL_LIMIT` | `5` | recall result count |
| `GRAPHITI_ARCHIVE_MIN_AGE_DAYS` | `30` | cold archive min age |
| `GRAPHITI_ARCHIVE_MAX_QUERY_RATE` | `3` | cold archive low-frequency threshold |

## Repository Layout

```text
graphiti-memory-lifecycle/
├── index.js
├── openclaw.plugin.json
├── SKILL.md
├── README.md
└── scripts/
    ├── conf.py
    ├── graphiti-server.py
    ├── graphiti_hook_capture.py
    ├── graphiti-add-memory.py
    ├── graphiti-add-skill.py
    ├── graphiti-add-resource.py
    ├── graphiti-cold-archive.py
    ├── graphiti-daily-sync.py
    ├── graphiti-recall.py  # optional utility
    └── gguf_local.py
```

## Operator Runbook

| Task | Command |
|---|---|
| Start server | `python3 scripts/graphiti-server.py --daemon` |
| Check server | `python3 scripts/graphiti-recall.py "health check"` |
| Manual recall | `python3 scripts/graphiti-recall.py "<query>"` |
| Distill one day | `python3 scripts/graphiti-add-memory.py --date YYYY-MM-DD` |
| Sync all skills | `python3 scripts/graphiti-add-skill.py --all --skills-root ~/.openclaw/workspace/skills` |
| Add one resource | `python3 scripts/graphiti-add-resource.py --file <path>` |
| Cold archive | `python3 scripts/graphiti-cold-archive.py --min-age 30 --max-query-rate 3` |
| Full daily chain | `python3 scripts/graphiti-daily-sync.py --date YYYY-MM-DD` |
