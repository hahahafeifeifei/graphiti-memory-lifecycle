# Before-Agent Hook Configuration

## How It Works

Before each agent turn, the hook runs `graphiti-recall.py` with the user's message as query.
Relevant facts from Graphiti are injected as a `<graphiti_memories>` system block before the
agent sees the message.

## Option A: OpenClaw Plugin (recommended)

Add to `${HOME}/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "graphiti-memory-lifecycle": {
      "path": "<workspace>/skills/graphiti-memory-lifecycle",
      "config": {
        "baseUrl": "http://localhost:8001",
        "groupId": "main-agent",
        "maxFacts": 10,
        "autoRecall": true,
        "autoIndex": false
      }
    }
  }
}
```

This uses `index.ts` which registers a `beforeAgent` hook via the OpenClaw plugin API.

Plugin config options:

| Key | Default | Description |
|-----|---------|-------------|
| `baseUrl` | `http://localhost:8001` | Graphiti API URL |
| `groupId` | `main-agent` | Graphiti group_id for this agent |
| `maxFacts` | `10` | Max facts to inject per turn |
| `autoRecall` | `true` | Search Graphiti before each turn |
| `autoIndex` | `false` | Sync memory files on startup |
| `requestTimeoutMs` | `15000` | HTTP timeout for Graphiti requests |

## Option B: Shell Hook (simpler, no TypeScript)

If the plugin system is not available, use a shell-based hook:

```json
{
  "agents": {
    "main": {
      "hooks": {
        "beforeAgent": {
          "command": "python3 <workspace>/skills/graphiti-memory-lifecycle/scripts/graphiti-recall.py",
          "injectAs": "system",
          "wrapTag": "graphiti_memories",
          "timeoutMs": 5000,
          "failSilently": true
        }
      }
    }
  }
}
```

The hook receives the user message as the first argument. Output is prepended to system context.

## Verifying the Hook

After configuration, test with:

```bash
python3 scripts/graphiti-recall.py "test query"
```

Should output a `<graphiti_memories>` block if Graphiti has matching facts, or nothing if empty.
