/**
 * Graphiti Memory Lifecycle RAG Plugin for OpenClaw
 *
 * Hook pipeline:
 * - before_compaction / before_reset: capture raw memories + update NOW.md
 * - after_compaction / session_start: mark pending NOW recall
 * - before_agent_start: inject NOW.md first (if pending), then Graphiti recall
 */

import { createConnection } from 'node:net';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';

const SKILL_DIR = dirname(fileURLToPath(import.meta.url));
const DEFAULT_SOCKET_PATH = join(SKILL_DIR, 'runtime', 'graphiti-recall.sock');
const HOME_DIR = process.env.HOME || '/root';

const DEFAULTS = {
  maxResults: 5,
  maxContextChars: 5000,
  timeoutMs: 600000,
  logInjections: true,
  skipShortPrompts: 10,
  socketPath: DEFAULT_SOCKET_PATH,
  captureNowEnabled: true,
  nowFilePath: 'NOW.md',
};

const VERSION = '0.5.0';
const RESET_CAPTURE_TTL_MS = 10 * 60 * 1000;

let globalPendingNowRecall = false;
const pendingNowRecallBySession = new Set();
const capturedResetSessions = new Map();

function resolveWorkspaceDir(ctx) {
  if (ctx?.workspaceDir && typeof ctx.workspaceDir === 'string') {
    return ctx.workspaceDir;
  }
  return join(HOME_DIR, '.openclaw', 'workspace');
}

function expandEnvPath(inputPath) {
  if (!inputPath || typeof inputPath !== 'string') return inputPath;
  let out = inputPath.replaceAll('${HOME}', HOME_DIR);
  if (out === '~') return HOME_DIR;
  if (out.startsWith('~/')) return join(HOME_DIR, out.slice(2));
  return out;
}

function resolvePath(inputPath, workspaceDir) {
  const expanded = expandEnvPath(inputPath);
  if (!expanded || typeof expanded !== 'string') return null;
  return isAbsolute(expanded) ? expanded : join(workspaceDir, expanded);
}

function getSessionId(event, ctx) {
  return (
    ctx?.sessionId ||
    event?.sessionId ||
    event?.session_id ||
    null
  );
}

function normalizeSessionId(input) {
  if (typeof input !== 'string') return '';
  const trimmed = input.trim();
  if (!trimmed) return '';
  return trimmed.endsWith('.jsonl') ? trimmed.slice(0, -6) : trimmed;
}

function pruneCapturedResetSessions(nowMs = Date.now()) {
  for (const [sessionId, ts] of capturedResetSessions.entries()) {
    if (nowMs - ts > RESET_CAPTURE_TTL_MS) {
      capturedResetSessions.delete(sessionId);
    }
  }
}

function rememberCapturedResetSession(sessionId) {
  const normalized = normalizeSessionId(sessionId);
  if (!normalized) return;
  pruneCapturedResetSessions();
  capturedResetSessions.set(normalized, Date.now());
}

function wasResetRecentlyCaptured(sessionId) {
  const normalized = normalizeSessionId(sessionId);
  if (!normalized) return false;
  pruneCapturedResetSessions();
  const ts = capturedResetSessions.get(normalized);
  if (!ts) return false;
  return Date.now() - ts <= RESET_CAPTURE_TTL_MS;
}

function findLatestResetTranscript(sessionId, ctx) {
  const normalizedSessionId = normalizeSessionId(sessionId);
  if (!normalizedSessionId) return null;

  const workspaceDir = resolveWorkspaceDir(ctx);
  const stateDir = dirname(workspaceDir);
  const agentId = typeof ctx?.agentId === 'string' && ctx.agentId.trim()
    ? ctx.agentId.trim()
    : 'main';

  const dirs = [
    join(stateDir, 'agents', agentId, 'sessions'),
    join(stateDir, 'agents', 'main', 'sessions'),
  ];

  let latestName = '';
  let latestPath = null;

  for (const dirPath of dirs) {
    if (!existsSync(dirPath)) continue;

    let entries;
    try {
      entries = readdirSync(dirPath);
    } catch {
      continue;
    }

    for (const name of entries) {
      const isExactReset = name.startsWith(`${normalizedSessionId}.jsonl.reset.`);
      const isTopicReset =
        name.startsWith(`${normalizedSessionId}-topic-`) &&
        name.includes('.jsonl.reset.');

      if (!isExactReset && !isTopicReset) continue;
      if (name <= latestName) continue;

      latestName = name;
      latestPath = join(dirPath, name);
    }
  }

  return latestPath;
}

function markNowRecallPending(event, ctx) {
  const sessionId = getSessionId(event, ctx);
  if (sessionId) {
    pendingNowRecallBySession.add(sessionId);
    return;
  }
  globalPendingNowRecall = true;
}

function consumeNowRecallPending(event, ctx) {
  const sessionId = getSessionId(event, ctx);
  if (sessionId && pendingNowRecallBySession.has(sessionId)) {
    pendingNowRecallBySession.delete(sessionId);
    return true;
  }
  if (globalPendingNowRecall) {
    globalPendingNowRecall = false;
    return true;
  }
  return false;
}

function readNowContext(config, ctx) {
  const workspaceDir = resolveWorkspaceDir(ctx);
  const nowPath = resolvePath(config.nowFilePath, workspaceDir);
  if (!nowPath || !existsSync(nowPath)) return '';

  try {
    const text = readFileSync(nowPath, 'utf8').trim();
    if (!text) return '';
    return `<now_context>\n${text}\n</now_context>`;
  } catch {
    return '';
  }
}

function sendSocketRequest(request, socketPath, timeoutMs) {
  return new Promise((resolve) => {
    if (!existsSync(socketPath)) {
      resolve(null);
      return;
    }

    const socket = createConnection(socketPath);
    const timer = setTimeout(() => {
      socket.destroy();
      resolve(null);
    }, timeoutMs);

    let data = '';

    socket.on('connect', () => {
      socket.write(`${JSON.stringify(request)}\n`);
    });

    socket.on('data', (chunk) => {
      data += chunk.toString();
      const nlIdx = data.indexOf('\n');
      if (nlIdx !== -1) {
        clearTimeout(timer);
        socket.destroy();
        try {
          resolve(JSON.parse(data.slice(0, nlIdx)));
        } catch {
          resolve(null);
        }
      }
    });

    socket.on('error', () => {
      clearTimeout(timer);
      resolve(null);
    });

    socket.on('timeout', () => {
      clearTimeout(timer);
      socket.destroy();
      resolve(null);
    });
  });
}

async function recallViaSocket(query, limit, socketPath, timeoutMs) {
  const resp = await sendSocketRequest(
    { action: 'recall', query, limit },
    socketPath,
    timeoutMs
  );
  if (!resp?.ok) return null;
  return resp.result || '';
}

async function captureFromHook(config, logger, hookName, event, ctx, options = {}) {
  if (!config.captureNowEnabled) return;

  const workspaceDir = resolveWorkspaceDir(ctx);
  const nowFilePath = resolvePath(config.nowFilePath, workspaceDir);
  const sessionId = options.sessionId || getSessionId(event, ctx);

  const payload = {
    action: 'capture_from_hook',
    trigger: hookName,
    session_file: options.sessionFile || event?.sessionFile || null,
    // before_reset is fire-and-forget in OpenClaw; the transcript file may be
    // archived before this plugin reads it. Keep in-memory messages as fallback.
    messages:
      options.messages !== undefined
        ? options.messages
        : (Array.isArray(event?.messages) ? event.messages : null),
    session_id: sessionId,
    agent_id: ctx?.agentId || null,
    now_file_path: nowFilePath,
  };

  const resp = await sendSocketRequest(payload, config.socketPath, config.timeoutMs);
  if (!resp?.ok) {
    logger.warn(`graphiti-memory-lifecycle: ${hookName} capture skipped (${resp?.error || 'socket unavailable'})`);
    return;
  }

  if (resp.note) {
    logger.warn(`graphiti-memory-lifecycle: ${hookName} capture note: ${resp.note}`);
    return;
  }

  if (sessionId && (hookName === 'before_reset' || hookName === 'session_start')) {
    rememberCapturedResetSession(sessionId);
  }

  logger.info(
    `graphiti-memory-lifecycle: ${hookName} captured ${resp.memory_count || 0} raw memories (NOW: ${resp.now_file || 'n/a'})`
  );
}

export default {
  id: 'graphiti-memory-lifecycle',
  name: 'Graphiti Memory Lifecycle RAG',
  description: 'Hook-driven graph memory with NOW recall + Graphiti recall',
  version: VERSION,

  register(api) {
    const config = { ...DEFAULTS, ...api.pluginConfig };
    config.socketPath = expandEnvPath(config.socketPath);
    const logger = api.logger;

    logger.info(
      `graphiti-memory-lifecycle: registered (v${VERSION}, socket=${config.socketPath})`
    );

    api.on('before_compaction', async (event, ctx) => {
      try {
        await captureFromHook(config, logger, 'before_compaction', event, ctx);
      } catch (err) {
        logger.warn(`graphiti-memory-lifecycle: before_compaction failed: ${String(err)}`);
      }
    }, { priority: 10 });

    api.on('before_reset', async (event, ctx) => {
      try {
        await captureFromHook(config, logger, 'before_reset', event, ctx);
      } catch (err) {
        logger.warn(`graphiti-memory-lifecycle: before_reset failed: ${String(err)}`);
      }
    }, { priority: 10 });

    api.on('after_compaction', (event, ctx) => {
      markNowRecallPending(event, ctx);
      logger.info('graphiti-memory-lifecycle: NOW recall scheduled after compaction');
    }, { priority: 10 });

    api.on('session_start', async (event, ctx) => {
      markNowRecallPending(event, ctx);
      logger.info('graphiti-memory-lifecycle: NOW recall scheduled on session start');

      const resumedFrom = normalizeSessionId(event?.resumedFrom);
      if (!resumedFrom) return;

      if (wasResetRecentlyCaptured(resumedFrom)) {
        logger.debug(`graphiti-memory-lifecycle: session_start fallback skipped (already captured ${resumedFrom})`);
        return;
      }

      const fallbackSessionFile = findLatestResetTranscript(resumedFrom, ctx);
      if (!fallbackSessionFile) {
        logger.warn(`graphiti-memory-lifecycle: session_start fallback skipped (no reset transcript for ${resumedFrom})`);
        return;
      }

      try {
        await captureFromHook(
          config,
          logger,
          'session_start',
          event,
          ctx,
          {
            sessionId: resumedFrom,
            sessionFile: fallbackSessionFile,
            messages: null,
          }
        );
      } catch (err) {
        logger.warn(`graphiti-memory-lifecycle: session_start fallback capture failed: ${String(err)}`);
      }
    }, { priority: 10 });

    api.on('before_agent_start', async (event, ctx) => {
      const prompt = event?.prompt || '';
      const blocks = [];

      // 1) NOW first (if pending), then clear marker
      if (consumeNowRecallPending(event, ctx)) {
        const nowContext = readNowContext(config, ctx);
        if (nowContext) {
          blocks.push(nowContext);
          if (config.logInjections) {
            logger.info(`graphiti-memory-lifecycle: injecting NOW context (${nowContext.length} chars)`);
          }
        }
      }

      // 2) Graphiti recall
      if (prompt && prompt.length >= config.skipShortPrompts) {
        const output = await recallViaSocket(
          prompt,
          config.maxResults,
          config.socketPath,
          config.timeoutMs
        );

        if (output) {
          blocks.push(output);
          if (config.logInjections) {
            const previewLines = output
              .split('\n')
              .map((line) => line.trim())
              .filter(Boolean)
              .filter((line) => line.startsWith('- '))
              .slice(0, 5);
            logger.info(`graphiti-memory-lifecycle: injecting graph recall (${output.length} chars)`);
            logger.info(
              `graphiti-memory-lifecycle: graph recall preview ${JSON.stringify({
                items: previewLines.length,
                preview: previewLines,
              })}`
            );
          }
        } else if (config.logInjections) {
          logger.debug('graphiti-memory-lifecycle: no graph recall results');
        }
      }

      if (blocks.length === 0) return;

      let context = blocks.join('\n\n');
      if (context.length > config.maxContextChars) {
        context = context.slice(0, config.maxContextChars);
      }

      return { appendContext: context };
    }, { priority: 10 });
  },
};
