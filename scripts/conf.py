"""Load config-full.env and expose config constants."""

import os, re
import socket
from urllib.parse import urlparse

def _load():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preexisting_keys = set(os.environ.keys())
    config_path = os.path.join(script_dir, "config-full.env")

    if not os.path.exists(config_path):
        raise FileNotFoundError("Missing config file: config-full.env")

    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            v = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), v)
            # Keep explicit runtime environment highest priority.
            if k in preexisting_keys:
                continue
            os.environ[k] = v

_load()

def _r(k, default=None):
    v = os.environ.get(k)
    if not v:
        if default is not None:
            return default
        raise ValueError(f"Missing: {k}. Check config-full.env")
    return v

def _opt(k, default=""):
    v = os.environ.get(k)
    return v if v is not None else default

# Graph database config (Neo4j)
NEO4J_URI      = _r("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = _r("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _opt("NEO4J_PASSWORD", "")
NEO4J_AUTH_ENABLED = _opt("NEO4J_AUTH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

GROUP_ID       = _r("GRAPHITI_GROUP_ID")
MEMORY_DIR     = os.path.expanduser(_r("MEMORY_DIR"))
WORKSPACE_DIR  = os.path.expanduser(_r("WORKSPACE_DIR"))

LLM_API_KEY    = _r("LLM_API_KEY")
LLM_BASE_URL   = _r("LLM_BASE_URL")
LLM_MODEL      = _r("LLM_MODEL")

GRAPHITI_RECALL_LIMIT = int(_opt("GRAPHITI_RECALL_LIMIT", "5"))
GRAPHITI_ARCHIVE_MIN_AGE_DAYS = int(_opt("GRAPHITI_ARCHIVE_MIN_AGE_DAYS", "30"))
GRAPHITI_ARCHIVE_MAX_QUERY_RATE = float(_opt("GRAPHITI_ARCHIVE_MAX_QUERY_RATE", "3"))

# Unified embedding config for both local and API modes.
# Auto mode:
# - EMBED_MODEL_PATH exists -> local llama.cpp embedding mode
# - otherwise -> API embedding mode
EMBED_MODEL_PATH = os.path.expanduser(
    _opt("EMBED_MODEL_PATH", "~/.openclaw/models/gguf/Qwen3-Embedding-0.6B-Q8_0.gguf")
)
EMBED_DIMS = int(_opt("EMBED_DIMS", "512"))
EMBED_BASE_URL = _opt("EMBED_BASE_URL", "http://127.0.0.1:8011")
EMBED_ENDPOINT = _opt("EMBED_ENDPOINT", "/v1/embeddings")
EMBED_MODEL = _opt("EMBED_MODEL", "")
EMBED_API_KEY = _opt("EMBED_API_KEY", "")
EMBED_TIMEOUT = float(_opt("EMBED_TIMEOUT", "300"))

# ---------------------------------------------------------------------------
# Shared LLM client and Graphiti factory
# ---------------------------------------------------------------------------
import json
import logging
from typing import Any
from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import RateLimitError

_logger = logging.getLogger(__name__)

class JsonLLMClient(LLMClient):
    """Prompt-based JSON client for OpenAI-compatible chat APIs."""
    def __init__(self, config=None, cache=False):
        super().__init__(config or LLMConfig(), cache)
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model

    async def _generate_response(self, messages, response_model=None, max_tokens=32768, **kw):
        msgs = []
        for m in messages:
            c = m.content
            if m.role == "system" and response_model:
                c += f"\n\nJSON schema:\n{json.dumps(response_model.model_json_schema(), indent=2)}\nJSON only."
            msgs.append({"role": m.role, "content": c})
        r = await self.client.chat.completions.create(
            model=self.model, messages=msgs, max_tokens=max_tokens)
        c = (r.choices[0].message.content or "").strip()
        for fence in ("```json", "```"):
            if c.startswith(fence):
                c = c[len(fence):]
        if c.endswith("```"):
            c = c[:-3]
        return json.loads(c.strip())

    async def generate_response(self, messages, response_model=None, max_tokens=None, **kw):
        try:
            result = await self._generate_response(messages, response_model, max_tokens or 32768)
            if response_model:
                try:
                    return response_model(**result).model_dump()
                except Exception as e:
                    _logger.warning(f"Schema validation: {e}")
            return result
        except json.JSONDecodeError:
            raise
        except Exception as e:
            if "rate limit" in str(e).lower():
                raise RateLimitError from e
            raise


class BatchOpenAIEmbedder(OpenAIEmbedder):
    """OpenAIEmbedder with batch size limit for DashScope compatibility."""
    MAX_BATCH = 6  # DashScope limit is 10, use 6 for safety

    async def create(self, input_data):
        if isinstance(input_data, str):
            result = await self.client.embeddings.create(
                input=input_data, model=self.config.embedding_model
            )
            return result.data[0].embedding[: self.config.embedding_dim]

        # input_data is a list — split into batches
        if isinstance(input_data, list) and len(input_data) > self.MAX_BATCH:
            all_embeddings = []
            for i in range(0, len(input_data), self.MAX_BATCH):
                batch = input_data[i:i + self.MAX_BATCH]
                result = await self.client.embeddings.create(
                    input=batch, model=self.config.embedding_model
                )
                all_embeddings.extend(result.data)
            return all_embeddings[0].embedding[: self.config.embedding_dim]

        result = await self.client.embeddings.create(
            input=input_data, model=self.config.embedding_model
        )
        return result.data[0].embedding[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Batch embed with chunking to respect DashScope's batch limit."""
        if len(input_data_list) <= self.MAX_BATCH:
            result = await self.client.embeddings.create(
                input=input_data_list, model=self.config.embedding_model
            )
            return [e.embedding[: self.config.embedding_dim] for e in result.data]

        all_embeddings = []
        for i in range(0, len(input_data_list), self.MAX_BATCH):
            batch = input_data_list[i:i + self.MAX_BATCH]
            result = await self.client.embeddings.create(
                input=batch, model=self.config.embedding_model
            )
            all_embeddings.extend(
                [e.embedding[: self.config.embedding_dim] for e in result.data]
            )
        return all_embeddings


# Singleton Graphiti instance kept open to avoid repeated initialization
_graphiti_instance = None
_graphiti_mode = None

def get_pipeline_mode() -> str:
    """
    Resolve pipeline mode with auto-detection only.

    Rule:
    - EMBED_MODEL_PATH exists -> llama_cpp
    - otherwise -> api
    """
    if EMBED_MODEL_PATH and os.path.exists(EMBED_MODEL_PATH):
        return "llama_cpp"
    return "api"

def _assert_neo4j_reachable(uri: str, timeout_seconds: float = 1.5) -> None:
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return
    except OSError as e:
        raise ConnectionError(
            f"Neo4j is unreachable at {host}:{port}. "
            "Start Neo4j and verify NEO4J_URI/port 7687."
        ) from e

def get_graphiti_driver():
    """Return the driver from the singleton Graphiti instance (for raw queries)."""
    if _graphiti_instance is not None:
        return _graphiti_instance.driver
    return None

def create_graphiti() -> Graphiti:
    """Create or return cached Graphiti instance (Neo4j backend only)."""
    global _graphiti_instance, _graphiti_mode
    pipeline_mode = get_pipeline_mode()

    if _graphiti_instance is not None and _graphiti_mode == pipeline_mode:
        return _graphiti_instance

    # Mode changed: force re-create
    _graphiti_instance = None

    _assert_neo4j_reachable(NEO4J_URI)

    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    password = NEO4J_PASSWORD if NEO4J_AUTH_ENABLED else ""
    driver = Neo4jDriver(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=password,
        database="neo4j",
    )

    if pipeline_mode == "api":
        missing = [
            key for key, value in (
                ("EMBED_API_KEY", EMBED_API_KEY),
                ("EMBED_BASE_URL", EMBED_BASE_URL),
                ("EMBED_MODEL", EMBED_MODEL),
            ) if not value
        ]
        if missing:
            raise ValueError(
                "API mode selected but required config is missing: "
                + ", ".join(missing)
            )

        embedder = BatchOpenAIEmbedder(config=OpenAIEmbedderConfig(
            api_key=EMBED_API_KEY,
            base_url=EMBED_BASE_URL,
            embedding_model=EMBED_MODEL,
            embedding_dim=EMBED_DIMS,
        ))
    else:
        # GGUF Embedder (local llama-cpp server)
        from gguf_local import LlamaCppHttpEmbedder

        embedder = LlamaCppHttpEmbedder(
            embedding_dim=EMBED_DIMS,
            base_url=EMBED_BASE_URL,
            endpoint=EMBED_ENDPOINT,
            model=EMBED_MODEL,
            api_key=EMBED_API_KEY,
            timeout_seconds=EMBED_TIMEOUT,
        )

    _graphiti_instance = Graphiti(
        graph_driver=driver,
        llm_client=JsonLLMClient(config=LLMConfig(
            api_key=LLM_API_KEY, base_url=LLM_BASE_URL, model=LLM_MODEL)),
        embedder=embedder,
    )
    _graphiti_mode = pipeline_mode
    return _graphiti_instance


async def init_graphiti_schema():
    """Initialize schema migration (idempotent). Neo4j doesn't need manual FTS indexes."""
    driver = get_graphiti_driver()
    if driver is None:
        return

    # Neo4j: properties created automatically when set; nothing to do here
    # Schema initialization handled by build_indices_and_constraints()
    _logger.debug("Neo4j schema ready")


# ---------------------------------------------------------------------------
# Socket client for server-based operations (avoids DB lock conflicts)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
RUNTIME_DIR = os.path.join(SKILL_DIR, "runtime")
DEFAULT_SOCKET = os.path.join(RUNTIME_DIR, "graphiti-recall.sock")
DEFAULT_SOCKET_TIMEOUT = 600.0
LONG_SOCKET_TIMEOUT = 600.0

def send_to_server(
    request: dict,
    socket_path: str = DEFAULT_SOCKET,
    timeout_seconds: float = DEFAULT_SOCKET_TIMEOUT,
) -> dict:
    """Send JSON request to graphiti-server via Unix socket, return response."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout_seconds)
        sock.connect(socket_path)
        
        # Send request
        sock.sendall((json.dumps(request) + "\n").encode())
        
        # Receive response
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
            if b"\n" in chunk:
                break
        
        sock.close()
        return json.loads(response.decode().strip())
    except Exception as e:
        return {"ok": False, "error": f"Socket error: {e}"}


def add_episode_via_server(name: str, body: str, source: str = "text", 
                           source_description: str = "") -> tuple[bool, str]:
    """Add episode via server socket (no direct DB access needed)."""
    response = send_to_server({
        "action": "add_episode",
        "name": name,
        "body": body,
        "source": source,
        "source_description": source_description
    }, timeout_seconds=LONG_SOCKET_TIMEOUT)
    
    if response.get("ok"):
        return True, f"{response.get('nodes', 0)} nodes, {response.get('edges', 0)} edges"
    else:
        return False, response.get("error", "Unknown error")
