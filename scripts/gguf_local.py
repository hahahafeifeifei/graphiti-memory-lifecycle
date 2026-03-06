"""
Local llama.cpp HTTP embedding client used by Graphiti in auto mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import Iterable, Union

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_LLAMA_CPP_EMBED_BASE_URL = "http://127.0.0.1:8011"
DEFAULT_LLAMA_CPP_EMBED_ENDPOINT = "/v1/embeddings"
DEFAULT_LLAMA_CPP_EMBED_TIMEOUT = 20.0
_WARNED_MULTI_INPUT_CREATE = False


class LlamaCppHttpEmbedder(EmbedderClient):
    """Embed via llama.cpp server HTTP endpoint (/v1/embeddings)."""

    def __init__(
        self,
        embedding_dim: int,
        base_url: str = DEFAULT_LLAMA_CPP_EMBED_BASE_URL,
        endpoint: str = DEFAULT_LLAMA_CPP_EMBED_ENDPOINT,
        model: str = "",
        api_key: str = "",
        timeout_seconds: float = DEFAULT_LLAMA_CPP_EMBED_TIMEOUT,
    ):
        self.config = EmbedderConfig(embedding_dim=embedding_dim)
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _fit_dim(self, vector: list[float]) -> list[float]:
        target_dim = int(self.config.embedding_dim)
        if len(vector) >= target_dim:
            return vector[:target_dim]
        return vector + [0.0] * (target_dim - len(vector))

    def _parse_embeddings(self, parsed: object, expected_count: int) -> list[list[float]]:
        rows: list[tuple[int, list[float]]] = []

        if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
            for idx, item in enumerate(parsed["data"]):
                if not isinstance(item, dict):
                    continue
                emb = item.get("embedding")
                if isinstance(emb, list):
                    out_idx = int(item.get("index", idx))
                    rows.append((out_idx, [float(x) for x in emb]))
        elif isinstance(parsed, list):
            # llama.cpp /embeddings format (non-OpenAI compatible)
            for idx, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue
                emb = item.get("embedding")
                if isinstance(emb, list):
                    out_idx = int(item.get("index", idx))
                    # when pooling=none, embedding is token-level 2D; reject for this use-case
                    if emb and isinstance(emb[0], list):
                        raise RuntimeError("llama.cpp embedding response is token-level; require pooled embeddings")
                    rows.append((out_idx, [float(x) for x in emb]))

        if not rows:
            raise RuntimeError(f"Unexpected llama.cpp embeddings response: {parsed}")

        rows.sort(key=lambda x: x[0])
        vectors = [self._fit_dim(v) for _, v in rows]
        if len(vectors) != expected_count:
            raise RuntimeError(
                f"llama.cpp embeddings count mismatch: expected={expected_count}, got={len(vectors)}"
            )
        return vectors

    def _request_embed_sync(self, input_texts: list[str]) -> list[list[float]]:
        payload: dict[str, object] = {"input": input_texts}
        if self.model:
            payload["model"] = self.model

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url=f"{self.base_url}{self.endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return self._parse_embeddings(parsed, expected_count=len(input_texts))

    async def create(
        self,
        input_data: Union[str, list[str], Iterable[int], Iterable[Iterable[int]]],
    ) -> list[float]:
        global _WARNED_MULTI_INPUT_CREATE

        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list):
            if not input_data:
                raise ValueError("LlamaCppHttpEmbedder.create() received empty list[str]")
            if not all(isinstance(item, str) for item in input_data):
                raise ValueError("LlamaCppHttpEmbedder.create() expects list[str]")
            text = input_data[0]
            # graphiti-core often calls create(input_data=[single_text]); this is normal.
            # Only warn once for unexpected multi-item usage on single-vector API.
            if len(input_data) > 1 and not _WARNED_MULTI_INPUT_CREATE:
                logger.warning(
                    "LlamaCppHttpEmbedder.create() received list[str] with len>1; "
                    "using the first item only. Use create_batch() for multi-input embedding."
                )
                _WARNED_MULTI_INPUT_CREATE = True
        else:
            raise ValueError(f"LlamaCppHttpEmbedder only supports str input, got: {type(input_data)}")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._request_embed_sync, [text])
        return result[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not input_data_list:
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._request_embed_sync, input_data_list)
