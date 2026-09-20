"""OpenAI embeddings with batching and an on-disk cache.

Plugs straight into `src.EmbeddingStore(embedding_fn=...)`: it is callable text -> vector,
and exposes `prefetch(texts)`, which EmbeddingStore.add_documents calls to embed a whole
batch in a few API requests instead of one request per chunk. Re-running the benchmark
costs nothing: every vector is cached by sha256(model + text).

When no API key is available, the benchmark uses a deterministic lexical feature-hashing
embedder.  It is deliberately labelled as an offline baseline: unlike the old random
fallback, it gives related Vietnamese text a meaningful score while making no claim of
being a semantic model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "embeddings"
BATCH_SIZE = 96
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*%?|[^\W\d_]+", re.UNICODE)


class OfflineHashEmbedder:
    """Dependency-free lexical vectors for reproducible offline benchmark runs.

    Unigrams, adjacent word pairs and accent-folded variants are projected into a fixed
    vector with signed feature hashing, then L2-normalized.  This is not a replacement
    for a multilingual embedding model; it is a useful, honest fallback for CI and for
    students who do not have an API key on the machine running the lab.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._backend_name = f"offline lexical hashing ({dim}d)"

    @staticmethod
    def _fold(text: str) -> str:
        return "".join(
            char for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )

    def __call__(self, text: str) -> list[float]:
        normalized = unicodedata.normalize("NFC", text).lower()
        words = _TOKEN.findall(normalized)
        folded = [self._fold(word) for word in words]
        features: list[tuple[str, float]] = []
        features.extend((f"w:{word}", 1.0) for word in words)
        features.extend((f"f:{word}", 0.35) for word in folded if word)
        features.extend((f"b:{left}_{right}", 1.35) for left, right in zip(words, words[1:]))

        vector = [0.0] * self.dim
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * weight
        return self._normalize(vector)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class CachedOpenAIEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
        self._backend_name = f"{self.model_name} (cached)"
        self.client = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except Exception:
                self.client = None

        if self.client is None:
            self._offline = OfflineHashEmbedder()
            self._backend_name = self._offline._backend_name
            self._cache_namespace = "offline-lexical-v1"
        else:
            self._cache_namespace = self.model_name

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_namespace = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._cache_namespace)
        self._path = CACHE_DIR / f"{safe_namespace}.jsonl"
        self._cache: dict[str, list[float]] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    key, vector = json.loads(line)
                    self._cache[key] = vector

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._cache_namespace}\n{text}".encode()).hexdigest()

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def prefetch(self, texts: list[str]) -> None:
        missing = list(dict.fromkeys(t for t in texts if self._key(t) not in self._cache))
        if not missing:
            return
        if self.client is not None:
            with self._path.open("a", encoding="utf-8") as handle:
                for start in range(0, len(missing), BATCH_SIZE):
                    batch = missing[start : start + BATCH_SIZE]
                    response = self.client.embeddings.create(model=self.model_name, input=batch)
                    for text, item in zip(batch, response.data):
                        key = self._key(text)
                        self._cache[key] = self._normalize(item.embedding)
                        handle.write(json.dumps([key, self._cache[key]]) + "\n")
        else:
            for text in missing:
                key = self._key(text)
                self._cache[key] = self._offline(text)

    def __call__(self, text: str) -> list[float]:
        key = self._key(text)
        if key not in self._cache:
            if self.client is not None:
                self.prefetch([text])
            else:
                self._cache[key] = self._offline(text)
        return self._cache[key]
