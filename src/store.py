from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A lightweight in-memory vector store for text chunks.

    The embedding_fn parameter allows injection of mock, local or API-backed
    embeddings without changing the storage/search contract.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": dict(doc.metadata or {}),
            "embedding": embedding,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records or top_k <= 0:
            return []
        query_vec = self._embedding_fn(query)
        scored: list[dict[str, Any]] = []
        for r in records:
            score = _dot(query_vec, r["embedding"])
            scored.append({
                "id": r["id"],
                "content": r["content"],
                "metadata": r.get("metadata", {}),
                "score": float(score),
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _matches(meta: dict, filt: dict) -> bool:
        return all(meta.get(k) == v for k, v in (filt or {}).items())

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        Embeddings are prefetched in a batch when the backend supports it, then
        each document is appended to the in-memory store.
        """
        prefetch = getattr(self._embedding_fn, "prefetch", None)
        if callable(prefetch):
            prefetch([doc.content for doc in docs])
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        filtered = [
            r for r in self._store
            if all(r.get("metadata", {}).get(k) == v for k, v in metadata_filter.items())
        ]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        initial_len = len(self._store)
        self._store = [
            r for r in self._store
            if r.get("id") != doc_id and r.get("metadata", {}).get("doc_id") != doc_id
        ]
        return len(self._store) < initial_len
