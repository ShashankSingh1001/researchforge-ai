import os
import numpy as np
from backend.rag.embedder import embed_text, EmbedError
from backend.rag.retriever import Retriever, RetrieverError

# supported store type identifiers
SHORT_TERM = "short_term"
LONG_TERM = "long_term"

# default directory for persisting long-term store files
DEFAULT_STORE_DIR = os.path.join("backend", "memory", "store")

# embedding dimension for all-MiniLM-L6-v2; update if model changes
EMBEDDING_DIMENSION = 384


class VectorStoreError(Exception):
    # raised for any failure during vector store operations
    pass


class VectorStore:
    # manages two named FAISS-backed stores for short-term and long-term memory

    def __init__(self, store_dir: str = DEFAULT_STORE_DIR, model_name: str = None):
        # initializes both stores; model_name is passed through to embedder
        self.store_dir = store_dir
        self.model_name = model_name
        self._stores: dict[str, Retriever] = {
            SHORT_TERM: Retriever(EMBEDDING_DIMENSION),
            LONG_TERM: Retriever(EMBEDDING_DIMENSION),
        }

    def _resolve_store(self, store_type: str) -> Retriever:
        # returns the Retriever for the given store type or raises on unknown type
        if store_type not in self._stores:
            raise VectorStoreError(
                f"Unknown store type '{store_type}'. Use '{SHORT_TERM}' or '{LONG_TERM}'."
            )
        return self._stores[store_type]

    def _store_base_path(self, store_type: str) -> str:
        # builds the base file path for a given store type under store_dir
        return os.path.join(self.store_dir, store_type)

    def store(self, text: str, source: str, store_type: str = SHORT_TERM) -> None:
        # embeds text and adds it with source metadata to the chosen store
        retriever = self._resolve_store(store_type)
        try:
            kwargs = {"model_name": self.model_name} if self.model_name else {}
            vector = embed_text(text, **kwargs)
        except EmbedError as e:
            raise VectorStoreError(f"Embedding failed during store: {e}") from e
        metadata = {"text": text, "source": source}
        try:
            retriever.add(vector.reshape(1, -1), [metadata])
        except RetrieverError as e:
            raise VectorStoreError(f"Failed to add to store: {e}") from e

    def query(self, text: str, top_k: int = 5, store_type: str = SHORT_TERM) -> list[dict]:
        # embeds query text and returns top_k matching entries from the chosen store
        retriever = self._resolve_store(store_type)
        try:
            kwargs = {"model_name": self.model_name} if self.model_name else {}
            vector = embed_text(text, **kwargs)
        except EmbedError as e:
            raise VectorStoreError(f"Embedding failed during query: {e}") from e
        try:
            return retriever.search(vector, top_k=top_k)
        except RetrieverError as e:
            raise VectorStoreError(f"Search failed: {e}") from e

    def persist(self, store_type: str = LONG_TERM) -> None:
        # saves the specified store to disk under store_dir
        retriever = self._resolve_store(store_type)
        os.makedirs(self.store_dir, exist_ok=True)
        try:
            retriever.save(self._store_base_path(store_type))
        except RetrieverError as e:
            raise VectorStoreError(f"Failed to persist store '{store_type}': {e}") from e

    def load_store(self, store_type: str = LONG_TERM) -> None:
        # loads a previously persisted store from disk into memory
        retriever = self._resolve_store(store_type)
        base_path = self._store_base_path(store_type)
        try:
            retriever.load(base_path)
        except RetrieverError as e:
            raise VectorStoreError(f"Failed to load store '{store_type}': {e}") from e

    def clear(self, store_type: str = SHORT_TERM) -> None:
        # resets the specified store to an empty index, discarding all entries
        self._resolve_store(store_type)
        self._stores[store_type] = Retriever(EMBEDDING_DIMENSION)

    def count(self, store_type: str = SHORT_TERM) -> int:
        # returns the number of vectors currently held in the specified store
        return self._resolve_store(store_type).index.ntotal