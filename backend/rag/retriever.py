import pickle
import os
import numpy as np
import faiss

# file extensions for index and metadata persistence
INDEX_EXTENSION = ".faiss"
METADATA_EXTENSION = ".pkl"


class RetrieverError(Exception):
    # raised for any failure during retrieval operations
    pass


class Retriever:
    # wraps a FAISS Flat L2 index with parallel metadata storage

    def __init__(self, dimension: int):
        # dimension must match the embedding model output size
        if not isinstance(dimension, int) or dimension <= 0:
            raise RetrieverError(f"Dimension must be a positive integer, got: {dimension}")
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        # adds float32 vectors and corresponding metadata dicts to the index
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
            raise RetrieverError("Vectors must be a 2-D numpy array.")
        if vectors.shape[1] != self.dimension:
            raise RetrieverError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}."
            )
        if len(vectors) != len(metadata):
            raise RetrieverError("Number of vectors and metadata entries must match.")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        try:
            self.index.add(vectors)
            self.metadata.extend(metadata)
        except Exception as e:
            raise RetrieverError(f"Failed to add vectors: {e}") from e

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        # returns top_k closest entries as list of dicts with 'score' and metadata fields
        if self.index.ntotal == 0:
            raise RetrieverError("Index is empty. Add vectors before searching.")
        if not isinstance(query_vector, np.ndarray) or query_vector.ndim != 1:
            raise RetrieverError("Query vector must be a 1-D numpy array.")
        if query_vector.shape[0] != self.dimension:
            raise RetrieverError(
                f"Query dimension {query_vector.shape[0]} does not match index dimension {self.dimension}."
            )
        if not isinstance(top_k, int) or top_k <= 0:
            raise RetrieverError("top_k must be a positive integer.")
        top_k = min(top_k, self.index.ntotal)
        query = query_vector.astype(np.float32).reshape(1, -1)
        try:
            distances, indices = self.index.search(query, top_k)
        except Exception as e:
            raise RetrieverError(f"Search failed: {e}") from e
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            entry = {"score": float(dist)}
            entry.update(self.metadata[idx])
            results.append(entry)
        return results

    def save(self, path: str) -> None:
        # saves FAISS index to <path>.faiss and metadata to <path>.pkl
        if not path:
            raise RetrieverError("Save path must be a non-empty string.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            faiss.write_index(self.index, path + INDEX_EXTENSION)
            with open(path + METADATA_EXTENSION, "wb") as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            raise RetrieverError(f"Failed to save index: {e}") from e

    def load(self, path: str) -> None:
        # loads FAISS index and metadata from files at the given base path
        index_path = path + INDEX_EXTENSION
        meta_path = path + METADATA_EXTENSION
        if not os.path.exists(index_path):
            raise RetrieverError(f"Index file not found: {index_path}")
        if not os.path.exists(meta_path):
            raise RetrieverError(f"Metadata file not found: {meta_path}")
        try:
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            self.dimension = self.index.d
        except Exception as e:
            raise RetrieverError(f"Failed to load index: {e}") from e