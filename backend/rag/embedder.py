import numpy as np
from sentence_transformers import SentenceTransformer

# default model name, override by passing model_name to get_embedder()
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# module-level cache to avoid reloading the model on repeated calls
_model_cache: dict[str, SentenceTransformer] = {}


class EmbedError(Exception):
    # raised for any failure during embedding
    pass


def get_embedder(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    # returns cached model instance, loads once per model name
    if model_name not in _model_cache:
        try:
            _model_cache[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbedError(f"Failed to load model '{model_name}': {e}") from e
    return _model_cache[model_name]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    # L2-normalizes rows so cosine similarity equals dot product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def embed_text(text: str, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    # embeds a single string and returns a 1-D normalized float32 vector
    if not isinstance(text, str) or not text.strip():
        raise EmbedError("Input text must be a non-empty string.")
    try:
        model = get_embedder(model_name)
        vector = model.encode([text], convert_to_numpy=True).astype(np.float32)
        return _normalize(vector)[0]
    except EmbedError:
        raise
    except Exception as e:
        raise EmbedError(f"Embedding failed: {e}") from e


def embed_batch(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    # embeds a list of strings and returns a 2-D normalized float32 matrix
    if not isinstance(texts, list) or not texts:
        raise EmbedError("Input must be a non-empty list of strings.")
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise EmbedError("Every element in the batch must be a non-empty string.")
    try:
        model = get_embedder(model_name)
        vectors = model.encode(texts, convert_to_numpy=True).astype(np.float32)
        return _normalize(vectors)
    except EmbedError:
        raise
    except Exception as e:
        raise EmbedError(f"Batch embedding failed: {e}") from e