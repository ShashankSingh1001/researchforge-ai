import os
import sys
import numpy as np
import pytest
import tempfile

# ---------------------------------------------------------------------------
# embedder tests
# ---------------------------------------------------------------------------

def test_embed_text_shape_and_dtype(monkeypatch):
    # single text returns a 1-D float32 vector of length 384
    import backend.rag.embedder as embedder_mod
    monkeypatch.setattr(embedder_mod, "embed_text", _mock_embed_text)
    v = embedder_mod.embed_text("hello world")
    assert v.ndim == 1
    assert v.dtype == np.float32
    assert v.shape[0] == 384


def test_embed_text_normalized(monkeypatch):
    # output vector must have L2 norm of 1.0
    import backend.rag.embedder as embedder_mod
    monkeypatch.setattr(embedder_mod, "embed_text", _mock_embed_text)
    v = embedder_mod.embed_text("normalization check")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_embed_batch_shape(monkeypatch):
    # batch of N texts returns a 2-D matrix with N rows of length 384
    import backend.rag.embedder as embedder_mod
    monkeypatch.setattr(embedder_mod, "embed_batch", _mock_embed_batch)
    m = embedder_mod.embed_batch(["first", "second", "third"])
    assert m.ndim == 2
    assert m.shape == (3, 384)
    assert m.dtype == np.float32


def test_embed_batch_normalized(monkeypatch):
    # every row in the batch output must have L2 norm of 1.0
    import backend.rag.embedder as embedder_mod
    monkeypatch.setattr(embedder_mod, "embed_batch", _mock_embed_batch)
    m = embedder_mod.embed_batch(["a", "b", "c"])
    for row in m:
        assert abs(np.linalg.norm(row) - 1.0) < 1e-5


def test_embed_text_empty_string_raises():
    # empty or whitespace-only string must raise EmbedError
    from backend.rag.embedder import embed_text, EmbedError
    with pytest.raises(EmbedError):
        embed_text("   ")


def test_embed_batch_empty_list_raises():
    # empty list must raise EmbedError
    from backend.rag.embedder import embed_batch, EmbedError
    with pytest.raises(EmbedError):
        embed_batch([])


def test_embed_batch_invalid_element_raises():
    # list containing a whitespace-only element must raise EmbedError
    from backend.rag.embedder import embed_batch, EmbedError
    with pytest.raises(EmbedError):
        embed_batch(["valid", "   "])


# ---------------------------------------------------------------------------
# retriever tests
# ---------------------------------------------------------------------------

def test_retriever_add_increases_count():
    # adding N vectors increases index total by N
    from backend.rag.retriever import Retriever
    r = Retriever(8)
    r.add(_rand_vecs(3, 8), [{"text": f"c{i}"} for i in range(3)])
    assert r.index.ntotal == 3


def test_retriever_search_returns_top_k():
    # search returns exactly top_k results with required fields
    from backend.rag.retriever import Retriever
    r = Retriever(8)
    vecs = _rand_vecs(5, 8)
    r.add(vecs, [{"text": f"chunk {i}"} for i in range(5)])
    results = r.search(vecs[0], top_k=3)
    assert len(results) == 3
    assert all("score" in res and "text" in res for res in results)


def test_retriever_search_top_result_is_self():
    # querying with a stored vector should return itself as the closest match
    from backend.rag.retriever import Retriever
    r = Retriever(8)
    vecs = _rand_vecs(5, 8)
    r.add(vecs, [{"text": f"chunk {i}"} for i in range(5)])
    results = r.search(vecs[2], top_k=1)
    assert results[0]["text"] == "chunk 2"


def test_retriever_save_load_roundtrip():
    # index and metadata survive a save/load cycle unchanged
    from backend.rag.retriever import Retriever
    r = Retriever(8)
    vecs = _rand_vecs(4, 8)
    r.add(vecs, [{"text": f"item {i}"} for i in range(4)])
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "idx")
        r.save(base)
        r2 = Retriever(8)
        r2.load(base)
        assert r2.index.ntotal == 4
        assert r2.metadata[0]["text"] == "item 0"


def test_retriever_wrong_dimension_raises():
    # searching with a vector of wrong dimension must raise RetrieverError
    from backend.rag.retriever import Retriever, RetrieverError
    r = Retriever(8)
    r.add(_rand_vecs(2, 8), [{"text": "a"}, {"text": "b"}])
    with pytest.raises(RetrieverError):
        r.search(np.ones(16, dtype=np.float32))


def test_retriever_mismatched_metadata_raises():
    # vectors and metadata list of different lengths must raise RetrieverError
    from backend.rag.retriever import Retriever, RetrieverError
    r = Retriever(8)
    with pytest.raises(RetrieverError):
        r.add(_rand_vecs(3, 8), [{"text": "only one"}])


def test_retriever_search_empty_index_raises():
    # searching an empty index must raise RetrieverError
    from backend.rag.retriever import Retriever, RetrieverError
    r = Retriever(8)
    with pytest.raises(RetrieverError):
        r.search(np.ones(8, dtype=np.float32))


# ---------------------------------------------------------------------------
# vector_store tests
# ---------------------------------------------------------------------------

def test_vector_store_short_term_store_and_query(monkeypatch):
    # stored text is retrievable via short-term query
    _patch_vector_store_embedder(monkeypatch)
    from backend.memory.vector_store import VectorStore, SHORT_TERM
    with tempfile.TemporaryDirectory() as tmp:
        vs = VectorStore(store_dir=tmp)
        vs.store("LLMs are changing fintech", "src_a", SHORT_TERM)
        assert vs.count(SHORT_TERM) == 1
        results = vs.query("fintech", top_k=1, store_type=SHORT_TERM)
        assert results[0]["text"] == "LLMs are changing fintech"
        assert results[0]["source"] == "src_a"


def test_vector_store_long_term_persist_and_load(monkeypatch):
    # long-term store survives persist and reload into a fresh instance
    _patch_vector_store_embedder(monkeypatch)
    from backend.memory.vector_store import VectorStore, LONG_TERM
    with tempfile.TemporaryDirectory() as tmp:
        vs = VectorStore(store_dir=tmp)
        vs.store("FAISS enables similarity search", "src_b", LONG_TERM)
        vs.persist(LONG_TERM)
        vs2 = VectorStore(store_dir=tmp)
        vs2.load_store(LONG_TERM)
        assert vs2.count(LONG_TERM) == 1
        results = vs2.query("similarity search", top_k=1, store_type=LONG_TERM)
        assert results[0]["text"] == "FAISS enables similarity search"


def test_vector_store_clear_resets_count(monkeypatch):
    # clearing a store drops its entry count to zero
    _patch_vector_store_embedder(monkeypatch)
    from backend.memory.vector_store import VectorStore, SHORT_TERM
    with tempfile.TemporaryDirectory() as tmp:
        vs = VectorStore(store_dir=tmp)
        vs.store("some text", "src", SHORT_TERM)
        vs.clear(SHORT_TERM)
        assert vs.count(SHORT_TERM) == 0


def test_vector_store_unknown_type_raises(monkeypatch):
    # using an invalid store type must raise VectorStoreError
    _patch_vector_store_embedder(monkeypatch)
    from backend.memory.vector_store import VectorStore, VectorStoreError
    with tempfile.TemporaryDirectory() as tmp:
        vs = VectorStore(store_dir=tmp)
        with pytest.raises(VectorStoreError):
            vs.store("text", "src", "invalid_type")


def test_vector_store_query_empty_raises(monkeypatch):
    # querying an empty store must raise VectorStoreError
    _patch_vector_store_embedder(monkeypatch)
    from backend.memory.vector_store import VectorStore, VectorStoreError, SHORT_TERM
    with tempfile.TemporaryDirectory() as tmp:
        vs = VectorStore(store_dir=tmp)
        with pytest.raises(VectorStoreError):
            vs.query("anything", store_type=SHORT_TERM)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rand_vecs(n: int, dim: int) -> np.ndarray:
    # returns an (n, dim) float32 matrix of random unit vectors
    vecs = np.random.rand(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _mock_embed_text(text: str, **kwargs) -> np.ndarray:
    # deterministic mock: hashes text to seed a unit vector of dim 384
    np.random.seed(abs(hash(text)) % (2 ** 31))
    v = np.random.rand(384).astype(np.float32)
    return v / np.linalg.norm(v)


def _mock_embed_batch(texts: list, **kwargs) -> np.ndarray:
    # stacks per-text mock embeddings into a 2-D matrix
    return np.stack([_mock_embed_text(t) for t in texts])


def _patch_vector_store_embedder(monkeypatch):
    # replaces embed_text inside vector_store module with the deterministic mock
    import backend.memory.vector_store as vs_mod
    monkeypatch.setattr(vs_mod, "embed_text", _mock_embed_text)