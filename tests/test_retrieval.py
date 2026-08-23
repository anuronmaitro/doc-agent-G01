"""Unit test home for retrieval. IMPLEMENT — CI runs these."""

import json

import numpy as np
import pytest
import sentence_transformers

from doc_agent.contracts import Chunk
from doc_agent.index import store
from doc_agent.retrieval import rerank as rerank_mod
from doc_agent.retrieval import retriever as retriever_mod


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    """Redirect every store path into tmp_path so tests never touch the real data/index/."""
    d = tmp_path / "index"
    monkeypatch.setattr(store, "INDEX_DIR", d)
    monkeypatch.setattr(store, "FAISS_PATH", d / "faiss.index")
    monkeypatch.setattr(store, "CHUNKS_PATH", d / "chunks.jsonl")
    monkeypatch.setattr(store, "META_PATH", d / "index_meta.json")
    return d


CFG = {
    "index": {"type": "faiss:flat", "chunk_tokens": 512, "overlap": 0},
    "embed": {"model": "BAAI/bge-m3", "dim": 8},
}


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _corpus(n=4, dim=8, seed=0):
    """n chunks with distinct unit vectors, ids/doc_ids/page_ids shaped like the real ones."""
    rng = np.random.default_rng(seed)
    chunks, vecs = [], []
    pages = ["as_p0255", "as_p0255", "as_p0360", "as_p0243"]
    chapters = ["ch06_gamma", "ch06_gamma", "ch09_bessel", "ch05_expint"]
    formulas = ["6.1.8", "6.1.9", "9.1.12", ""]
    for i in range(n):
        page, chapter, fid = pages[i % 4], chapters[i % 4], formulas[i % 4]
        cid = f"{chapter}|{page}|r{i:02d}" + (f"|{fid}" if fid else "")
        chunks.append(
            Chunk(id=cid, doc_id=chapter, text=f"formula body {i}", page_ids=[page], score=0.0)
        )
        vecs.append(_unit(rng.normal(size=dim)))
    return chunks, np.stack(vecs).astype(np.float32)


class TestBuild:
    def test_writes_all_three_artifacts(self, index_dir):
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        assert (index_dir / "faiss.index").exists()
        assert (index_dir / "chunks.jsonl").exists()
        assert (index_dir / "index_meta.json").exists()

    def test_meta_records_the_stats_the_form_needs(self, index_dir):
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        assert meta["n_chunks"] == 4
        assert meta["embedding_dim"] == 8
        assert meta["index_type"] == "faiss:flat"
        assert meta["pages_covered"] == 3  # 0255 twice, 0360, 0243
        assert meta["chapters_covered"] == 3
        assert meta["index_size_bytes"] > 0

    def test_rejects_vector_count_mismatch(self, index_dir):
        chunks, vectors = _corpus(4)
        with pytest.raises(ValueError, match="one row per chunk"):
            store.build(chunks, vectors[:3], CFG)

    def test_rejects_non_2d_vectors(self, index_dir):
        chunks, _ = _corpus(4)
        with pytest.raises(ValueError, match="2-D"):
            store.build(chunks, np.zeros(4, dtype=np.float32), CFG)

    def test_rejects_unknown_index_type(self, index_dir):
        chunks, vectors = _corpus()
        with pytest.raises(ValueError, match="unsupported index type"):
            store.build(chunks, vectors, {"index": {"type": "annoy:hnsw"}, "embed": {}})

    def test_unnormalised_vectors_are_normalised(self, index_dir):
        """Otherwise 'cosine' silently becomes a dot product that ranks by magnitude."""
        chunks, vectors = _corpus()
        loud = vectors * np.array([[1.0], [50.0], [1.0], [1.0]], dtype=np.float32)
        store.build(chunks, loud, CFG)
        loaded = store.load(CFG)
        # query with chunk 0's own vector: it must win, not the 50x-longer chunk 1
        scores, ids = loaded.index.search(vectors[:1], 4)
        assert loaded.chunks[ids[0][0]].id == chunks[0].id
        assert scores[0][0] == pytest.approx(1.0, abs=1e-4)

    def test_empty_corpus_does_not_crash(self, index_dir):
        store.build([], np.zeros((0, 8), dtype=np.float32), CFG)
        assert (index_dir / "index_meta.json").exists()
        assert json.loads((index_dir / "index_meta.json").read_text())["n_chunks"] == 0


class TestLoad:
    def test_round_trips_chunks_faithfully(self, index_dir):
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(CFG)

        assert loaded.index.ntotal == len(chunks)
        assert loaded.dim == 8
        assert loaded.index_type == "faiss:flat"
        assert [c.id for c in loaded.chunks] == [c.id for c in chunks]
        assert [c.doc_id for c in loaded.chunks] == [c.doc_id for c in chunks]
        assert [c.text for c in loaded.chunks] == [c.text for c in chunks]
        assert [c.page_ids for c in loaded.chunks] == [c.page_ids for c in chunks]

    def test_row_order_matches_chunk_order(self, index_dir):
        """The one contract A3's retriever depends on: index row i IS chunks[i]."""
        chunks, vectors = _corpus(4)
        store.build(chunks, vectors, CFG)
        loaded = store.load(CFG)
        for i in range(len(chunks)):
            _scores, ids = loaded.index.search(vectors[i : i + 1], 1)
            assert ids[0][0] == i
            assert loaded.chunks[ids[0][0]].id == chunks[i].id

    def test_is_exact_cosine(self, index_dir):
        """Flat + unit vectors => the top score for a chunk's own vector is exactly 1.0."""
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(CFG)
        scores, _ids = loaded.index.search(vectors[2:3], 1)
        assert scores[0][0] == pytest.approx(1.0, abs=1e-5)

    def test_unpacks_positionally_and_by_name(self, index_dir):
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(CFG)
        index, got_chunks, dim, index_type = loaded
        assert index is loaded.index and got_chunks == loaded.chunks
        assert dim == loaded.dim and index_type == loaded.index_type

    def test_missing_index_raises_with_the_fix(self, index_dir):
        with pytest.raises(FileNotFoundError, match="build_index.sh"):
            store.load(CFG)

    def test_inconsistent_sidecar_is_caught(self, index_dir):
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        rows = (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        (index_dir / "chunks.jsonl").write_text("\n".join(rows[:2]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="inconsistent"):
            store.load(CFG)

    def test_citation_metadata_survives_the_round_trip(self, index_dir):
        """A retrieved chunk must still carry its page and formula id, or we cannot cite."""
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(CFG)
        gamma = next(c for c in loaded.chunks if c.id.endswith("|6.1.8"))
        assert gamma.page_ids == ["as_p0255"]
        assert gamma.doc_id == "ch06_gamma"


class TestContextExpand:
    """2026-08-23: neighbour-context enrichment, off by default (existing callers/tests
    above must see unexpanded text unchanged), on in the real configs/config.yaml."""

    CFG_ON = {**CFG, "retrieve": {"context_expand": True}}
    CFG_OFF = {**CFG, "retrieve": {"context_expand": False}}

    def test_default_off_does_not_change_existing_behaviour(self, index_dir):
        chunks, vectors = _corpus()  # CFG (module-level) has no "retrieve" key at all
        store.build(chunks, vectors, CFG)
        loaded = store.load(CFG)
        assert [c.text for c in loaded.chunks] == [c.text for c in chunks]

    def test_explicit_off_does_not_change_text(self, index_dir):
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(self.CFG_OFF)
        assert [c.text for c in loaded.chunks] == [c.text for c in chunks]

    def test_same_page_neighbours_get_merged_in(self, index_dir):
        """_corpus()'s first two chunks (r00, r01) share page as_p0255 -- adjacent in the
        list, same as real chunking's own reading order."""
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(self.CFG_ON)

        first = loaded.chunks[0]  # only a same-page successor (no predecessor, i=0)
        assert first.text == "formula body 0\nformula body 1"

        second = loaded.chunks[1]  # both a same-page predecessor AND successor... but
        # chunk 2 is on a DIFFERENT page (as_p0360), so only the predecessor merges in.
        assert second.text == "formula body 0\nformula body 1"

    def test_page_boundary_is_not_crossed(self, index_dir):
        """Chunk 2 (as_p0360) sits between chunk 1 (as_p0255) and chunk 3 (as_p0243) --
        both real neighbours in list position, neither a real neighbour on the page."""
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(self.CFG_ON)
        assert loaded.chunks[2].text == "formula body 2"  # unchanged -- no same-page neighbour

    def test_does_not_change_chunk_count_or_row_alignment(self, index_dir):
        """The one contract that must survive expansion untouched: index row i is still
        chunks[i] -- only .text content changes, never the list shape."""
        chunks, vectors = _corpus()
        store.build(chunks, vectors, CFG)
        loaded = store.load(self.CFG_ON)
        assert len(loaded.chunks) == len(chunks)
        assert [c.id for c in loaded.chunks] == [c.id for c in chunks]
        for i in range(len(chunks)):
            _scores, ids = loaded.index.search(vectors[i : i + 1], 1)
            assert ids[0][0] == i  # embeddings still align to the ORIGINAL (unexpanded) text

    def test_single_chunk_page_is_unaffected(self, index_dir):
        chunks = [Chunk(id="solo|p1", doc_id="d", text="alone", page_ids=["p1"], score=0.0)]
        vectors = np.stack([_unit(np.random.default_rng(1).normal(size=8))]).astype(np.float32)
        store.build(chunks, vectors, CFG)
        loaded = store.load(self.CFG_ON)
        assert loaded.chunks[0].text == "alone"


class TestRebuild:
    def test_rebuild_replaces_rather_than_appends(self, index_dir):
        chunks, vectors = _corpus(4)
        store.build(chunks, vectors, CFG)
        store.build(chunks[:2], vectors[:2], CFG)
        loaded = store.load(CFG)
        assert loaded.index.ntotal == 2
        assert len(loaded.chunks) == 2

    def test_unicode_maths_survives_the_sidecar(self, index_dir):
        chunks, vectors = _corpus(1)
        chunks[0] = Chunk(
            id=chunks[0].id,
            doc_id=chunks[0].doc_id,
            text=r"\Gamma(\tfrac12)=\pi^{1/2} — Γ(½)=√π",
            page_ids=chunks[0].page_ids,
            score=0.0,
        )
        store.build(chunks, vectors, CFG)
        assert store.load(CFG).chunks[0].text == chunks[0].text


RETRIEVE_CFG = {"k": 4, "k_step": 10, "k_max": 40, "weak_threshold": 0.35}


class _FakeEncoder:
    """Returns a caller-supplied vector regardless of the input text, so a test can pin
    exactly which corpus chunk a "query" should match -- no real BGE-M3 download in CI."""

    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector

    def encode(self, texts, **kwargs):
        return np.stack([self._vector for _ in texts]).astype(np.float32)


class TestRetrieve:
    def test_k_defaults_from_config(self, index_dir, monkeypatch):
        chunks, vectors = _corpus(4)
        store.build(chunks, vectors, CFG)
        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            lambda name, **kw: _FakeEncoder(vectors[0]),
        )
        cfg = {**CFG, "retrieve": {**RETRIEVE_CFG, "k": 2}}
        r = retriever_mod.Retriever(cfg)
        assert len(r.retrieve("anything")) == 2  # not hardcoded, read from cfg["retrieve"]["k"]

    def test_scores_are_set_and_descending(self, index_dir, monkeypatch):
        chunks, vectors = _corpus(4)
        store.build(chunks, vectors, CFG)
        # query vector == chunk 2's own vector -> chunk 2 must win with score ~1.0
        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            lambda name, **kw: _FakeEncoder(vectors[2]),
        )
        cfg = {**CFG, "retrieve": RETRIEVE_CFG}
        r = retriever_mod.Retriever(cfg)
        results = r.retrieve("query", k=4)
        assert results[0].id == chunks[2].id
        assert results[0].score == pytest.approx(1.0, abs=1e-4)
        scores = [c.score for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_larger_than_corpus_drops_padding_not_fake_chunks(self, index_dir, monkeypatch):
        chunks, vectors = _corpus(3)
        store.build(chunks, vectors, CFG)
        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            lambda name, **kw: _FakeEncoder(vectors[0]),
        )
        cfg = {**CFG, "retrieve": RETRIEVE_CFG}
        r = retriever_mod.Retriever(cfg)
        results = r.retrieve("query", k=10)  # FAISS pads the extra 7 rows with -1
        assert len(results) == 3

    def test_empty_index_returns_empty_list(self, index_dir, monkeypatch, tmp_path):
        store.build([], np.zeros((0, 8), dtype=np.float32), CFG)
        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            lambda name, **kw: _FakeEncoder(np.zeros(8, dtype=np.float32)),
        )
        # An empty text index is unconditionally "weak", so without this the visual
        # fallback would fire for real -- against whatever real image_embed_cache.npz
        # happens to sit at data/index/ on this machine (real after Step 3's Kaggle run),
        # making this test's result depend on local dev-machine state instead of being
        # deterministic. Same isolation pattern as TestVisualFallback below.
        monkeypatch.setattr(retriever_mod, "IMAGE_CACHE_PATH", tmp_path / "missing.npz")
        cfg = {**CFG, "retrieve": RETRIEVE_CFG}
        r = retriever_mod.Retriever(cfg)
        assert r.retrieve("query") == []

    def test_index_and_encoder_load_once_and_are_reused(self, index_dir, monkeypatch):
        chunks, vectors = _corpus(4)
        store.build(chunks, vectors, CFG)
        calls = {"n": 0}

        def _fake_ctor(name, **kw):
            calls["n"] += 1
            return _FakeEncoder(vectors[0])

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _fake_ctor)
        cfg = {**CFG, "retrieve": RETRIEVE_CFG}
        r = retriever_mod.Retriever(cfg)
        r.retrieve("first")
        r.retrieve("second")
        r.retrieve("third")
        assert calls["n"] == 1  # constructed once, cached and reused across all three calls

    def test_result_score_mutation_does_not_leak_across_calls(self, index_dir, monkeypatch):
        """The cached loaded.chunks list must never be mutated in place -- otherwise one
        query's score would bleed into the next query's results for the same chunk, since
        the Retriever instance (and its loaded index) is reused across the whole re-search
        loop in decide() (Step 10)."""
        chunks, vectors = _corpus(4)
        store.build(chunks, vectors, CFG)
        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            lambda name, **kw: _FakeEncoder(vectors[1]),
        )
        cfg = {**CFG, "retrieve": RETRIEVE_CFG}
        r = retriever_mod.Retriever(cfg)
        first = r.retrieve("q1")
        next(c for c in first if c.id == chunks[1].id).score = -999.0  # mutate caller's copy
        second = r.retrieve("q2")
        assert next(c for c in second if c.id == chunks[1].id).score != -999.0


# ============================================================================================
# Step 3 — rerank() (E6) and the scoped visual-retrieval fallback (DECISION D2)
# ============================================================================================


class _FakeCrossEncoder:
    """Scores every (query, text) pair by looking the text up in a caller-supplied map --
    lets a test assert the reranker actually changed the order, not just passed it through."""

    def __init__(self, score_by_text: dict[str, float]) -> None:
        self._score_by_text = score_by_text

    def predict(self, pairs):
        return [self._score_by_text[text] for _query, text in pairs]


@pytest.fixture(autouse=True)
def _clear_rerank_cache():
    """rerank._MODEL_CACHE is module-level state shared across the whole test session --
    clear it around every test so one test's fake reranker can't leak into another's."""
    rerank_mod._MODEL_CACHE.clear()
    yield
    rerank_mod._MODEL_CACHE.clear()


class TestRerank:
    RERANK_ON = {"retrieve": {"rerank": True, "reranker": "fake/reranker"}}
    RERANK_OFF = {"retrieve": {"rerank": False, "reranker": "fake/reranker"}}

    def _candidates(self) -> list[Chunk]:
        return [
            Chunk(id="a", doc_id="ch01", text="alpha", page_ids=["as_p0001"], score=0.0),
            Chunk(id="b", doc_id="ch01", text="beta", page_ids=["as_p0002"], score=0.0),
        ]

    def test_noop_passthrough_when_rerank_disabled(self):
        candidates = self._candidates()
        result = rerank_mod.rerank("q", candidates, self.RERANK_OFF)
        assert result == candidates

    def test_empty_candidates_returns_empty(self):
        assert rerank_mod.rerank("q", [], self.RERANK_ON) == []

    def test_reorders_and_overwrites_score_by_cross_encoder_output(self, monkeypatch):
        # "beta" scores higher than "alpha" despite arriving first in the input list --
        # the reranker, not the input order, must decide the output order.
        monkeypatch.setattr(
            sentence_transformers,
            "CrossEncoder",
            lambda name, **kw: _FakeCrossEncoder({"alpha": 0.2, "beta": 0.9}),
        )
        result = rerank_mod.rerank("q", self._candidates(), self.RERANK_ON)
        assert [c.id for c in result] == ["b", "a"]
        assert result[0].score == pytest.approx(0.9)
        assert result[1].score == pytest.approx(0.2)

    def test_cross_encoder_loaded_once_and_cached_across_calls(self, monkeypatch):
        calls = {"n": 0}

        def _ctor(name, **kw):
            calls["n"] += 1
            return _FakeCrossEncoder({"alpha": 0.5, "beta": 0.5})

        monkeypatch.setattr(sentence_transformers, "CrossEncoder", _ctor)
        candidates = self._candidates()
        rerank_mod.rerank("q1", candidates, self.RERANK_ON)
        rerank_mod.rerank("q2", candidates, self.RERANK_ON)
        assert calls["n"] == 1


class _FakeClipTextModel:
    """Fakes the two stable submodules search() actually calls -- text_model then
    text_projection -- rather than the get_text_features() convenience wrapper.
    Production code stopped trusting that wrapper after the real Kaggle Step 3 run found
    its return type isn't stable across transformers versions (a newer pre-installed
    version on Kaggle returned the raw model output instead of the projected tensor its
    own docstring promises). Always yields a fixed, caller-supplied vector regardless of
    the input, so a test can pin exactly what the query "looks like" to the fallback."""

    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector

    def eval(self):
        return self

    def text_model(self, **kwargs):
        return (None, None)  # only index [1] is read; text_projection below ignores it

    def text_projection(self, pooled_output):
        import torch

        return torch.from_numpy(np.array([self._vector], dtype=np.float32))


class _FakeClipProcessor:
    def __call__(self, text, return_tensors, padding, truncation):
        return {}


def _patch_fake_clip(monkeypatch, vector: np.ndarray) -> None:
    """Patches _ImageIndex._ensure_clip() directly rather than transformers.CLIPModel/
    CLIPProcessor -- transformers' top-level namespace is a lazy-loading module
    (_LazyModule), so monkeypatch.setattr(transformers, "CLIPModel", ...) does NOT
    reliably intercept the `from transformers import CLIPModel` inside _ensure_clip()
    (confirmed the hard way: two tests silently downloaded and ran the real 512-dim
    model instead of the fake, then crashed matmul-ing it against an 8-dim corpus).
    Patching our own method sidesteps that import machinery entirely."""

    def _fake_ensure_clip(self):
        return _FakeClipTextModel(vector), _FakeClipProcessor()

    monkeypatch.setattr(retriever_mod._ImageIndex, "_ensure_clip", _fake_ensure_clip)


def _write_image_cache(path, page_ids: list[str], vectors: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        page_ids=np.array(page_ids),
        vectors=np.stack(vectors).astype(np.float32),
    )


VISUAL_RETRIEVE_CFG = {"k": 5, "k_step": 10, "k_max": 40, "weak_threshold": 0.35}
E0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # matches the one
ORTHOGONAL = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # weak vs E0


class TestVisualFallback:
    def _build_one_chunk_index(self, index_dir):
        chunk = Chunk(
            id="ch01|as_p0200|r00",
            doc_id="ch01",
            text="known text",
            page_ids=["as_p0200"],
            score=0.0,
        )
        store.build([chunk], np.array([E0]), CFG)
        return chunk

    def test_no_cache_degrades_to_dense_only(self, index_dir, monkeypatch, tmp_path):
        """A missing image_embed_cache.npz (CI, or before Step 4's Kaggle run) must not
        change dense-only behaviour at all -- the fallback is additive, never required."""
        chunk = self._build_one_chunk_index(index_dir)
        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            lambda name, **kw: _FakeEncoder(ORTHOGONAL),
        )
        monkeypatch.setattr(retriever_mod, "IMAGE_CACHE_PATH", tmp_path / "missing.npz")
        cfg = {**CFG, "retrieve": VISUAL_RETRIEVE_CFG}
        r = retriever_mod.Retriever(cfg)
        results = r.retrieve(
            "q"
        )  # orthogonal query -> weak (score 0.0) -> triggers fallback attempt
        assert len(results) == 1
        assert results[0].id == chunk.id
        assert results[0].score == pytest.approx(0.0)  # untouched, no visual boost applied

    def test_strong_dense_result_never_loads_clip_at_all(self, index_dir, monkeypatch, tmp_path):
        """Only pay the CLIP cost when text retrieval is actually struggling -- confirmed
        by making _ensure_clip() raise if it's ever called."""
        self._build_one_chunk_index(index_dir)
        monkeypatch.setattr(
            sentence_transformers, "SentenceTransformer", lambda name, **kw: _FakeEncoder(E0)
        )

        def _exploding_ensure_clip(self):
            raise AssertionError("CLIP must not load when dense retrieval isn't weak")

        monkeypatch.setattr(retriever_mod._ImageIndex, "_ensure_clip", _exploding_ensure_clip)
        cache_path = tmp_path / "cache.npz"
        _write_image_cache(cache_path, ["as_p0999"], [E0])
        monkeypatch.setattr(retriever_mod, "IMAGE_CACHE_PATH", cache_path)
        cfg = {**CFG, "retrieve": VISUAL_RETRIEVE_CFG}
        r = retriever_mod.Retriever(cfg)
        results = r.retrieve("q")  # query == E0 -> score 1.0 -> strong, must not touch CLIP
        assert results[0].score == pytest.approx(1.0)

    def test_visual_hit_synthesises_a_citable_placeholder_for_a_page_with_no_chunk(
        self, index_dir, monkeypatch, tmp_path
    ):
        """The core reason D2 exists: one of the 53 OCR-failure pages has zero text chunks,
        so it can never surface from dense search no matter how weak the threshold -- the
        visual fallback must still make it discoverable and citable by page id."""
        self._build_one_chunk_index(index_dir)  # only covers as_p0200
        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            lambda name, **kw: _FakeEncoder(ORTHOGONAL),
        )
        _patch_fake_clip(monkeypatch, E0)
        cache_path = tmp_path / "cache.npz"
        _write_image_cache(cache_path, ["as_p0509"], [E0])  # as_p0509 has NO text chunk
        monkeypatch.setattr(retriever_mod, "IMAGE_CACHE_PATH", cache_path)
        cfg = {**CFG, "retrieve": VISUAL_RETRIEVE_CFG}
        r = retriever_mod.Retriever(cfg)
        results = r.retrieve("q")
        pseudo = next((c for c in results if c.id == "visual|as_p0509"), None)
        assert pseudo is not None
        assert pseudo.page_ids == ["as_p0509"]
        assert pseudo.score == pytest.approx(1.0)
        assert "image similarity only" in pseudo.text
        assert pseudo.doc_id != "unknown"  # real chapter, resolved from the printed page number

    def test_visual_hit_raises_score_of_an_already_present_weak_chunk(
        self, index_dir, monkeypatch, tmp_path
    ):
        """A chunk that's already in the weak top-k gets its score RAISED on visual
        confirmation, not silently skipped just because it was already present."""
        chunk = self._build_one_chunk_index(index_dir)  # covers as_p0200, dense score 0.0
        monkeypatch.setattr(
            sentence_transformers,
            "SentenceTransformer",
            lambda name, **kw: _FakeEncoder(ORTHOGONAL),
        )
        _patch_fake_clip(monkeypatch, E0)
        cache_path = tmp_path / "cache.npz"
        _write_image_cache(cache_path, ["as_p0200"], [E0])  # visual also finds as_p0200
        monkeypatch.setattr(retriever_mod, "IMAGE_CACHE_PATH", cache_path)
        cfg = {**CFG, "retrieve": VISUAL_RETRIEVE_CFG}
        r = retriever_mod.Retriever(cfg)
        results = r.retrieve("q")
        boosted = next(c for c in results if c.id == chunk.id)
        assert boosted.score == pytest.approx(1.0)  # raised from 0.0, not left at 0.0
