"""Stage 5 — dense retrieval"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import *  # noqa
from ..index import store
from ..logging_conf import get_logger

logger = get_logger(__name__)


def _pick_device() -> str:
    """2026-08-23: promoted from a one-off Kaggle notebook script into production code --
    every local-model loader in this module (and rerank.py's reranker) now uses this,
    instead of trusting sentence-transformers'/transformers' own default device selection.

    torch.cuda.is_available() only checks a driver is present -- it does NOT check the
    GPU's compute capability is actually supported by this PyTorch build. Kaggle can assign
    an older GPU (confirmed in practice at Step 3: a Tesla P100, sm_60) that a newer PyTorch
    wheel (built for sm_70+) can detect but not actually run a kernel on -- that combination
    reports "cuda available" and then hard-crashes on the first real op. Try a real,
    trivial CUDA op and fall back to CPU on failure, rather than trusting availability
    alone. This matters more now than when it only guarded the one-time CLIP embedding
    pass: Step 22's GPU note (plan_a3.md, revised 2026-08-23) budgets real GPU time for the
    reranker specifically, on the assumption the code actually uses whatever GPU Kaggle
    assigns -- that assumption was previously unverified in the actual pipeline code.
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    try:
        probe = torch.zeros(1, device="cuda")
        _ = probe + 1
        return "cuda"
    except RuntimeError as exc:
        logger.warning(
            f"retriever: CUDA reports available but a real op failed ({exc}) -- using CPU"
        )
        return "cpu"


# --- Visual-retrieval fallback (DECISION D2, plan_a3.md §5) --------------------------------
# Scoped deliberately small: NOT full ColPali (README.md/STRUCTURE.md forbid new top-level
# modules, so there's no home for a genuine separate late-interaction system). Instead, a
# lightweight CLIP text/image embedding, folded into this class, triggered only when dense
# text retrieval already came back weak -- covers both of A2's real failures: the 53 pages
# with NO text chunk at all (unreachable by text search no matter what), and the
# under-transcribed-table case where the right page exists but scored low.
IMAGE_CACHE_PATH = Path("data/index/image_embed_cache.npz")
DEFAULT_VISUAL_MODEL = "openai/clip-vit-base-patch32"


class _ImageIndex:
    """Page-level CLIP embeddings, held in memory as a plain array -- 1040 vectors is far
    too small to justify a second FAISS index file on top of the text one."""

    def __init__(self, page_ids: list[str], vectors: np.ndarray, model_name: str) -> None:
        self.page_ids = page_ids
        self.vectors = vectors  # (n_pages, dim), L2-normalised
        self._model_name = model_name
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cpu"

    def _ensure_clip(self) -> tuple[Any, Any]:
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor

            # Pinned to the exact commit resolved for openai/clip-vit-base-patch32 when this
            # was written (bandit B615: from_pretrained() without a revision is a real
            # supply-chain risk -- the same finding and the same fix A2 used for TATR in
            # vision/layout.py). Re-resolve via HfApi().model_info(...).sha if the model
            # string in config.yaml's retrieve.visual_model ever changes.
            revision = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
            self._device = _pick_device()
            self._model = (
                CLIPModel.from_pretrained(self._model_name, revision=revision)
                .to(self._device)
                .eval()
            )
            self._processor = CLIPProcessor.from_pretrained(self._model_name, revision=revision)
        return self._model, self._processor

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Cosine similarity between the query text and every cached page image.

        NOTE (flagged for Step 22 to measure, same discipline as the rerank/cosine scale
        mismatch in Step 3's rerank()): CLIP's cross-modal (text-vs-image) cosine scores are
        NOT guaranteed to sit on the same scale as BGE-M3's same-modal text-vs-text cosine
        scores -- this is CLIP's well-documented "modality gap." Merging the two score sets
        by raw magnitude (as retrieve() below does) is a real, reportable assumption, not a
        proven-safe one.
        """
        if not self.page_ids:
            return []
        import torch

        model, processor = self._ensure_clip()
        inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            # NOT model.get_text_features(**inputs) -- the Kaggle embedding job (Step 3)
            # hit a real version-dependent break where a newer transformers release
            # returned the raw BaseModelOutputWithPooling from this convenience method
            # instead of the projected tensor its own docstring promises. text_model +
            # text_projection are the stable core submodules get_text_features() itself
            # wraps (verified by reading its source), so call them directly instead of
            # trusting the wrapper's contract across versions.
            text_out = model.text_model(**inputs)
            pooled_output = text_out[1]
            qvec = model.text_projection(pooled_output)[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm
        scores = self.vectors @ qvec
        top_idx = np.argsort(-scores)[:k]
        return [(self.page_ids[i], float(scores[i])) for i in top_idx]


def _load_image_index(cfg: dict) -> _ImageIndex | None:
    """None (not an error) when the cache hasn't been built yet -- the visual fallback is
    additive, so a clean clone / CI without it must fall back to dense-only, unchanged."""
    if not IMAGE_CACHE_PATH.exists():
        return None
    data = np.load(IMAGE_CACHE_PATH)
    page_ids = data["page_ids"].tolist()
    vectors = data["vectors"].astype(np.float32)
    model_name = cfg.get("retrieve", {}).get("visual_model", DEFAULT_VISUAL_MODEL)
    return _ImageIndex(page_ids, vectors, model_name)


class Retriever:
    """Loads the A2 index and the BGE-M3 encoder once, lazily, and reuses both across
    calls -- decide()'s evidence-gated re-search loop (A3 Step 10) calls retrieve()
    multiple times per query, so reloading a 16 MB index and a 568M-param encoder on
    every call would make the eval runs (Steps 22-24) unaffordable."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["retrieve"]
        self._full_cfg = cfg
        self._loaded: store.LoadedIndex | None = None
        self._encoder: Any | None = None
        self._image_index: _ImageIndex | None = None
        self._image_index_checked = False

    def _ensure_loaded(self) -> store.LoadedIndex:
        if self._loaded is None:
            self._loaded = store.load(self._full_cfg)
        return self._loaded

    def _ensure_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            model_name = self._full_cfg["embed"]["model"]
            # Explicit device, not sentence-transformers' own default selection -- same
            # P100-incompatibility risk _pick_device() exists to catch (Step 3's real
            # Kaggle finding), now checked here too rather than trusted blindly.
            self._encoder = SentenceTransformer(model_name, device=_pick_device())
        return self._encoder

    def _ensure_image_index(self) -> _ImageIndex | None:
        if not self._image_index_checked:
            self._image_index = _load_image_index(self._full_cfg)
            self._image_index_checked = True
            if self._image_index is None:
                logger.info(
                    "retriever: no image_embed_cache.npz -- visual fallback disabled, "
                    "dense-only retrieval unaffected"
                )
        return self._image_index

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Top-k dense retrieval. Set chunk.score (relevance) on every result so decide() can judge
        whether the evidence is weak.

        Embeds the query with the SAME model the index was built with (cfg['embed']['model'],
        BGE-M3) and L2-normalises it, exactly like index/embed.py does for chunks -- a mismatched
        model or a raw (un-normalised) query would silently turn cosine into a meaningless score
        (store.py's own _NORM_TOL comment explains why the index side already guards this).

        If the dense result comes back weak, automatically blends in the CLIP visual fallback
        (DECISION D2) before returning -- decide()'s widen-and-retry loop (Step 10) calls this
        plain method at increasing k and never needs to know the visual system exists.
        """
        k = k if k is not None else self.cfg["k"]
        loaded = self._ensure_loaded()
        encoder = self._ensure_encoder()

        qvec = encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, ids = loaded.index.search(qvec, k)

        results: list[Chunk] = []
        for score, i in zip(scores[0], ids[0], strict=True):
            if i < 0:
                continue  # FAISS pads with -1 when k exceeds ntotal
            # A fresh copy per result, never mutate loaded.chunks in place -- that list is
            # cached on the instance and shared across every call this Retriever makes, so
            # writing .score onto it directly would let one query's scores leak into another's.
            results.append(loaded.chunks[i].model_copy(update={"score": float(score)}))

        if is_weak(results, self._full_cfg):
            results = self._augment_with_visual_fallback(query, results, k, loaded)
        return results

    def _augment_with_visual_fallback(
        self, query: str, text_results: list[Chunk], k: int, loaded: store.LoadedIndex
    ) -> list[Chunk]:
        image_index = self._ensure_image_index()
        if image_index is None:
            return text_results  # cache not built (e.g. CI, or before Step 4's Kaggle run)

        visual_hits = image_index.search(query, k)
        if not visual_hits:
            return text_results

        # Index every chunk in the WHOLE corpus by page, not just this call's top-k text
        # results -- a visual hit should be able to "rescue" a real chunk that exists but
        # didn't make the cut on text score alone, not only pages with zero chunks at all.
        chunks_by_page: dict[str, list[Chunk]] = {}
        for c in loaded.chunks:
            for pid in c.page_ids:
                chunks_by_page.setdefault(pid, []).append(c)

        # Keyed by id, not a plain list -- a chunk already present (with a weak text score)
        # gets its score RAISED on visual confirmation instead of being silently skipped,
        # since being found by both modalities is itself stronger evidence than either alone.
        merged: dict[str, Chunk] = {c.id: c for c in text_results}
        for page_id, vscore in visual_hits:
            page_chunks = chunks_by_page.get(page_id)
            if page_chunks:
                for c in page_chunks:
                    existing = merged.get(c.id)
                    best = max(vscore, existing.score) if existing is not None else vscore
                    merged[c.id] = c.model_copy(update={"score": best})
            else:
                # No text chunk exists for this page at all (one of the 53 OCR failures) --
                # synthesise a minimal, honestly-labelled placeholder so the page is at least
                # discoverable and citable by page id. This is the whole reason D2 exists.
                pseudo_id = f"visual|{page_id}"
                if pseudo_id not in merged:
                    from ..ingest.loader import _chapter_of

                    printed = int(page_id[4:]) if page_id.startswith("as_p") else 0
                    merged[pseudo_id] = Chunk(
                        id=pseudo_id,
                        doc_id=_chapter_of(printed) if printed else "unknown",
                        text=(
                            "[Found via image similarity only -- OCR produced no usable "
                            "text for this page. Use read_page/enhance_page to inspect "
                            "it directly.]"
                        ),
                        page_ids=[page_id],
                        score=vscore,
                    )

        ranked = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        return ranked[:k]


# --- evidence-strength policy: read by agent.decide() for evidence-gated re-search ---
def top_score(chunks: list[Chunk]) -> float:
    """Strength of the current evidence = best chunk score (0.0 if empty)."""
    return max((c.score for c in chunks), default=0.0)


def is_weak(chunks: list[Chunk], cfg: dict) -> bool:
    """Weak evidence = best score below cfg.retrieve.weak_threshold."""
    return top_score(chunks) < cfg["retrieve"]["weak_threshold"]


def next_k(k: int, cfg: dict) -> int | None:
    """Widen the net: k + k_step, or None once it would exceed k_max (signal to ABSTAIN)."""
    nk = k + cfg["retrieve"]["k_step"]
    return nk if nk <= cfg["retrieve"]["k_max"] else None
