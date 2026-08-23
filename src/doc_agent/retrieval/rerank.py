"""Stage 5 — reranking"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa

# Module-level cache: the cross-encoder is a real download + GPU/CPU load, and rerank()
# gets called once per retrieve() in decide()'s loop (Step 10) -- loading it fresh every
# call would make the eval runs (Steps 22-24) unaffordable, same reasoning as
# retriever.py's cached index/encoder.
_MODEL_CACHE: dict[str, Any] = {}


def _get_cross_encoder(model_name: str) -> Any:
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder

        # Explicit device, not CrossEncoder's own default selection -- reuses
        # retriever.py's _pick_device() (Step 3's real P100-incompatibility finding: CUDA
        # can report available and still crash on the first real op) rather than
        # duplicating that probe here. This module is directly downstream of retriever.py
        # already (agent.py imports both), so importing it here isn't a new dependency
        # direction.
        from .retriever import _pick_device

        _MODEL_CACHE[model_name] = CrossEncoder(model_name, device=_pick_device())
    return _MODEL_CACHE[model_name]


def rerank(query: str, candidates: list[Chunk], cfg: dict) -> list[Chunk]:
    """Cross-encoder rerank if cfg['retrieve']['rerank'].

    Overwrites chunk.score with the cross-encoder's score and re-sorts. NOTE (flagged for
    Step 22 to measure, not silently assumed fine): the cross-encoder's score sits on a
    different scale than the cosine similarity is_weak() was tuned against
    (weak_threshold=0.35). If the two scales disagree badly once real numbers exist,
    weak_threshold needs re-tuning -- that's a real, reportable finding, not a bug to hide.

    The top-1 vs top-2 score gap the Explainable NFR needs (Step 20) is just
    results[0].score - results[1].score on the list this returns -- not carried as a
    separate return value, since the caller already has everything needed to compute it.
    """
    if not cfg["retrieve"]["rerank"] or not candidates:
        return candidates

    model_name = cfg["retrieve"]["reranker"]
    model = _get_cross_encoder(model_name)
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    rescored = [
        c.model_copy(update={"score": float(s)}) for c, s in zip(candidates, scores, strict=True)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored
