"""Stage 4 — vector store"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# All three artifacts live in data/index/, already gitignored: the index is fully
# rebuildable by `bash scripts/build_index.sh`, so it is never committed (plan.md 11.6).
# index/embed.py caches to embed_cache.npz in this same directory; these names are
# deliberately distinct from it.
INDEX_DIR = Path("data/index")
FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
META_PATH = INDEX_DIR / "index_meta.json"

# cfg["index"]["type"] is locked to "faiss:flat" (summary.md 7a). Flat = IndexFlatIP, an
# exhaustive inner-product scan. On unit-length vectors inner product IS cosine
# similarity, which is why index/embed.py L2-normalises. Chosen over HNSW/IVF because a
# few thousand chunks is far below the scale where approximate search pays for itself,
# and an approximate index would fold its own recall error into every number we report
# in form Section 5 -- we would not be able to tell a retrieval bug from ANN drift.
DEFAULT_INDEX_TYPE = "faiss:flat"

# Unit-norm tolerance. embed.py normalises, but build() is a public entry point and a
# caller passing un-normalised vectors would silently turn "cosine" into a dot product
# that ranks long vectors first -- a wrong answer that still looks like a working index.
_NORM_TOL = 1e-3


class LoadedIndex(NamedTuple):
    """What `load()` hands back.

    A NamedTuple so A3's `retrieval/retriever.py` can either unpack it positionally
    (`index, chunks, dim, index_type = store.load(cfg)`) or read it by name
    (`loaded.index`), whichever reads better there.

    `index` row i corresponds to `chunks[i]` -- that alignment is the whole contract
    between this module and the retriever: FAISS returns row numbers, and the only way
    back to a `Chunk` (and so to a citable page and formula id) is this ordering.
    """

    index: Any
    chunks: list[Chunk]
    dim: int
    index_type: str


def _faiss() -> Any:
    import faiss

    return faiss


def _as_matrix(vectors: Any, n_chunks: int) -> np.ndarray:
    """Coerce to the contiguous float32 (n, dim) matrix FAISS requires."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"index.store: expected a 2-D (n_chunks, dim) array, got {matrix.shape}")
    if matrix.shape[0] != n_chunks:
        raise ValueError(
            f"index.store: {matrix.shape[0]} vectors for {n_chunks} chunks -- "
            "embed.encode() must return one row per chunk, in the same order"
        )
    return np.ascontiguousarray(matrix)


def _pages_covered(chunks: list[Chunk]) -> int:
    pages: set[str] = set()
    for c in chunks:
        pages.update(c.page_ids)
    return len(pages)


def _dir_size_bytes(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.exists())


def build(chunks: list[Chunk], vectors: Any, cfg: dict) -> None:
    """Persist a flat vector index plus the chunk sidecar `load()` rebuilds `Chunk`s from.

    Writes three files under data/index/:
      * faiss.index     -- IndexFlatIP over the (already normalised) embeddings
      * chunks.jsonl    -- one row per chunk, in index-row order, so row i of the FAISS
                           index maps back to a real Chunk (id, doc_id, text, page_ids)
      * index_meta.json -- the statistics printed below, kept on disk so Step 30 and the
                           form's Section 5 can be re-read without rebuilding

    FAISS stores only vectors, never the text or the ids -- without the sidecar a search
    returns row numbers and nothing else, so the index alone could not produce a citation.

    Logs the stats plan.md Step 15 asks for: n_chunks, embedding dim, index type, on-disk
    size, and pages covered.
    """
    index_cfg = cfg.get("index") or {}
    index_type = str(index_cfg.get("type", DEFAULT_INDEX_TYPE))
    if not index_type.startswith("faiss:"):
        raise ValueError(
            f"index.store: unsupported index type {index_type!r}; "
            f"config.yaml locks {DEFAULT_INDEX_TYPE!r} (summary.md 7a)"
        )

    matrix = _as_matrix(vectors, len(chunks))
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if len(chunks) == 0:
        # Nothing to index. Still write the sidecar + meta so load() fails with our clear
        # message rather than a FAISS one, and so a caller can tell "built, but empty"
        # from "never built".
        dim = int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[1] else 0
        logger.warning("index.store: no chunks to index -- writing an empty index")
    else:
        dim = int(matrix.shape[1])
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=_NORM_TOL):
            worst = float(np.max(np.abs(norms - 1.0)))
            logger.warning(
                f"index.store: vectors are not unit-length (max deviation {worst:.4f}); "
                "normalising so inner product is cosine similarity"
            )
            matrix = matrix / np.clip(norms, 1e-12, None)[:, None]
            matrix = np.ascontiguousarray(matrix.astype(np.float32))

    faiss = _faiss()
    index = faiss.IndexFlatIP(dim) if dim else faiss.IndexFlatIP(1)
    if len(chunks):
        index.add(matrix)
    faiss.write_index(index, str(FAISS_PATH))

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {"id": c.id, "doc_id": c.doc_id, "text": c.text, "page_ids": c.page_ids},
                    ensure_ascii=False,
                )
                + "\n"
            )

    size_bytes = _dir_size_bytes([FAISS_PATH, CHUNKS_PATH])
    meta = {
        "n_chunks": len(chunks),
        "embedding_dim": dim,
        "index_type": index_type,
        "index_size_bytes": size_bytes,
        "pages_covered": _pages_covered(chunks),
        "chapters_covered": len({c.doc_id for c in chunks}),
        "embed_model": (cfg.get("embed") or {}).get("model", ""),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # These are the numbers that go into form Section 5 (plan.md Step 15).
    logger.info(
        "index.store: built\n"
        f"    n_chunks       : {meta['n_chunks']}\n"
        f"    embedding dim  : {meta['embedding_dim']}\n"
        f"    index type     : {meta['index_type']}\n"
        f"    on-disk size   : {size_bytes / 1e6:.2f} MB  ({size_bytes} bytes)\n"
        f"    pages covered  : {meta['pages_covered']}\n"
        f"    chapters       : {meta['chapters_covered']}\n"
        f"    embed model    : {meta['embed_model']}\n"
        f"    written to     : {INDEX_DIR}/"
    )


def load(cfg: dict) -> Any:
    """Read the index and its chunk sidecar back.

    Returns a `LoadedIndex` whose `index` row i is `chunks[i]` -- see that class for why
    the alignment matters. Raises FileNotFoundError with the command that fixes it if the
    index has not been built, since "index missing" and "index corrupt" want very
    different responses from a caller.
    """
    if not FAISS_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"index.store: no index at {INDEX_DIR}/. Build it first: bash scripts/build_index.sh"
        )

    faiss = _faiss()
    index = faiss.read_index(str(FAISS_PATH))

    chunks: list[Chunk] = []
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        chunks.append(
            Chunk(
                id=row["id"],
                doc_id=row["doc_id"],
                text=row["text"],
                page_ids=row["page_ids"],
                score=0.0,
            )
        )

    if index.ntotal != len(chunks):
        raise ValueError(
            f"index.store: index has {index.ntotal} vectors but the sidecar has "
            f"{len(chunks)} chunks -- data/index/ is inconsistent, rebuild with "
            "bash scripts/build_index.sh"
        )

    index_type = DEFAULT_INDEX_TYPE
    if META_PATH.exists():
        try:
            index_type = json.loads(META_PATH.read_text(encoding="utf-8")).get(
                "index_type", index_type
            )
        except (OSError, json.JSONDecodeError):
            pass

    if cfg.get("retrieve", {}).get("context_expand", False):
        chunks = _expand_context(chunks)

    logger.info(f"index.store: loaded {index.ntotal} vectors (dim {index.d}) from {INDEX_DIR}/")
    return LoadedIndex(index=index, chunks=chunks, dim=int(index.d), index_type=index_type)


def _expand_context(chunks: list[Chunk]) -> list[Chunk]:
    """Append each chunk's immediate same-page neighbours' text into its own `.text`.

    This corpus chunks fine-grained (as_p0255 alone splits into 16 chunks -- one per
    numbered formula plus separate prose paragraphs), so a single retrieved chunk often
    carries almost no surrounding context. A real live query ("Gamma(1/2)") found the
    correct chunk (formula 6.1.8) sitting right next to a chunk it needed but didn't
    contain -- retrieved in total isolation, that formula's own OCR'd text
    (`\\Gamma(1)=...`, a transcription artefact) reads as unrelated to the question asking
    about `\\Gamma(1/2)`. Neighbour context gives the LLM the surrounding page's own words
    to recover from exactly that kind of isolated-chunk gap.

    Applied ONCE, here at load time -- not duplicated inside `Retriever.retrieve()` alone --
    so every consumer that resolves a chunk_id back to its text (`Retriever`,
    `agent/tools.py`'s citation/extract lookups, `eval/metrics.py`'s groundedness check)
    sees the IDENTICAL enriched text. Splitting this differently between the retrieval path
    and the citation-validation path would silently break `citation_accuracy`'s span-bounds
    check: a span computed against enriched text would then fail bounds-checking against a
    shorter, un-enriched copy loaded separately elsewhere.

    Adjacency is positional in `chunks` (the same list `index` row i already aligns with),
    not a separate ordering field -- `LoadedIndex`'s own contract already guarantees chunks
    from the same page appear in reading order next to each other, since that's how
    `index/chunk.py` produced them in the first place. Guarded by `page_ids` equality so a
    chunk at a page boundary never pulls in a neighbouring PAGE's unrelated text.

    Does not change `Chunk`'s shape, the list's length, or FAISS-row alignment -- only
    `.text` content -- so `index` row i is still `chunks[i]` exactly as `LoadedIndex`
    requires, and the embeddings already computed against the ORIGINAL (unexpanded) text
    stay valid; only what the LLM eventually reads at synthesis time changes, not what was
    embedded for search. (A well-known, legitimate RAG technique -- "sentence-window" /
    "small-to-big" retrieval: precise small chunks for search ranking, wider context
    windows for the model that actually has to answer from them.)
    """
    expanded: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        parts = [chunk.text]
        if i > 0 and chunks[i - 1].page_ids == chunk.page_ids:
            parts.insert(0, chunks[i - 1].text)
        if i + 1 < len(chunks) and chunks[i + 1].page_ids == chunk.page_ids:
            parts.append(chunks[i + 1].text)
        expanded.append(
            chunk.model_copy(update={"text": "\n".join(parts)}) if len(parts) > 1 else chunk
        )
    return expanded
