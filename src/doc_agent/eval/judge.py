"""Stage 9 — LLM-as-judge for non-verifiable inference"""

from __future__ import annotations

import re

from .. import config
from ..contracts import *  # noqa
from ..llm import client, prompts

# --- shared lazy chunk lookup -------------------------------------------------------------
#
# Small, module-local cache -- deliberately NOT imported from eval/metrics.py or
# agent/tools.py, which each need the identical chunk_id -> Chunk lookup for their own
# purposes. This project's stub convention (see tools.py's own comment on this exact
# pattern) is that each file owns its own body rather than share one, at the cost of ~8
# duplicated lines -- kept here for consistency with that established precedent, not an
# oversight.
_CHUNK_LOOKUP_CACHE: dict[str, Chunk] | None = None


def _get_chunk_lookup() -> dict[str, Chunk]:
    global _CHUNK_LOOKUP_CACHE
    if _CHUNK_LOOKUP_CACHE is None:
        from ..index import store

        loaded = store.load(config.load())
        _CHUNK_LOOKUP_CACHE = {c.id: c for c in loaded.chunks}
    return _CHUNK_LOOKUP_CACHE


def _evidence_block(answer: Answer) -> str:
    """One line per cited chunk: `[chunk_id] chunk text`, matching the JUDGE prompt's own
    documented evidence format (no scores -- those aren't relevant to judging correctness/
    completeness/groundedness, unlike SYNTHESIZE's evidence block). `judge()`'s own FIXED
    signature only receives the `Answer`, not the full retrieved chunk list, so the
    citations the answer itself relied on are the only evidence available to reconstruct --
    a citation that fails to resolve (a fabricated or stale chunk_id) is dropped rather than
    guessed at, same as `eval/metrics.py::_cited_excerpt`'s own "unresolvable -> None"
    contract."""
    lookup = _get_chunk_lookup()
    lines = []
    for citation in answer.citations:
        chunk = lookup.get(citation.chunk_id)
        if chunk is not None:
            lines.append(f"[{citation.chunk_id}] {chunk.text}")
    return "\n".join(lines) if lines else "(no cited evidence)"


_SCORE_RE = re.compile(
    r"CORRECTNESS:\s*([0-2])\D*COMPLETENESS:\s*([0-2])\D*GROUNDEDNESS:\s*([0-2])", re.DOTALL
)


def _parse_score(raw: str) -> float | None:
    """Extract the three 0-2 criterion scores and sum them -- recomputed from the individual
    fields, not trusted from the model's own `TOTAL:` line, since an LLM can add three small
    integers wrong. `None` (not 0.0) on a genuinely unparseable reply, so the caller can tell
    "the judge scored this 0" apart from "the judge's reply couldn't be read at all" -- the
    two get the same fail-closed numeric treatment, but the distinction matters for anyone
    debugging a malformed-reply report later."""
    m = _SCORE_RE.search(raw)
    if not m:
        return None
    return float(sum(int(g) for g in m.groups()))


def judge(query: Query, answer: Answer) -> float:
    """LLM-as-judge for non-verifiable (judged-kind) questions — DECISION D5 (plan_a3.md
    §5): hybrid method, this function is the LLM half; `reports/a3_judge_spotcheck.md`
    (Step 16 Do item 3) is the human-agreement evidence for the other half.

    **Method:** one call to `llm.client.LLM` (Groq, D1) using the fixed `JUDGE` prompt
    (`llm/prompts.py`, Step 6), temperature 0 for reproducibility. The evidence shown to the
    judge is reconstructed from `answer.citations` (resolved against the real built index) —
    the judge checks the answer's claims against what it actually cited, not a broader
    context this function doesn't have access to.

    **Rubric** (verbatim from the `JUDGE` prompt, restated here since §5 of the form asks
    for both the method and the rubric to be stated, not just referenced): three criteria,
    each scored 0/1/2 by the model —
      - CORRECTNESS: does the conclusion actually and validly follow from the evidence, and
        is it mathematically sound?
      - COMPLETENESS: does the answer address the full question, not just part of it?
      - GROUNDEDNESS: is every claim traceable to the evidence, with no unsupported addition?
    A correct "insufficient evidence" abstention scores 2 on every criterion — correctly
    abstaining is not penalised.

    **Return value:** the three criteria summed, range **[0.0, 6.0]** (not normalised to
    0-1 — the prompt's own natural `TOTAL: <sum>/6` scale, so a reader cross-checking a
    score against a raw judge transcript sees the same number in both places).
    `grading_kit/success_check.py`'s `JUDGE_PASS_THRESHOLD = 4.0` was set against this exact
    range (Step 15's own RESULT block flags this dependency).

    **Never raises** — a malformed reply, a request/parsing failure, or any other exception
    all fail closed to `0.0` (the worst possible score, not a crash) rather than taking down
    whatever eval loop called this, same "never raise at this seam" contract every other
    cross-cutting piece of this codebase already holds to (`postprocess.py`'s `_ground`,
    `hitl.py`'s `escalate`, `guardrails.py`'s `check`).
    """
    try:
        llm = client.LLM(config.load())
        evidence = _evidence_block(answer)
        prompt = prompts.JUDGE.format(query=query.text, evidence=evidence, answer=answer.text)
        raw = llm.complete(prompt, temperature=0)
        score = _parse_score(raw)
        return score if score is not None else 0.0
    except Exception:
        return 0.0
