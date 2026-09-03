"""Stage 9 -- EXPLAINABLE (E32), PRIMARY NFR -- retrieval-level attribution, not pixel Grad-CAM.

A1's own commitment (plan_a3.md Sec.5) is stricter than plain source attribution: a grounded
answer must say WHY the reference it cited beats the runner-up it didn't, grounded in the
real rerank score gap (Step 3/7's `results[0].score - results[1].score`), not an unsupported
preference. Two headline numbers, both required by the form (Sec.3):

  - **rationale coverage**   (target 1.0)  -- of the answers that actually cite something,
    the fraction whose text carries a non-empty rationale. `postprocess.format_answer` only
    appends one when the model's own RATIONALE field was non-empty -- measured here, not
    assumed 1.0 just because the SYNTHESIZE prompt always asks for one.
  - **rationale faithfulness** (target >= 0.90) -- of the answers whose rationale makes a
    checkable "beat the runner-up" claim, the fraction where that claim is actually true of
    what retrieval really ranked. Checked mechanically against `traces/run.jsonl` (Step 12),
    not judged by an LLM reading the prose -- "checkable against the trace, not judged by
    vibes" is the whole point of this being the metrics file's job and not eval/judge.py's.

**Why this is retrieval-level attribution, not the Grad-CAM the old stub named.** Grad-CAM
back-propagates a model's output gradient onto input PIXELS to produce a heatmap over an
image -- a vision-level interpretability technique. This system does not do that, and not by
omission: `enhance.enabled: false` (configs/config.yaml, A1's own trade-off, Step 7's
tools.py) means A2's enhancement stage never runs, and nothing downstream of OCR keeps a
differentiable path back to page pixels at all -- by the time an `Answer` exists, "the
evidence" is TEXT chunks (`index/chunk.py`), reranked scores, and citation spans, none of
which a pixel gradient could attach to even if we wanted one. What "explainable" means here
instead is retrieval-level attribution: which chunk, out of which real alternatives, by which
actually-measured score -- itself a genuine, falsifiable claim (this file's entire job), just
not a heatmap over a scanned page.

**How the trace makes this checkable at all.** `traces/run.jsonl` is truncated per run
(`logging_conf.py`'s own documented design) -- it is NOT a durable log across a whole eval
batch, only the most recently completed answer's trail. Since 2026-08-26 (Step 20), its final
`answer` TraceStep also carries the real primary/runner-up score breakdown (`primary_chunk_id`
/`runner_up_chunk_id`/`score_gap`), read from `decide()`'s own final ranked chunk list -- see
`logging_conf.register()`'s own docstring and `traces/README.md`. `explain()` below reads
that file immediately after the `pipeline.answer()` call that produced `answer`, exactly the
way `scripts/run_eval.py`'s per-task loop already calls `read_trace()` -- call it any later
(after the next task's answer) and it verifies against the WRONG task's trace.
"""

from __future__ import annotations

import json
from typing import Any

from .. import logging_conf
from ..contracts import *  # noqa

RATIONALE_COVERAGE_TARGET = 1.0
RATIONALE_FAITHFULNESS_TARGET = 0.90

_RATIONALE_MARKER = "\n\nRationale: "


def _extract_rationale(answer_text: str) -> str:
    """The rationale `postprocess.format_answer` appended, or "" if none was appended --
    either the model left its RATIONALE field empty, or this is an abstention (format_answer
    never appends a rationale to an abstained answer's text at all: it returns the bare
    `INSUFFICIENT_EVIDENCE` sentinel before the rationale is ever used)."""
    if _RATIONALE_MARKER not in answer_text:
        return ""
    return answer_text.split(_RATIONALE_MARKER, 1)[1].strip()


def _read_last_answer_obs() -> dict[str, Any]:
    """The most recently written "answer" TraceStep's `obs`. Never raises: a missing,
    unreadable, or corrupt trace file degrades to `{}` -- "nothing to verify against" --
    same fail-soft contract every other hook/trace reader in this codebase holds to, not a
    crash in what is meant to be a reporting-only code path."""
    path = logging_conf.TRACE_PATH
    if not path.exists():
        return {}
    last_answer: dict[str, Any] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            step = json.loads(line)
            if step.get("tool") == "answer":
                last_answer = step.get("obs", {})
    except (OSError, json.JSONDecodeError):
        return {}
    return last_answer


def explain(answer: Answer, cfg: dict) -> dict:
    """Explain ONE answer: does it carry a rationale, and -- when that rationale makes a
    checkable "beat the runner-up" claim -- is the claim actually true of the real trace.

    `cfg` is accepted, not read, to match this file's FIXED signature (plan_a3.md Step 20's
    own Do item 1) -- everything needed comes from `answer` and the trace `pipeline.answer()`
    just wrote for it, not from cfg.

    MUST be called before the next `pipeline.answer()` call in the same process -- see this
    module's own docstring on why the trace file is per-answer, not a durable log.

    Returns a dict, never raises:
      answered            -- True iff this answer is grounded with >=1 citation (an
                              abstention cites nothing, so there is no "why this reference"
                              claim to make in the first place -- scoping detail both
                              aggregate functions below read directly off this field).
      has_rationale        -- True iff answer.text actually carries a non-empty rationale.
      rationale             -- the rationale text itself (empty string if none).
      primary_chunk_id       -- the answer's own first-cited chunk id, or None if unanswered.
      checkable              -- True iff the trace recorded a real runner-up to compare
                              against (False when only one chunk was ever retrieved, or no
                              trace was found -- nothing to falsify either way).
      faithful                -- None until checkable is True; then True/False.
      runner_up_chunk_id / score_gap -- the trace's own recorded numbers, when checkable.
      reason                   -- one human-readable sentence explaining the verdict above.
    """
    rationale = _extract_rationale(answer.text)
    result: dict[str, Any] = {
        "answered": False,
        "has_rationale": bool(rationale),
        "rationale": rationale,
        "primary_chunk_id": None,
        "checkable": False,
        "faithful": None,
        "runner_up_chunk_id": None,
        "score_gap": None,
        "reason": "",
    }

    if not answer.grounded or not answer.citations:
        result["reason"] = "abstained/ungrounded -- no cited reference to explain"
        return result

    result["answered"] = True
    primary = answer.citations[0].chunk_id
    result["primary_chunk_id"] = primary

    trace_obs = _read_last_answer_obs()
    trace_primary = trace_obs.get("primary_chunk_id")
    trace_runner_up = trace_obs.get("runner_up_chunk_id")
    trace_gap = trace_obs.get("score_gap")

    if trace_primary is None or trace_runner_up is None or trace_gap is None:
        result["reason"] = (
            "no runner-up recorded in the trace (fewer than two chunks retrieved, or no "
            "matching trace found) -- nothing to check a 'beat the runner-up' claim against"
        )
        return result

    result["checkable"] = True
    result["runner_up_chunk_id"] = trace_runner_up
    result["score_gap"] = trace_gap

    if primary != trace_primary:
        result["faithful"] = False
        result["reason"] = (
            f"cited primary chunk {primary!r} does not match the trace's real top-scoring "
            f"chunk {trace_primary!r} -- the 'why this reference' premise does not match "
            f"what retrieval actually ranked best"
        )
        return result

    if trace_gap <= 0:
        result["faithful"] = False
        result["reason"] = (
            f"cited chunk did not actually outscore the runner-up (recorded gap={trace_gap:.4f})"
        )
        return result

    result["faithful"] = True
    result["reason"] = f"cited chunk is the trace's real top scorer, gap={trace_gap:.4f} > 0"
    return result


def rationale_coverage(explanations: list[dict]) -> dict[str, Any]:
    """target 1.0 (`RATIONALE_COVERAGE_TARGET`). Scoped to `answered` explanations only --
    an abstention cites nothing, so there is no "reference" for a rationale to be about;
    counting it against coverage would conflate "correctly declined to answer" with "failed
    to explain an answer it gave." Returns `{"rate", "n", "n_covered"}` rather than a bare
    float -- `rate` is None (not a silently vacuous 1.0 or 0.0) when `n == 0`, so a caller
    with e.g. an all-abstention task set reports that honestly instead of a misleading
    number, matching this codebase's own established precedent for an honestly-reported
    out-of-band result over a flattering reframe."""
    answered = [e for e in explanations if e["answered"]]
    if not answered:
        return {"rate": None, "n": 0, "n_covered": 0}
    n_covered = sum(1 for e in answered if e["has_rationale"])
    return {
        "rate": round(n_covered / len(answered), 4),
        "n": len(answered),
        "n_covered": n_covered,
    }


def rationale_faithfulness(explanations: list[dict]) -> dict[str, Any]:
    """target >= 0.90 (`RATIONALE_FAITHFULNESS_TARGET`). Scoped to `checkable` explanations
    only -- an answer with no real runner-up in the trace made no comparative claim, so there
    is nothing to falsify; including it would move the number for a reason unrelated to
    whether any actual claim was faithful. `rate` is None (not 0.0/1.0) when `n == 0`, same
    honest-reporting reasoning as `rationale_coverage` above."""
    checkable = [e for e in explanations if e["checkable"]]
    if not checkable:
        return {"rate": None, "n": 0, "n_faithful": 0}
    n_faithful = sum(1 for e in checkable if e["faithful"])
    return {
        "rate": round(n_faithful / len(checkable), 4),
        "n": len(checkable),
        "n_faithful": n_faithful,
    }
