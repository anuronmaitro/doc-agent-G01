"""Per-task verifier. FIXED signature."""

from __future__ import annotations

import sys
from pathlib import Path

# `doc_agent` is only importable via the pinned pythonpath (`pyproject.toml`, `src`) --
# pytest picks that up on its own, but this module can also be imported directly by a
# non-pytest caller (Step 17's run_eval.py, or a grader running this standalone), so it
# ensures its own path rather than assuming the caller already set one up. Repo-relative via
# `__file__`, not cwd-relative -- same reasoning as logging_conf.py's TRACE_PATH.
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from doc_agent.contracts import Answer, Query  # noqa: E402
from doc_agent.eval import judge as judge_module  # noqa: E402
from doc_agent.eval.metrics import normalize_latex  # noqa: E402

# Step 16 (not yet implemented) decides judge()'s real return range and documents it there --
# assumed here to be the JUDGE prompt's own natural "TOTAL: <sum>/6" scale (3 criteria, 0-2
# each), since that is what the prompt (Step 6, already written) actually asks the model to
# emit, not an arbitrary 0-1 normalisation nothing in the prompt calls for. >=4/6 requires an
# average of "mostly correct" across all three criteria (e.g. 2+1+1), not just one strong
# criterion carrying the others. If Step 16 picks a different range, this constant is the one
# place to update -- flagged forward on Step 16's own header in plan_a3.md so its author
# either confirms this assumption or coordinates a change here.
JUDGE_PASS_THRESHOLD = 4.0


def _check_verifiable(task: dict, answer: dict) -> bool:
    """Exact match, after `eval/metrics.py`'s own LaTeX normalisation, of the gold value
    appearing *within* the answer's text -- not the whole text being nothing but the gold
    value. A real synthesized answer legitimately carries surrounding prose (the SYNTHESIZE
    prompt asks for an answer sentence, not a bare value); requiring whole-string equality
    would mark a correct, well-formed answer wrong for being readable. The value match itself
    is still exact (post-normalisation substring, not a fuzzy/partial one), and the answer
    must actually be `grounded` -- an abstained answer's text is never treated as satisfying
    a verifiable task just because a value happens to appear in unrelated boilerplate."""
    gold = task.get("gold")
    text = answer.get("text")
    if not isinstance(gold, str) or not gold.strip() or not isinstance(text, str):
        return False
    if answer.get("grounded") is not True:
        return False
    return normalize_latex(gold) in normalize_latex(text)


def _check_judged(task: dict, answer: dict) -> bool:
    """Delegate to `eval.judge.judge()`, thresholded. Until Step 16 lands, `judge()` itself
    raises `NotImplementedError` -- caught by `check()`'s own outer `try`/`except`, so a
    judged-kind task simply returns `False` (not yet gradable) rather than crashing whatever
    called `check()`, same "never raise" contract every other cross-cutting piece of this
    codebase already holds to."""
    query = Query(text=task["question"], verifiable=False, judged=True)
    ans = Answer(**answer)
    score = judge_module.judge(query, ans)
    return score >= JUDGE_PASS_THRESHOLD


def _check_abstention(answer: dict) -> bool:
    """Passes when the agent abstained, fails when it answered -- the one polarity that must
    never be gotten backwards (Step 15's own ORDER). `grounded` alone is the signal: every
    abstention path in `Agent.synthesize()` (Step 11) -- a self-reported "no evidence", zero
    citations resolving, decide()'s own k_max abstain, or a verify-and-correct retry that's
    still ungrounded -- converges on `grounded=False` by the time synthesize() returns, so
    checking it directly is both sufficient and simpler than string-matching the abstention
    sentinel text. `grounded is False` (identity, not a truthiness check): a missing or
    malformed `grounded` field must fail closed, not be silently treated as an abstention --
    that's exactly the kind of gap that would quietly reward a hallucination."""
    return answer.get("grounded") is False


def check(task: dict, answer: dict) -> bool:
    """Return True if `answer` satisfies `task`. Never raises -- a malformed `task`/`answer`,
    an exception from `judge()` (including "not implemented yet", pre-Step-16), or a bad
    `Answer(**answer)` construction all fail closed to `False` rather than crashing whatever
    eval loop called this."""
    try:
        kind = task.get("kind")
        if kind == "verifiable":
            return _check_verifiable(task, answer)
        if kind == "judged":
            return _check_judged(task, answer)
        if kind == "abstention":
            return _check_abstention(answer)
        return False
    except Exception:
        return False
