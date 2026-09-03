"""Unit test home for eval/interpret.py (Step 20, A3) -- our primary NFR (Explainable).

Builds the trace the way the real pipeline does: register the real `logging_conf` hooks,
fire ON_STEP/AFTER_ANSWER with a real `state["chunks"]` list, then call `explain()` against
whatever that wrote to (a tmp-path-redirected) `traces/run.jsonl` -- exercises the real
`logging_conf` <-> `interpret` contract end to end, not a hand-crafted JSON fixture that
could silently drift from what `logging_conf._trace` actually writes.
"""

from __future__ import annotations

import pytest

from doc_agent import hooks, logging_conf
from doc_agent.contracts import Answer, Chunk, Citation
from doc_agent.eval import interpret

CFG: dict = {}


@pytest.fixture(autouse=True)
def _trace_to_tmp_path(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_conf, "TRACE_PATH", tmp_path / "run.jsonl")
    hooks.clear()
    logging_conf.register(hooks)
    yield
    hooks.clear()


def _write_trace(chunks: list[Chunk], obs: list[dict] | None = None) -> None:
    """Fires the real hooks a real Agent.run() would, so the trace file is genuinely
    produced by logging_conf's own code, not assembled by hand."""
    state = {
        "query": "q",
        "obs": obs or [{"top_score": chunks[0].score if chunks else 0.0, "k": 10}],
    }
    if chunks:
        state["chunks"] = chunks
    hooks.run(hooks.ON_STEP, {"state": state})
    grounded = bool(chunks)
    ans = Answer(
        text="placeholder",
        citations=[Citation(chunk_id=chunks[0].id, span=(0, 1))] if chunks else [],
        grounded=grounded,
        confidence=0.5,
    )
    hooks.run(hooks.AFTER_ANSWER, {"answer": ans})


def _answer(text: str, chunk_id: str | None, grounded: bool = True) -> Answer:
    citations = [Citation(chunk_id=chunk_id, span=(0, 1))] if chunk_id else []
    return Answer(text=text, citations=citations, grounded=grounded, confidence=0.7)


class TestExplainNotAnswered:
    def test_abstained_answer_is_not_answered_and_not_checkable(self):
        result = interpret.explain(
            Answer(text="INSUFFICIENT EVIDENCE", citations=[], grounded=False, confidence=0.0),
            CFG,
        )
        assert result["answered"] is False
        assert result["checkable"] is False
        assert result["faithful"] is None
        assert "no cited reference" in result["reason"]

    def test_grounded_flag_true_but_no_citations_is_still_not_answered(self):
        """Shouldn't happen given format_answer()'s own contract, but explain() must not
        crash or misreport if it ever does -- citations, not the flag alone, gate 'answered'."""
        result = interpret.explain(
            Answer(text="x", citations=[], grounded=True, confidence=0.5), CFG
        )
        assert result["answered"] is False


class TestExplainFaithful:
    def test_cited_chunk_matches_trace_top_scorer_with_positive_gap_is_faithful(self):
        c1 = Chunk(id="c1", doc_id="d0", text="best", page_ids=["p0"], score=0.91)
        c2 = Chunk(id="c2", doc_id="d0", text="runner up", page_ids=["p0"], score=0.34)
        _write_trace([c1, c2])

        ans = _answer("ANSWER: x\n\nRationale: c1 beats c2 by a wide margin.", "c1")
        result = interpret.explain(ans, CFG)

        assert result["answered"] is True
        assert result["has_rationale"] is True
        assert result["checkable"] is True
        assert result["faithful"] is True
        assert result["runner_up_chunk_id"] == "c2"
        assert result["score_gap"] == pytest.approx(0.91 - 0.34)


class TestExplainUnfaithful:
    def test_cited_chunk_is_not_the_real_top_scorer(self):
        """The model cited c2 as primary, but c1 actually outranked it in the real trace --
        the rationale's 'why this reference' premise doesn't match what retrieval ranked."""
        c1 = Chunk(id="c1", doc_id="d0", text="best", page_ids=["p0"], score=0.91)
        c2 = Chunk(id="c2", doc_id="d0", text="runner up", page_ids=["p0"], score=0.34)
        _write_trace([c1, c2])

        ans = _answer("ANSWER: x\n\nRationale: chose c2.", "c2")
        result = interpret.explain(ans, CFG)

        assert result["checkable"] is True
        assert result["faithful"] is False
        assert "does not match the trace" in result["reason"]

    def test_zero_gap_to_the_runner_up_is_unfaithful(self):
        """A tie is not 'beat the runner-up' -- the comparative claim the prompt asks for
        requires a real, positive gap, not an equal score."""
        c1 = Chunk(id="c1", doc_id="d0", text="tied a", page_ids=["p0"], score=0.5)
        c2 = Chunk(id="c2", doc_id="d0", text="tied b", page_ids=["p0"], score=0.5)
        _write_trace([c1, c2])

        ans = _answer("ANSWER: x\n\nRationale: chose c1.", "c1")
        result = interpret.explain(ans, CFG)

        assert result["checkable"] is True
        assert result["faithful"] is False
        assert "did not actually outscore" in result["reason"]


class TestExplainNotCheckable:
    def test_single_retrieved_chunk_has_no_runner_up_to_check_against(self):
        c1 = Chunk(id="c1", doc_id="d0", text="only one", page_ids=["p0"], score=0.7)
        _write_trace([c1])

        ans = _answer("ANSWER: x\n\nRationale: only evidence available.", "c1")
        result = interpret.explain(ans, CFG)

        assert result["answered"] is True
        assert result["checkable"] is False
        assert result["faithful"] is None
        assert "no runner-up" in result["reason"]

    def test_no_trace_file_at_all_is_not_checkable_not_a_crash(self):
        # deliberately skip _write_trace() -- no trace exists yet
        ans = _answer("ANSWER: x\n\nRationale: whatever.", "c1")
        result = interpret.explain(ans, CFG)
        assert result["checkable"] is False
        assert result["faithful"] is None


class TestExplainRationaleCoverage:
    def test_no_rationale_text_is_recorded_as_missing_even_when_checkable(self):
        c1 = Chunk(id="c1", doc_id="d0", text="best", page_ids=["p0"], score=0.9)
        c2 = Chunk(id="c2", doc_id="d0", text="second", page_ids=["p0"], score=0.2)
        _write_trace([c1, c2])

        ans = _answer("ANSWER: x", "c1")  # no "\n\nRationale: ..." suffix at all
        result = interpret.explain(ans, CFG)

        assert result["has_rationale"] is False
        assert result["rationale"] == ""
        # still checkable/faithful -- coverage and faithfulness are independent measurements
        assert result["checkable"] is True
        assert result["faithful"] is True


class TestAggregateCoverage:
    def test_scoped_to_answered_only_abstentions_excluded(self):
        explanations = [
            interpret.explain(
                Answer(text="INSUFFICIENT EVIDENCE", citations=[], grounded=False, confidence=0.0),
                CFG,
            ),
            {"answered": True, "has_rationale": True},
            {"answered": True, "has_rationale": False},
        ]
        result = interpret.rationale_coverage(explanations)
        assert result == {"rate": 0.5, "n": 2, "n_covered": 1}

    def test_empty_input_reports_n_zero_rate_none_not_a_vacuous_number(self):
        assert interpret.rationale_coverage([]) == {"rate": None, "n": 0, "n_covered": 0}

    def test_all_abstained_reports_n_zero_rate_none(self):
        explanations = [
            interpret.explain(
                Answer(text="INSUFFICIENT EVIDENCE", citations=[], grounded=False, confidence=0.0),
                CFG,
            )
        ]
        assert interpret.rationale_coverage(explanations) == {"rate": None, "n": 0, "n_covered": 0}

    def test_full_coverage_reports_1_0(self):
        explanations = [{"answered": True, "has_rationale": True} for _ in range(5)]
        result = interpret.rationale_coverage(explanations)
        assert result["rate"] == 1.0
        assert result["n"] == 5


class TestAggregateFaithfulness:
    def test_scoped_to_checkable_only_single_chunk_cases_excluded(self):
        explanations = [
            {"checkable": True, "faithful": True},
            {"checkable": True, "faithful": False},
            {"checkable": False, "faithful": None},  # excluded -- nothing to falsify
        ]
        result = interpret.rationale_faithfulness(explanations)
        assert result == {"rate": 0.5, "n": 2, "n_faithful": 1}

    def test_empty_input_reports_n_zero_rate_none(self):
        assert interpret.rationale_faithfulness([]) == {"rate": None, "n": 0, "n_faithful": 0}

    def test_meets_the_090_target_on_a_realistic_mixed_batch(self):
        explanations = [{"checkable": True, "faithful": True} for _ in range(9)] + [
            {"checkable": True, "faithful": False}
        ]
        result = interpret.rationale_faithfulness(explanations)
        assert result["rate"] == pytest.approx(0.9)
        assert result["rate"] >= interpret.RATIONALE_FAITHFULNESS_TARGET
