"""Unit test home for logging_conf.py's tracer. IMPLEMENT — CI runs these."""

import json

import pytest

from doc_agent import hooks, logging_conf
from doc_agent.contracts import Answer, Chunk


@pytest.fixture(autouse=True)
def _trace_to_tmp_path(tmp_path, monkeypatch):
    """Every test here writes to an isolated tmp file, never the real repo's
    traces/run.jsonl -- that file is reserved for Step 25's real committed evidence, not
    test output."""
    monkeypatch.setattr(logging_conf, "TRACE_PATH", tmp_path / "run.jsonl")
    hooks.clear()
    logging_conf.register(hooks)
    yield
    hooks.clear()


def _lines() -> list[dict]:
    if not logging_conf.TRACE_PATH.exists():
        return []
    return [json.loads(line) for line in logging_conf.TRACE_PATH.read_text().splitlines()]


class TestNeverRaises:
    """The one hard requirement: a trace handler that throws takes the agent down with it."""

    def test_on_step_with_state_never_raises(self):
        hooks.run(hooks.ON_STEP, {"state": {"query": "q", "obs": []}})

    def test_on_tool_call_with_action_never_raises(self):
        hooks.run(hooks.ON_TOOL_CALL, {"action": {"tool": "calculator", "args": {"expr": "1+1"}}})

    def test_after_answer_never_raises(self):
        ans = Answer(text="x", citations=[], grounded=True, confidence=0.5)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})

    def test_malformed_action_missing_tool_key_never_raises(self):
        hooks.run(hooks.ON_TOOL_CALL, {"action": {}})

    def test_after_answer_with_no_prior_on_step_never_raises(self):
        """A handler that only ever sees AFTER_ANSWER (no state captured yet) must still
        degrade gracefully, not crash."""
        ans = Answer(text="x", citations=[], grounded=False, confidence=0.0)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})


class TestTraceContent:
    def test_creates_the_traces_directory_if_missing(self, tmp_path, monkeypatch):
        nested = tmp_path / "does" / "not" / "exist" / "yet" / "run.jsonl"
        monkeypatch.setattr(logging_conf, "TRACE_PATH", nested)
        assert not nested.parent.exists()
        hooks.run(hooks.ON_TOOL_CALL, {"action": {"tool": "calculator", "args": {}}})
        assert nested.parent.is_dir()

    def test_tool_call_is_written_as_its_own_step(self):
        hooks.run(hooks.ON_TOOL_CALL, {"action": {"tool": "calculator", "args": {"expr": "1+1"}}})
        lines = _lines()
        assert len(lines) == 1
        assert lines[0]["tool"] == "calculator"
        assert lines[0]["args"] == {"expr": "1+1"}
        assert lines[0]["step"] == 1

    def test_decide_widen_trail_flushes_at_after_answer_as_retrieve_steps(self):
        """The core Step 12 requirement: obs must carry the top_score/k decide() saw, so the
        grader's gate can confirm the path actually depended on the evidence."""
        state = {
            "query": "What is Gamma(1/2)?",
            "obs": [{"top_score": 0.2, "k": 10}, {"top_score": 0.4, "k": 20}],
        }
        hooks.run(hooks.ON_STEP, {"state": state})
        ans = Answer(text="sqrt(pi)", citations=[], grounded=True, confidence=0.8)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})

        lines = _lines()
        retrieve_lines = [ln for ln in lines if ln["tool"] == "retrieve"]
        assert [ln["obs"] for ln in retrieve_lines] == [
            {"top_score": 0.2, "k": 10},
            {"top_score": 0.4, "k": 20},
        ]
        assert [ln["args"]["k"] for ln in retrieve_lines] == [10, 20]
        assert all(ln["args"]["query"] == "What is Gamma(1/2)?" for ln in retrieve_lines)
        # steps are sequential and end with exactly one answer step
        assert [ln["step"] for ln in lines] == list(range(1, len(lines) + 1))
        assert lines[-1]["tool"] == "answer"

    def test_grounded_answer_step_marks_abstained_false(self):
        state = {"query": "q", "obs": [{"top_score": 0.9, "k": 10}]}
        hooks.run(hooks.ON_STEP, {"state": state})
        ans = Answer(text="a real answer", citations=[], grounded=True, confidence=0.7)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})
        answer_line = _lines()[-1]
        assert answer_line["tool"] == "answer"
        assert answer_line["obs"] == {"abstained": False}

    def test_abstained_answer_step_marks_abstained_true(self):
        state = {"query": "q", "obs": [{"top_score": 0.1, "k": 40}]}
        hooks.run(hooks.ON_STEP, {"state": state})
        ans = Answer(text="INSUFFICIENT EVIDENCE", citations=[], grounded=False, confidence=0.0)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})
        answer_line = _lines()[-1]
        assert answer_line["obs"] == {"abstained": True}


class TestAnswerStepScoreBreakdown:
    """Step 20: the primary/runner-up score breakdown eval/interpret.py's
    rationale-faithfulness check reads -- read from state["chunks"], decide()'s own final
    ranked list, never from state["obs"] (that only ever carries the aggregate top_score)."""

    def test_two_or_more_chunks_records_the_full_breakdown(self):
        chunk1 = Chunk(id="c1", doc_id="d0", text="best", page_ids=["p0"], score=0.91)
        chunk2 = Chunk(id="c2", doc_id="d0", text="runner up", page_ids=["p0"], score=0.34)
        chunk3 = Chunk(id="c3", doc_id="d0", text="third", page_ids=["p0"], score=0.10)
        state = {
            "query": "q",
            "obs": [{"top_score": 0.91, "k": 10}],
            "chunks": [chunk1, chunk2, chunk3],
        }
        hooks.run(hooks.ON_STEP, {"state": state})
        ans = Answer(text="a", citations=[], grounded=True, confidence=0.8)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})

        obs = _lines()[-1]["obs"]
        assert obs["abstained"] is False
        assert obs["primary_chunk_id"] == "c1"
        assert obs["primary_score"] == pytest.approx(0.91)
        assert obs["runner_up_chunk_id"] == "c2"
        assert obs["runner_up_score"] == pytest.approx(0.34)
        assert obs["score_gap"] == pytest.approx(0.91 - 0.34)
        # the third-ranked chunk plays no part in the breakdown -- only top-2 matter.
        assert "c3" not in json.dumps(obs)

    def test_single_chunk_records_primary_only_no_runner_up_or_gap(self):
        chunk1 = Chunk(id="c1", doc_id="d0", text="only one", page_ids=["p0"], score=0.7)
        state = {"query": "q", "obs": [{"top_score": 0.7, "k": 10}], "chunks": [chunk1]}
        hooks.run(hooks.ON_STEP, {"state": state})
        ans = Answer(text="a", citations=[], grounded=True, confidence=0.6)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})

        obs = _lines()[-1]["obs"]
        assert obs["primary_chunk_id"] == "c1"
        assert "runner_up_chunk_id" not in obs
        assert "runner_up_score" not in obs
        assert "score_gap" not in obs

    def test_no_chunks_key_at_all_records_only_abstained_unchanged(self):
        """The k_max-abstain short-circuit in synthesize() never sets state["chunks"] to a
        real ranked list before the LLM would have run -- must degrade to exactly the old
        two-field shape, not crash or invent chunk ids from nothing."""
        state = {"query": "q", "obs": [{"top_score": 0.1, "k": 40}]}
        hooks.run(hooks.ON_STEP, {"state": state})
        ans = Answer(text="INSUFFICIENT EVIDENCE", citations=[], grounded=False, confidence=0.0)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})
        assert _lines()[-1]["obs"] == {"abstained": True}

    def test_empty_chunks_list_records_only_abstained_unchanged(self):
        state = {"query": "q", "obs": [], "chunks": []}
        hooks.run(hooks.ON_STEP, {"state": state})
        ans = Answer(text="INSUFFICIENT EVIDENCE", citations=[], grounded=False, confidence=0.0)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})
        assert _lines()[-1]["obs"] == {"abstained": True}


class TestTruncateThenAppendPerRun:
    def test_a_new_run_state_object_truncates_the_previous_runs_trail(self):
        state1 = {"query": "first", "obs": [{"top_score": 0.9, "k": 10}]}
        hooks.run(hooks.ON_STEP, {"state": state1})
        ans1 = Answer(text="a1", citations=[], grounded=True, confidence=0.7)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans1})
        first_run_lines = _lines()
        assert len(first_run_lines) == 2  # one retrieve + one answer

        state2 = {"query": "second", "obs": [{"top_score": 0.8, "k": 10}]}
        hooks.run(hooks.ON_STEP, {"state": state2})
        ans2 = Answer(text="a2", citations=[], grounded=True, confidence=0.6)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans2})
        second_run_lines = _lines()

        assert len(second_run_lines) == 2  # not 4 -- the first run's lines are gone
        assert all(ln["args"].get("query", "second") == "second" for ln in second_run_lines)
        assert second_run_lines[0]["step"] == 1  # step counter also resets per run

    def test_within_one_run_multiple_writes_append_not_truncate(self):
        state = {"query": "q", "obs": [{"top_score": 0.9, "k": 10}]}
        hooks.run(hooks.ON_STEP, {"state": state})
        hooks.run(hooks.ON_TOOL_CALL, {"action": {"tool": "cite", "args": {"chunk_id": "c1"}}})
        ans = Answer(text="a", citations=[], grounded=True, confidence=0.7)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})

        lines = _lines()
        # tool call + one retrieve (from state["obs"]) + answer = 3, all in one file
        assert len(lines) == 3
        assert [ln["tool"] for ln in lines] == ["cite", "retrieve", "answer"]
