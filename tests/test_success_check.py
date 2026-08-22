"""Unit test home for grading_kit/success_check.py. IMPLEMENT — CI runs these."""

import pytest

from doc_agent.eval import judge as judge_mod
from grading_kit import success_check


def _answer(**overrides) -> dict:
    base = {"text": "", "citations": [], "grounded": True, "confidence": 0.5}
    base.update(overrides)
    return base


class TestVerifiable:
    def _task(self, gold: str) -> dict:
        return {"kind": "verifiable", "gold": gold}

    def test_gold_value_present_in_a_grounded_answer_passes(self):
        task = self._task("1.41421\\,35624")
        answer = _answer(text="sqrt(2) is 1.41421\\,35624 to ten places.", grounded=True)
        assert success_check.check(task, answer) is True

    def test_different_latex_spelling_of_the_same_value_still_passes(self):
        # \tfrac vs \frac -- exactly the example the ORDER names.
        task = self._task("\\tfrac12")
        answer = _answer(text="The result is \\frac{1}{2} exactly.", grounded=True)
        assert success_check.check(task, answer) is True

    def test_wrong_value_fails(self):
        task = self._task("512")
        answer = _answer(text="The answer is 513.", grounded=True)
        assert success_check.check(task, answer) is False

    def test_right_value_but_answer_not_grounded_fails(self):
        """A value coincidentally present in an ungrounded/abstained answer must not count --
        only a real, grounded answer satisfies a verifiable task."""
        task = self._task("512")
        answer = _answer(text="512 is mentioned but INSUFFICIENT EVIDENCE.", grounded=False)
        assert success_check.check(task, answer) is False

    def test_missing_gold_fails_closed(self):
        task = {"kind": "verifiable"}
        answer = _answer(text="anything", grounded=True)
        assert success_check.check(task, answer) is False

    def test_non_string_answer_text_fails_closed(self):
        task = self._task("512")
        answer = _answer(text=None, grounded=True)
        assert success_check.check(task, answer) is False


class TestJudged:
    def _task(self) -> dict:
        return {"kind": "judged", "question": "Explain the reflection formula."}

    def test_score_at_or_above_threshold_passes(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "judge", lambda query, answer: 4.0)
        assert success_check.check(self._task(), _answer()) is True

    def test_score_above_threshold_passes(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "judge", lambda query, answer: 6.0)
        assert success_check.check(self._task(), _answer()) is True

    def test_score_below_threshold_fails(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "judge", lambda query, answer: 3.5)
        assert success_check.check(self._task(), _answer()) is False

    def test_judge_raising_any_exception_fails_closed(self, monkeypatch):
        def _boom(query, answer):
            raise RuntimeError("LLM unavailable")

        monkeypatch.setattr(judge_mod, "judge", _boom)
        assert success_check.check(self._task(), _answer()) is False

    def test_malformed_answer_dict_fails_closed(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "judge", lambda query, answer: 6.0)
        assert success_check.check(self._task(), {"text": "x"}) is False  # missing required fields


class TestAbstention:
    """The polarity that must never be backwards -- both directions get their own test."""

    def _task(self) -> dict:
        return {"kind": "abstention", "question": "What is the FFT's time complexity?"}

    def test_agent_abstaining_passes(self):
        answer = _answer(text="INSUFFICIENT EVIDENCE", citations=[], grounded=False, confidence=0.0)
        assert success_check.check(self._task(), answer) is True

    def test_agent_answering_instead_of_abstaining_fails(self):
        """The critical direction: a hallucinated answer to an abstention question must not
        quietly pass."""
        answer = _answer(text="It runs in O(n log n) time.", grounded=True, confidence=0.8)
        assert success_check.check(self._task(), answer) is False

    def test_missing_grounded_field_fails_closed_not_treated_as_abstention(self):
        answer = {"text": "something"}
        assert success_check.check(self._task(), answer) is False

    def test_non_boolean_grounded_value_fails_closed(self):
        # Defensive: only the real boolean False counts, not a falsy stand-in.
        answer = _answer(grounded=0)
        assert success_check.check(self._task(), answer) is False


class TestJudgedIntegrationWithTheRealJudge:
    """Step 16 landed after this file's judged tests above (all mocking judge_module.judge
    directly). This class proves the real wiring end to end instead: the real judge(), with
    only the LLM client mocked, called through check()'s own dispatch."""

    def _task(self) -> dict:
        return {"kind": "judged", "question": "Explain the reflection formula."}

    def test_real_judge_through_check_passes_a_good_answer(self, monkeypatch):
        from types import SimpleNamespace

        from doc_agent.llm import client as client_mod

        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: {})
        monkeypatch.setattr(client_mod.settings, "llm_api_key", "fake-test-key")

        reply = "CORRECTNESS: 2\nCOMPLETENESS: 2\nGROUNDEDNESS: 2\nTOTAL: 6/6\nVERDICT: good.\n"

        class _FakeCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
                    usage=SimpleNamespace(total_tokens=10),
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
        monkeypatch.setattr(client_mod, "Groq", lambda api_key: fake_client)

        assert success_check.check(self._task(), _answer()) is True

    def test_real_judge_through_check_fails_a_bad_answer(self, monkeypatch):
        from types import SimpleNamespace

        from doc_agent.llm import client as client_mod

        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: {})
        monkeypatch.setattr(client_mod.settings, "llm_api_key", "fake-test-key")

        reply = "CORRECTNESS: 0\nCOMPLETENESS: 1\nGROUNDEDNESS: 0\nTOTAL: 1/6\nVERDICT: weak.\n"

        class _FakeCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
                    usage=SimpleNamespace(total_tokens=10),
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
        monkeypatch.setattr(client_mod, "Groq", lambda api_key: fake_client)

        assert success_check.check(self._task(), _answer()) is False


class TestUnknownKindAndNeverRaises:
    def test_unknown_kind_fails_closed(self):
        assert success_check.check({"kind": "mystery"}, _answer()) is False

    def test_missing_kind_fails_closed(self):
        assert success_check.check({}, _answer()) is False

    def test_none_task_never_raises(self):
        assert success_check.check(None, _answer()) is False

    def test_none_answer_never_raises(self):
        assert success_check.check({"kind": "verifiable", "gold": "x"}, None) is False

    def test_completely_empty_inputs_never_raise(self):
        assert success_check.check({}, {}) is False


@pytest.fixture(autouse=True)
def _clear_metrics_chunk_cache(monkeypatch):
    """normalize_latex() itself needs no index, but keep this file isolated from any lazy
    module-global cache other eval/metrics.py tests might leave behind."""
    from doc_agent.eval import metrics

    monkeypatch.setattr(metrics, "_CHUNK_LOOKUP_CACHE", None)
