"""Unit test home for eval/judge.py. IMPLEMENT — CI runs these."""

from types import SimpleNamespace

import httpx
import pytest
from groq import RateLimitError

from doc_agent.contracts import Answer, Chunk, Citation, Query
from doc_agent.eval import judge as judge_mod
from doc_agent.llm import client as client_mod

_REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _fake_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(total_tokens=10),
    )


class _FakeCompletions:
    def __init__(self, texts: list) -> None:
        self._texts = list(texts)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self._texts.pop(0)
        if isinstance(text, BaseException):
            raise text
        return _fake_response(text)


class _FakeGroqClient:
    def __init__(self, texts: list) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(texts))


@pytest.fixture(autouse=True)
def _wire_fake_key(monkeypatch):
    monkeypatch.setattr(client_mod.settings, "llm_api_key", "fake-test-key")


def _wire_llm(monkeypatch, texts: list) -> _FakeGroqClient:
    fake = _FakeGroqClient(texts)
    monkeypatch.setattr(client_mod, "Groq", lambda api_key: fake)
    return fake


def _query() -> Query:
    return Query(text="What does the reflection formula say?", verifiable=False, judged=True)


def _answer(citations=None, text="The reflection formula relates psi(1-z) to psi(z).") -> Answer:
    return Answer(text=text, citations=citations or [], grounded=True, confidence=0.7)


WELL_FORMED_REPLY = (
    "CORRECTNESS: 2\nCOMPLETENESS: 1\nGROUNDEDNESS: 2\nTOTAL: 5/6\n"
    "VERDICT: Correct and grounded, slightly narrow.\n"
)


class TestParseScore:
    def test_well_formed_reply_sums_the_three_criteria(self):
        assert judge_mod._parse_score(WELL_FORMED_REPLY) == 5.0

    def test_recomputes_the_sum_rather_than_trusting_a_wrong_total_line(self):
        raw = "CORRECTNESS: 2\nCOMPLETENESS: 2\nGROUNDEDNESS: 2\nTOTAL: 3/6\nVERDICT: x\n"
        assert judge_mod._parse_score(raw) == 6.0  # not the (wrong) TOTAL: 3

    def test_perfect_score(self):
        raw = "CORRECTNESS: 2\nCOMPLETENESS: 2\nGROUNDEDNESS: 2\nTOTAL: 6/6\nVERDICT: x\n"
        assert judge_mod._parse_score(raw) == 6.0

    def test_zero_score(self):
        raw = "CORRECTNESS: 0\nCOMPLETENESS: 0\nGROUNDEDNESS: 0\nTOTAL: 0/6\nVERDICT: x\n"
        assert judge_mod._parse_score(raw) == 0.0

    def test_garbled_reply_returns_none(self):
        assert judge_mod._parse_score("the model said something unrelated entirely") is None

    def test_missing_one_criterion_returns_none(self):
        raw = "CORRECTNESS: 2\nCOMPLETENESS: 1\nVERDICT: forgot groundedness\n"
        assert judge_mod._parse_score(raw) is None

    def test_out_of_range_digit_is_rejected(self):
        # only 0/1/2 are valid per the rubric -- a "3" must not silently parse.
        raw = "CORRECTNESS: 3\nCOMPLETENESS: 1\nGROUNDEDNESS: 2\nTOTAL: 6/6\nVERDICT: x\n"
        assert judge_mod._parse_score(raw) is None


class TestEvidenceBlock:
    def test_resolves_citations_against_the_chunk_lookup(self, monkeypatch):
        lookup = {"c1": Chunk(id="c1", doc_id="d0", text="Gamma(1/2)=sqrt(pi).", page_ids=["p0"])}
        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: lookup)
        answer = _answer(citations=[Citation(chunk_id="c1", span=(0, 10))])
        evidence = judge_mod._evidence_block(answer)
        assert "[c1]" in evidence
        assert "Gamma(1/2)=sqrt(pi)." in evidence

    def test_unresolvable_citation_is_dropped_not_crashing(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: {})
        answer = _answer(citations=[Citation(chunk_id="ghost", span=(0, 1))])
        evidence = judge_mod._evidence_block(answer)
        assert evidence == "(no cited evidence)"

    def test_no_citations_at_all(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: {})
        evidence = judge_mod._evidence_block(_answer(citations=[]))
        assert evidence == "(no cited evidence)"


class TestJudge:
    def setup_method(self):
        judge_mod._CHUNK_LOOKUP_CACHE = None

    def test_well_formed_reply_returns_the_summed_score(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: {})
        _wire_llm(monkeypatch, [WELL_FORMED_REPLY])
        score = judge_mod.judge(_query(), _answer())
        assert score == 5.0

    def test_score_is_bounded_zero_to_six(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: {})
        for text, expected in [
            ("CORRECTNESS: 0\nCOMPLETENESS: 0\nGROUNDEDNESS: 0\nTOTAL: 0/6\nVERDICT: x\n", 0.0),
            ("CORRECTNESS: 2\nCOMPLETENESS: 2\nGROUNDEDNESS: 2\nTOTAL: 6/6\nVERDICT: x\n", 6.0),
        ]:
            _wire_llm(monkeypatch, [text])
            score = judge_mod.judge(_query(), _answer())
            assert score == expected
            assert 0.0 <= score <= 6.0

    def test_malformed_reply_fails_closed_to_zero_not_raise(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: {})
        _wire_llm(monkeypatch, ["garbage, not the expected format at all"])
        assert judge_mod.judge(_query(), _answer()) == 0.0

    def test_llm_call_failing_fails_closed_not_raise(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: {})
        error = RateLimitError(
            "rate limited", response=httpx.Response(429, request=_REQ), body=None
        )
        _wire_llm(monkeypatch, [error, error, error, error])  # exhaust all retries
        assert judge_mod.judge(_query(), _answer()) == 0.0

    def test_missing_api_key_fails_closed_not_raise(self, monkeypatch):
        monkeypatch.setattr(client_mod.settings, "llm_api_key", "")
        assert judge_mod.judge(_query(), _answer()) == 0.0

    def test_temperature_zero_is_passed_for_reproducibility(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: {})
        fake = _wire_llm(monkeypatch, [WELL_FORMED_REPLY])
        judge_mod.judge(_query(), _answer())
        assert fake.chat.completions.calls[0]["temperature"] == 0

    def test_prompt_carries_the_query_evidence_and_answer(self, monkeypatch):
        lookup = {"c1": Chunk(id="c1", doc_id="d0", text="a real cited excerpt", page_ids=["p0"])}
        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: lookup)
        fake = _wire_llm(monkeypatch, [WELL_FORMED_REPLY])
        q = _query()
        a = _answer(citations=[Citation(chunk_id="c1", span=(0, 5))], text="the real answer text")
        judge_mod.judge(q, a)
        sent_prompt = fake.chat.completions.calls[0]["messages"][0]["content"]
        assert q.text in sent_prompt
        assert "a real cited excerpt" in sent_prompt
        assert "the real answer text" in sent_prompt

    def test_correct_abstention_can_score_full_marks(self, monkeypatch):
        """The JUDGE rubric's own rule: a correct 'insufficient evidence' scores 2 on every
        criterion. judge() itself must not special-case or block a perfect score for an
        abstained answer -- the prompt handles that, not this function."""
        monkeypatch.setattr(judge_mod, "_get_chunk_lookup", lambda: {})
        perfect = "CORRECTNESS: 2\nCOMPLETENESS: 2\nGROUNDEDNESS: 2\nTOTAL: 6/6\nVERDICT: x\n"
        _wire_llm(monkeypatch, [perfect])
        abstained = Answer(
            text="INSUFFICIENT EVIDENCE", citations=[], grounded=False, confidence=0.0
        )
        assert judge_mod.judge(_query(), abstained) == 6.0
