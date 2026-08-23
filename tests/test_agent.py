"""Unit test home for agent. IMPLEMENT — CI runs these."""

from types import SimpleNamespace

import httpx
import pytest
from groq import AuthenticationError, RateLimitError

from doc_agent import hooks
from doc_agent.agent import hitl_store
from doc_agent.agent.agent import Agent
from doc_agent.contracts import Chunk, ToolResult
from doc_agent.eval import metrics
from doc_agent.llm import client as client_mod
from doc_agent.llm import postprocess

CFG = {"agent": {"model": "openai/gpt-oss-120b"}}


def _make_agent() -> Agent:
    # act() never touches self.retriever -- it only dispatches through tools.REGISTRY --
    # so a bare cfg/None retriever is enough here. synthesize() also never touches
    # self.retriever (it reads state["chunks"], set by decide()), so this is fine for
    # synthesize()-only tests too, as long as the test builds its own state dict.
    return Agent(cfg={"agent": {"max_steps": 8}}, retriever=None)


# decide()'s own cfg -- small k/k_step/k_max so a widening test takes 2-3 calls, not 4
# (real configs/config.yaml uses k=10/k_step=10/k_max=40; the *shape* of the policy is
# what's under test here, not the real numbers).
DECIDE_RETRIEVE_CFG = {
    "k": 1,
    "k_step": 1,
    "k_max": 3,
    "weak_threshold": 0.5,
    # False -- these tests are about the widen/abstain loop's own logic (real, controlled
    # dense scores from _StubRetriever), not about reranking, which decide() now also calls
    # (2026-08-23) before every is_weak() check. rerank.rerank() is itself a no-op
    # passthrough when this is False, so chunk.score here stays exactly what the stub set.
    "rerank": False,
}


class _StubRetriever:
    """Controllable stand-in for retrieval.retriever.Retriever -- returns one canned
    top_score per call, in order, so a test can script an exact weak/strong sequence
    without a real index or encoder. Mirrors this file's own _FakeCompletions pattern."""

    def __init__(self, top_scores: list) -> None:
        self._top_scores = list(top_scores)
        self.calls: list = []

    def retrieve(self, query: str, k: int) -> list:
        self.calls.append((query, k))
        score = self._top_scores.pop(0)
        return [Chunk(id="c0", doc_id="d0", text="chunk text", page_ids=["p0"], score=score)]


def _agent_with_stub(top_scores: list):
    stub = _StubRetriever(top_scores)
    agent = Agent(cfg={"agent": {"max_steps": 8}, "retrieve": DECIDE_RETRIEVE_CFG}, retriever=stub)
    return agent, stub


_REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _fake_response(text: str = "answer", total_tokens: int = 42) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def _rate_limit_error() -> RateLimitError:
    return RateLimitError("rate limited", response=httpx.Response(429, request=_REQ), body=None)


def _auth_error() -> AuthenticationError:
    return AuthenticationError("bad key", response=httpx.Response(401, request=_REQ), body=None)


class _FakeCompletions:
    """Pops one canned response/exception per call, in order -- lets a test script
    exactly the sequence of failures-then-success it wants to exercise."""

    def __init__(self, side_effects: list) -> None:
        self._side_effects = list(side_effects)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        effect = self._side_effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect


class _FakeGroqClient:
    def __init__(self, side_effects: list) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(side_effects))


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Every test here gets a fake key regardless of the real .env -- CI has no key,
    and a test that needs one is a test that fails on a grader's clean clone."""
    monkeypatch.setattr(client_mod.settings, "llm_api_key", "fake-test-key")


def _make_llm(monkeypatch, side_effects: list) -> client_mod.LLM:
    monkeypatch.setattr(client_mod, "Groq", lambda api_key: _FakeGroqClient(side_effects))
    return client_mod.LLM(CFG)


class TestLLMClient:
    def test_missing_key_raises_clear_error(self, monkeypatch):
        monkeypatch.setattr(client_mod.settings, "llm_api_key", "")
        with pytest.raises(RuntimeError, match="LLM_API_KEY"):
            client_mod.LLM(CFG)

    def test_successful_call_returns_text(self, monkeypatch):
        llm = _make_llm(monkeypatch, [_fake_response("the answer")])
        assert llm.complete("a question") == "the answer"

    def test_temperature_zero_by_default(self, monkeypatch):
        llm = _make_llm(monkeypatch, [_fake_response()])
        llm.complete("q")
        assert llm._client.chat.completions.calls[0]["temperature"] == 0

    def test_temperature_override_is_respected(self, monkeypatch):
        llm = _make_llm(monkeypatch, [_fake_response()])
        llm.complete("q", temperature=0.7)
        assert llm._client.chat.completions.calls[0]["temperature"] == 0.7

    def test_model_comes_from_config_not_hardcoded(self, monkeypatch):
        llm = _make_llm(monkeypatch, [_fake_response()])
        llm.complete("q")
        assert llm._client.chat.completions.calls[0]["model"] == "openai/gpt-oss-120b"

    def test_call_count_and_token_count_accumulate(self, monkeypatch):
        llm = _make_llm(
            monkeypatch, [_fake_response(total_tokens=10), _fake_response(total_tokens=15)]
        )
        llm.complete("q1")
        llm.complete("q2")
        assert llm.call_count == 2
        assert llm.total_tokens == 25

    def test_retries_on_rate_limit_then_succeeds(self, monkeypatch):
        llm = _make_llm(monkeypatch, [_rate_limit_error(), _fake_response("recovered")])
        result = llm.complete("q", max_retries=3)
        assert result == "recovered"
        assert len(llm._client.chat.completions.calls) == 2  # one failure, one success

    def test_gives_up_after_max_retries_and_raises(self, monkeypatch):
        llm = _make_llm(
            monkeypatch,
            [_rate_limit_error(), _rate_limit_error(), _rate_limit_error()],
        )
        with pytest.raises(RateLimitError):
            llm.complete("q", max_retries=2)
        # 1 initial attempt + 2 retries = 3 calls total, then it gives up
        assert len(llm._client.chat.completions.calls) == 3

    def test_auth_error_is_not_retried(self, monkeypatch):
        """A bad key is a config problem, not a transient one -- retrying it would just
        burn the free-tier rate-limit budget on calls that can never succeed."""
        llm = _make_llm(monkeypatch, [_auth_error(), _fake_response("should never be reached")])
        with pytest.raises(AuthenticationError):
            llm.complete("q", max_retries=3)
        assert len(llm._client.chat.completions.calls) == 1  # no retry attempted

    def test_max_retries_kwarg_is_not_forwarded_to_the_api_call(self, monkeypatch):
        """max_retries is this wrapper's own control knob -- Groq's API has no such
        parameter, so it must be popped before the kwargs reach chat.completions.create."""
        llm = _make_llm(monkeypatch, [_fake_response()])
        llm.complete("q", max_retries=1)
        assert "max_retries" not in llm._client.chat.completions.calls[0]


class TestAgentAct:
    """Step 9: act() only -- registry dispatch. decide()/synthesize() land at Steps 10/11."""

    def test_dispatches_to_registered_tool_by_name(self):
        agent = _make_agent()
        result = agent.act({"tool": "calculator", "args": {"expr": "2 + 2"}})
        assert isinstance(result, ToolResult)
        assert result.ok is True
        assert result.payload["value"] == 4

    def test_unknown_tool_returns_ok_false_not_raise(self):
        agent = _make_agent()
        result = agent.act({"tool": "not_a_real_tool", "args": {}})
        assert result.ok is False
        assert "not_a_real_tool" in result.payload["reason"]

    def test_missing_args_key_defaults_to_empty_kwargs(self):
        # aggregate's "count" op accepts an empty list, so a tool with no required args
        # exercises the action.get("args", {}) default without needing a real payload.
        agent = _make_agent()
        result = agent.act({"tool": "aggregate", "args": {"op": "count", "items": []}})
        assert result.ok is True
        assert result.payload["value"] == 0
        # decide() (Step 10) and synthesize() (Step 11) are both implemented as of this
        # step -- their own behaviour is covered by TestAgentDecide/TestAgentSynthesize
        # below, not re-asserted here.


class TestAgentDecide:
    """Step 10: evidence-gated re-search -- the mandatory agentic behaviour.

    The trace, for each case below (walked through in the PR description, per the ORDER):
    each retrieval attempt decide() makes appends one {"top_score": ..., "k": ...} entry to
    state["obs"] BEFORE either widening or stopping -- so state["obs"] after decide() returns
    is the full widen-and-recheck trail, in order, with the branch on the real number visible
    directly (a strong score right away = one entry and done; weak scores show k climbing
    entry by entry until either a strong score appears or k_max is hit and state["abstain"]
    flips true). Step 12 (not this step) is what later turns this into traces/run.jsonl.
    """

    def test_strong_evidence_is_a_single_pass_no_widening(self):
        agent, stub = _agent_with_stub([0.9])
        state = {"query": "q", "obs": []}
        action = agent.decide(state)

        assert action == {"tool": "stop", "args": {}}
        assert [k for _, k in stub.calls] == [1]
        assert state["obs"] == [{"top_score": 0.9, "k": 1}]
        assert state["abstain"] is False

    def test_weak_then_strong_widens_exactly_once(self):
        agent, stub = _agent_with_stub([0.2, 0.9])
        state = {"query": "q", "obs": []}
        agent.decide(state)

        assert [k for _, k in stub.calls] == [1, 2]  # k + k_step, once
        assert state["obs"] == [
            {"top_score": 0.2, "k": 1},
            {"top_score": 0.9, "k": 2},
        ]
        assert state["abstain"] is False

    def test_weak_all_the_way_widens_to_k_max_then_abstains(self):
        agent, stub = _agent_with_stub([0.1, 0.1, 0.1])
        state = {"query": "q", "obs": []}
        action = agent.decide(state)

        assert [k for _, k in stub.calls] == [1, 2, 3]  # climbs to k_max, never further
        assert state["obs"] == [
            {"top_score": 0.1, "k": 1},
            {"top_score": 0.1, "k": 2},
            {"top_score": 0.1, "k": 3},
        ]
        assert state["abstain"] is True
        assert state["abstain_reason"] == "insufficient evidence"
        assert action == {"tool": "stop", "args": {}}  # never fabricates, never loops forever

    def test_k_never_exceeds_k_max(self):
        agent, stub = _agent_with_stub([0.1, 0.1, 0.1])
        state = {"query": "q", "obs": []}
        agent.decide(state)

        assert all(k <= DECIDE_RETRIEVE_CFG["k_max"] for _, k in stub.calls)

    def test_reranks_before_checking_weak(self, monkeypatch):
        """2026-08-23 fix: before this, rerank.rerank() was never called anywhere in the
        live loop -- is_weak() only ever saw raw dense scores. Prove decide() now actually
        calls it, and that is_weak() judges the RERANKED score, not the dense one the stub
        set. A stub that reports a dense score of 0.9 (comfortably strong) but whose
        reranked score comes back 0.1 (weak) must widen -- if decide() were still checking
        the dense score, it would wrongly stop after one pass."""
        import doc_agent.agent.agent as agent_mod

        calls: list = []

        def _fake_rerank(query, candidates, cfg):
            calls.append((query, [c.id for c in candidates]))
            return [c.model_copy(update={"score": 0.1}) for c in candidates]

        monkeypatch.setattr(agent_mod.rerank, "rerank", _fake_rerank)

        cfg = {**DECIDE_RETRIEVE_CFG, "rerank": True}
        stub = _StubRetriever([0.9, 0.9, 0.9])  # dense scores alone would never widen
        agent = Agent(cfg={"agent": {"max_steps": 8}, "retrieve": cfg}, retriever=stub)
        state = {"query": "q", "obs": []}
        agent.decide(state)

        assert len(calls) == 3  # once per retrieve attempt, including the widen calls
        assert state["abstain"] is True  # reranked score (0.1) stayed under weak_threshold
        assert all(c.score == 0.1 for c in state["chunks"])  # is_weak saw the reranked score


class TestAgentSynthesize:
    """Step 11: synthesize() -- grounded answer + D6 verify-and-correct retry.

    hooks.clear()/postprocess.register(hooks) per test, matching test_crosscutting.py's own
    pattern: `_ground` is a real registered BEFORE_ANSWER handler, not mocked out, so these
    tests exercise the actual hook wiring. `metrics.groundedness` (the deep, index-backed
    check `_ground` calls) is monkeypatched so the outcome is deterministic without a real
    built index -- format_answer()'s own structural parsing/resolution runs for real.
    """

    def setup_method(self):
        hooks.clear()
        postprocess.register(hooks)

    def teardown_method(self):
        hooks.clear()

    def _agent_and_state(self):
        agent = Agent(
            cfg={"agent": {"max_steps": 8, "model": "openai/gpt-oss-120b"}}, retriever=None
        )
        chunks = [
            Chunk(id="c1", doc_id="d0", text="Gamma(1/2)=sqrt(pi).", page_ids=["p0"], score=0.9)
        ]
        state = {"query": "What is Gamma(1/2)?", "obs": [], "chunks": chunks, "abstain": False}
        return agent, state

    def _wire_fake_llm(self, monkeypatch, texts: list) -> _FakeGroqClient:
        fake = _FakeGroqClient([_fake_response(t) for t in texts])
        monkeypatch.setattr(client_mod, "Groq", lambda api_key: fake)
        monkeypatch.setattr(client_mod.settings, "llm_api_key", "fake-test-key")
        return fake

    def test_decide_abstain_short_circuits_before_any_llm_call(self, monkeypatch, tmp_path):
        # No usable key at all -- if synthesize() ever tried to build an LLM here despite the
        # abstain flag, LLM.__init__ would raise loudly rather than silently reaching a real
        # API, so this also proves no call is attempted.
        monkeypatch.setattr(client_mod.settings, "llm_api_key", "")
        monkeypatch.setattr(hitl_store, "QUEUE_PATH", tmp_path / "hitl_queue.json")
        agent, state = self._agent_and_state()
        state["abstain"] = True
        state["abstain_reason"] = "insufficient evidence"

        ans = agent.synthesize(state)

        assert ans.grounded is False
        assert ans.citations == []
        assert ans.text == postprocess.INSUFFICIENT_EVIDENCE
        # A1's HITL trigger (Step 13): confidence (always 0.0 here) under tau=0.50 after
        # decide() has already re-searched to k_max -- this is exactly that case.
        pending = hitl_store.pending()
        assert len(pending) == 1
        assert "0.50" in pending[0]["reason"] or "tau" in pending[0]["reason"].lower()

    def test_first_pass_grounded_answer_is_returned_as_is(self, monkeypatch):
        monkeypatch.setattr(metrics, "groundedness", lambda ans: 0.95)
        fake = self._wire_fake_llm(
            monkeypatch,
            ["ANSWER: Gamma(1/2) = sqrt(pi)\nCITATIONS: c1\nRATIONALE: c1 says so directly.\n"],
        )
        agent, state = self._agent_and_state()

        ans = agent.synthesize(state)

        assert ans.grounded is True
        assert "Gamma(1/2) = sqrt(pi)" in ans.text
        assert len(fake.chat.completions.calls) == 1  # no retry needed

    def test_verify_and_correct_retry_recovers_a_downgraded_answer(self, monkeypatch):
        scores = iter([0.1, 0.9])  # first attempt fails the deep check, the retry passes
        monkeypatch.setattr(metrics, "groundedness", lambda ans: next(scores))
        fake = self._wire_fake_llm(
            monkeypatch,
            [
                "ANSWER: a first guess\nCITATIONS: c1\nRATIONALE: r1\n",
                "ANSWER: a corrected answer\nCITATIONS: c1\nRATIONALE: r2\n",
            ],
        )
        agent, state = self._agent_and_state()

        ans = agent.synthesize(state)

        assert ans.grounded is True
        assert "corrected answer" in ans.text
        assert len(fake.chat.completions.calls) == 2  # exactly one retry
        retry_prompt = fake.chat.completions.calls[1]["messages"][0]["content"]
        assert "retry" in retry_prompt.lower()  # _ground's complaint actually reached the model

    def test_retry_exhausted_and_still_ungrounded_abstains_for_real(self, monkeypatch):
        monkeypatch.setattr(metrics, "groundedness", lambda ans: 0.1)  # never passes
        fake = self._wire_fake_llm(
            monkeypatch,
            [
                "ANSWER: a first guess\nCITATIONS: c1\nRATIONALE: r1\n",
                "ANSWER: still not backed by anything\nCITATIONS: c1\nRATIONALE: r2\n",
            ],
        )
        agent, state = self._agent_and_state()

        ans = agent.synthesize(state)

        assert ans.grounded is False
        assert ans.citations == []
        assert ans.text == postprocess.INSUFFICIENT_EVIDENCE  # the ungrounded text never ships
        assert len(fake.chat.completions.calls) == 2  # capped at ONE retry, not a loop

    def test_self_reported_abstention_on_first_pass_skips_the_retry(self, monkeypatch):
        monkeypatch.setattr(metrics, "groundedness", lambda ans: 0.0)
        fake = self._wire_fake_llm(
            monkeypatch,
            ["ANSWER: INSUFFICIENT EVIDENCE\nCITATIONS: NONE\nRATIONALE: nothing supports this.\n"],
        )
        agent, state = self._agent_and_state()

        ans = agent.synthesize(state)

        assert ans.grounded is False
        assert ans.text == postprocess.INSUFFICIENT_EVIDENCE
        assert len(fake.chat.completions.calls) == 1  # already-correct "I don't know" -- no retry
