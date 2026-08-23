"""Stage 6 - FIXED loop - perceive -> decide -> act -> observe, with cross-cutting seams.
All three of decide()/act()/synthesize() are now implemented. Security, grounding, PII, and
tracing run via hooks at the marked seams - do NOT inline them here."""

from __future__ import annotations

from typing import Any

from .. import hooks
from ..contracts import *  # noqa
from ..llm import client, postprocess, prompts
from ..retrieval import rerank
from ..retrieval import retriever as retriever_mod
from . import hitl, tools
from .memory import Memory


class Agent:
    """FIXED loop. decide()/act()/synthesize() are all implemented."""

    def __init__(self, cfg: dict, retriever: Any) -> None:
        self.cfg = cfg["agent"]
        # decide()'s evidence-gated re-search needs cfg["retrieve"] (k/k_step/k_max/
        # weak_threshold), which self.cfg above doesn't carry -- same "keep the whole cfg
        # around" pattern retrieval/retriever.py's own Retriever.__init__ already uses
        # (its self._full_cfg) for the identical reason.
        self._full_cfg = cfg
        self.retriever = retriever
        self.mem = Memory()

    def run(self, query_text: str) -> Answer:
        state: dict[str, Any] = {"query": query_text, "obs": []}
        for _ in range(self.cfg["max_steps"]):
            hooks.run(hooks.ON_STEP, {"state": state})
            action = self.decide(state)  # IMPLEMENT (policy)
            if action["tool"] == "stop":
                break
            hooks.run(hooks.ON_TOOL_CALL, {"action": action})  # guardrails/injection/trace
            result = self.act(action)  # runs the tool via REGISTRY
            state["obs"].append(result)
            self.mem.add(result)
        hooks.run(hooks.BEFORE_ANSWER, {"state": state})  # grounding gate / PII redact
        ans = self.synthesize(state)  # IMPLEMENT (grounded answer)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})  # trace / metrics
        return ans

    def decide(self, state: dict) -> dict:
        """Evidence-gated re-search — the MANDATORY agentic behaviour (A3 gate, fail-closed).
        Read the last observation (top_score, k) and branch on the NUMBER, using retrieval.retriever:
          1. retrieve at k = cfg.retrieve.k, then RERANK it (see the 2026-08-23 note below)
          2. if is_weak(chunks, cfg):  k2 = next_k(k, cfg)
               - k2 is not None -> retrieve AGAIN at the wider k2 (widen the net), rerank, re-check
               - k2 is None (hit k_max) and still weak -> ABSTAIN ("insufficient evidence")
          3. else -> synthesize a grounded, cited answer
        Emit obs {"top_score": ..., "k": ...} on each step. A fixed retrieve->answer path is NOT agentic
        and caps the grade. Rule-based (baseline) or RL policy (Stage 7).

        2026-08-23 follow-up (plan_a3.md Step 10's own RESULT has the full record): before
        this, `rerank.rerank()` was never actually called anywhere in the live loop —
        `is_weak()` only ever saw raw dense-cosine scores, and `chunk.score` never became a
        cross-encoder score outside the standalone `Rerank` tool (which this rule-based
        `decide()` never dispatches). Reranking is now called here explicitly, at every
        retrieve (initial and each widen), so `is_weak()`/`top_score()` compare against the
        reranked score exactly as Step 22's own already-flagged open item asked for —
        `weak_threshold` was retuned in `config.yaml` accordingly (0.35 was tuned for cosine,
        not this scale) but that number is PROVISIONAL, from one real data point, not a
        calibrated distribution — real calibration is still Step 22's job over the full
        suite. Real, measured cost of this on CPU: ~187s to rerank 40 candidates for one
        query (bge-reranker-v2-m3 is a full cross-encoder, not a bi-encoder — O(n) full
        forward passes, not O(1) per candidate the way dense search is) — Step 22's own GPU
        note has been updated to reflect this is no longer a safe CPU-optional default the
        way it was before this change.
        """
        rcfg = self._full_cfg
        query = state["query"]

        def _emit(chunks: list, k: int) -> None:
            state["obs"].append({"top_score": retriever_mod.top_score(chunks), "k": k})

        k = rcfg["retrieve"]["k"]
        chunks = rerank.rerank(query, self.retriever.retrieve(query, k=k), rcfg)
        while retriever_mod.is_weak(chunks, rcfg):
            _emit(chunks, k)
            k2 = retriever_mod.next_k(k, rcfg)
            if k2 is None:
                # Hit k_max still weak -> ABSTAIN. state["chunks"] keeps the last (still
                # weak) attempt so synthesize() can show its work / cite why it abstained,
                # not because those chunks are meant to ground an answer.
                state["chunks"] = chunks
                state["abstain"] = True
                state["abstain_reason"] = "insufficient evidence"
                return {"tool": "stop", "args": {}}
            k = k2
            chunks = rerank.rerank(query, self.retriever.retrieve(query, k=k), rcfg)
        _emit(chunks, k)
        state["chunks"] = chunks
        state["abstain"] = False
        return {"tool": "stop", "args": {}}

    def act(self, action: dict) -> ToolResult:
        """Look `action["tool"]` up in `tools.REGISTRY` by name and call it with
        `action["args"]`. An unknown tool name returns `ToolResult(ok=False, ...)`, same
        "never raise mid-run" contract every tool in the registry already honours (Step 7) --
        a bad dispatch must not crash the loop any more than a bad tool call would."""
        name = action["tool"]
        for tool_cls in tools.REGISTRY:
            if tool_cls.name == name:
                # REGISTRY's element type widens to type[Tool] (the abstract base) once
                # mixed concrete subclasses join in a list literal -- every entry is
                # concrete in practice (test_tools.py's own test_registry_is_tool_subclasses
                # asserts issubclass), mypy just can't see that through the join.
                return tool_cls()(**action.get("args", {}))  # type: ignore[abstract]
        return ToolResult(ok=False, payload={"reason": f"unknown tool: {name!r}"})

    def synthesize(self, state: dict) -> Answer:
        """Grounded, cited answer; abstain if unsupported (no-hallucination).

        decide()'s own abstention (evidence never got strong enough, even at k_max) short-
        circuits here -- no LLM call for a question decide() already determined has no real
        evidence. Otherwise: one LLM synthesis attempt, run through the BEFORE_ANSWER
        grounding gate (postprocess._ground). D6 (verify-and-correct): if `_ground` downgrades
        a real, cited answer, retry exactly once with its specific complaint fed back; if
        still unsupported, abstain for real -- discard the retry's own ungrounded text rather
        than ship it with a warning label, since "abstain" means the claim never reaches a
        user. An answer format_answer() *itself* already abstained on (the model
        self-reporting no evidence, or zero citations surviving resolution) skips the retry
        entirely -- there is nothing to correct in an already-correct "I don't know".

        A1's HITL trigger (guardrails.ESCALATION_CONFIDENCE_THRESHOLD, τ=0.50): fires exactly
        here, on decide()'s own k_max abstention -- confidence is always 0.0 in this branch,
        always under τ, and `state["abstain"]` being set is precisely "we've re-searched to
        k_max". The OTHER abstain path below (a retry that's still ungrounded) is a different
        failure mode -- the evidence was strong enough for decide(), the LLM's answer just
        wasn't grounded in it -- and isn't what A1's own trigger names, so it isn't escalated
        here."""
        if state.get("abstain"):
            hitl.escalate(
                "confidence below A1's tau=0.50 threshold after re-search to k_max",
                {"query": state.get("query"), "abstain_reason": state.get("abstain_reason")},
            )
            return Answer(
                text=postprocess.INSUFFICIENT_EVIDENCE, citations=[], grounded=False, confidence=0.0
            )

        llm = client.LLM(self._full_cfg)
        ans = self._synthesize_attempt(llm, state, feedback=None, retried=False)
        ctx = hooks.run(hooks.BEFORE_ANSWER, {"state": state, "answer": ans})
        ans = ctx.get("answer", ans)

        if not ans.grounded and ans.text != postprocess.INSUFFICIENT_EVIDENCE:
            complaint = ctx.get(
                "grounding_complaint", "the previous answer was not grounded in the cited evidence"
            )
            ans = self._synthesize_attempt(llm, state, feedback=complaint, retried=True)
            ctx2 = hooks.run(hooks.BEFORE_ANSWER, {"state": state, "answer": ans})
            ans = ctx2.get("answer", ans)
            if not ans.grounded and ans.text != postprocess.INSUFFICIENT_EVIDENCE:
                ans = Answer(
                    text=postprocess.INSUFFICIENT_EVIDENCE,
                    citations=[],
                    grounded=False,
                    confidence=0.0,
                )
        return ans

    def _synthesize_attempt(
        self, llm: Any, state: dict, feedback: str | None, retried: bool
    ) -> Answer:
        """One SYNTHESIZE call + format_answer(). `feedback` (the retry's grounding
        complaint) rides in the QUERY placeholder, not the EVIDENCE one -- the evidence block
        is untrusted corpus data the model must never treat as an instruction (prompts.py's
        own anti-injection contract), so a trusted, system-authored retry instruction belongs
        in the instruction channel, not mixed into the data channel."""
        chunks = state["chunks"]
        evidence = "\n".join(f"[{c.id}] (score={c.score:.3f}) {c.text}" for c in chunks)
        query = state["query"]
        if feedback:
            query = (
                f"{query}\n\n(This is a retry. Your previous answer was rejected: {feedback}. "
                f"Answer again using ONLY the evidence below, or say "
                f"{postprocess.INSUFFICIENT_EVIDENCE} if it truly does not support one.)"
            )
        raw = llm.complete(prompts.SYNTHESIZE.format(query=query, evidence=evidence))

        top_score = retriever_mod.top_score(chunks)
        score_gap = chunks[0].score - chunks[1].score if len(chunks) >= 2 else 0.0
        # Always False under the current wiring (decide() never routes through act(), so
        # state["obs"] never holds a calculator ToolResult) -- kept as a real, general check
        # rather than a hardcoded False so it activates automatically if a future step ever
        # adds tool-routed calls ahead of synthesize().
        calculator_verified = any(
            isinstance(o, ToolResult) and o.ok and "expr" in o.payload and "value" in o.payload
            for o in state["obs"]
        )
        return postprocess.format_answer(
            raw,
            chunks,
            top_score=top_score,
            score_gap=score_gap,
            calculator_verified=calculator_verified,
            retried=retried,
        )
