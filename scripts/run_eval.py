"""Run tasks.jsonl through the agent and score.

Step 17 (plan_a3.md). Loads `grading_kit/tasks.jsonl`, runs each task through
`pipeline.answer()`, scores it with `grading_kit/success_check.check` and `doc_agent.eval.
metrics`, and writes BOTH a machine-readable JSON and a readable markdown summary into
`reports/` -- the same shape A2's `reports/step29_test_eval.{md,json}` used, which is what
made those numbers auditable when the grounding gate came for them.

Results break down by **split** (train/val/test) and by **kind** (verifiable/judged/
abstention). Form sections 3 and 5 both need those breakdowns and neither is recoverable from
an aggregate afterwards, so they are computed on the way in, not derived later.

RESUMABLE. Every task's row is written to the JSON the moment it completes, so a killed
session loses one task, not the whole run. This is not hypothetical: `plan.md` 11.4 records
A2's Kaggle session dying ~4.4h into an OCR run, and `reports/a3_step14_probe_rerun.md`
records a probe of mine throwing away seven completed GPU measurements for exactly this
reason -- it wrote its output once, at the end. Use `--resume` to continue an interrupted run.

--- three implementation choices worth stating, because none is the obvious one ---

1. **The agentic branch is read from `traces/run.jsonl`, not inferred.** `logging_conf.
   register()` truncates that file at the start of each `Agent.run()` and appends one
   `TraceStep` per retrieval, so immediately after `pipeline.answer()` returns it holds
   exactly that task's trajectory. Reading it gives the real `k` sequence the agent walked --
   the same file the A3 agentic gate itself reads. Re-deriving the branch from the answer
   would be a guess about behaviour we can simply observe.

2. **recall@k re-runs retrieval rather than reaching into the agent.** `pipeline.answer()`
   returns an `Answer`; the retrieved chunks live on `decide()`'s internal `state`, and the
   only way to reach them from outside is a `BEFORE_ANSWER` hook -- which `pipeline.answer()`
   itself wipes, because it calls `wiring.register_all()` (and therefore `hooks.clear()`) on
   every call. Rather than monkeypatch a FIXED module, this re-runs `Retriever.retrieve()` +
   `rerank.rerank()` at the final `k` the trace reports. Dense search and the cross-encoder
   are both deterministic, so the chunks are the ones the agent actually saw, and the cost is
   one extra rerank per task (~3s on the T4 Step 22 runs on; measured 187s on CPU per
   `agent.py`'s own note, hence `--no-recall` for laptop smoke runs).

3. **Token spend is metered by registering `LLM` instances, not by wrapping `complete()`.**
   `Agent.synthesize()` constructs its own `client.LLM` per call and `eval.judge` constructs
   another, so there is no single instance to interrogate. `_LLMMeter` patches `LLM.__init__`
   to keep a reference to every instance built inside its `with` block, then sums the
   `call_count` / `total_tokens` counters those instances already maintain. Observation only
   -- it changes no behaviour and restores the original `__init__` on exit.

USD is reported as 0.00: DECISION D1 (plan_a3.md 5) put us on Groq's free tier, which makes
the locked `budget_usd: 0.05` a non-binding ceiling rather than a real spend. Tokens are the
number that actually constrains us (rate limits), so tokens are what this reports.

Usage:
    python scripts/run_eval.py                          # full suite
    python scripts/run_eval.py --limit 3                # smoke run
    python scripts/run_eval.py --tasks t01,t02,te13     # specific ids
    python scripts/run_eval.py --resume                 # continue an interrupted run
    python scripts/run_eval.py --no-recall              # skip the extra rerank pass
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doc_agent import config, pipeline  # noqa: E402
from doc_agent.contracts import Answer, Chunk, Query  # noqa: E402
from doc_agent.eval import calibration, interpret, judge, metrics  # noqa: E402
from doc_agent.llm import client  # noqa: E402
from doc_agent.logging_conf import TRACE_PATH, get_logger  # noqa: E402
from doc_agent.retrieval import rerank  # noqa: E402
from doc_agent.retrieval import retriever as retriever_mod  # noqa: E402
from grading_kit import success_check  # noqa: E402

logger = get_logger(__name__)

DEFAULT_TASKS = REPO_ROOT / "grading_kit" / "tasks.jsonl"
DEFAULT_JSON = REPO_ROOT / "reports" / "a3_eval_results.json"
DEFAULT_MD = REPO_ROOT / "reports" / "a3_eval_report.md"

SPLITS = ("train", "val", "test")
KINDS = ("verifiable", "judged", "abstention")


def _relpath(path: Path) -> str:
    """`path` relative to REPO_ROOT for display, falling back to the resolved absolute
    path if it lives outside the repo. `Path.relative_to` raises on a path that was never
    resolved (a bare `--out-json reports/x.json` from the CLI is relative to the caller's
    cwd, not to `path`) or that is genuinely outside REPO_ROOT -- both real inputs, not
    edge cases worth crashing the run over at the very last line, after every task already
    ran."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


# --------------------------------------------------------------------------- loading


def load_tasks(path: Path) -> list[dict]:
    """Read tasks.jsonl, skipping blank lines and `#` comments.

    The `#` handling is not defensive boilerplate: as of 2026-08-23 the committed file ends
    with a 21-line comment block after its last JSON row, documenting the `needs_research`
    convention. That is useful documentation and invalid JSONL at the same time -- a stock
    reader doing `json.loads` per line crashes on it. Tolerated here and warned about on
    every run, but it belongs in `grading_kit/README.md` -- a grader's own loader will
    not be this forgiving, and this script's tolerance is not evidence that theirs is.
    """
    rows: list[dict] = []
    n_comment = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            n_comment += 1
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: not valid JSON -- {exc}") from exc
    if n_comment:
        logger.warning(
            f"run_eval: {path.name} contains {n_comment} '#' comment lines. Valid as notes, "
            "NOT valid JSONL -- a grader's own loader may not skip them. Move them to "
            "grading_kit/README.md."
        )
    return rows


def select_tasks(rows: list[dict], limit: int | None, ids: str | None) -> list[dict]:
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        by_id = {r["id"]: r for r in rows}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"unknown task ids: {', '.join(missing)}")
        return [by_id[i] for i in wanted]
    return rows[:limit] if limit else rows


# --------------------------------------------------------------------------- metering


class _LLMMeter:
    """Sum `call_count`/`total_tokens` across every `LLM` built inside the block, and record
    each raw `complete()` response verbatim.

    Observation only -- wraps `__init__` (for instance registration, so per-instance
    call_count/total_tokens can be summed) AND `complete` (2026-08-26: added after Step 22's
    real Kaggle run showed 40/41 successfully-scored tasks came back with zero citations and
    the bare INSUFFICIENT_EVIDENCE text, despite retrieval finding strong evidence in most
    cases -- with no raw LLM text captured anywhere, there was no way to tell "the model
    genuinely said insufficient evidence" apart from "the model answered fine but its cited
    chunk ids didn't string-match the pool." `raw_outputs` closes that gap for future runs,
    at zero extra LLM cost -- it only ever reads text `complete()` already produced, never
    makes an additional call.
    """

    def __init__(self) -> None:
        self.instances: list[Any] = []
        self.raw_outputs: list[str] = []
        self._orig_init: Any = None
        self._orig_complete: Any = None

    def __enter__(self) -> _LLMMeter:
        self._orig_init = client.LLM.__init__
        self._orig_complete = client.LLM.complete
        meter = self

        def _init(inner_self: Any, cfg: dict) -> None:
            meter._orig_init(inner_self, cfg)
            meter.instances.append(inner_self)

        def _complete(inner_self: Any, prompt: str, **kw: Any) -> str:
            text = meter._orig_complete(inner_self, prompt, **kw)
            meter.raw_outputs.append(text)
            return text

        client.LLM.__init__ = _init  # type: ignore[method-assign]
        client.LLM.complete = _complete  # type: ignore[method-assign]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        client.LLM.__init__ = self._orig_init  # type: ignore[method-assign]
        client.LLM.complete = self._orig_complete  # type: ignore[method-assign]

    @property
    def calls(self) -> int:
        return sum(getattr(i, "call_count", 0) for i in self.instances)

    @property
    def tokens(self) -> int:
        return sum(getattr(i, "total_tokens", 0) for i in self.instances)


# --------------------------------------------------------------------------- trace


def read_trace() -> dict[str, Any]:
    """Parse the trace `pipeline.answer()` just wrote.

    `logging_conf.register()` truncates `traces/run.jsonl` per run and appends one
    `retrieve` TraceStep per retrieval attempt, so this reports the real evidence-gated
    re-search path -- the same file the A3 agentic gate reads.
    """
    steps: list[dict] = []
    if TRACE_PATH.exists():
        for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    retrieves = [s for s in steps if s.get("tool") == "retrieve"]
    ks = [s["obs"]["k"] for s in retrieves if "k" in s.get("obs", {})]
    scores = [s["obs"]["top_score"] for s in retrieves if "top_score" in s.get("obs", {})]
    answers = [s for s in steps if s.get("tool") == "answer"]
    return {
        "n_retrieve_steps": len(retrieves),
        "k_sequence": ks,
        "top_scores": [round(float(s), 4) for s in scores],
        "final_k": ks[-1] if ks else None,
        "re_searched": len(retrieves) > 1,
        "trace_abstained": bool(answers[-1]["obs"].get("abstained")) if answers else None,
        "n_trace_steps": len(steps),
    }


# --------------------------------------------------------------------------- scoring


def measure_recall(
    query: str, gold_pages: list[str], k: int, cfg: dict, retriever: Any
) -> float | None:
    """recall@k over the same chunks the agent saw, by re-running deterministic retrieval."""
    if not gold_pages or not k:
        return None
    chunks: list[Chunk] = rerank.rerank(query, retriever.retrieve(query, k=k), cfg)
    return round(metrics.recall_at_k(chunks, gold_pages, k), 4)


def run_one(task: dict, cfg: dict, retriever: Any, want_recall: bool) -> dict:
    """One task, end to end. Never raises -- a crashed task is recorded and the run goes on."""
    started = time.time()
    row: dict[str, Any] = {
        "id": task["id"],
        "question": task["question"],
        "split": task.get("split"),
        "kind": task.get("kind"),
        "gold": task.get("gold"),
        "gold_pages": task.get("gold_pages", []),
        "declared_needs_research": task.get("needs_research"),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    # judge.judge() (below) makes its own real LLM call for judged-kind tasks -- kept INSIDE
    # this meter's scope, not after it exits, so llm_calls/llm_tokens report the true total
    # cost of scoring this task, not just pipeline.answer()'s own share of it.
    with _LLMMeter() as meter:
        try:
            answer: Answer = pipeline.answer(task["question"], cfg)
        except Exception as exc:  # noqa: BLE001 -- one bad task must not end the run
            row.update(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "success": False,
                    "elapsed_s": round(time.time() - started, 2),
                    "llm_calls": meter.calls,
                    "llm_tokens": meter.tokens,
                }
            )
            logger.error(f"run_eval: {task['id']} raised {type(exc).__name__}: {exc}")
            return row

        trace = read_trace()
        # interpret.explain() MUST run before the next task's pipeline.answer() call
        # truncates traces/run.jsonl (its own docstring) -- right here, immediately after
        # read_trace() read the same still-fresh trace, is exactly that window.
        explanation = interpret.explain(answer, cfg)

        # judged-kind tasks: call judge.judge() directly, ONCE, rather than through
        # success_check.check() -- that would call judge() a SECOND time internally
        # (grading_kit/success_check.py::_check_judged) and discard the raw score, doubling
        # the LLM-judge cost for every judged task and leaving no number to report as
        # "judged quality" beyond a bare pass/fail. success is derived from the SAME score
        # using success_check's own threshold constant, not a second independent judge call.
        judge_score: float | None = None
        if task.get("kind") == "judged":
            query = Query(text=task["question"], verifiable=False, judged=True)
            judge_score = judge.judge(query, answer)
            success = judge_score >= success_check.JUDGE_PASS_THRESHOLD
        else:
            success = success_check.check(task, answer.model_dump())

        llm_calls, llm_tokens = meter.calls, meter.tokens
        raw_llm_outputs = list(meter.raw_outputs)

    row.update(
        {
            "answer_text": answer.text,
            "grounded": answer.grounded,
            "confidence": round(answer.confidence, 4),
            "n_citations": len(answer.citations),
            "success": success,
            "judge_score": round(judge_score, 4) if judge_score is not None else None,
            "groundedness": round(metrics.groundedness(answer), 4),
            "citation_accuracy": round(metrics.citation_accuracy(answer), 4),
            "llm_calls": llm_calls,
            "llm_tokens": llm_tokens,
            "raw_llm_outputs": raw_llm_outputs,
            "interpret": explanation,
            **trace,
        }
    )
    # The gate's own wording is that re-search must fire ONLY when needed. A declared
    # control that widens is a failed control, not a cosmetic mislabel -- so the comparison
    # is recorded per task rather than left for someone to notice later.
    declared = row["declared_needs_research"]
    row["measured_needs_research"] = trace["re_searched"]
    row["needs_research_label_ok"] = (
        None if declared is None else bool(declared) == bool(trace["re_searched"])
    )
    if want_recall and trace["final_k"]:
        try:
            row["recall_at_k"] = measure_recall(
                task["question"], row["gold_pages"], int(trace["final_k"]), cfg, retriever
            )
            row["recall_k"] = trace["final_k"]
        except Exception as exc:  # noqa: BLE001
            row["recall_at_k"] = None
            row["recall_error"] = f"{type(exc).__name__}: {exc}"
    row["elapsed_s"] = round(time.time() - started, 2)
    return row


# --------------------------------------------------------------------------- aggregation


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarise(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]

    def block(subset: list[dict]) -> dict[str, Any]:
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "success_rate": round(sum(1 for r in subset if r.get("success")) / len(subset), 4),
            "groundedness": _mean([r.get("groundedness") for r in subset]),
            "citation_accuracy": _mean([r.get("citation_accuracy") for r in subset]),
            "recall_at_k": _mean([r.get("recall_at_k") for r in subset]),
            "mean_confidence": _mean([r.get("confidence") for r in subset]),
            "mean_judge_score": _mean([r.get("judge_score") for r in subset]),
            "abstain_rate": round(
                sum(1 for r in subset if r.get("grounded") is False) / len(subset), 4
            ),
            "re_search_rate": round(
                sum(1 for r in subset if r.get("re_searched")) / len(subset), 4
            ),
            "llm_calls": sum(r.get("llm_calls", 0) for r in subset),
            "llm_tokens": sum(r.get("llm_tokens", 0) for r in subset),
            "wall_clock_s": round(sum(r.get("elapsed_s", 0.0) for r in subset), 1),
        }

    label_checked = [r for r in ok if r.get("needs_research_label_ok") is not None]
    mislabelled = [r for r in label_checked if r["needs_research_label_ok"] is False]
    explanations = [r["interpret"] for r in ok if "interpret" in r]
    # ECE (Step 21, second NFR): "correct" = success_check's own pass/fail per task -- the
    # same notion of "was this answer actually right" confidence is supposed to predict.
    # `n=0` reports ece=None rather than a vacuous 0.0, same honest-reporting rule
    # `interpret.py`'s coverage/faithfulness already follow.
    ece_value = (
        calibration.ece([r["confidence"] for r in ok], [bool(r.get("success")) for r in ok])
        if ok
        else None
    )
    return {
        "overall": block(ok),
        "explainability": {
            "rationale_coverage": interpret.rationale_coverage(explanations),
            "rationale_faithfulness": interpret.rationale_faithfulness(explanations),
        },
        "calibration": {"ece": ece_value, "target": calibration.ECE_TARGET, "n": len(ok)},
        "by_split": {s: block([r for r in ok if r.get("split") == s]) for s in SPLITS},
        "by_kind": {k: block([r for r in ok if r.get("kind") == k]) for k in KINDS},
        "by_split_kind": {
            f"{s}/{k}": block([r for r in ok if r.get("split") == s and r.get("kind") == k])
            for s in SPLITS
            for k in KINDS
        },
        "errors": [{"id": r["id"], "error": r["error"]} for r in rows if "error" in r],
        "agentic_gate": {
            "n_labels_checked": len(label_checked),
            "n_mislabelled": len(mislabelled),
            "mislabelled_ids": [r["id"] for r in mislabelled],
            "n_declared_triggers": sum(1 for r in ok if r.get("declared_needs_research") is True),
            "n_measured_triggers": sum(1 for r in ok if r.get("measured_needs_research")),
            "branches": dict(
                Counter(
                    (
                        "abstain"
                        if r.get("grounded") is False and r.get("re_searched")
                        else "re-searched" if r.get("re_searched") else "single_pass"
                    )
                    for r in ok
                )
            ),
        },
    }


# --------------------------------------------------------------------------- reporting


def _row(cells: list[Any]) -> str:
    out = []
    for c in cells:
        out.append("—" if c is None else (f"{c:.4f}" if isinstance(c, float) else str(c)))
    return "| " + " | ".join(out) + " |"


def _metric_table(title: str, blocks: dict[str, dict]) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append(
        "| group | n | success | groundedness | citation acc | recall@k | mean conf | "
        "judge /6 | abstain | re-search | LLM calls | tokens | wall-clock |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, b in blocks.items():
        if not b.get("n"):
            lines.append(
                _row([name, 0, None, None, None, None, None, None, None, None, 0, 0, None])
            )
            continue
        lines.append(
            _row(
                [
                    name,
                    b["n"],
                    b["success_rate"],
                    b["groundedness"],
                    b["citation_accuracy"],
                    b["recall_at_k"],
                    b["mean_confidence"],
                    b["mean_judge_score"],
                    b["abstain_rate"],
                    b["re_search_rate"],
                    b["llm_calls"],
                    b["llm_tokens"],
                    f"{b['wall_clock_s']}s",
                ]
            )
        )
    lines.append("")
    return lines


def write_markdown(rows: list[dict], summary: dict, cfg: dict, meta: dict, path: Path) -> None:
    ov = summary["overall"]
    gate = summary["agentic_gate"]
    expl = summary["explainability"]
    cal = summary["calibration"]
    cov, faith = expl["rationale_coverage"], expl["rationale_faithfulness"]
    by_kind = summary["by_kind"]
    lines: list[str] = [
        "# A3 eval run — `scripts/run_eval.py`",
        "",
        f"**Generated:** {meta['generated_at']} · **Tasks:** {meta['n_selected']} of "
        f"{meta['n_available']} · **Wall-clock:** {meta['wall_clock_s']}s",
        f"**Config:** `{cfg['agent']['model']}` · retrieve k={cfg['retrieve']['k']} "
        f"step={cfg['retrieve']['k_step']} max={cfg['retrieve']['k_max']} "
        f"weak_threshold={cfg['retrieve']['weak_threshold']} "
        f"rerank={cfg['retrieve']['rerank']}",
        f"**Raw results:** `{meta['json_path']}` (one row per task, written as each completed)",
        "",
        "> Every number here is regenerated by `bash scripts/run.sh`, or by "
        "`python scripts/run_eval.py` alone if the index already exists. Nothing in this file "
        "is hand-entered.",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "|---|---|",
        _row(["tasks scored", ov.get("n", 0)]),
        _row(["success rate", ov.get("success_rate")]),
        _row(["groundedness (headline, target ≥ 0.90)", ov.get("groundedness")]),
        _row(["citation accuracy", ov.get("citation_accuracy")]),
        _row(["recall@k", ov.get("recall_at_k")]),
        _row(["mean confidence", ov.get("mean_confidence")]),
        _row(["abstention rate", ov.get("abstain_rate")]),
        _row(["LLM calls", ov.get("llm_calls", 0)]),
        _row(["LLM tokens", ov.get("llm_tokens", 0)]),
        _row(["USD spend", "0.00 (Groq free tier, DECISION D1)"]),
        _row(
            [
                "verifiable accuracy (by_kind/verifiable)",
                by_kind.get("verifiable", {}).get("success_rate"),
            ]
        ),
        _row(
            [
                "judged quality, mean judge score /6 (by_kind/judged)",
                by_kind.get("judged", {}).get("mean_judge_score"),
            ]
        ),
        _row(
            [
                "judged pass rate, ≥4/6 (by_kind/judged)",
                by_kind.get("judged", {}).get("success_rate"),
            ]
        ),
        _row(
            [
                "abstention correctness (by_kind/abstention)",
                by_kind.get("abstention", {}).get("success_rate"),
            ]
        ),
        "",
        "## Calibrated — SECOND NFR (eval/calibration.py, Step 21)",
        "",
        f"ECE, binned over all {cal['n']} scored tasks' (confidence, success) pairs. Confidence is "
        "Step 11's own hand-built formula (retrieval strength, score gap, calculator "
        "verification, retry penalty), not a classifier softmax -- see `eval/calibration.py`'s "
        "own docstring.",
        "",
        "| metric | target | measured | n |",
        "|---|---|---|---|",
        _row(["ECE", f"≤ {cal['target']}", cal["ece"], cal["n"]]),
        "",
        "## Explainable — PRIMARY NFR (eval/interpret.py, Step 20)",
        "",
        "Retrieval-level attribution, not pixel Grad-CAM — see `eval/interpret.py`'s own "
        "docstring for why. Faithfulness is checked mechanically against `traces/run.jsonl`'s "
        "real primary/runner-up score breakdown, not judged by an LLM reading the prose.",
        "",
        "| metric | target | measured | n |",
        "|---|---|---|---|",
        _row(
            [
                "rationale coverage",
                interpret.RATIONALE_COVERAGE_TARGET,
                cov["rate"],
                f"{cov['n_covered']}/{cov['n']}" if cov["n"] else "0/0 (no answered tasks)",
            ]
        ),
        _row(
            [
                "rationale faithfulness",
                f"≥ {interpret.RATIONALE_FAITHFULNESS_TARGET}",
                faith["rate"],
                f"{faith['n_faithful']}/{faith['n']}" if faith["n"] else "0/0 (no checkable tasks)",
            ]
        ),
        "",
    ]
    lines += _metric_table("By split (form §3)", summary["by_split"])
    lines += _metric_table("By kind (form §5)", summary["by_kind"])
    lines += _metric_table("By split × kind", summary["by_split_kind"])

    lines += [
        "## Agentic gate — evidence-gated re-search",
        "",
        "Read from `traces/run.jsonl`, the same file the A3 gate reads. The gate requires that "
        "re-search fires **only when needed**: a trigger widens `k` and then recovers or "
        "abstains at `k_max`; a control single-passes.",
        "",
        "| | |",
        "|---|---|",
        _row(["branches observed", ", ".join(f"{k}={v}" for k, v in gate["branches"].items())]),
        _row(["tasks declared `needs_research: true`", gate["n_declared_triggers"]]),
        _row(["tasks that actually re-searched", gate["n_measured_triggers"]]),
        _row(["labels checked", gate["n_labels_checked"]]),
        _row(["**labels contradicted by measurement**", gate["n_mislabelled"]]),
        "",
    ]
    if gate["n_mislabelled"]:
        lines += [
            f"> ⚠️ **{gate['n_mislabelled']} task(s) carry a `needs_research` label the agent's "
            "own behaviour contradicts.** A declared control that widens is a *failed control* "
            "under the gate's own wording, not a cosmetic mislabel. Ids: "
            + ", ".join(f"`{i}`" for i in gate["mislabelled_ids"]),
            "",
        ]

    worst = sorted(
        (r for r in rows if "error" not in r and not r.get("success")),
        key=lambda r: (r.get("groundedness") or 0.0),
    )[:5]
    if worst:
        lines += ["## Worst failures", ""]
        for r in worst:
            lines += [
                f"**`{r['id']}`** ({r.get('split')}/{r.get('kind')}) — grounded="
                f"{r.get('grounded')}, groundedness={r.get('groundedness')}, "
                f"conf={r.get('confidence')}, k={r.get('k_sequence')}",
                "",
                f"> Q: {r['question']}",
                "",
                f"> gold: `{r.get('gold')}`",
                "",
                f"> got: {(r.get('answer_text') or '')[:400]}",
                "",
            ]

    if summary["errors"]:
        lines += ["## Tasks that raised", ""]
        for e in summary["errors"]:
            lines.append(f"- `{e['id']}` — {e['error']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    ap.add_argument("--tasks", type=str, default=None, help="comma-separated task ids")
    ap.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--resume", action="store_true", help="skip ids already in --out-json")
    ap.add_argument("--no-recall", action="store_true", help="skip the extra rerank pass")
    args = ap.parse_args()

    cfg = config.load()
    all_tasks = load_tasks(args.tasks_file)
    selected = select_tasks(all_tasks, args.limit, args.tasks)

    rows: list[dict] = []
    done: set[str] = set()
    if args.resume and args.out_json.exists():
        # Two shapes can be on disk: the bare list written after each task (an interrupted
        # run), or the full {meta, summary, results} payload written at the end (a completed
        # run being extended). Both must resume, or --resume works only on crashes and not
        # on "run 3 more tasks than last time", which is the same flag people will reach for.
        loaded = json.loads(args.out_json.read_text(encoding="utf-8"))
        rows = loaded["results"] if isinstance(loaded, dict) else loaded
        done = {r["id"] for r in rows}
        logger.info(f"run_eval: resuming, {len(done)} task(s) already recorded")

    todo = [t for t in selected if t["id"] not in done]
    logger.info(f"run_eval: {len(todo)} task(s) to run ({len(selected)} selected)")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    retriever = retriever_mod.Retriever(cfg) if not args.no_recall else None

    started = time.time()
    for i, task in enumerate(todo, 1):
        logger.info(f"run_eval: [{i}/{len(todo)}] {task['id']}")
        rows.append(run_one(task, cfg, retriever, want_recall=not args.no_recall))
        # Written every iteration, not at the end. See this module's docstring.
        args.out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    wall = round(time.time() - started, 1)

    summary = summarise(rows)
    payload = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_available": len(all_tasks),
            "n_selected": len(selected),
            "n_run_this_session": len(todo),
            "wall_clock_s": wall,
            "json_path": _relpath(args.out_json),
            "recall_measured": not args.no_recall,
            "config": {
                "model": cfg["agent"]["model"],
                "retrieve": cfg["retrieve"],
                "seed": cfg.get("seed"),
            },
        },
        "summary": summary,
        "results": rows,
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(rows, summary, cfg, payload["meta"], args.out_md)

    ov = summary["overall"]
    logger.info(
        f"run_eval: done in {wall}s -- n={ov.get('n', 0)} "
        f"success={ov.get('success_rate')} groundedness={ov.get('groundedness')} "
        f"tokens={ov.get('llm_tokens', 0)}"
    )
    logger.info(f"run_eval: wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
