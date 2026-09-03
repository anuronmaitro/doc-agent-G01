"""Step 22 main eval run -- generates KAGGLE/step22_main_eval/kaggle_step22_main_eval.ipynb.

Runs the REAL, full 49-task suite end to end (scripts/run_eval.py, no --limit) against the
REAL, already-published Step 4 index (mounted, not rebuilt), with a REAL Groq LLM key read
from Kaggle Secrets. GPU (T4): decide() reranks on every retrieve, including every widen --
Step 10's own measured cost is 186.7s/query on CPU, and Step 22's own GPU note (plan_a3.md)
flips the earlier "CPU is tolerable" verdict once that became true.

WHY THIS NEEDS AN OVERLAY, NOT JUST `git clone`:

Steps 14/17/18/19/20/21's work (the corrected `tasks.jsonl`, `run_eval.py`'s interpret/judge/
ECE wiring, `agent.py`'s calibrated HITL check, `calibration.py`/`interpret.py` themselves,
the `groundedness()` fix, the trace score-breakdown enrichment, `configs/config.yaml`'s
`weak_threshold: 0.50`/`calibration.temperature`) is all still local, uncommitted work on
this machine -- the team's standing rule this whole session is that the AI never commits.
A plain `git clone` would run against STALE code: the old 15-mislabelled task suite, `ece()`
raising `NotImplementedError`, an unwired `interpret.py`, the pre-fix `groundedness()`, and
`weak_threshold: 0.35`. This notebook clones `main` for the real corpus/index-building
history, then overlays the files below with their exact current local content -- the same
pattern Step 18's notebook already established, extended to the larger file set this step
actually needs.

Regenerate with `python scripts/build_kaggle_notebook_step22_main_eval.py`, push with
`kaggle kernels push -p KAGGLE/step22_main_eval/`.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

OWNER = "eliasmainur"
KERNEL_SLUG = "mathscholar-step22-main-eval"
REPO_URL = "https://github.com/anuronmaitro/doc-agent-1.git"
INDEX_DATASET = "himadribiswas0904/mathscholar-index-mirror"

# Everything actually reachable from scripts/run_eval.py's own import graph (directly, or
# via pipeline.py's top-level `from .index import chunk, embed, store`) that has real,
# uncommitted local changes this session. Files with no local changes (agent/guardrails.py,
# retrieval/*, llm/postprocess.py, etc.) are already correct on `main` -- not overlaid.
OVERLAY_FILES = {
    "grading_kit/tasks.jsonl": REPO_ROOT / "grading_kit" / "tasks.jsonl",
    "scripts/run_eval.py": REPO_ROOT / "scripts" / "run_eval.py",
    "src/doc_agent/agent/agent.py": REPO_ROOT / "src" / "doc_agent" / "agent" / "agent.py",
    "src/doc_agent/eval/calibration.py": REPO_ROOT
    / "src"
    / "doc_agent"
    / "eval"
    / "calibration.py",
    "src/doc_agent/eval/interpret.py": REPO_ROOT / "src" / "doc_agent" / "eval" / "interpret.py",
    "src/doc_agent/eval/metrics.py": REPO_ROOT / "src" / "doc_agent" / "eval" / "metrics.py",
    "src/doc_agent/index/chunk.py": REPO_ROOT / "src" / "doc_agent" / "index" / "chunk.py",
    "src/doc_agent/logging_conf.py": REPO_ROOT / "src" / "doc_agent" / "logging_conf.py",
    "configs/config.yaml": REPO_ROOT / "configs" / "config.yaml",
}

# The needs_research pair for the agentic-gate verification (Do item 5) -- t04 is one of the
# 15 tasks Step 14 measured as a genuine trigger against the CURRENT (reranked) decide() path;
# t01 is a genuine, never-flipped control. Picked from real measurement, not guessed.
GATE_TRIGGER_ID = "t04"
GATE_CONTROL_ID = "t01"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


GATE_VERIFICATION_SCRIPT = f"""
log_progress("cell 7: gate verification starting (zero extra LLM calls) ...")
try:
    import json
    from pathlib import Path

    # v8's real run hit Groq's free-tier TPD cap (199,116/200,000 tokens) from the 49-task
    # batch ALONE -- two extra live pipeline.answer() calls here for {GATE_TRIGGER_ID!r}/
    # {GATE_CONTROL_ID!r} tipped it over and failed. Fixed by NOT calling the LLM again: t04
    # and t01 are both already inside the 49-task suite cell 6 just ran, and run_eval.py's
    # own read_trace() already read traces/run.jsonl for EACH task individually, right after
    # that task's own pipeline.answer() call, before the next task's call overwrote it (see
    # run_eval.py's own docstring on this) -- so their real, trace-derived k_sequence/
    # top_scores/re_searched are already sitting in reports/a3_eval_results.json. Pulling
    # them from there is reading the SAME real trace data, at zero extra token cost, not a
    # weaker substitute for reading it live.
    gate_log_path = Path("/kaggle/working/gate_verification_log.txt")

    def glog(msg):
        with open(gate_log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\\n")
        print(msg)

    results = json.loads(Path("reports/a3_eval_results.json").read_text(encoding="utf-8"))
    rows_by_id = {{r["id"]: r for r in results["results"]}}

    GATE_TASKS = {{"trigger": {GATE_TRIGGER_ID!r}, "control": {GATE_CONTROL_ID!r}}}
    glog("=== Do item 5: agentic-gate verification, read from each task's own real trace "
         "(captured during the main run, before the next task's trace overwrote it) ===\\n")
    for role, tid in GATE_TASKS.items():
        row = rows_by_id.get(tid)
        if row is None:
            glog(f"--- {{role}} ({{tid}}): NOT FOUND in reports/a3_eval_results.json ---\\n")
            continue
        glog(f"--- {{role}} ({{tid}}), declared needs_research={{row.get('declared_needs_research')}} ---")
        glog(f"Q: {{row['question']}}")
        glog(f"  k sequence      : {{row.get('k_sequence')}}")
        glog(f"  top_score seq   : {{row.get('top_scores')}}")
        glog(f"  re-searched     : {{row.get('re_searched')}}")
        glog(f"  answer.grounded : {{row.get('grounded')}}")

        re_searched = bool(row.get("re_searched"))
        matches_role = (role == "trigger" and re_searched) or (role == "control" and not re_searched)
        verdict = "CONFIRMED" if matches_role else "MISMATCH -- see note above the gate cell"
        glog(f"  VERDICT: {{verdict}}\\n")
    log_progress("cell 7: gate verification done")
except Exception:
    log_progress("cell 7 FAILED:\\n" + traceback.format_exc())
    raise
"""

WEAK_THRESHOLD_ANALYSIS_SCRIPT = """
log_progress("cell 8: weak_threshold analysis starting ...")
try:
    import json
    import statistics
    from pathlib import Path

    results = json.loads(Path("reports/a3_eval_results.json").read_text(encoding="utf-8"))
    rows = results["results"]
    current_weak_threshold = results["meta"]["config"]["retrieve"]["weak_threshold"]

    all_top_scores = []
    for r in rows:
        all_top_scores.extend(r.get("top_scores", []))

    all_top_scores.sort()
    n = len(all_top_scores)
    print(f"=== Do item 6: real cross-encoder score distribution, {n} retrieve attempts "
          f"across {len(rows)} tasks ===\\n")
    if all_top_scores:
        print(f"  min    : {all_top_scores[0]:.4f}")
        print(f"  median : {statistics.median(all_top_scores):.4f}")
        print(f"  mean   : {statistics.mean(all_top_scores):.4f}")
        print(f"  max    : {all_top_scores[-1]:.4f}")
        print(f"  stdev  : {statistics.pstdev(all_top_scores):.4f}" if n > 1 else "")
        for thr in (0.35, 0.50):
            below = sum(1 for s in all_top_scores if s < thr)
            print(f"  scores < {thr}: {below}/{n} ({below/n:.1%})"
                  + ("  <-- CURRENT weak_threshold" if thr == current_weak_threshold else ""))
        print()
        print("  full sorted distribution:")
        print(" ", [round(s, 3) for s in all_top_scores])
    else:
        print("  no retrieve steps recorded -- nothing to analyse")

    print(f"\\n  re_search_rate (from summary): {results['summary']['overall'].get('re_search_rate')}")
    print(f"  n_declared_triggers: {results['summary']['agentic_gate']['n_declared_triggers']}")
    print(f"  n_measured_triggers: {results['summary']['agentic_gate']['n_measured_triggers']}")
    print(f"  n_mislabelled: {results['summary']['agentic_gate']['n_mislabelled']}")

    Path("/kaggle/working/weak_threshold_analysis.json").write_text(
        json.dumps({"n": n, "sorted_top_scores": all_top_scores,
                    "current_weak_threshold": current_weak_threshold}, indent=2),
        encoding="utf-8",
    )
    log_progress("cell 8: weak_threshold analysis done")
except Exception:
    log_progress("cell 8 FAILED:\\n" + traceback.format_exc())
    raise
"""


def build_cells() -> list[dict]:
    cells: list[dict] = []

    cells.append(md("""# MathScholar (team 1) — Step 22: the main Kaggle eval run

Run by: Elias Mainur (S3, 2105058), at the user's request, 2026-08-26.

The full 49-task suite (`scripts/run_eval.py`), no `--limit`, against the real published
Step 4 index (mounted, not rebuilt), with a real Groq key read from Kaggle Secrets — never
pasted into this notebook (the repo is public). Produces `reports/a3_eval_results.json` +
`reports/a3_eval_report.md`, verifies the `needs_research` agentic gate directly from
`traces/run.jsonl`, and settles whether `weak_threshold` still makes sense against the real
cross-encoder score distribution.

## Why this notebook overlays nine files instead of just cloning `main`

Steps 14/17–21's work is still local, uncommitted (this team's standing rule: the AI never
commits). A plain clone would run stale code — the old mislabelled task suite, `ece()`
raising `NotImplementedError`, an unwired `interpret.py`, the pre-fix `groundedness()`,
`weak_threshold: 0.35`. This notebook clones `main` for the real corpus/index history, then
overlays those nine files with their exact current local content, embedded below.

**Do not edit this notebook on kaggle.com** — regenerate from
`scripts/build_kaggle_notebook_step22_main_eval.py`.
"""))

    cells.append(code('REPO_URL = "' + REPO_URL + '"\n' 'BRANCH = "main"\n'))

    cells.append(md("""## 0. Progress log

`kaggle kernels output` only pulls files actually written to disk -- if a cell dies with no
prior file write, there is nothing to diagnose from outside the (JS-only) run log after the
fact. Every subsequent risky cell appends one flushed line to `/kaggle/working/progress.log`
BEFORE and AFTER its real work, and writes any exception's full traceback there too before
re-raising -- so a failure is diagnosable from the downloaded output alone, not just the
kernel's live console."""))

    cells.append(code("""import traceback
from pathlib import Path

PROGRESS_LOG = Path("/kaggle/working/progress.log")


def log_progress(msg: str) -> None:
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\\n")
    print(msg)


log_progress("=== progress log started ===")
"""))

    cells.append(md("""## 1. Clone `main`"""))

    cells.append(code("""import os
import subprocess

log_progress("cell 1: cloning main ...")
if not os.path.exists("/kaggle/working/repo"):
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, "/kaggle/working/repo"],
        check=True,
    )
%cd /kaggle/working/repo
!git log --oneline -3
log_progress("cell 1: clone done")
"""))

    overlay_lines = [
        "import os\n",
        "\n",
        'log_progress("cell 2: overlaying files ...")\n',
        "\n",
        "OVERLAYS = {\n",
    ]
    for rel_path, local_path in OVERLAY_FILES.items():
        content = local_path.read_text(encoding="utf-8")
        overlay_lines.append(f"    {rel_path!r}: {content!r},\n")
    overlay_lines.append("}\n\n")
    overlay_lines.append(
        "try:\n"
        "    for rel_path, content in OVERLAYS.items():\n"
        '        p = os.path.join("/kaggle/working/repo", rel_path)\n'
        "        os.makedirs(os.path.dirname(p), exist_ok=True)\n"
        '        with open(p, "w", encoding="utf-8") as f:\n'
        "            f.write(content)\n"
        '        log_progress(f"overlaid {rel_path}  ({len(content)} chars)")\n'
        '    log_progress("cell 2: overlay done")\n'
        "except Exception:\n"
        '    log_progress("cell 2 FAILED:\\n" + traceback.format_exc())\n'
        "    raise\n"
    )

    cells.append(md("""## 2. Overlay the nine uncommitted local files

Exact content of each file as it stands on the author's machine at generation time — see
this notebook's own top cell for why."""))
    cells.append(code("".join(overlay_lines)))

    cells.append(md("""## 3. LLM API key — Kaggle Secrets, never in this notebook

🔴 Requires a Kaggle Secret named exactly `LLM_API_KEY` (Add-ons → Secrets → attach to this
notebook) holding a real Groq key. Written to `.env` at runtime — `.env` is gitignored and
never committed, and this notebook never prints the key value itself."""))

    cells.append(code("""log_progress("cell 3: reading LLM_API_KEY from Kaggle Secrets ...")
try:
    from kaggle_secrets import UserSecretsClient

    llm_api_key = UserSecretsClient().get_secret("LLM_API_KEY")
    with open(".env", "w", encoding="utf-8") as f:
        f.write(f"LLM_API_KEY={llm_api_key}\\n")
    log_progress(f"cell 3: .env written -- key length {len(llm_api_key)} chars (value not printed)")
except Exception:
    log_progress("cell 3 FAILED:\\n" + traceback.format_exc())
    raise
"""))

    cells.append(md("""## 4. Dependencies

Not `requirements.lock` — the repo's own `faiss-cpu`/`numpy<2` pins fight Kaggle's
pre-installed numpy 2.x stack (Steps 4/14's already-documented failure). Same proven fix:
let pip choose current `faiss-cpu`/`sentence-transformers`/`pydantic`/`groq`/
`pydantic-settings`, rely on Kaggle's own `torch`/`numpy`/`scipy`/`transformers`.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set before `import torch` — the exact
remedy Step 18's v1 OOM error message named for allocator fragmentation across many small
alloc/free cycles (one query embed + one rerank pass per task, ~150+ times over this run).
Must be set before CUDA initialises.

`opencv-python-headless` added (v5) — `run_eval.py`'s own top-level `from doc_agent import
config, pipeline` pulls in `pipeline.py`'s full import graph, including `vision/layout.py`
and `ingest/preprocess.py`, both of which `import cv2` at module level even though this run
never calls layout detection or preprocessing (Step 18's notebook avoided this by importing
`ablation`/`Retriever` directly, never `pipeline` — `run_eval.py` cannot avoid it, since it
genuinely needs `pipeline.answer()`)."""))

    cells.append(code("""log_progress("cell 4: installing dependencies ...")
try:
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "faiss-cpu", "sentence-transformers", "pydantic", "groq", "pydantic-settings",
         "opencv-python-headless"],
        check=True,
    )
    import torch  # noqa: E402 -- must import AFTER the pip install above AND the env var above

    log_progress("cell 4: dependencies installed")
    log_progress(f"cell 4: CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_progress(f"cell 4: GPU: {torch.cuda.get_device_name(0)}")
except Exception:
    log_progress("cell 4 FAILED:\\n" + traceback.format_exc())
    raise
"""))

    cells.append(
        md(
            """## 5. Mount the real Step 4 index — not rebuilt

Source: `"""
            + INDEX_DATASET
            + """` (public). Recursive search rather than an assumed mount depth.

Unlike Step 18's ablation notebook, `image_embed_cache.npz` **is** copied here — this run's
whole point is the real, production-accurate headline numbers, and DECISION D2's visual
fallback is a real part of that production path (`configs/config.yaml`'s
`retrieve.visual_model`). The OOM Step 18 hit came from four ablation arms plus a full index
rebuild sharing one session; this run is one baseline pass over 49 tasks, closer in shape to
Step 18's own baseline arm alone (677s, no issue) — mitigated further by the allocator fix
above."""
        )
    )

    cells.append(code("""log_progress("cell 5: mounting index dataset ...")
try:
    import shutil
    import zipfile
    from pathlib import Path

    input_listing = list(Path("/kaggle/input").rglob("*"))
    log_progress(f"cell 5: /kaggle/input has {len(input_listing)} entries: "
                 f"{[str(p) for p in input_listing[:20]]}")

    faiss_files = sorted(Path("/kaggle/input").rglob("faiss.index"))
    if not faiss_files:
        zips = sorted(Path("/kaggle/input").rglob("*.zip"))
        assert zips, "no faiss.index and no .zip found anywhere under /kaggle/input/"
        extract_dir = Path("/kaggle/working/index_extracted")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(extract_dir)
        faiss_files = sorted(extract_dir.rglob("faiss.index"))
    assert faiss_files, "still no faiss.index found after checking for a zip"
    mounted_index_dir = faiss_files[0].parent
    log_progress(f"cell 5: mounted index dir: {mounted_index_dir}")

    target_dir = Path("data/index")
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("faiss.index", "chunks.jsonl", "index_meta.json", "embed_cache.npz",
                 "image_embed_cache.npz"):
        src = mounted_index_dir / name
        if src.exists():
            shutil.copy(src, target_dir / name)
            log_progress(f"cell 5: copied {name}  ({src.stat().st_size:,} bytes)")
        else:
            log_progress(f"cell 5: not present in mount: {name} (ok if this index predates it)")
    log_progress("cell 5: mount done")
except Exception:
    log_progress("cell 5 FAILED:\\n" + traceback.format_exc())
    raise
"""))

    cells.append(md("""## 6. Run the full 49-task suite

`scripts/run_eval.py`, no `--limit`. Resumable (Step 17) — a session death is survivable via
the fallback resume cell below, not a lost run."""))

    cells.append(code("""log_progress("cell 6: starting full run_eval.py ...")
# --resume is a no-op on a truly fresh run (run_eval.py only activates its resume path if
# reports/a3_eval_results.json already exists) -- kept on unconditionally as a safety net:
# if this same kernel session gets re-run after a partial failure (e.g. Groq's daily token
# cap, which the 49-task run alone measured out at 199,116/200,000 tokens in one real run),
# a retry in the SAME session picks up where it left off instead of re-spending tokens on
# already-completed tasks.
run_eval_exit = subprocess.run(
    [sys.executable, "scripts/run_eval.py", "--resume"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
Path("/kaggle/working/run_eval_stdout.log").write_text(run_eval_exit.stdout, encoding="utf-8")
print(run_eval_exit.stdout[-8000:])  # tail in the cell output too, for a quick look
log_progress(f"cell 6: run_eval.py exited with code {run_eval_exit.returncode} "
             f"(full stdout/stderr saved to run_eval_stdout.log, "
             f"{len(run_eval_exit.stdout)} chars)")
if run_eval_exit.returncode != 0:
    raise RuntimeError(f"scripts/run_eval.py exited {run_eval_exit.returncode} -- see "
                        f"run_eval_stdout.log")
"""))

    cells.append(
        md("""### 6b. Resume fallback (only run this cell if cell 6 above died partway through)

`--resume` picks up from whatever's already in `reports/a3_eval_results.json`, per-task
resumability (Step 17's own design, exactly for this).""")
    )

    cells.append(code("""# !python scripts/run_eval.py --resume
"""))

    cells.append(
        md(
            """## 7. Do item 5 — verify the agentic gate from the real trace, at zero extra cost

**Revised after a real finding:** the first version of this cell called `pipeline.answer()`
twice more, live, for `"""
            + GATE_TRIGGER_ID
            + """`/`"""
            + GATE_CONTROL_ID
            + """`, to capture each raw `traces/run.jsonl`
before the next call overwrote it. That worked exactly once, then failed — the full 49-task
run alone measured out at 199,116 of Groq's free-tier 200,000 tokens-per-day cap, and those
two extra calls tipped it over (`groq.RateLimitError: 429`, real evidence in its own right
about actual token spend, not a bug). Since `"""
            + GATE_TRIGGER_ID
            + """`/`"""
            + GATE_CONTROL_ID
            + """` are both already inside the
49-task suite cell 6 just ran, and `run_eval.py`'s own `read_trace()` already read each
task's real `traces/run.jsonl` individually, right after that task's own `pipeline.answer()`
call and before the next task's call overwrote it — this cell now pulls their real,
trace-derived `k_sequence`/`top_scores`/`re_searched` straight out of
`reports/a3_eval_results.json` instead. Same real trace data, zero extra tokens spent. If the
trigger does NOT show `k` widening here, that is reported plainly — per the user's own
instruction, the fix in that case is Step 14's task labels, not a `weak_threshold` retune."""
        )
    )

    cells.append(code(GATE_VERIFICATION_SCRIPT))

    cells.append(md("""## 8. Do item 6 — settle `weak_threshold` against the real score distribution

The plan's own text still says "0.35" (Step 3's original, cosine-tuned value) — the LIVE
config value as of this run is actually **0.50** already (retuned 2026-08-23 from exactly
ONE real data point, explicitly flagged PROVISIONAL pending this step's full-suite
measurement). This cell reports where the REAL distribution, across every retrieve attempt
in the full run, actually sits relative to both numbers."""))

    cells.append(code(WEAK_THRESHOLD_ANALYSIS_SCRIPT))

    cells.append(md("""## 9. Package output"""))

    cells.append(code("""import shutil as _shutil

log_progress("cell 9: packaging output ...")
out_dir = "/kaggle/working/out"
os.makedirs(out_dir, exist_ok=True)
for name in ("reports/a3_eval_results.json", "reports/a3_eval_report.md",
             "/kaggle/working/progress.log", "/kaggle/working/run_eval_stdout.log",
             "/kaggle/working/gate_verification_log.txt",
             "/kaggle/working/weak_threshold_analysis.json"):
    src = Path(name)
    if src.exists():
        _shutil.copy(src, out_dir)
    else:
        print(f"(not present, skipped) {name}")
gate_dir = Path("/kaggle/working/gate_verification")
if gate_dir.exists():
    _shutil.copytree(gate_dir, os.path.join(out_dir, "gate_verification"), dirs_exist_ok=True)
archive_path = _shutil.make_archive("/kaggle/working/step22_main_eval_output", "zip", out_dir)
log_progress(f"cell 9: wrote {archive_path} ({os.path.getsize(archive_path) / 1e6:.2f} MB)")
"""))

    return cells


def main() -> None:
    cells = build_cells()
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_dir = REPO_ROOT / "KAGGLE" / "step22_main_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "kaggle_step22_main_eval.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(nb['cells'])} cells, {out.stat().st_size} bytes)")

    kernel_metadata = {
        "id": f"{OWNER}/{KERNEL_SLUG}",
        "title": "MathScholar Step22 Main Eval",
        "code_file": "kaggle_step22_main_eval.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",  # plan_a3.md 12.7 -- avoids P100
        "enable_internet": True,
        "dataset_sources": [INDEX_DATASET],
        "competition_sources": [],
        "kernel_sources": [],
    }
    meta_out = out_dir / "kernel-metadata.json"
    meta_out.write_text(json.dumps(kernel_metadata, indent=1), encoding="utf-8")
    print(f"wrote {meta_out}")


if __name__ == "__main__":
    main()
