# Changelog
## 0.1.0 — starter skeleton

### A2 progress ledger
> One line per merged step. This is how anyone resumes: last `[x]` = where we are,
> first `[ ]` = what to do next. Owners and step numbers follow `plan.md`.

- [x] Step 01 — repo bootstrap: deps, lockfiles, gitignore, green CI baseline — S1 (2105037) — 2026-08-08
- [x] Step 02 — configs/task.yaml + config.yaml — S2 (2105047)
- [x] Step 03 — data/provenance.md + scripts/get_data.sh — S3 (2105058)
- [x] Step 04 — notebooks/eda.ipynb — S1 (2105037)
- [x] Step 05 — grading_kit manifest + labels reconcile + 3 gold pages — S2
- [x] Step 06 — data/validate.py + data/versioning.py — S3
- [x] Step 07 — governance/pii.py (pipeline blocker) — S1
- [x] Step 08 — ingest/loader.py — S2
- [x] Step 09 — ingest/preprocess.py — S3
- [x] Step 10 — vision/layout.py — S1
- [x] Step 11 — vision/ocr.py (baseline reader) — S2
- [x] Step 12 — eval/metrics.py — S3
- [x] Step 13 — index/chunk.py (semantic chunking, bonus E4) — S1
- [x] Step 14 — index/embed.py — S2
- [x] Step 15 — index/store.py + scripts/build_index.sh — S3
- [x] Step 16 — Kaggle: baseline full-book OCR + BEFORE number--S1
- [x] Step 17 — annotation conventions + tooling — S2
- [x] Step 18 — render the 164 annotation pages — S3
- [x]Step 18b — Repair the baseline reader, then re-run it — S1
- [x] Step 19 — annotate TEST batch A (18) — S1
- [x] Step 20 — annotate TEST batch B (18) — S2
- [x] Step 21 — annotate VALIDATION (20) — S3
- [x] Step 22 — annotate TRAIN A (41) — S1
- [x] Step 23 — annotate TRAIN B (41) — S2; revised
- [x] Step 24 — annotate TRAIN C (40) — S3
- [x] Step 25 — NIST extraction (Stage A pairs) — S1
- [x] Step 26 — synthetic degradation pipeline — S2
- [x] Step 27 — training/ + LoRA adapt — S1-(modified)
- [x] Step 28 — Kaggle: fine-tune + learning curve — S3(modified)
- [x] Step 29 — final TEST measurement (once) — S2
- [x] Step 30 — Kaggle: re-OCR + rebuild index — S3
- [x] Step 31 — notebooks/kb_demo.ipynb — S1
- [x] Step 32 — reports/pipeline_diagram.md — S2
- [x] Step 33 — configs/design_choices.md — S3
- [x] Step 34 — A2 form sections 4 + 5 — S1
- [x] Step 35 — A2 form sections 2 + 3 — S2
- [x] Step 36 — A2 form sections 1 + 6 + 7 — S3
- [x] Step 37 — transcripts x3 — all
- [x] Step 38 — final checks + `a2-submit` tag — S1

### A3 progress ledger
> One line per merged step. This is how anyone resumes: last `[x]` = where we are,
> first `[ ]` = what to do next. Owners and step numbers follow `plan_a3.md`.

- [x] Step 01 — retrieval/retriever.py::retrieve() (dense + cached index + scores) — S1 (2105037)
- [x] Step 02 — llm/client.py::LLM.complete() — Groq, `openai/gpt-oss-120b` (DECISION D1) — S1 (2105037)
- [x] Step 03 — retrieval/rerank.py + scoped CLIP visual fallback (DECISION D2); real Kaggle
  GPU embedding run, `KAGGLE/a3_step03_visual_embed/` — S1 (2105037)
- [x] Step 04 — Kaggle: publish `data/index/` as `mathscholar-index` dataset + recall@k probe
  (39 gold TEST pages, dense vs. reranked), `KAGGLE/a3_step04_index_probe/` +
  `reports/a3_retrieval_probe.md` — S1 (2105037)
- [x] Step 05 — eval/metrics.py — recall_at_k, groundedness, citation_accuracy, subgroup_gap — S1 (2105037)
- [x] Step 06 — llm/prompts.py — DECIDE, SYNTHESIZE, JUDGE — S1
- [x] Step 07 — agent/tools.py — all nine tools — S1
- [x] Step 08 — agent/memory.py::recall() — S2
- [x] Step 09 — agent/agent.py::act() — tool dispatch — S3
- [x] Step 10 — 🔴 agent/agent.py::decide() — THE MANDATORY AGENTIC BEHAVIOUR — S1
- [x] Step 11 — synthesize() + llm/postprocess.py — grounding & abstention (DECISION D6:
  verify-and-correct) — S2
- [x] Step 12 — logging_conf.register() + traces/run.jsonl — S3
- [x] Step 13 — guardrails.check() + hitl.escalate() — S1
- [x] Step 14 — 🔴 grading_kit/tasks.jsonl (DECISION D3) — S2
- [] Step 15 — grading_kit/success_check.py — S3
- [ ] Step 16 — eval/judge.py (DECISION D5: hybrid LLM-judge + human spot-check) — S1
- [ ] Step 17 — scripts/run_eval.py + scripts/run.sh — S2
- [ ] Step 18 — eval/ablation.py — S3
- [ ] Step 19 — Tests: un-skip everything and close the CI gates — S1
- [ ] Step 20 — eval/interpret.py — 🔴 PRIMARY NFR: Explainable — S2
- [ ] Step 21 — eval/calibration.py — second NFR: Calibrated — S3
- [ ] Step 22 — ☁️ Kaggle: the main eval run — S1
- [ ] Step 23 — ☁️ Kaggle: baseline + ablations — S2
- [ ] Step 24 — ☁️ Kaggle: NFR measurement + reproducibility proof — S3
- [ ] Step 25 — traces/ + reports/eval_report.md — S1
- [ ] Step 26 — ☁️ Kaggle: RL / RLVR (DECISION D4: attempting it) — S2
- [ ] Step 27 — configs/design_choices.md — Stages 5-9 — S3
- [ ] Step 28 — Form §1 + §2 + §7 — S1
- [ ] Step 29 — Form §3 + §4 — S2
- [ ] Step 30 — Form §5 + §6 — S3
- [ ] Step 31 — transcripts x3 — all
- [ ] Step 32 — 🏁 final checks + `a3-submit` tag — S1
