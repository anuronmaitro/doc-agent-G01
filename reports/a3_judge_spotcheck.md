# A3 Step 16 — LLM-judge human spot-check (DECISION D5, second half)

**Status: PENDING real data — not yet run.** This is a template with the real methodology and
the real 10 selected questions, not a placeholder with invented numbers. `eval/judge.py::judge()`
itself is fully implemented and tested (`tests/test_judge.py`, 18 tests, LLM mocked, real
`_parse_score()`/`_evidence_block()` logic exercised directly) — that part of Step 16 is done.
This spot-check specifically needs **real agent-generated answers** to judged questions, scored by
both the real LLM-judge and a human, and none exist yet: Steps 17–21 (the eval harness, ablations,
calibration) haven't been built, and Step 22 (the main Kaggle eval run — the source Do item 3 names
explicitly: *"do it once against the main eval run's judge outputs"*) hasn't happened. Fabricating
an agreement number here would misrepresent evidence backing a graded form section, so this stays
honestly marked pending rather than filled with invented data.

**To complete this**: once Step 22 (or an earlier lightweight run producing real `Answer` objects
for these 10 questions) exists, fill in the LLM-judge and human columns below from the real run's
output, then compute the agreement rate and replace this status line.

## Method

- **Selection**: 10 of the 17 `kind: "judged"` tasks in `grading_kit/tasks.jsonl` (Step 14),
  covering all three splits proportionally (4 train / 3 val / 3 test, matching the tasks' own
  20/12/17 split ratio) rather than picked to flatter any one split's agreement rate.
- **Human grading**: a teammate scores each of the 10 real agent answers by hand against the
  *exact same* `JUDGE` rubric (`src/doc_agent/llm/prompts.py`) `eval/judge.py::judge()` uses — same
  three criteria (CORRECTNESS / COMPLETENESS / GROUNDEDNESS, each 0/1/2), same evidence shown, no
  separate human-only rubric.
- **Agreement definition**: human total and LLM-judge total (both on the same 0–6 scale —
  `judge()`'s own documented range) agree if they're within **1 point** of each other. Reported as
  agreement count / 10, plus the raw score pairs so a reader can see the actual spread, not just
  the summary rate.

## The 10 selected questions

| id | split | question |
|---|---|---|
| t09 | train | Explain the asymptotic behavior described by formula 11.1.14 for x^(1/2) e^(-x) times the integral of I_0(t) from 0 to x, as x becomes large. |
| t10 | train | Explain the asymptotic expansion of H_nu(z) - Y_nu(z) for large \|z\| given in formula 12.1.34, including what the leading term depends on. |
| t12 | train | Explain what happens to the Weierstrass functions wp, wp', zeta, and sigma at the half-periods, per section 18.3. |
| t14 | train | Explain the orthogonality property of Bernoulli polynomials described by formula 23.1.12, including the condition on m+n. |
| v06 | val | Explain the notational conventions for associated Legendre functions given on p.332, including what P^n(x) and P_{nm}(x) denote. |
| v08 | val | Explain the Airy-function asymptotic approximation for M(a,b,x) given in formula 13.5.19, and when it applies. |
| v09 | val | Explain the Fuchsian classification of solutions to the hypergeometric differential equation based on the singular points 0, 1, infinity. |
| te07 | test | Explain the auxiliary functions f(z) and g(z) used to express the sine and cosine integrals Si(z) and Ci(z), per formulas 5.2.6-5.2.9. |
| te09 | test | Describe the series expansions for the Fresnel integral C(z) given in formulas 7.3.11-7.3.12. |
| te11 | test | Describe the polynomial approximation given in formula 9.8.5 for the modified Bessel function K_0(x) on the interval 0 < x <= 2. |

## Results (fill in once real answers + judge outputs exist)

| id | LLM-judge total (/6) | Human total (/6) | \|diff\| | Agree (≤1)? |
|---|---|---|---|---|
| t09 | — | — | — | — |
| t10 | — | — | — | — |
| t12 | — | — | — | — |
| t14 | — | — | — | — |
| v06 | — | — | — | — |
| v08 | — | — | — | — |
| v09 | — | — | — | — |
| te07 | — | — | — | — |
| te09 | — | — | — | — |
| te11 | — | — | — | — |

**Agreement rate: — / 10 (pending)**
