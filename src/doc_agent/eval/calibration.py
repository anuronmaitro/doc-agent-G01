"""Stage 9 -- CALIBRATED (E17), second NFR -- expected calibration error + temperature scaling.

A1's target: ECE <= 0.05 (Sec.3/Sec.5). Confidence comes from Step 11's own
`llm/postprocess.py::_confidence()` -- a hand-built weighted formula (retrieval strength,
score gap, calculator verification, retry penalty), not a classifier's softmax output. That
matters for what "temperature scaling" means here: the classic technique divides a model's
pre-softmax LOGITS by a fitted scalar T before the softmax/sigmoid. We have no logits -- only
an already-in-[0,1] confidence score -- so `temperature_scale`/`apply_temperature` below work
in an equivalent space instead: inverse-sigmoid (`_logit`) the confidence to get a logit-like
value, scale by 1/T, sigmoid back. `T == 1.0` is the identity transform (today's honest
default -- see `IDENTITY_TEMPERATURE` below), so nothing changes until a real T is actually
fit and configured.

**Fit `temperature_scale` on VALIDATION only, never TEST** -- same one-time-TEST-opening
discipline `plan.md`'s A2 Step 28/29 already committed to (test data is opened exactly once,
for the final reported number, never used to tune anything upstream of that). This module has
no notion of which split its inputs came from -- enforcing that is the CALLER's job, same
split-discipline boundary every other place in this codebase that touches train/val/test
already holds to (e.g. `eval/judge.py`'s human spot-check, kept separate from what actually
grades).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ..contracts import *  # noqa

DEFAULT_N_BINS = 10
IDENTITY_TEMPERATURE = 1.0
ECE_TARGET = 0.05  # A1's own commitment (plan_a3.md Sec.3/Sec.5)
# search grid for temperature_scale() -- fine enough (0.01 step) to be a real fit, cheap
# enough (~500 ECE evaluations, each O(n)) to run on a laptop, no new dependency needed.
_T_MIN, _T_MAX, _T_STEP = 0.05, 5.0, 0.01
_EPS = 1e-6


def ece(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = DEFAULT_N_BINS
) -> float:
    """Expected Calibration Error, equal-width binned. Target <= 0.05 (A1's own commitment).

    ECE = sum over bins of (|bin| / N) * |accuracy(bin) - mean_confidence(bin)| -- the
    standard binned estimator. An empty bin contributes 0 (nothing measured there), not NaN.
    `n_bins=10` is the conventional default; exposed as a parameter since a small eval set
    (this project's ~49 tasks) can make finer binning noisy -- a caller measuring on few
    points may deliberately choose fewer bins.

    Raises ValueError on mismatched lengths (a caller bug, not something to silently
    misalign and report a wrong number for). Returns 0.0 on empty input -- there is nothing
    to be miscalibrated about with zero data points, not an error.
    """
    conf = [float(c) for c in confidences]
    ok = [bool(c) for c in correct]
    if len(conf) != len(ok):
        raise ValueError(
            f"confidences and correct must be the same length, got {len(conf)} and {len(ok)}"
        )
    if not conf:
        return 0.0
    if any(c < 0.0 or c > 1.0 for c in conf):
        raise ValueError("confidences must lie in [0.0, 1.0]")

    n = len(conf)
    edges = [i / n_bins for i in range(n_bins + 1)]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        is_last = hi >= 1.0 - 1e-12
        in_bin = [
            (c, o) for c, o in zip(conf, ok, strict=True) if lo <= c < hi or (is_last and c == hi)
        ]
        if not in_bin:
            continue
        bin_conf = sum(c for c, _ in in_bin) / len(in_bin)
        bin_acc = sum(1 for _, o in in_bin if o) / len(in_bin)
        total += (len(in_bin) / n) * abs(bin_acc - bin_conf)
    return round(total, 6)


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def apply_temperature(confidence: float, temperature: float) -> float:
    """Rescale ONE confidence value in probability space: logit -> divide by T -> sigmoid
    back. `temperature == IDENTITY_TEMPERATURE` (1.0) is the identity transform -- this is
    what every live call site in this codebase uses until a real T has actually been fit and
    configured (see `configs/config.yaml`'s `calibration.temperature`, and this module's own
    docstring on why 1.0, not some other number, is the honest un-calibrated default)."""
    if temperature == IDENTITY_TEMPERATURE:
        return confidence
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return _sigmoid(_logit(confidence) / temperature)


def temperature_scale(logits: Sequence[float], labels: Sequence[bool]) -> dict[str, Any]:
    """Fit ONE scalar temperature T on (confidence, correct) pairs by grid-searching T over
    (0, 5] in 0.01 steps and keeping whichever minimises `ece()` on these same inputs --
    the direct, honest objective (A1's own target metric), not a proxy like NLL that could
    minimise something else and still miss ECE. `logits` here means Step 11's `Answer.
    confidence` values (already in [0,1]) -- see this module's own docstring for why we
    don't have real classifier logits to scale.

    **MUST be called with VALIDATION data only** -- see this module's docstring.

    Returns `{"temperature", "ece_before", "ece_after", "n"}`, not a bare number: reports
    BOTH the pre- and post-fit ECE so a caller can see whether fitting actually helped
    rather than trusting the returned T blindly (Do item 3's own discipline: "if the
    confidence signal turns out to be nearly constant, say so and report the honest ECE
    rather than manufacturing spread" -- a near-constant confidence signal has essentially
    nothing for temperature scaling to fix, and `ece_after` approx= `ece_before` is exactly
    how that shows up here, not hidden inside a single opaque number).

    Empty input returns the identity temperature and `ece_before == ece_after == 0.0` --
    there is nothing to fit, not an error.
    """
    conf = [float(c) for c in logits]
    lab = [bool(c) for c in labels]
    if len(conf) != len(lab):
        raise ValueError(
            f"logits and labels must be the same length, got {len(conf)} and {len(lab)}"
        )
    if not conf:
        return {"temperature": IDENTITY_TEMPERATURE, "ece_before": 0.0, "ece_after": 0.0, "n": 0}

    ece_before = ece(conf, lab)
    best_t, best_ece = IDENTITY_TEMPERATURE, ece_before

    steps = round((_T_MAX - _T_MIN) / _T_STEP)
    for i in range(steps + 1):
        t = round(_T_MIN + i * _T_STEP, 4)
        scaled = [apply_temperature(c, t) for c in conf]
        e = ece(scaled, lab)
        if e < best_ece:
            best_ece, best_t = e, t

    return {"temperature": best_t, "ece_before": ece_before, "ece_after": best_ece, "n": len(conf)}
