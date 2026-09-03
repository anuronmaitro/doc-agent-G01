"""Unit test home for eval/calibration.py (Step 21, A3) -- second NFR (Calibrated, target
ECE <= 0.05). Synthetic confidence/correctness arrays with a hand-computed, known ECE, per
the ORDER's own requirement -- not just "does it run," but "does it compute the right number."
"""

from __future__ import annotations

import pytest

from doc_agent.eval import calibration, metrics


class TestEceKnownValues:
    def test_perfectly_calibrated_confidence_scores_zero(self):
        """confidence == accuracy in every bin -> ECE is exactly 0."""
        # bin [0.0,0.1): 10 points at conf=0.05, 1 correct -> acc=0.1, matches conf=0.05?
        # Simpler: use exact bin-centre confidences with the SAME fraction correct.
        confidences = [0.05] * 10 + [0.95] * 10
        # 1 of the low-confidence 10 correct (10% accuracy, conf~5%) -- not exact, so build
        # an exact case instead: confidence == 1.0 or 0.0, correctness matches exactly.
        confidences = [0.0] * 5 + [1.0] * 5
        correct = [False] * 5 + [True] * 5
        assert calibration.ece(confidences, correct) == pytest.approx(0.0, abs=1e-9)

    def test_hand_computed_two_bin_example(self):
        """10 points, 2 bins worth of mass, ECE computed by hand:
        bin A: 5 points at confidence 0.9, 4 correct -> conf=0.9, acc=0.8, |diff|=0.1, weight 0.5
        bin B: 5 points at confidence 0.3, 1 correct -> conf=0.3, acc=0.2, |diff|=0.1, weight 0.5
        ECE = 0.5*0.1 + 0.5*0.1 = 0.1
        """
        confidences = [0.9] * 5 + [0.3] * 5
        correct = [True] * 4 + [False] + [True] + [False] * 4
        assert calibration.ece(confidences, correct) == pytest.approx(0.1, abs=1e-9)

    def test_maximally_overconfident_wrong_every_time(self):
        """confidence=1.0 but always wrong -> ECE == 1.0, the worst possible value."""
        confidences = [1.0] * 20
        correct = [False] * 20
        assert calibration.ece(confidences, correct) == pytest.approx(1.0, abs=1e-9)

    def test_empty_input_is_zero_not_an_error(self):
        assert calibration.ece([], []) == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            calibration.ece([0.5, 0.5], [True])

    def test_out_of_range_confidence_raises(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            calibration.ece([1.5], [True])

    def test_meets_the_005_target_on_a_well_calibrated_synthetic_set(self):
        confidences = [0.1] * 20 + [0.5] * 20 + [0.9] * 20
        correct = (
            [False] * 18
            + [True] * 2  # ~10% accuracy at conf 0.1
            + [False] * 10
            + [True] * 10  # 50% accuracy at conf 0.5
            + [False] * 2
            + [True] * 18  # 90% accuracy at conf 0.9
        )
        assert calibration.ece(confidences, correct) <= 0.05


class TestApplyTemperature:
    def test_identity_temperature_is_a_no_op(self):
        assert calibration.apply_temperature(0.73, 1.0) == 0.73

    def test_temperature_greater_than_one_pulls_toward_0_5(self):
        """Dividing the logit by T>1 shrinks it toward 0 -- softens an over/under-confident
        score toward the uninformative midpoint, the whole point of temperature scaling."""
        softened = calibration.apply_temperature(0.9, 2.0)
        assert 0.5 < softened < 0.9

    def test_temperature_less_than_one_pushes_away_from_0_5(self):
        sharpened = calibration.apply_temperature(0.6, 0.5)
        assert sharpened > 0.6

    def test_non_positive_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature must be > 0"):
            calibration.apply_temperature(0.5, 0.0)

    def test_extremes_stay_within_bounds(self):
        assert 0.0 < calibration.apply_temperature(0.0, 2.0) < 1.0
        assert 0.0 < calibration.apply_temperature(1.0, 2.0) < 1.0


class TestTemperatureScale:
    def test_fitting_on_systematically_overconfident_data_improves_ece(self):
        """Confidence is always 0.9 regardless of correctness (a classic overconfidence
        pattern) -- fitting T should soften it toward the real ~50% accuracy and lower ECE."""
        confidences = [0.9] * 40
        correct = [True] * 20 + [False] * 20  # true accuracy is 50%, not 90%
        result = calibration.temperature_scale(confidences, correct)
        assert result["ece_after"] < result["ece_before"]
        assert result["temperature"] > 1.0  # softening requires T > 1
        assert result["n"] == 40

    def test_nearly_constant_confidence_reports_the_honest_ece_not_manufactured_spread(self):
        """A near-constant confidence signal has essentially nothing for temperature scaling
        to fix -- report ece_after close to ece_before, not a suspiciously perfect number."""
        confidences = [0.62] * 30
        correct = [True] * 15 + [False] * 15  # 50% accuracy at a fixed ~0.62 confidence
        result = calibration.temperature_scale(confidences, correct)
        # a single scalar T cannot fix a systematic 0.12 miscalibration into 0 -- the honest
        # post-fit ECE stays meaningfully above 0, not driven to a manufactured-looking 0.0.
        assert result["ece_after"] > 0.01

    def test_already_well_calibrated_data_keeps_temperature_near_identity(self):
        confidences = [0.1] * 20 + [0.5] * 20 + [0.9] * 20
        correct = [False] * 18 + [True] * 2 + [False] * 10 + [True] * 10 + [False] * 2 + [True] * 18
        result = calibration.temperature_scale(confidences, correct)
        assert result["ece_before"] <= 0.05
        assert result["ece_after"] <= result["ece_before"] + 1e-9

    def test_empty_input_returns_identity_and_zero_ece(self):
        result = calibration.temperature_scale([], [])
        assert result == {
            "temperature": calibration.IDENTITY_TEMPERATURE,
            "ece_before": 0.0,
            "ece_after": 0.0,
            "n": 0,
        }

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            calibration.temperature_scale([0.5, 0.5], [True])


class TestMetricsDelegatesToCalibration:
    """Step 21's own instruction: one ece must delegate to the other, not drift apart."""

    def test_metrics_ece_matches_calibration_ece_exactly(self):
        confidences = [0.9] * 5 + [0.3] * 5
        correct = [True] * 4 + [False] + [True] + [False] * 4
        assert metrics.ece(confidences, correct) == calibration.ece(confidences, correct)

    def test_metrics_ece_is_the_same_function_behaviourally_on_edge_cases(self):
        assert metrics.ece([], []) == calibration.ece([], [])
        with pytest.raises(ValueError):
            metrics.ece([0.5], [True, False])
