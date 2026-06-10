import numpy as np
import pytest

from vision.core.comparator import (
    CORRECT,
    PARTIAL,
    CompareResult,
    _band,
    compare_gesture,
)
from vision.gestures.samples import GESTURES


# ── _band ─────────────────────────────────────────────────────────────────────

def test_band_at_correct_threshold_is_correct():
    assert _band(CORRECT) == "correct"


def test_band_above_correct_is_correct():
    assert _band(CORRECT + 0.01) == "correct"


def test_band_between_partial_and_correct_is_partial():
    midpoint = (PARTIAL + CORRECT) / 2
    assert _band(midpoint) == "partial"


def test_band_at_partial_threshold_is_partial():
    assert _band(PARTIAL) == "partial"


def test_band_just_below_partial_is_incorrect():
    assert _band(PARTIAL - 0.001) == "incorrect"


def test_band_zero_is_incorrect():
    assert _band(0.0) == "incorrect"


# ── compare_gesture ───────────────────────────────────────────────────────────

def test_compare_identical_inputs_high_accuracy():
    ref = GESTURES["hello"]
    result = compare_gesture(ref.tolist(), ref)
    assert result.accuracy > 0.9


def test_compare_identical_band_is_correct():
    ref = GESTURES["hello"]
    result = compare_gesture(ref.tolist(), ref)
    assert result.band == "correct"


def test_compare_identical_no_incorrect_points():
    ref = GESTURES["hello"]
    result = compare_gesture(ref.tolist(), ref)
    assert result.incorrect_points == []


def test_compare_returns_compare_result():
    ref = GESTURES["hello"]
    result = compare_gesture(ref.tolist(), ref)
    assert isinstance(result, CompareResult)


def test_compare_result_accuracy_in_range():
    ref = GESTURES["hello"]
    result = compare_gesture(ref.tolist(), ref)
    assert 0.0 <= result.accuracy <= 1.0


def test_compare_wrong_reference_shape_raises():
    bad_ref = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="reference must be"):
        compare_gesture(GESTURES["hello"].tolist(), bad_ref)


def test_compare_very_different_gesture_low_accuracy():
    ref = GESTURES["hello"]
    user = GESTURES["sorry"]
    result = compare_gesture(user.tolist(), ref, allow_mirror=False)
    assert result.accuracy < 0.95


def test_compare_different_gestures_have_incorrect_points():
    ref = GESTURES["hello"]
    user = GESTURES["sorry"]
    result = compare_gesture(user.tolist(), ref, allow_mirror=False)
    assert len(result.incorrect_points) > 0


def test_compare_to_dict_has_expected_keys():
    ref = GESTURES["hello"]
    result = compare_gesture(ref.tolist(), ref)
    d = result.to_dict()
    assert set(d.keys()) == {"accuracy", "band", "incorrect_points"}


def test_compare_to_dict_accuracy_rounded():
    ref = GESTURES["hello"]
    result = compare_gesture(ref.tolist(), ref)
    d = result.to_dict()
    assert isinstance(d["accuracy"], float)


def test_compare_mirrored_input_matches_with_mirror_enabled():
    ref = GESTURES["hello"]
    flipped = ref.copy()
    flipped[:, 0] *= -1.0
    result = compare_gesture(flipped.tolist(), ref, allow_mirror=True)
    assert result.accuracy > 0.9


def test_compare_mirrored_input_lower_without_mirror():
    ref = GESTURES["hello"]
    flipped = ref.copy()
    flipped[:, 0] *= -1.0
    with_mirror = compare_gesture(flipped.tolist(), ref, allow_mirror=True)
    without_mirror = compare_gesture(flipped.tolist(), ref, allow_mirror=False)
    assert with_mirror.accuracy >= without_mirror.accuracy


def test_compare_accepts_numpy_array_input():
    ref = GESTURES["hello"]
    result = compare_gesture(ref, ref)
    assert result.accuracy > 0.9


def test_compare_incorrect_points_are_valid_landmark_indices():
    ref = GESTURES["hello"]
    user = GESTURES["sorry"]
    result = compare_gesture(user.tolist(), ref, allow_mirror=False)
    for idx in result.incorrect_points:
        assert 0 <= idx <= 20
