"""
Integration tests for the vision pipeline.

These tests exercise multiple modules together (gestures/samples → core/utils →
core/comparator) without mocking any layer, verifying that the full pipeline
produces consistent, correct results end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision import compare_gesture, get_reference, normalize_landmarks
from vision.core.comparator import CORRECT, PARTIAL, CompareResult, _band
from vision.core.utils import LANDMARK_COUNT, mirror_horizontal
from vision.gestures.samples import GESTURES


# ── helpers ───────────────────────────────────────────────────────────────────

def _perturb(arr: np.ndarray, scale: float) -> np.ndarray:
    """Add uniform noise to a landmark array to simulate imperfect signing."""
    rng = np.random.default_rng(seed=42)
    return arr + rng.uniform(-scale, scale, arr.shape).astype(np.float32)


# ── normalize → compare pipeline ──────────────────────────────────────────────

def test_pipeline_identical_gesture_returns_correct_band():
    ref = get_reference("hello")
    result = compare_gesture(ref.tolist(), ref)
    assert result.band == "correct"


def test_pipeline_identical_gesture_high_accuracy():
    ref = get_reference("hello")
    result = compare_gesture(ref.tolist(), ref)
    assert result.accuracy > 0.9


def test_pipeline_produces_compare_result():
    ref = get_reference("hello")
    result = compare_gesture(ref.tolist(), ref)
    assert isinstance(result, CompareResult)


def test_pipeline_normalize_then_compare_identity():
    ref = get_reference("water")
    normalized = normalize_landmarks(ref.tolist())
    result = compare_gesture(normalized.tolist(), ref)
    assert result.band == "correct"


def test_pipeline_slight_perturbation_still_correct():
    ref = get_reference("hello")
    noisy = _perturb(ref, scale=0.05)
    result = compare_gesture(noisy.tolist(), ref)
    assert result.band == "correct"


def test_pipeline_heavy_perturbation_degrades_accuracy():
    ref = get_reference("hello")
    noisy = _perturb(ref, scale=1.0)
    result_clean = compare_gesture(ref.tolist(), ref)
    result_noisy = compare_gesture(noisy.tolist(), ref)
    assert result_noisy.accuracy < result_clean.accuracy


def test_pipeline_wrong_gesture_lower_accuracy():
    ref = get_reference("hello")
    user = get_reference("sorry")
    result = compare_gesture(user.tolist(), ref, allow_mirror=False)
    assert result.accuracy < 0.9


def test_pipeline_wrong_gesture_produces_incorrect_points():
    ref = get_reference("hello")
    user = get_reference("sorry")
    result = compare_gesture(user.tolist(), ref, allow_mirror=False)
    assert len(result.incorrect_points) > 0


def test_pipeline_incorrect_points_are_valid_indices():
    ref = get_reference("hello")
    user = get_reference("stop")
    result = compare_gesture(user.tolist(), ref, allow_mirror=False)
    for idx in result.incorrect_points:
        assert 0 <= idx <= 20


def test_pipeline_mirrored_user_matched_with_mirror_flag():
    ref = get_reference("hello")
    flipped = mirror_horizontal(ref)
    result_with = compare_gesture(flipped.tolist(), ref, allow_mirror=True)
    result_without = compare_gesture(flipped.tolist(), ref, allow_mirror=False)
    assert result_with.accuracy >= result_without.accuracy


def test_pipeline_to_dict_shape():
    ref = get_reference("water")
    result = compare_gesture(ref.tolist(), ref)
    d = result.to_dict()
    assert set(d.keys()) == {"accuracy", "band", "incorrect_points"}
    assert isinstance(d["accuracy"], float)
    assert isinstance(d["band"], str)
    assert isinstance(d["incorrect_points"], list)


# ── all gestures round-trip ───────────────────────────────────────────────────

@pytest.mark.parametrize("gesture_id", list(GESTURES.keys()))
def test_every_gesture_self_compares_as_correct(gesture_id):
    ref = get_reference(gesture_id)
    result = compare_gesture(ref.tolist(), ref)
    assert result.band == "correct", (
        f"'{gesture_id}' self-comparison returned band='{result.band}' "
        f"(accuracy={result.accuracy:.4f})"
    )


@pytest.mark.parametrize("gesture_id", list(GESTURES.keys()))
def test_every_gesture_self_accuracy_above_threshold(gesture_id):
    ref = get_reference(gesture_id)
    result = compare_gesture(ref.tolist(), ref)
    assert result.accuracy >= CORRECT, (
        f"'{gesture_id}' self-accuracy {result.accuracy:.4f} below CORRECT={CORRECT}"
    )


@pytest.mark.parametrize("gesture_id", list(GESTURES.keys()))
def test_every_gesture_self_no_incorrect_points(gesture_id):
    ref = get_reference(gesture_id)
    result = compare_gesture(ref.tolist(), ref)
    assert result.incorrect_points == [], (
        f"'{gesture_id}' self-comparison flagged joints {result.incorrect_points}"
    )


# ── cross-gesture discrimination ──────────────────────────────────────────────

_DISTINCT_PAIRS = [
    ("hello", "sorry"),
    ("water", "stop"),
    ("help", "no"),
    ("yes", "sleep"),
    ("letter_a", "letter_b"),
    ("letter_u", "letter_v"),
]


@pytest.mark.parametrize("gesture_a,gesture_b", _DISTINCT_PAIRS)
def test_distinct_gestures_differ_in_accuracy(gesture_a, gesture_b):
    ref = get_reference(gesture_a)
    same = compare_gesture(get_reference(gesture_a).tolist(), ref)
    diff = compare_gesture(get_reference(gesture_b).tolist(), ref, allow_mirror=False)
    assert same.accuracy > diff.accuracy, (
        f"'{gesture_a}' vs '{gesture_b}': same={same.accuracy:.4f} diff={diff.accuracy:.4f}"
    )


# ── get_reference / normalize integration ────────────────────────────────────

def test_get_reference_feeds_directly_into_compare():
    ref = get_reference("please")
    result = compare_gesture(ref, ref)
    assert result.band == "correct"


def test_normalize_output_shape_matches_landmark_count():
    ref = get_reference("food")
    normalized = normalize_landmarks(ref)
    assert normalized.shape == (LANDMARK_COUNT, 3)


def test_normalize_wrist_is_zero_after_normalization():
    ref = get_reference("hello")
    normalized = normalize_landmarks(ref)
    np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0], atol=1e-5)


def test_normalize_palm_span_is_one_after_normalization():
    ref = get_reference("hello")
    normalized = normalize_landmarks(ref)
    palm_span = float(np.linalg.norm(normalized[9]))
    assert abs(palm_span - 1.0) < 1e-4


def test_normalize_accepts_list_input():
    ref = get_reference("water")
    normalized = normalize_landmarks(ref.tolist())
    assert normalized.shape == (LANDMARK_COUNT, 3)


def test_normalize_accepts_numpy_input():
    ref = get_reference("water")
    normalized = normalize_landmarks(ref)
    assert normalized.shape == (LANDMARK_COUNT, 3)


def test_normalize_wrong_shape_raises_value_error():
    bad = [[0.1, 0.2, 0.3]] * 10  # only 10 points
    with pytest.raises(ValueError):
        normalize_landmarks(bad)


# ── public api surface ────────────────────────────────────────────────────────

def test_public_exports_compare_gesture_callable():
    from vision import compare_gesture as cg
    ref = get_reference("hello")
    result = cg(ref.tolist(), ref)
    assert isinstance(result, CompareResult)


def test_public_exports_get_reference_callable():
    from vision import get_reference as gr
    ref = gr("hello")
    assert ref.shape == (LANDMARK_COUNT, 3)


def test_public_exports_normalize_callable():
    from vision import normalize_landmarks as nl
    ref = get_reference("hello")
    out = nl(ref)
    assert out.shape == (LANDMARK_COUNT, 3)


def test_public_exports_gestures_dict_available():
    from vision import GESTURES as G
    assert len(G) > 0
    assert "hello" in G
