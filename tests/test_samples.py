import numpy as np
import pytest

from vision.core.utils import LANDMARK_COUNT
from vision.gestures.samples import GESTURES, get_reference


# ── get_reference ─────────────────────────────────────────────────────────────

def test_get_reference_returns_numpy_array():
    ref = get_reference("hello")
    assert isinstance(ref, np.ndarray)


def test_get_reference_correct_shape():
    ref = get_reference("hello")
    assert ref.shape == (LANDMARK_COUNT, 3)


def test_get_reference_dtype_is_float32():
    ref = get_reference("hello")
    assert ref.dtype == np.float32


def test_get_reference_unknown_raises_key_error():
    with pytest.raises(KeyError, match="no reference gesture"):
        get_reference("not-a-gesture")


def test_get_reference_key_error_contains_id():
    with pytest.raises(KeyError, match="unknown-id"):
        get_reference("unknown-id")


# ── GESTURES catalog ──────────────────────────────────────────────────────────

def test_all_gestures_have_correct_shape():
    for name, arr in GESTURES.items():
        assert arr.shape == (LANDMARK_COUNT, 3), f"wrong shape for '{name}'"


def test_all_gestures_dtype_is_float32():
    for name, arr in GESTURES.items():
        assert arr.dtype == np.float32, f"wrong dtype for '{name}'"


def test_catalog_is_nonempty():
    assert len(GESTURES) > 0


def test_at_least_30_gestures_registered():
    assert len(GESTURES) >= 30


def test_core_vocabulary_present():
    expected = [
        "hello", "thank_you", "please", "sorry", "water", "food",
        "help", "stop", "yes", "no", "bathroom", "pain", "tired",
        "sleep", "more", "finished",
    ]
    for gesture_id in expected:
        assert gesture_id in GESTURES, f"core gesture missing: {gesture_id}"


def test_all_26_alphabet_letters_present():
    for letter in "abcdefghijklmnopqrstuvwxyz":
        key = f"letter_{letter}"
        assert key in GESTURES, f"missing alphabet gesture: {key}"


def test_gestures_wrist_at_origin():
    for name, arr in GESTURES.items():
        wrist = arr[0]
        np.testing.assert_allclose(
            wrist, [0.0, 0.0, 0.0], atol=1e-5,
            err_msg=f"wrist of '{name}' is not at origin",
        )


def test_get_reference_returns_same_object_as_gestures():
    ref = get_reference("hello")
    assert ref is GESTURES["hello"]
