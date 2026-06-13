from __future__ import annotations

# golden-frame fixtures for the comparator. every reference in the catalog is
# treated as a canonical landmark set and held to the properties the app
# relies on: a correct hand scores correct, a mirrored hand scores correct,
# distance from the camera doesn't matter, and no reference is closer to a
# different gesture than to itself. pure read-only checks — nothing here
# touches comparator math or thresholds.
#
# run directly (`python3 tests/test_comparator.py` from /vision) or via pytest.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from vision import GESTURES, compare_gesture, get_reference
from vision.core.comparator import EXCELLENT
from vision.core.utils import mirror_horizontal


def test_every_reference_matches_itself() -> None:
    for gesture_id, reference in GESTURES.items():
        result = compare_gesture(reference.tolist(), reference)
        assert result.band == "correct", f"{gesture_id}: band {result.band}"
        assert result.accuracy >= EXCELLENT, f"{gesture_id}: accuracy {result.accuracy:.4f}"
        assert result.incorrect_points == [], f"{gesture_id}: joints flagged {result.incorrect_points}"


def test_mirrored_hand_still_matches() -> None:
    # the left-handed version of every reference must score as well as the
    # right-handed one — this is the lefty guarantee the mirror pass exists for.
    for gesture_id, reference in GESTURES.items():
        result = compare_gesture(mirror_horizontal(reference).tolist(), reference)
        assert result.band == "correct", f"{gesture_id} mirrored: band {result.band}"
        assert result.accuracy >= EXCELLENT, f"{gesture_id} mirrored: accuracy {result.accuracy:.4f}"


def test_scale_and_position_do_not_change_the_score() -> None:
    # the same hand seen from twice the distance, anywhere in the frame, must
    # score identically — normalization owns this.
    offset = np.array([0.4, -1.2, 0.8], dtype=np.float32)
    for gesture_id, reference in GESTURES.items():
        base = compare_gesture(reference.tolist(), reference).accuracy
        moved = np.asarray(reference) * 3.0 + offset
        again = compare_gesture(moved.tolist(), reference).accuracy
        assert abs(base - again) < 1e-4, f"{gesture_id}: {base:.6f} vs {again:.6f}"


def test_each_reference_is_its_own_best_match() -> None:
    # separation: a perfect frame of gesture A must never score higher against
    # gesture B. some pairs sit close (m/n differ only in palm tilt), so this
    # is the property that keeps live-mode matching honest.
    for gesture_id, reference in GESTURES.items():
        frame = reference.tolist()
        self_accuracy = compare_gesture(frame, reference).accuracy
        for other_id, other_reference in GESTURES.items():
            if other_id == gesture_id:
                continue
            other_accuracy = compare_gesture(frame, other_reference).accuracy
            assert other_accuracy < self_accuracy, (
                f"{gesture_id} scores {other_accuracy:.4f} against {other_id}, "
                f"only {self_accuracy:.4f} against itself"
            )


def test_degenerate_input_never_scores_correct() -> None:
    # a hand that collapsed to a point (tracking glitch) must not pass anything.
    flat = [[0.0, 0.0, 0.0]] * 21
    for gesture_id, reference in GESTURES.items():
        result = compare_gesture(flat, reference)
        assert result.band != "correct", f"{gesture_id}: zero hand scored {result.accuracy:.4f}"


def test_malformed_input_is_rejected() -> None:
    reference = get_reference("hello")
    try:
        compare_gesture([[0.0, 0.0, 0.0]] * 20, reference)
        raise AssertionError("20-point hand was accepted")
    except ValueError:
        pass
    try:
        get_reference("not_a_gesture")
        raise AssertionError("unknown gesture id was accepted")
    except KeyError:
        pass


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok — {name}")
    print(f"\n{len(tests)} checks passed over {len(GESTURES)} references.")
