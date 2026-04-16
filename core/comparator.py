from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import LANDMARK_COUNT, mirror_horizontal, normalize_landmarks

# per-joint distance above which we flag the joint as "off".
_JOINT_DRIFT_THRESHOLD = 0.22

# accuracy bands — used by callers to pick feedback copy.
CORRECT = 0.8
PARTIAL = 0.5


@dataclass
class CompareResult:
    accuracy: float  # 0..1
    incorrect_points: list[int]
    band: str  # "correct" | "partial" | "incorrect"

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "incorrect_points": self.incorrect_points,
            "band": self.band,
        }


def _band(accuracy: float) -> str:
    if accuracy >= CORRECT:
        return "correct"
    if accuracy >= PARTIAL:
        return "partial"
    return "incorrect"


def _similarity(user: np.ndarray, ref: np.ndarray) -> tuple[float, np.ndarray]:
    """
    returns (similarity in [0, 1], per-joint distance).
    uses euclidean per-joint distance mapped through an exponential to stay bounded.
    """
    diffs = user - ref
    per_joint = np.linalg.norm(diffs, axis=1)
    mean_dist = float(per_joint.mean())
    # distance 0 → sim 1; distance ~0.7 → sim ~0.5; large distance → sim → 0.
    similarity = float(np.exp(-mean_dist * 1.5))
    return similarity, per_joint


def compare_gesture(
    landmarks: list[list[float]] | np.ndarray,
    reference: np.ndarray,
    *,
    allow_mirror: bool = True,
) -> CompareResult:
    """
    compare a user's raw landmarks against a normalized reference vector.

    reference is expected to be (21, 3), already normalized.
    if allow_mirror, we also try the horizontally flipped reference and take the best
    — lets lefties practice right-handed references without failing.
    """
    if reference.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"reference must be (21, 3), got {reference.shape}")

    user_norm = normalize_landmarks(landmarks)

    sim, per_joint = _similarity(user_norm, reference)
    best_joint = per_joint

    if allow_mirror:
        mirrored = mirror_horizontal(reference)
        sim_m, per_joint_m = _similarity(user_norm, mirrored)
        if sim_m > sim:
            sim, best_joint = sim_m, per_joint_m

    incorrect = [int(i) for i, d in enumerate(best_joint) if d > _JOINT_DRIFT_THRESHOLD]
    return CompareResult(accuracy=sim, incorrect_points=incorrect, band=_band(sim))
