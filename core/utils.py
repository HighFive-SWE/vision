from __future__ import annotations

import numpy as np

# mediapipe returns 21 points per hand. we keep that contract explicit.
LANDMARK_COUNT = 21


def _as_array(landmarks: list[list[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim != 2 or arr.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"expected ({LANDMARK_COUNT}, 3) landmarks, got {arr.shape}")
    return arr


def normalize_landmarks(landmarks: list[list[float]] | np.ndarray) -> np.ndarray:
    """
    translate relative to wrist, then scale so the palm span is 1.
    output is position- and scale-invariant — identical gesture from any distance
    should map to (almost) the same vector.
    """
    pts = _as_array(landmarks)
    wrist = pts[0]
    centered = pts - wrist

    # palm span = wrist → middle-finger MCP (landmark 9). non-zero in any real hand.
    palm_span = float(np.linalg.norm(centered[9]))
    if palm_span < 1e-6:
        return centered
    return centered / palm_span


def to_vector(normalized: np.ndarray) -> np.ndarray:
    """flatten a normalized (21, 3) landmark array into a 63-d gesture vector."""
    return normalized.reshape(-1).astype(np.float32)


def mirror_horizontal(landmarks: np.ndarray) -> np.ndarray:
    """flip x so a right-handed reference compares fairly against a left-handed user."""
    flipped = landmarks.copy()
    flipped[:, 0] *= -1.0
    return flipped
