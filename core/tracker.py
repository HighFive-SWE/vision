from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

try:
    import mediapipe as mp
except ImportError:  # mediapipe is optional server-side if the frontend tracks.
    mp = None  # type: ignore[assignment]


@dataclass
class TrackedHand:
    landmarks: np.ndarray  # (21, 3)
    handedness: str  # "Left" | "Right"
    score: float


class HandTracker:
    """
    thin wrapper around mediapipe hands for server-side or offline use.
    the frontend runs the same model in-browser via @mediapipe/tasks-vision —
    this class exists for tooling, batch eval, and tests.
    """

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
    ) -> None:
        if mp is None:
            raise RuntimeError(
                "mediapipe is not installed — install vision/requirements.txt to use HandTracker."
            )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def detect(self, frame_rgb: np.ndarray) -> list[TrackedHand]:
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return []

        hands: list[TrackedHand] = []
        for lm, meta in zip(
            results.multi_hand_landmarks,
            results.multi_handedness or [],
            strict=False,
        ):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            label = meta.classification[0].label if meta.classification else "Right"
            score = float(meta.classification[0].score) if meta.classification else 0.0
            hands.append(TrackedHand(landmarks=pts, handedness=label, score=score))
        return hands

    def stream(self, frames: Iterator[np.ndarray]) -> Iterator[list[TrackedHand]]:
        for frame in frames:
            yield self.detect(frame)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
