from __future__ import annotations

import numpy as np

# reference poses are hand-authored as normalized landmarks:
#   wrist at origin, palm span (wrist -> middle-MCP/landmark 9) scaled to 1.
# keep these in sync with frontend/modules/mirror/gestures.ts.

# open flat palm, fingers extended upward — greeting wave frozen at mid-swing.
HELLO = np.array(
    [
        [0.00, 0.00, 0.00],   # 0  wrist
        [-0.25, -0.15, 0.00], # 1  thumb cmc
        [-0.55, -0.30, -0.02],# 2  thumb mcp
        [-0.75, -0.50, -0.04],# 3  thumb ip
        [-0.90, -0.70, -0.06],# 4  thumb tip
        [-0.22, -0.95, 0.00], # 5  index mcp
        [-0.22, -1.30, 0.00], # 6  index pip
        [-0.22, -1.55, 0.00], # 7  index dip
        [-0.22, -1.75, 0.00], # 8  index tip
        [0.00, -1.00, 0.00],  # 9  middle mcp (palm span anchor)
        [0.00, -1.40, 0.00],  # 10 middle pip
        [0.00, -1.65, 0.00],  # 11 middle dip
        [0.00, -1.85, 0.00],  # 12 middle tip
        [0.22, -0.95, 0.00],  # 13 ring mcp
        [0.22, -1.30, 0.00],  # 14 ring pip
        [0.22, -1.55, 0.00],  # 15 ring dip
        [0.22, -1.72, 0.00],  # 16 ring tip
        [0.40, -0.90, 0.00],  # 17 pinky mcp
        [0.40, -1.20, 0.00],  # 18 pinky pip
        [0.40, -1.40, 0.00],  # 19 pinky dip
        [0.40, -1.55, 0.00],  # 20 pinky tip
    ],
    dtype=np.float32,
)

# "w" shape — index, middle, ring extended; thumb tucks across palm; pinky curls.
WATER = np.array(
    [
        [0.00, 0.00, 0.00],
        [-0.20, -0.20, 0.00],
        [-0.10, -0.40, -0.05],
        [0.00, -0.50, -0.10],
        [0.10, -0.60, -0.15],
        [-0.22, -0.95, 0.00],
        [-0.22, -1.30, 0.00],
        [-0.22, -1.55, 0.00],
        [-0.22, -1.75, 0.00],
        [0.00, -1.00, 0.00],
        [0.00, -1.40, 0.00],
        [0.00, -1.65, 0.00],
        [0.00, -1.85, 0.00],
        [0.22, -0.95, 0.00],
        [0.22, -1.30, 0.00],
        [0.22, -1.55, 0.00],
        [0.22, -1.72, 0.00],
        [0.40, -0.90, 0.00],
        [0.35, -0.70, -0.10],
        [0.25, -0.55, -0.15],
        [0.15, -0.55, -0.18],
    ],
    dtype=np.float32,
)


GESTURES: dict[str, np.ndarray] = {
    "hello": HELLO,
    "water": WATER,
}


def get_reference(gesture_id: str) -> np.ndarray:
    try:
        return GESTURES[gesture_id]
    except KeyError as exc:
        raise KeyError(f"no reference gesture registered for '{gesture_id}'") from exc
