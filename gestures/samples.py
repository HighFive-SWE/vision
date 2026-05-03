from __future__ import annotations

import numpy as np

# reference poses are composed from a small library of thumb + finger states
# so every gesture shares the same skeleton. keep these perfectly in sync with
# frontend/modules/mirror/gestures.ts — same ids, same vectors.

# landmark 0 is the wrist. every reference is wrist-centered and palm-span
# (wrist -> middle-MCP/landmark 9) scaled to 1.

_FINGER_BASE_X: tuple[float, float, float, float] = (-0.22, 0.00, 0.22, 0.40)

# thumb: 4 points — cmc, mcp, ip, tip (landmarks 1..4).
_THUMB_STATES: dict[str, list[list[float]]] = {
    "side":         [[-0.25, -0.15, 0.00], [-0.55, -0.30, -0.02], [-0.75, -0.50, -0.04], [-0.90, -0.70, -0.06]],
    "across":       [[-0.20, -0.20, 0.00], [-0.10, -0.40, -0.05], [0.00, -0.50, -0.10],  [0.10, -0.60, -0.15]],
    "up":           [[-0.20, -0.25, 0.00], [-0.15, -0.55, -0.02], [-0.12, -0.80, -0.02], [-0.10, -1.00, -0.02]],
    "between":      [[-0.20, -0.30, 0.00], [-0.15, -0.55, -0.05], [-0.10, -0.75, -0.05], [-0.08, -0.90, -0.05]],
    "pinch_index":  [[-0.20, -0.25, 0.00], [-0.25, -0.50, -0.10], [-0.25, -0.75, -0.15], [-0.22, -0.90, -0.20]],
    "touch_middle": [[-0.15, -0.35, -0.02], [-0.05, -0.60, -0.05], [0.05, -0.80, -0.10],  [0.15, -0.95, -0.12]],
    "corner":       [[-0.30, -0.20, 0.00], [-0.55, -0.30, -0.02], [-0.75, -0.40, -0.04], [-0.95, -0.55, -0.06]],
    "inline":       [[-0.20, -0.20, 0.00], [-0.35, -0.30, -0.02], [-0.45, -0.50, -0.04], [-0.50, -0.70, -0.06]],
}


def _finger(base_x: float, state: str) -> list[list[float]]:
    if state == "ext":
        return [
            [base_x, -0.95, 0.00],
            [base_x, -1.30, 0.00],
            [base_x, -1.55, 0.00],
            [base_x, -1.75, 0.00],
        ]
    if state == "half":
        return [
            [base_x, -0.95, 0.00],
            [base_x, -1.20, -0.10],
            [base_x, -1.20, -0.30],
            [base_x, -1.10, -0.40],
        ]
    if state == "fold":
        return [
            [base_x, -0.95, 0.00],
            [base_x, -1.10, -0.10],
            [base_x, -1.00, -0.25],
            [base_x, -0.85, -0.30],
        ]
    if state == "tip_in":
        return [
            [base_x, -0.95, 0.00],
            [base_x, -1.25, -0.05],
            [base_x * 0.5, -1.40, -0.15],
            [0.00, -1.50, -0.20],
        ]
    if state == "bent_tip":
        return [
            [base_x, -0.95, 0.00],
            [base_x, -1.25, 0.00],
            [base_x * 1.1, -1.45, -0.10],
            [base_x * 1.2, -1.50, -0.20],
        ]
    raise ValueError(f"unknown finger state: {state}")


def _build(
    thumb: str,
    fingers: tuple[str, str, str, str],
    *,
    z_shift: float = 0.0,
) -> np.ndarray:
    pts: list[list[float]] = [[0.0, 0.0, 0.0]]
    pts.extend([p[:] for p in _THUMB_STATES[thumb]])
    for base_x, state in zip(_FINGER_BASE_X, fingers):
        pts.extend(_finger(base_x, state))
    arr = np.array(pts, dtype=np.float32)
    if z_shift != 0.0:
        # tilt the whole hand in depth. wrist stays at origin so normalization
        # still lines up user + reference.
        arr[1:, 2] += z_shift
    return arr


GESTURES: dict[str, np.ndarray] = {
    # phase 6 originals
    "hello":     _build("side",         ("ext",      "ext",      "ext",      "ext")),
    "thank_you": _build("side",         ("ext",      "ext",      "ext",      "ext"),      z_shift=-0.35),
    "please":    _build("side",         ("half",     "half",     "half",     "half")),
    "sorry":     _build("across",       ("fold",     "fold",     "fold",     "fold")),
    "water":     _build("across",       ("ext",      "ext",      "ext",      "fold")),
    "food":      _build("pinch_index",  ("tip_in",   "tip_in",   "tip_in",   "tip_in")),
    "help":      _build("up",           ("fold",     "fold",     "fold",     "fold")),
    "stop":      _build("inline",       ("ext",      "ext",      "ext",      "ext"),      z_shift=0.35),
    "yes":       _build("across",       ("fold",     "fold",     "fold",     "fold"),     z_shift=0.20),
    "no":        _build("touch_middle", ("ext",      "ext",      "fold",     "fold")),
    "bathroom":  _build("between",      ("fold",     "fold",     "fold",     "fold")),
    "pain":      _build("across",       ("ext",      "fold",     "fold",     "fold")),
    "tired":     _build("across",       ("bent_tip", "bent_tip", "bent_tip", "bent_tip")),
    "play":      _build("corner",       ("fold",     "fold",     "fold",     "ext")),
    "sleep":     _build("inline",       ("half",     "half",     "half",     "half"),     z_shift=-0.25),
    # phase 8 expansion
    "drink":     _build("side",         ("half",     "half",     "fold",     "fold"),     z_shift=0.15),
    "eat":       _build("inline",       ("half",     "half",     "half",     "half")),
    "friend":    _build("across",       ("ext",      "fold",     "fold",     "fold"),     z_shift=-0.15),
    "family":    _build("pinch_index",  ("fold",     "fold",     "fold",     "fold"),     z_shift=0.10),
    "doctor":    _build("touch_middle", ("ext",      "fold",     "fold",     "fold"),     z_shift=0.15),
    "school":    _build("side",         ("ext",      "ext",      "ext",      "ext"),      z_shift=0.15),
    "home":      _build("pinch_index",  ("tip_in",   "tip_in",   "tip_in",   "tip_in"),   z_shift=0.15),
    "wait":      _build("side",         ("half",     "half",     "half",     "half"),     z_shift=0.20),
    "come":      _build("across",       ("ext",      "fold",     "fold",     "fold"),     z_shift=0.25),
    "go":        _build("up",           ("ext",      "fold",     "fold",     "fold")),
    "more":      _build("pinch_index",  ("tip_in",   "tip_in",   "tip_in",   "tip_in"),   z_shift=-0.15),
    "finished":  _build("side",         ("ext",      "ext",      "ext",      "ext"),      z_shift=-0.15),
    # asl fingerspelling — 26-letter alphabet. shapes are composed from the
    # same thumb + finger state library; z_shifts disambiguate letters that
    # share a finger configuration but differ in palm tilt (e.g. e/yes,
    # w/water, o/food).
    "letter_a":  _build("up",           ("fold",     "fold",     "fold",     "fold"),     z_shift=-0.10),
    "letter_b":  _build("across",       ("ext",      "ext",      "ext",      "ext")),
    "letter_c":  _build("side",         ("half",     "half",     "half",     "half"),     z_shift=-0.10),
    "letter_d":  _build("pinch_index",  ("ext",      "fold",     "fold",     "fold")),
    "letter_e":  _build("across",       ("fold",     "fold",     "fold",     "fold"),     z_shift=-0.10),
    "letter_f":  _build("pinch_index",  ("fold",     "ext",      "ext",      "ext")),
    "letter_g":  _build("up",           ("ext",      "fold",     "fold",     "fold"),     z_shift=0.20),
    "letter_h":  _build("inline",       ("ext",      "ext",      "fold",     "fold")),
    "letter_i":  _build("across",       ("fold",     "fold",     "fold",     "ext")),
    "letter_j":  _build("across",       ("fold",     "fold",     "fold",     "ext"),      z_shift=0.15),
    "letter_k":  _build("between",      ("ext",      "half",     "fold",     "fold")),
    "letter_l":  _build("up",           ("ext",      "fold",     "fold",     "fold"),     z_shift=-0.05),
    "letter_m":  _build("between",      ("fold",     "fold",     "fold",     "fold"),     z_shift=0.10),
    "letter_n":  _build("between",      ("fold",     "fold",     "fold",     "fold"),     z_shift=-0.10),
    "letter_o":  _build("pinch_index",  ("tip_in",   "tip_in",   "tip_in",   "tip_in"),   z_shift=0.05),
    "letter_p":  _build("between",      ("ext",      "half",     "fold",     "fold"),     z_shift=-0.25),
    "letter_q":  _build("up",           ("ext",      "fold",     "fold",     "fold"),     z_shift=-0.20),
    "letter_r":  _build("across",       ("ext",      "half",     "fold",     "fold")),
    "letter_s":  _build("across",       ("fold",     "fold",     "fold",     "fold"),     z_shift=0.10),
    "letter_t":  _build("between",      ("fold",     "fold",     "fold",     "fold"),     z_shift=0.25),
    "letter_u":  _build("across",       ("ext",      "ext",      "fold",     "fold")),
    "letter_v":  _build("across",       ("ext",      "ext",      "fold",     "fold"),     z_shift=0.15),
    "letter_w":  _build("across",       ("ext",      "ext",      "ext",      "fold"),     z_shift=0.10),
    "letter_x":  _build("across",       ("bent_tip", "fold",     "fold",     "fold")),
    "letter_y":  _build("corner",       ("fold",     "fold",     "fold",     "ext"),      z_shift=0.05),
    "letter_z":  _build("across",       ("ext",      "fold",     "fold",     "fold"),     z_shift=0.05),
}


def get_reference(gesture_id: str) -> np.ndarray:
    try:
        return GESTURES[gesture_id]
    except KeyError as exc:
        raise KeyError(f"no reference gesture registered for '{gesture_id}'") from exc
