from __future__ import annotations

# parity check between this catalog and frontend/modules/mirror/gestures.ts.
# both files compose every reference from the same thumb + finger state
# recipes, and the comment in each says "keep perfectly in sync" — this test
# makes that a contract instead of a hope. it parses the ts recipe table,
# rebuilds each gesture through the python composer, and compares the result
# numerically against the python catalog. the thumb-state table and finger
# base positions are compared value-for-value as well.
#
# not covered: the finger-state vectors inside the ts `finger()` switch — they
# are expressions, not literals. a drift there shows up here anyway the moment
# any recipe uses the edited state, via the numeric rebuild.
#
# run directly (`python3 tests/test_catalog_parity.py` from /vision) or via
# pytest. skips quietly when the frontend isn't checked out next to /vision.

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from vision.gestures.samples import _FINGER_BASE_X, _THUMB_STATES, _build, GESTURES

_GESTURES_TS = Path(__file__).resolve().parents[2] / "frontend" / "modules" / "mirror" / "gestures.ts"

_RECIPE_LINE = re.compile(
    r'^\s*(\w+):\s*build\(\s*"(\w+)",\s*\[([^\]]*)\]\s*(?:,\s*(-?\d*\.?\d+))?\s*\),?\s*$'
)
_THUMB_LINE = re.compile(r"^\s*(\w+):\s*(\[\[.*\]\]),?\s*$")


def _frontend_source() -> str | None:
    if not _GESTURES_TS.exists():
        return None
    return _GESTURES_TS.read_text()


def _parse_recipes(source: str) -> dict[str, tuple[str, tuple[str, ...], float]]:
    recipes: dict[str, tuple[str, tuple[str, ...], float]] = {}
    for line in source.splitlines():
        match = _RECIPE_LINE.match(line)
        if not match:
            continue
        gesture_id, thumb, fingers_raw, z_shift = match.groups()
        fingers = tuple(s.strip().strip('"') for s in fingers_raw.split(","))
        recipes[gesture_id] = (thumb, fingers, float(z_shift) if z_shift else 0.0)
    return recipes


def _parse_thumb_states(source: str) -> dict[str, list[list[float]]]:
    block = source.split("const THUMB_STATES", 1)[1].split("};", 1)[0]
    states: dict[str, list[list[float]]] = {}
    for line in block.splitlines():
        match = _THUMB_LINE.match(line)
        if match:
            states[match.group(1)] = json.loads(match.group(2))
    return states


def test_catalogs_agree() -> None:
    source = _frontend_source()
    if source is None:
        print("skipped — frontend checkout not found next to /vision")
        return

    recipes = _parse_recipes(source)
    assert len(recipes) >= 50, f"only parsed {len(recipes)} recipes — ts format changed?"

    ts_ids = set(recipes)
    py_ids = set(GESTURES)
    assert ts_ids == py_ids, (
        f"id drift — only in ts: {sorted(ts_ids - py_ids)}, "
        f"only in python: {sorted(py_ids - ts_ids)}"
    )

    # rebuild each ts recipe with the python composer; any difference in thumb
    # state, finger states, or z_shift lands here as a numeric mismatch.
    for gesture_id, (thumb, fingers, z_shift) in recipes.items():
        rebuilt = _build(thumb, fingers, z_shift=z_shift)
        assert np.allclose(rebuilt, GESTURES[gesture_id], atol=1e-6), (
            f"{gesture_id}: ts recipe {thumb}/{fingers}/z={z_shift} "
            "builds a different vector than the python catalog"
        )


def test_state_libraries_agree() -> None:
    source = _frontend_source()
    if source is None:
        print("skipped — frontend checkout not found next to /vision")
        return

    base_x = re.search(r"FINGER_BASE_X\s*=\s*(\[[^\]]+\])", source)
    assert base_x, "couldn't find FINGER_BASE_X in gestures.ts"
    assert tuple(json.loads(base_x.group(1))) == _FINGER_BASE_X, "finger base x drifted"

    ts_thumbs = _parse_thumb_states(source)
    assert set(ts_thumbs) == set(_THUMB_STATES), (
        f"thumb state names drifted: {sorted(set(ts_thumbs) ^ set(_THUMB_STATES))}"
    )
    for name, points in _THUMB_STATES.items():
        assert np.allclose(ts_thumbs[name], points, atol=1e-6), f"thumb state '{name}' drifted"


if __name__ == "__main__":
    test_catalogs_agree()
    print("ok — test_catalogs_agree")
    test_state_libraries_agree()
    print("ok — test_state_libraries_agree")
    print(f"\ncatalogs in sync — {len(GESTURES)} gestures, both sides.")
