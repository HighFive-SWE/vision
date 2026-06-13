# vision

mediapipe-based hand tracking and gesture comparison.

runs in two places:

- **browser**: `@mediapipe/tasks-vision` inside the frontend's practice
  pages, for low-latency feedback.
- **python**: this package, used by the backend's `/cv/evaluate` endpoint
  and any offline tooling.

## install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

the comparator and the tests only need numpy. mediapipe and opencv are
only required for `core/tracker.py`.

## layout

```
core/
  tracker.py      mediapipe hands wrapper (batch and streaming)
  utils.py        normalize_landmarks, to_vector, mirror_horizontal
  comparator.py   compare_gesture -> { accuracy, incorrect_points, band }
gestures/
  samples.py      hand-authored reference poses (45 words, 26 letters)
tests/
  test_comparator.py      golden-frame checks over every reference
  test_catalog_parity.py  ts/python catalog drift contract
```

## contract

`normalize_landmarks` expects `(21, 3)`, the mediapipe hand landmark
layout, and returns landmarks translated to the wrist with palm span
scaled to 1.

`compare_gesture` returns accuracy in `[0, 1]`. bands: `correct >= 0.60`,
`partial 0.40 to 0.60`, `incorrect < 0.40` (phase 6 recalibration, see
`core/comparator.py`). a mirrored reference is also tried so left-handed
signers score fairly against right-handed references.

the same normalization and scoring runs in typescript at
`frontend/modules/mirror/comparator.ts`. the two gesture catalogs are kept
identical, and `tests/test_catalog_parity.py` fails if they drift.

## tests

```bash
python3 tests/test_comparator.py
python3 tests/test_catalog_parity.py
```

no test dependencies, plain asserts, also collectable by pytest. the
parity test skips itself when the frontend isn't checked out alongside
this folder.
