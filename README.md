# vision

mediapipe-based hand tracking and gesture comparison.

runs in two places:

- **browser** — `@mediapipe/tasks-vision` inside the frontend's `/mirror` page for low-latency feedback.
- **python** — this module, used by the backend `/cv/evaluate` endpoint and any offline tooling.

## install (python side)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## layout

```
core/
  tracker.py      mediapipe hands wrapper (batch + streaming)
  utils.py        normalize_landmarks, to_vector
  comparator.py   compare_gesture → { accuracy, incorrect_points, band }
gestures/
  samples.py      hand-authored reference poses (hello, water)
```

## contract

`normalize_landmarks` expects `(21, 3)` — the mediapipe hand landmark layout — and returns landmarks translated to the wrist with palm-span scaled to 1.

`compare_gesture` returns accuracy in `[0, 1]`. bands: `correct >= 0.8`, `partial 0.5..0.8`, `incorrect < 0.5`.
