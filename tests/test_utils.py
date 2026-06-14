import numpy as np
import pytest

from vision.core.utils import (
    LANDMARK_COUNT,
    _as_array,
    mirror_horizontal,
    normalize_landmarks,
    to_vector,
)


def make_raw_landmarks(scale=1.0, offset=(0.0, 0.0, 0.0)):
    pts = []
    for i in range(LANDMARK_COUNT):
        pts.append([
            offset[0] + scale * (i * 0.05),
            offset[1] + scale * (-(i * 0.1)),
            offset[2] + scale * (i * 0.01),
        ])
    return pts


# ── _as_array ─────────────────────────────────────────────────────────────────

def test_as_array_wrong_coord_count_raises():
    with pytest.raises(ValueError):
        _as_array([[0.1, 0.2]] * LANDMARK_COUNT)


def test_as_array_wrong_point_count_raises():
    with pytest.raises(ValueError):
        _as_array([[0.1, 0.2, 0.3]] * (LANDMARK_COUNT - 1))


def test_as_array_accepts_list_of_21():
    arr = _as_array([[0.1, 0.2, 0.3]] * LANDMARK_COUNT)
    assert arr.shape == (LANDMARK_COUNT, 3)


def test_as_array_dtype_is_float32():
    arr = _as_array([[0.1, 0.2, 0.3]] * LANDMARK_COUNT)
    assert arr.dtype == np.float32


# ── normalize_landmarks ───────────────────────────────────────────────────────

def test_normalize_wrist_lands_at_origin():
    pts = make_raw_landmarks(offset=(5.0, 3.0, 1.0))
    result = normalize_landmarks(pts)
    np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=1e-5)


def test_normalize_output_shape():
    result = normalize_landmarks(make_raw_landmarks())
    assert result.shape == (LANDMARK_COUNT, 3)


def test_normalize_accepts_numpy_array():
    arr = np.array(make_raw_landmarks(), dtype=np.float32)
    result = normalize_landmarks(arr)
    assert result.shape == (LANDMARK_COUNT, 3)


def test_normalize_wrong_shape_raises():
    with pytest.raises(ValueError):
        normalize_landmarks([[0.1, 0.2, 0.3]] * 10)


def test_normalize_translation_invariant():
    pts_a = make_raw_landmarks(offset=(0.0, 0.0, 0.0))
    pts_b = make_raw_landmarks(offset=(100.0, 50.0, 25.0))
    na = normalize_landmarks(pts_a)
    nb = normalize_landmarks(pts_b)
    np.testing.assert_allclose(na, nb, atol=1e-4)


# ── to_vector ─────────────────────────────────────────────────────────────────

def test_to_vector_output_shape():
    pts = normalize_landmarks(make_raw_landmarks())
    v = to_vector(pts)
    assert v.shape == (63,)


def test_to_vector_dtype_is_float32():
    pts = normalize_landmarks(make_raw_landmarks())
    v = to_vector(pts)
    assert v.dtype == np.float32


def test_to_vector_is_flattened():
    pts = normalize_landmarks(make_raw_landmarks())
    v = to_vector(pts)
    assert v.ndim == 1


# ── mirror_horizontal ─────────────────────────────────────────────────────────

def test_mirror_flips_x_coordinates():
    pts = np.array(make_raw_landmarks(), dtype=np.float32)
    mirrored = mirror_horizontal(pts)
    np.testing.assert_allclose(mirrored[:, 0], -pts[:, 0], atol=1e-6)


def test_mirror_preserves_y_coordinates():
    pts = np.array(make_raw_landmarks(), dtype=np.float32)
    mirrored = mirror_horizontal(pts)
    np.testing.assert_allclose(mirrored[:, 1], pts[:, 1], atol=1e-6)


def test_mirror_preserves_z_coordinates():
    pts = np.array(make_raw_landmarks(), dtype=np.float32)
    mirrored = mirror_horizontal(pts)
    np.testing.assert_allclose(mirrored[:, 2], pts[:, 2], atol=1e-6)


def test_mirror_does_not_mutate_original():
    pts = np.array(make_raw_landmarks(), dtype=np.float32)
    original_x = pts[:, 0].copy()
    mirror_horizontal(pts)
    np.testing.assert_allclose(pts[:, 0], original_x)


def test_mirror_twice_returns_original():
    pts = np.array(make_raw_landmarks(), dtype=np.float32)
    double_mirrored = mirror_horizontal(mirror_horizontal(pts))
    np.testing.assert_allclose(double_mirrored, pts, atol=1e-6)
