"""Unit tests for x2_common.transforms (pure numpy geometry)."""
import numpy as np
import pytest

from x2_common import transforms as tf


def test_identity_quaternion_is_identity_rotation():
    r = tf.quat_to_rotation_matrix(0, 0, 0, 1)
    np.testing.assert_allclose(r, np.eye(3), atol=1e-9)


def test_zero_quaternion_raises():
    with pytest.raises(ValueError):
        tf.quat_to_rotation_matrix(0, 0, 0, 0)


def test_transform_points_translation():
    t = tf.make_transform(np.eye(3), [1.0, 2.0, 3.0])
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = tf.transform_points(t, pts)
    np.testing.assert_allclose(out, [[1, 2, 3], [2, 3, 4]])


def test_transform_points_rotation_90deg_z():
    # 90 deg about z: x->y, y->-x
    r = tf.quat_to_rotation_matrix(0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    t = tf.make_transform(r, [0, 0, 0])
    out = tf.transform_points(t, np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0, 1, 0]], atol=1e-9)


def test_projected_gravity_upright():
    g = tf.projected_gravity(np.eye(3))
    np.testing.assert_allclose(g, [0, 0, -1])


def test_drop_nonfinite():
    pts = np.array([[0, 0, 0], [np.nan, 1, 1], [1, np.inf, 1], [2, 2, 2]])
    out = tf.drop_nonfinite(pts)
    np.testing.assert_allclose(out, [[0, 0, 0], [2, 2, 2]])


def test_drop_nonfinite_empty():
    out = tf.drop_nonfinite(np.empty((0, 3)))
    assert out.shape == (0, 3)


def test_crop_roi():
    pts = np.array([[0.5, 0.0, 0.0], [3.0, 0.0, 0.0], [0.5, 2.0, 0.0]])
    out = tf.crop_roi(pts, (0.0, 2.0), (-0.8, 0.8), (-0.5, 1.0))
    np.testing.assert_allclose(out, [[0.5, 0.0, 0.0]])
