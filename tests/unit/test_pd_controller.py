"""Unit tests for the PD joint controller (pure numpy, sim-agnostic)."""
import numpy as np
import pytest

from x2_locomotion.control.pd import PDController


def ctrl(n=3, kp=100.0, kd=5.0, eff=50.0):
    return PDController(kp=np.full(n, kp), kd=np.full(n, kd), effort_limit=np.full(n, eff))


def test_zero_torque_at_target_and_rest():
    c = ctrl()
    tau = c.torque(q=np.zeros(3), qdot=np.zeros(3), q_des=np.zeros(3))
    np.testing.assert_allclose(tau, 0.0)


def test_proportional_to_position_error():
    c = ctrl(kp=100.0, kd=0.0)
    tau = c.torque(q=np.zeros(3), qdot=np.zeros(3), q_des=np.full(3, 0.1))
    np.testing.assert_allclose(tau, 10.0)  # 100 * 0.1


def test_damping_opposes_velocity():
    c = ctrl(kp=0.0, kd=5.0)
    tau = c.torque(q=np.zeros(3), qdot=np.full(3, 2.0), q_des=np.zeros(3))
    np.testing.assert_allclose(tau, -10.0)  # -5 * 2


def test_torque_clamped_to_effort_limit():
    c = ctrl(kp=10000.0, kd=0.0, eff=50.0)
    tau = c.torque(q=np.zeros(3), qdot=np.zeros(3), q_des=np.full(3, 1.0))
    assert np.all(np.abs(tau) <= 50.0 + 1e-9)
    np.testing.assert_allclose(tau, 50.0)


def test_desired_velocity_term():
    c = ctrl(kp=0.0, kd=5.0)
    tau = c.torque(q=np.zeros(3), qdot=np.zeros(3), q_des=np.zeros(3),
                   qdot_des=np.full(3, 1.0))
    np.testing.assert_allclose(tau, 5.0)  # -kd*(0 - 1) = +5


def test_shape_mismatch_raises():
    c = ctrl(n=3)
    with pytest.raises(ValueError):
        c.torque(q=np.zeros(2), qdot=np.zeros(2), q_des=np.zeros(2))
