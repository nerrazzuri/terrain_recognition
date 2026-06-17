"""Unit tests for the factory cpgwalk adapter's pure logic (no onnxruntime / model needed)."""
import numpy as np

from x2_locomotion.cref import factory_cpgwalk as fc


def test_obs_dims_and_layout():
    obs = fc.build_obs(
        imu_omega=[1, 2, 3], imu_euler=[0.1, 0.2, 0.3], command4=[0.5, 0, 0, 0],
        dof_pos=fc.DEFAULT_DOF_POS, dof_vel=np.zeros(17), prev_action=np.zeros(17),
        cpg_q=[0, 1, 0, -1])
    assert obs.shape == (fc.OBS_DIM,)
    # dof_pos at default -> the pos block is ~zero (float32 rounding)
    assert np.allclose(obs[fc.SL_POS], 0.0, atol=1e-6)
    # command forward scaled by lin_vel (2.0)
    assert np.isclose(obs[6], 0.5 * fc.S_LIN_VEL)
    # cpg phase passes through unscaled
    assert np.allclose(obs[fc.SL_Q], [0, 1, 0, -1])


def test_obs_dof_pos_is_relative_to_default_and_scaled():
    dof = fc.DEFAULT_DOF_POS.copy()
    dof[3] += 0.2  # left_knee offset
    obs = fc.build_obs(np.zeros(3), np.zeros(3), np.zeros(4), dof, np.zeros(17),
                       np.zeros(17), np.zeros(4))
    assert np.isclose(obs[fc.SL_POS][3], 0.2 * fc.S_DOF_POS)


def test_obs_clip():
    obs = fc.build_obs(np.full(3, 1e3), np.zeros(3), np.zeros(4), fc.DEFAULT_DOF_POS,
                       np.zeros(17), np.zeros(17), np.zeros(4))
    assert obs.max() <= fc.OBS_CLIP and obs.min() >= -fc.OBS_CLIP


def test_action_to_targets():
    action = np.zeros(17)
    assert np.allclose(fc.action_to_targets(action), fc.DEFAULT_DOF_POS)
    action = np.ones(17)
    assert np.allclose(fc.action_to_targets(action), fc.DEFAULT_DOF_POS + fc.ACTION_SCALE)


def test_cpg_phase_antiphase():
    q = fc.cpg_phase(0.0)
    assert q.shape == (4,)
    # left (sin,cos) and right (sin,cos) are a half-cycle apart -> opposite signs
    assert np.isclose(q[0], -q[2]) and np.isclose(q[1], -q[3])


def test_joint_order_length():
    assert len(fc.JOINT_ORDER) == fc.ACTION_DIM == 17
    assert fc.ACTION_SCALE.shape == fc.DEFAULT_DOF_POS.shape == (17,)
