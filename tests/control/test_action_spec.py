import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mstack.comm.policy_protocol import ActionSpec, PolicyInfo, JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT
from mstack.robots.fr3_kinematics import fk
from mstack.robots.policy_control.reference import ChunkManager, TimedChunk

Q = np.array([0., -.7, 0., -2., 0., 1.4, .7])


def test_spec_roundtrip_and_no_dimension_guess():
    spec = ActionSpec(JOINT_ABSOLUTE, .05)
    assert ActionSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    info = PolicyInfo('smolvla', 'fixture', 20, 224, ['agent'], 8, JOINT_ABSOLUTE, 8, 50,
                      action_spec=spec.to_dict())
    assert PolicyInfo.from_dict(info.to_dict()).action_spec == spec.to_dict()
    for bad in [dict(kind='ee_delta', action_dt=.05),
                dict(kind=EE_RELATIVE_WAYPOINT, action_dt=.05),
                dict(kind=JOINT_ABSOLUTE, action_dt=0),
                dict(kind=JOINT_ABSOLUTE, action_dt=.05, typo=1),
                dict(kind=JOINT_ABSOLUTE, action_dt=.05, position_scale=.3)]:
        with pytest.raises((ValueError, TypeError)):
            ActionSpec.from_dict(bad)
    with pytest.raises(ValueError):
        spec.decode(np.zeros((10, 7)))
    with pytest.raises(ValueError):
        spec.decode(np.full((10, 8), np.nan))


def test_joint_targets_preserve_posture_and_timing():
    spec = ActionSpec(JOINT_ABSOLUTE, .05)
    actions = np.array([np.r_[Q, 0], np.r_[Q+.01, 1]])
    chunk = TimedChunk(spec, actions, 100., Q, 1, 'epoch')
    sample = chunk.sample(np.array([100., 100.025, 100.05]))
    np.testing.assert_allclose(sample.targets, [Q, Q+.005, Q+.01])
    np.testing.assert_allclose(sample.gripper, [0, 0, 1])


def test_relative_waypoints_use_one_anchor_and_explicit_scales():
    spec = ActionSpec(EE_RELATIVE_WAYPOINT, .05, reference_frame='observation_ee', rotation='rotvec',
                      position_scale=.3, rotation_scale=1.5, gripper_encoding='minus_one_one')
    actions = np.array([[.1, .2, .3, .1, 0, .2, -1], [.1, .2, .3, .1, 0, .2, 1]])
    chunk = TimedChunk(spec, actions, 100., Q, 1, 'e')
    sample = chunk.sample(np.array([100., 100.05]))
    T = fk(Q)
    expected = T.copy()
    expected[:3, 3] += T[:3, :3] @ (actions[0, :3]*.3)
    expected[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(actions[0, 3:6]*1.5).as_matrix()
    np.testing.assert_allclose(sample.targets, [expected, expected], atol=1e-12)
    np.testing.assert_allclose(sample.gripper, [0, 1])


def test_manager_rejects_late_out_of_order_and_wrong_session():
    mgr = ChunkManager(ActionSpec(JOINT_ABSOLUTE, .05), 'e')
    actions = np.tile(np.r_[Q, 0], (10, 1))
    assert mgr.submit(actions, 10., Q, 2, 'e', 10.1)
    assert not mgr.submit(actions, 10., Q, 1, 'e', 10.1)
    assert not mgr.submit(actions, 9., Q, 3, 'e', 10.1)
    assert not mgr.submit(actions, 11., Q, 3, 'e', 10.1)
    assert not mgr.submit(actions, 10., Q, 3, 'old', 10.1)
    assert mgr.reference(np.array([10.6]), 10.5) is None
