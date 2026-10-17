"""Timestamped targets; chunk replacement changes only the QP objective."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from mstack.comm.policy_protocol import ActionSpec, JOINT_ABSOLUTE
from mstack.robots.fr3_kinematics import fk


@dataclass(frozen=True)
class Reference:
    kind: str
    targets: np.ndarray
    gripper: np.ndarray


class TimedChunk:
    def __init__(self, spec: ActionSpec, actions, observation_time: float, anchor_q,
                 sequence: int, epoch: int):
        target, self.gripper = spec.decode(actions)
        anchor = np.asarray(anchor_q, dtype=float)
        if anchor.shape != (7,) or not np.isfinite(anchor).all():
            raise ValueError("chunk requires the seven measured joints at observation time")
        if not np.isfinite(observation_time) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("invalid observation timestamp/sequence")
        self.spec, self.sequence, self.epoch = spec, sequence, epoch
        self.observation_time = float(observation_time)
        self.start = self.observation_time + spec.first_action_offset
        # The last sample is valid for one action interval; do not replay expired
        # chunks indefinitely by clamping their index to the last waypoint.
        self.end = self.start + len(target) * spec.action_dt
        self.times = self.start + np.arange(len(target)) * spec.action_dt
        if spec.kind == JOINT_ABSOLUTE:
            self.targets = target
            self.rotations = None
        else:
            T = fk(anchor)
            self.targets = target[:, :3] @ T[:3, :3].T + T[:3, 3]
            self.rotations = Rotation.from_matrix(T[:3, :3]) * Rotation.from_rotvec(target[:, 3:])
            self.slerp = Slerp(self.times, self.rotations) if len(target) > 1 else None

    def sample(self, times) -> Reference:
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("reference sample times must be a finite vector")
        # Horizon preview may hold the last target; the manager stops replanning
        # at end, leaving the already validated terminal braking trajectory.
        t = np.clip(times, self.times[0], self.times[-1])
        p = np.column_stack([np.interp(t, self.times, self.targets[:, j])
                             for j in range(self.targets.shape[1])])
        ix = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.gripper)-1)
        if self.spec.kind != JOINT_ABSOLUTE:
            pose = np.tile(np.eye(4), (len(t), 1, 1))
            pose[:, :3, 3] = p
            pose[:, :3, :3] = (self.slerp(t).as_matrix() if self.slerp is not None
                                else self.rotations.as_matrix()[0])
            p = pose
        return Reference(self.spec.kind, p, self.gripper[ix])


class ChunkManager:
    def __init__(self, spec: ActionSpec, epoch: int):
        spec.validate()
        self.spec, self.epoch = spec, epoch
        self.current = None
        self.last_sequence = -1

    def submit(self, actions, observation_time, anchor_q, sequence, epoch, now) -> bool:
        chunk = TimedChunk(self.spec, actions, observation_time, anchor_q, sequence, epoch)
        if (epoch != self.epoch or sequence <= self.last_sequence or chunk.end <= now
                or observation_time > now + 0.002):
            return False
        if self.current is not None and observation_time < self.current.observation_time:
            return False
        self.current = chunk
        self.last_sequence = sequence
        return True

    def reference(self, times, now):
        if self.current is None or now >= self.current.end or now < self.current.start:
            return None
        return self.current.sample(times)
