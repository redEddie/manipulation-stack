"""HTTP policy protocol shared by the policy server (GPU machine) and the policy
client (FR3 controller computer).

Both ends import this module, so the wire format is defined once. It must stay
importable from either venv: standard library + numpy only, no mstack imports.

Layering (see docs/policy-server.md):

    policy server (apps/policy_server.py)       model + its pre/post-processing
        |  HTTP JSON -- this module
    policy client (apps/fr3_policy_client.py)   timing, chunk scheduling, safety clamp
        |  ZMQ
    robot node / camera node                    pylibfranka, RealSense

Endpoints:

    GET  /info     -> PolicyInfo as JSON. What the loaded checkpoint expects.
    POST /reset    {"instruction": str}             -> {"status": "ok", "instruction": str}
    POST /predict  {observation}                    -> {"actions": [[action_dim] x chunk]}
    POST /step     {"observation.state": [8]}       -> {"action": [8]}   (server-side step)

Observation keys:

    "observation.state"           list[state_dim]   joint1-7 rad + gripper 0..1
    "observation.images.<camera>" encode_image(...)  RGB uint8, image_size^2, already
                                                     square-cropped by the client with the
                                                     same resize_rgb used to write the data

Servers written before /info existed answer it with 404/405/501 (a bare
http.server handler without do_GET says 501); the client then falls back to
LEGACY_INFO, which is what those servers were built for.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field

import numpy as np

STATE_KEY = "observation.state"
IMAGE_KEY_PREFIX = "observation.images."

# Action conventions a client knows how to execute.
JOINT_ABSOLUTE = "joint_absolute"   # [action_dim=8] joint1-7 rad + gripper 0..1
EE_DELTA = "ee_delta"               # [action_dim=7] EE-frame delta, client re-anchors + IK
EE_RELATIVE_WAYPOINT = "ee_relative_waypoint"
ACTION_TYPES = (JOINT_ABSOLUTE, EE_DELTA, EE_RELATIVE_WAYPOINT)


@dataclass(frozen=True)
class ActionSpec:
    """Versioned meaning of *postprocessed* actions, independent of model family.

    Waypoint i is relative to the single observation EE anchor, NOT waypoint i-1.
    Translation and rotation are expressed in that anchor's axes. Rotations use
    R_target = R_anchor @ Exp(rotvec * rotation_scale). Joint actions are radians
    in FR3 joint1..7 order. Gripper is last, with 0=open and 1=closed after decoding.
    action_dt is the training action grid, independent of inference arrival rate.
    """

    kind: str
    action_dt: float
    version: int = 1
    first_action_offset: float = 0.0
    reference_frame: str = "joint"
    rotation: str = "none"
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    gripper_encoding: str = "zero_one"
    joint_order: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))

    @property
    def action_dim(self) -> int:
        return 8 if self.kind == JOINT_ABSOLUTE else 7

    def validate(self) -> None:
        if self.version != 1 or self.kind not in (JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT):
            raise ValueError("QP requires ActionSpec v1 joint_absolute or ee_relative_waypoint; "
                             "legacy ee_delta is a different action meaning")
        values = [self.action_dt, self.first_action_offset, self.position_scale, self.rotation_scale]
        if not np.isfinite(values).all() or self.action_dt <= 0 or self.first_action_offset < 0:
            raise ValueError("ActionSpec timing and scales must be finite, action_dt > 0, offset >= 0")
        if self.position_scale <= 0 or self.rotation_scale <= 0:
            raise ValueError("ActionSpec scales must be positive")
        if self.gripper_encoding not in ("zero_one", "minus_one_one"):
            raise ValueError("unsupported gripper encoding")
        if tuple(self.joint_order) != tuple(f"joint{i}" for i in range(1, 8)):
            raise ValueError("FR3 requires joint1..joint7 order")
        expected = ("joint", "none") if self.kind == JOINT_ABSOLUTE else ("observation_ee", "rotvec")
        if (self.reference_frame, self.rotation) != expected:
            raise ValueError(f"{self.kind} requires frame/rotation {expected}")
        if self.kind == JOINT_ABSOLUTE and (self.position_scale != 1 or self.rotation_scale != 1):
            raise ValueError("absolute joints must already be in radians (unit scales)")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "ActionSpec":
        if not isinstance(raw, dict):
            raise ValueError("ActionSpec must be an object")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown ActionSpec fields: {sorted(unknown)}")
        values = dict(raw)
        if "joint_order" in values:
            values["joint_order"] = tuple(values["joint_order"])
        spec = cls(**values)
        spec.validate()
        return spec

    def decode(self, actions) -> tuple[np.ndarray, np.ndarray]:
        self.validate()
        a = np.asarray(actions, dtype=float)
        if a.ndim != 2 or not 0 < len(a) <= 4096 or a.shape[1] != self.action_dim or not np.isfinite(a).all():
            raise ValueError(f"expected finite action chunk (1..4096, {self.action_dim})")
        grip = a[:, -1].copy()
        if self.gripper_encoding == "minus_one_one":
            grip = (grip + 1) / 2
        # Model regressors can slightly exceed their training range. Only the
        # bounded hand channel saturates; joint/pose targets remain objectives.
        grip = np.clip(grip, 0, 1)
        target = a[:, :-1].copy()
        if self.kind == EE_RELATIVE_WAYPOINT:
            target[:, :3] *= self.position_scale
            target[:, 3:] *= self.rotation_scale
        if not np.isfinite(target).all():
            raise ValueError("scaled action targets must be finite")
        return target, grip



@dataclass
class PolicyInfo:
    """What a loaded checkpoint expects. The server reports it; the client obeys it.

    fps is the rate the *training data* was recorded at -- chunk index i means
    observation time + i/fps. Running a policy at any other rate replays its chunk
    faster or slower than it was trained to.
    """

    policy: str                     # e.g. "smolvla", "pi0", "groot", "mamba-vlajepa"
    checkpoint: str
    fps: int
    image_size: int                 # square side the client must send
    cameras: list[str]              # camera roles, e.g. ["agent", "wrist"]
    state_dim: int
    action_type: str
    action_dim: int
    chunk_size: int                 # actions returned per /predict
    extra: dict = field(default_factory=dict)
    action_spec: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyInfo":
        names = {f for f in cls.__dataclass_fields__}
        info = cls(**{k: v for k, v in d.items() if k in names})
        info.validate()
        return info

    def validate(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise ValueError(f"unknown action_type {self.action_type!r}; expected one of {ACTION_TYPES}")
        expected_dim = 8 if self.action_type == JOINT_ABSOLUTE else 7
        if self.action_dim != expected_dim:
            raise ValueError(f"{self.action_type} needs action_dim {expected_dim}, got {self.action_dim}")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("training fps must be finite and positive")
        if self.action_spec is not None:
            spec = ActionSpec.from_dict(self.action_spec)
            if spec.kind != self.action_type or spec.action_dim != self.action_dim:
                raise ValueError("ActionSpec disagrees with action_type/action_dim")
            if not np.isclose(spec.action_dt, 1.0 / self.fps, rtol=1e-6):
                raise ValueError("ActionSpec action_dt disagrees with training fps")
        if self.fps <= 0 or self.image_size <= 0 or self.chunk_size <= 0:
            raise ValueError(f"fps/image_size/chunk_size must be positive: {self}")


# What servers without /info (mamba-embeddingvla real_deploy) were built for:
# 20 Hz data, 256^2 client-side resize, 10-step chunks. action_type/dim are
# read from the first chunk instead (7 = EE-delta, 8 = joint-absolute).
LEGACY_INFO = dict(policy="legacy", checkpoint="?", fps=20, image_size=256,
                   cameras=["agent", "wrist"], state_dim=8, chunk_size=10)


def encode_image(img: np.ndarray) -> dict:
    """uint8 HxWx3 -> JSON-safe dict (raw bytes, base64). ~0.15 MB at 224^2."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return {"base64": base64.b64encode(img.tobytes()).decode(), "shape": list(img.shape), "dtype": "uint8"}


def decode_image(v: dict) -> np.ndarray:
    buf = base64.b64decode(v["base64"])
    # copy: frombuffer is read-only, and torch.from_numpy warns about that on every request
    return np.frombuffer(buf, dtype=v["dtype"]).reshape(v["shape"]).copy()


def decode_observation(raw: dict) -> dict:
    """Request JSON -> {key: np.ndarray | value}; encoded images become uint8 arrays."""
    obs = {}
    for k, v in raw.items():
        obs[k] = decode_image(v) if isinstance(v, dict) and "base64" in v else v
    return obs


def image_key(camera: str) -> str:
    return IMAGE_KEY_PREFIX + camera
