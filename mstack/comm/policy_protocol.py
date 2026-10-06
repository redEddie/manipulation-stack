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
ACTION_TYPES = (JOINT_ABSOLUTE, EE_DELTA)


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
    return np.frombuffer(buf, dtype=v["dtype"]).reshape(v["shape"])


def decode_observation(raw: dict) -> dict:
    """Request JSON -> {key: np.ndarray | value}; encoded images become uint8 arrays."""
    obs = {}
    for k, v in raw.items():
        obs[k] = decode_image(v) if isinstance(v, dict) and "base64" in v else v
    return obs


def image_key(camera: str) -> str:
    return IMAGE_KEY_PREFIX + camera
