"""LIBERO-format HDF5 writer for real FR3 + GELLO teleop demonstrations.

Schema mirrors what LIBERO/robomimic loaders actually read, confirmed against the
official LIBERO repo's own reading code (``scripts/get_dataset_info.py``,
``libero/libero/utils/dataset_utils.py``) and OpenVLA's
``experiments/robot/libero/regenerate_libero_dataset.py`` (obs key names, 256x256
image convention)::

    <task_name>_demo.hdf5
      data/                         (attrs: env_args=json str, problem_info=json str)
        demo_0/                     (attrs: num_samples=int)
          obs/
            agentview_rgb           (T, H, W, 3) uint8
            eye_in_hand_rgb         (T, H, W, 3) uint8
            joint_states            (T, 7) float32
            gripper_states          (T, 1) float32   -- continuous width, 0=open..1=closed
            ee_states               (T, 6) float32   -- xyz + axis-angle
            ee_pos                  (T, 3) float32
            ee_ori                  (T, 3) float32   -- axis-angle
          actions                   (T, 7) float32   -- normalized [-1, 1]
          rewards                   (T,) float32
          dones                     (T,) float32

One HDF5 file per task/language-instruction, one ``demo_N`` group per episode,
matching the official per-task file layout (e.g.
``turn_on_the_stove_demo.hdf5``).

Deliberately dropped vs. the simulator-generated original: ``states`` and
``model_file`` (full MuJoCo sim state / scene XML -- meaningless for a real
robot) and a genuine ``env_args`` (there is no BDDL scene/controller config to
report). ``env_args``/``problem_info`` are still written as JSON so
``get_dataset_info.py``-style readers do not KeyError, but ``env_args`` is a
real-robot stub, not simulator-replayable metadata.

ACTION SPACE -- read this before trusting a trained policy
------------------------------------------------------------
GELLO teleop here drives the follower in **joint space** (the leader mirrors
the follower's joints 1:1); there is no native commanded end-effector delta.
To stay drop-in compatible with LIBERO-format consumers (which assume a
robosuite ``OSC_POSE`` controller), ``actions`` is reconstructed *after the
fact* from the realized Cartesian trajectory: at frame ``t`` it is the
world-frame delta pose that carried ``ee_pos_quat[t] -> ee_pos_quat[t+1]``,
normalized by the OSC_POSE defaults (``output_max`` = 0.05 m / 0.5 rad),
clipped to [-1, 1]. Gripper: -1=open / +1=close (robosuite Panda convention).
Frame convention and gripper sign are taken from robosuite's documented
defaults, not verified byte-for-byte against an official demo file (repeated
attempts to stream one in this sandbox hit transient network failures --
before training anything on this data, sanity-check a few saved episodes by
eye: gripper sign flips exactly when the operator's trigger did, and replaying
cumulative ``actions`` roughly reproduces ``ee_states``).

Realized-trajectory actions lose the operator's force intent wherever the
follower is in contact: the leader keeps commanding *through* the obstacle
while the realized pose barely moves, so a realized delta goes to ~zero
exactly where the demonstration is pressing. They also break closed-loop
execution outright -- see ``compute_joint_absolute_action`` for the measured
consequences (backward regression at every replan, 1.3-2.0x slowdown). The
``joint_absolute`` space therefore records the **leader's command**, matching
ACT/ALOHA. To keep that intent recoverable for the other spaces too,
every episode ALSO stores the raw teleop command stream (independent of the
selected action space; see ``save_episode``)::

    obs/commanded_joint_states    (T, 7) float32  -- GELLO leader joints (rad),
                                                     the command sent at frame t
    obs/commanded_gripper_states  (T, 1) float32  -- commanded gripper, 0=open..1=closed

Same principle for the robot's own force/torque estimates (2026-08-23) --
franka computes them at 1 kHz anyway, they are tiny (20 floats/frame), and
they are the only record of contact the realized trajectory hides, so they
are written whenever the caller supplies them, independent of the schema.
HDF5-original only; the LeRobot converter does not consume them
(_CONSUMED_OBS_KEYS)::

    obs/joint_torques             (T, 7) float32  -- tau_J, measured joint torque (N*m)
    obs/ext_joint_torques         (T, 7) float32  -- tau_ext_hat_filtered,
                                                     estimated external joint torque (N*m)
    obs/desired_joint_torques     (T, 7) float32  -- tau_J_d, commanded joint torque
                                                     (N*m); measured - desired is the
                                                     tracking error
    obs/ee_wrench                 (T, 6) float32  -- O_F_ext_hat_K, estimated external
                                                     wrench on the EE in base frame O
                                                     [Fx Fy Fz Tx Ty Tz] (N, N*m)
    obs/ee_wrench_ee              (T, 6) float32  -- K_F_ext_hat_K, the same wrench in
                                                     the stiffness frame K (defaults to
                                                     the EE frame)
    obs/joint_contact             (T, 7) float32  -- franka's own contact flag per joint
    obs/cartesian_contact         (T, 6) float32  -- ditto per Cartesian dimension

``scripts/convert/derive_commanded_ee_actions.py`` turns these into commanded EE
delta actions (``actions_ee`` / ``actions_world_cmd``) offline via FR3
forward kinematics -- no extra dependency in the collection loop.
"""

from __future__ import annotations

import fcntl
import json
import zlib
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from mstack.data.dataset_schema import (
    ACTION_SPACE_EE_ABSOLUTE,
    ACTION_SPACE_EE_DELTA,
    ACTION_SPACE_JOINT_ABSOLUTE,
    ACTION_SPACE_JOINT_DELTA,
    FT_OBS_KEYS,
    OBS_AGENTVIEW_RGB,
    OBS_COMMANDED_GRIPPER_STATES,
    OBS_COMMANDED_JOINT_STATES,
    OBS_EE_ORI,
    OBS_EE_POS,
    OBS_EE_STATES,
    OBS_EYE_IN_HAND_RGB,
    OBS_GRIPPER_STATES,
    OBS_JOINT_STATES,
    OBS_JOINT_VELOCITIES,
    REPACK_COUNT_ATTR,
    REPACK_MARKER_ATTR,
    TIMING_ACTION,
    TIMING_FRAME,
    TIMING_GROUP,
    TIMING_ROBOT_STATE,
    DatasetSchemaConfig,
)
from mstack.config.station import load_station
from mstack.data.actions import (
    _quat_to_axis_angle,
    compute_delta_action,
    compute_ee_absolute_action,
    compute_joint_absolute_action,
    compute_joint_delta_action,
)
from mstack.data.crop import (
    default_crop_params,
    load_crop_params,
    resize_rgb,
    save_crop_params,
)
from mstack.data.schema_description import (
    action_column_names,
    describe_episode,
    describe_schema,
    resolved_action_column_names,
)


#: 프레임 청크 하나를 gzip-4 로 압축한다. **모듈 최상단이어야 한다** --
#: ProcessPoolExecutor 가 pickle 로 보내므로 지역 함수나 람다는 못 쓴다.
#: HDF5 의 gzip 필터가 기대하는 것이 정확히 zlib 스트림이라, 여기서 만든
#: 바이트열을 write_direct_chunk 로 그대로 넣으면 create_dataset 이 쓴 것과
#: 같은 파일이 된다 (2026-09-22 왕복 일치 + 파일 크기 동일 확인).
def gzip4(payload: bytes) -> bytes:
    return zlib.compress(payload, 4)


#: 이 프레임 수 아래면 풀을 쓰지 않는다. 프레임당 IPC 왕복이 있어서 짧은
#: 에피소드는 직렬 쪽이 빠르고, 버려진 에피소드(n<2)까지 워커를 깨울 이유가 없다.
_POOL_MIN_FRAMES = 8


def write_image_dataset(grp: h5py.Group, name: str, data: np.ndarray,
                        pool: Any = None) -> None:
    """(T, ...) 이미지 계열을 gzip-4 + 프레임 청크로 쓴다.

    프레임 청크인 이유: h5py 의 guess_chunk() 는 (16, 60, 80, 1) 을 고른다 --
    16프레임 x 60x80 타일 x **채널 1개**. 실측(120프레임)으로 프레임 청크보다
    30% 느리고 22% 크다 (2.91s/57.6MB vs 2.25s/47.2MB). 한 프레임이 한 청크면
    episode_trim 의 청크 자르기도 첫 축이 항상 1 이라 단순해진다.

    shuffle 은 걸지 않는다. 타입 한 원소 **안의** 바이트를 평면별로 모으는
    필터라 1바이트 타입에는 섞을 것이 없고, 실측으로 출력이 바이트까지 같다.

    ``pool`` 을 주면 압축을 워커 프로세스로 돌리고 결과를 원시 청크로 넣는다.
    출력 파일은 직렬 경로와 동일하다 -- 같은 zlib 스트림을 같은 자리에 넣을
    뿐이다. 주지 않으면(테스트·오프라인 도구) 평범하게 create_dataset 한다.
    """
    chunks = (1,) + data.shape[1:]
    if pool is None or data.shape[0] < _POOL_MIN_FRAMES:
        grp.create_dataset(name, data=data, compression="gzip",
                           compression_opts=4, chunks=chunks)
        return

    # 워커가 죽었거나 풀이 닫혔으면 직렬로 되돌린다 -- 압축 하나 때문에
    # 조작자의 에피소드를 잃는 것이 훨씬 나쁘다.
    #
    # **압축을 먼저 하고 데이터셋은 그 뒤에 만든다.** 순서를 뒤집어 만들어
    # 놓고 폴백에서 지우면, HDF5 는 지운 자리를 회수하지 않아 파일이 그만큼
    # 부풀어 오른다 (실측으로 직렬 경로와 크기가 달라졌다).
    try:
        blobs = list(pool.map(gzip4,
                             (data[i].tobytes() for i in range(data.shape[0])),
                             chunksize=4))
    except Exception:  # noqa: BLE001
        grp.create_dataset(name, data=data, compression="gzip",
                           compression_opts=4, chunks=chunks)
        return

    ds = grp.create_dataset(name, shape=data.shape, dtype=data.dtype,
                            compression="gzip", compression_opts=4,
                            chunks=chunks)
    zero = (0,) * (data.ndim - 1)
    for i, blob in enumerate(blobs):
        ds.id.write_direct_chunk((i,) + zero, blob)


def renumber_episodes(data: Any) -> None:
    """Closes gaps left by deleted episodes so the remaining ones become a
    contiguous ``demo_0..demo_{n-1}`` run again (``data`` is the file's
    top-level ``data`` group). Renames on disk -- not just a display-only
    renumbering -- so LeRobot conversion, the dataset explorer, and
    anything else that reads names straight off disk all agree, at the
    cost of reversing ``save_episode``'s original "monotonic, never
    reused" numbering (see its ``next_demo_idx`` comment): resets
    ``next_demo_idx`` to the post-renumber count here, so the next
    ``save_episode`` continues right after the last renumbered episode
    instead of leaving a gap again immediately.

    Safe to call with no gaps (every rename below is then a no-op skip).
    Always renames in ascending order of CURRENT index, which is
    collision-free without a temporary name / two-pass shuffle: the k-th
    (0-indexed) episode in sorted order can only have current index >= k
    (deleting can only ever pull indices down, never up), so demo_k can
    never already belong to a later, not-yet-processed episode when it's
    that episode's turn.
    """
    names = sorted(data.keys(), key=lambda n: int(n.split("_")[1]))
    for new_idx, name in enumerate(names):
        new_name = f"demo_{new_idx}"
        if name != new_name:
            data.move(name, new_name)
    data.attrs["next_demo_idx"] = len(names)


class LiberoEpisodeBuffer:
    """Accumulates one episode's frames in memory before it is committed or discarded.

    ``joint_states``/``ee_pos_quat``/``gripper_closed`` are always buffered
    regardless of ``schema`` -- action computation needs them no matter which
    action space is selected, and no matter which obs fields end up written
    (see LiberoTaskWriter.save_episode). Only the genuinely optional/costly
    fields (images, joint velocities, timestamps) are gated on ``schema``.
    The commanded_* fields are buffered whenever the caller passes them
    (the GUI worker always does) and likewise bypass the schema.
    """

    def __init__(self, schema: Optional[DatasetSchemaConfig] = None,
                 crop_params: Optional[dict] = None) -> None:
        self.schema = schema or DatasetSchemaConfig()
        self.crop_params = crop_params or default_crop_params()
        self._reset_lists()

    def _reset_lists(self) -> None:
        self.agentview_rgb: list[np.ndarray] = []
        self.eye_in_hand_rgb: list[np.ndarray] = []
        self.joint_states: list[np.ndarray] = []
        self.gripper_states: list[np.ndarray] = []
        self.ee_pos_quat: list[np.ndarray] = []
        self.gripper_closed: list[bool] = []
        self.joint_velocities: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.agentview_depth: list[np.ndarray] = []
        self.eye_in_hand_depth: list[np.ndarray] = []
        self.commanded_joint_positions: list[np.ndarray] = []
        self.commanded_gripper: list[float] = []
        # 포스·토크·접촉. 키는 FT_OBS_KEYS -- 개별 속성으로 두면 필드가 늘 때마다
        # 버퍼/시그니처/기록 3곳을 같이 고쳐야 한다.
        self.ft: dict[str, list[np.ndarray]] = {k: [] for k in FT_OBS_KEYS}
        # Per-frame timing (knu-1.3.0): name -> one value per frame. Keys come
        # from the caller (dataset_schema.TIMING_*); string values such as a
        # camera's clock domain are kept per frame and written as group attrs.
        self.timing: dict[str, list] = {}
        #: 축 이름 -> [(t_host, 프레임, 노드 meta), ...]  (knu-2.0.0)
        #: 카메라가 **자기 주기로** 준 것 전부. 20 Hz 루프가 집어간 것과
        #: 무관하며, 이것이 있으면 기록기가 축을 나눠 쓴다 (없으면 옛 경로).
        self.capture: dict[str, list] = {}

    def __len__(self) -> int:
        return len(self.joint_states)

    def set_capture(self, axis: str, frames: list) -> None:
        """카메라 한 대가 이 에피소드 동안 준 프레임 전부를 싣는다.

        ``frames`` 는 ``NodeCamera.stop_capture()`` 가 준 그대로
        ``[(t_host, arr, meta), ...]`` 다. 여기서 이미지를 가공하지 않는다 --
        저장 스레드가 한 번에 처리하는 편이 20 Hz 루프에 붙는 일이 없다.
        """
        self.capture[axis] = frames

    @property
    def per_axis(self) -> bool:
        """축을 나눠 쓸 재료가 있는가. 없으면 옛 한 행 = 한 프레임으로 쓴다."""
        return bool(self.capture)

    def add_frame(
        self,
        agentview_rgb: np.ndarray,
        eye_in_hand_rgb: np.ndarray,
        joint_positions: np.ndarray,
        gripper_position: float,
        ee_pos_quat: np.ndarray,
        gripper_closed: bool,
        joint_velocities: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        commanded_joint_positions: Optional[np.ndarray] = None,
        commanded_gripper: Optional[float] = None,
        agentview_depth: Optional[np.ndarray] = None,
        eye_in_hand_depth: Optional[np.ndarray] = None,
        ft: Optional[dict] = None,
        timing: Optional[dict] = None,
    ) -> None:
        self.joint_states.append(np.asarray(joint_positions, dtype=np.float32))
        self.gripper_states.append(np.array([gripper_position], dtype=np.float32))
        self.ee_pos_quat.append(np.asarray(ee_pos_quat, dtype=np.float64))
        self.gripper_closed.append(bool(gripper_closed))
        if commanded_joint_positions is not None:
            self.commanded_joint_positions.append(
                np.asarray(commanded_joint_positions, dtype=np.float32)
            )
        if commanded_gripper is not None:
            self.commanded_gripper.append(float(commanded_gripper))
        # 포스·토크는 commanded_* 와 같은 규칙: 호출자가 주면 스키마와 무관하게
        # 버퍼링한다 (모듈 docstring 참조).
        for key in FT_OBS_KEYS:
            v = (ft or {}).get(key)
            if v is not None:
                self.ft[key].append(np.asarray(v, dtype=np.float32))
        if self.schema.save_agentview_rgb:
            self.agentview_rgb.append(self._process_image(agentview_rgb))
        if self.schema.save_eye_in_hand_rgb:
            self.eye_in_hand_rgb.append(self._process_image(
                eye_in_hand_rgb, role="wrist"))
        if self.schema.save_joint_velocities and joint_velocities is not None:
            self.joint_velocities.append(np.asarray(joint_velocities, dtype=np.float32))
        if self.schema.save_timestamp and timestamp is not None:
            self.timestamps.append(float(timestamp))
        # Timing follows the commanded_* rule: tiny, recorded whenever given.
        # A key missing on some frames leaves that column short, and a short
        # column is not written (same length check as commanded_*).
        for key, value in (timing or {}).items():
            if value is not None:
                self.timing.setdefault(key, []).append(value)
        # depth 는 crop/resize 없이 원본 해상도 그대로 (#17: 원본 보관소 원칙,
        # D455 는 RGB-depth 픽셀 대응이 원래 안 맞아 RGB 크롭을 따라가면
        # 오히려 거짓 정렬이 된다). .copy() 는 RGB 와 같은 이유 -- 드라이버
        # 프레임 풀에 대한 뷰를 그대로 쌓으면 몇 프레임 뒤 덮어써진다.
        if self.schema.save_agentview_depth and agentview_depth is not None:
            self.agentview_depth.append(
                np.asarray(agentview_depth, dtype=np.uint16).copy())
        if self.schema.save_eye_in_hand_depth and eye_in_hand_depth is not None:
            self.eye_in_hand_depth.append(
                np.asarray(eye_in_hand_depth, dtype=np.uint16).copy())

    def _process_image(self, img: np.ndarray, role: str = "agent") -> np.ndarray:
        """Returns a frame this buffer owns, resized if the schema asks for it.

        The copy is the point, and it is deliberate rather than a side effect.
        Camera drivers hand out a *view* into a small pool of frame buffers
        they recycle (librealsense via lerobot's RealSenseCamera does, and its
        _postprocess_image only copies when a colour conversion or rotation is
        configured -- this collector requests neither). Appending that view
        stores a window onto memory the driver overwrites a few frames later,
        so a 200-frame episode ends up holding the same handful of images
        repeated, at the right shape and count and with no error anywhere.

        This used to be safe only by accident: cv2.resize allocates its output,
        so the resize path copied for free while image_size=None returned the
        view untouched. Copying here instead means the guarantee no longer
        depends on which branch runs.

        ``.copy()`` and not ``np.ascontiguousarray``: the latter is a no-op
        when the input already is contiguous, which a driver frame buffer is
        -- it would have left the aliasing exactly as it was.
        """
        if self.schema.image_size is None:
            return img.copy()
        p = self.crop_params.get(role, {})
        return resize_rgb(img, size=self.schema.image_size,
                          zoom=p.get("zoom", 1.0), x_shift=p.get("x", 0),
                          y_shift=p.get("y", 0))

    def clear(self) -> None:
        self._reset_lists()


def _mark_close_on_exec(f: h5py.File) -> None:
    """Without this, a task .hdf5 opened here stays locked forever by any
    child process spawned (e.g. via QProcess) while the file is open --
    HDF5's C library doesn't set FD_CLOEXEC on the fd it opens, unlike
    Python's own open()/io, so e.g. restarting the robot node mid-session
    (scripts/launch/launch_nodes.py, spawned from the GUI) silently inherits
    and keeps holding the lock via fork+exec even after this writer's own
    session ends and closes its copy -- the file then can't be read/
    converted by anything until that unrelated child process is killed.
    Best-effort: not every HDF5 driver's fd is retrievable this way, so a
    failure here is not fatal.
    """
    try:
        fd = f.id.get_vfd_handle()
        flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
    except Exception:  # noqa: BLE001
        pass


def write_episode_payload(
    grp: h5py.Group,
    buf: LiberoEpisodeBuffer,
    schema: DatasetSchemaConfig,
    success: Optional[bool] = None,
    pool: Any = None,
) -> int:
    """One episode's shared on-disk payload: the ``actions``/``rewards``/
    ``dones`` datasets, the ``obs/`` group, and the per-episode provenance
    attrs (``num_samples``, ``action_space``, ``crop_params``, ``station``,
    ...).

    legacy 포맷(``<task>_demo.hdf5`` 의 ``demo_N``)과 scene 포맷
    (``scene_XXX.hdf5`` 의 ``episode_NNN``, mstack/scene/scene_format.py)이 이 함수를
    공유한다 -- 에피소드 안쪽 구조가 같아야 변환기가 두 포맷을 같은 코드로
    읽는다. 여기 없는 attrs(instruction, episode_uid 등)는 각 포맷의 writer 가
    이 함수 호출 뒤에 얹는다.

    Caller owns group creation, file flush and buffer lifetime. Returns the
    frame count ``n`` (the caller rejects buffers with fewer than 2 frames
    before creating the group).
    """
    n = len(buf)
    if schema.action_space == ACTION_SPACE_JOINT_DELTA:
        q = np.stack(buf.joint_states)  # (n, 7)
        actions = np.zeros((n, 8), dtype=np.float32)
        for t in range(n - 1):
            actions[t] = compute_joint_delta_action(
                q[t], q[t + 1], buf.gripper_closed[t]
            )
        # Terminal frame: no further motion recorded; hold gripper state.
        actions[n - 1, :7] = 0.0
        actions[n - 1, 7] = 1.0 if buf.gripper_closed[-1] else -1.0
    elif schema.action_space == ACTION_SPACE_JOINT_ABSOLUTE:
        # The leader's command, verbatim -- see compute_joint_absolute_action
        # for why the follower's realized joint_states must not be used here.
        # No terminal-frame special case is needed: unlike a realized-next-
        # state target, a command exists at every frame including the last.
        if len(buf.commanded_joint_positions) != n:
            raise ValueError(
                "action_space='joint_absolute' needs commanded_joint_positions "
                f"on every frame (got {len(buf.commanded_joint_positions)} of {n}). "
                "The GUI worker supplies them; a caller that does not must use "
                "a different action space."
            )
        q_cmd = np.stack(buf.commanded_joint_positions)  # (n, 7)
        actions = np.zeros((n, 8), dtype=np.float32)
        for t in range(n):
            actions[t] = compute_joint_absolute_action(
                q_cmd[t], buf.gripper_closed[t]
            )
    elif schema.action_space == ACTION_SPACE_EE_ABSOLUTE:
        ee = np.stack(buf.ee_pos_quat)  # (n, 7)
        actions = np.zeros((n, 7), dtype=np.float32)
        for t in range(n - 1):
            actions[t] = compute_ee_absolute_action(
                ee[t + 1], buf.gripper_closed[t]
            )
        # Terminal frame: no further target recorded; hold current pose.
        actions[n - 1, :3] = ee[n - 1, :3]
        actions[n - 1, 3:6] = _quat_to_axis_angle(*ee[n - 1, 3:7])
        actions[n - 1, 6] = 1.0 if buf.gripper_closed[-1] else -1.0
    else:
        ee = np.stack(buf.ee_pos_quat)  # (n, 7)
        actions = np.zeros((n, 7), dtype=np.float32)
        for t in range(n - 1):
            actions[t] = compute_delta_action(
                ee[t], ee[t + 1], buf.gripper_closed[t]
            )
        actions[n - 1, :6] = 0.0
        actions[n - 1, 6] = 1.0 if buf.gripper_closed[-1] else -1.0

    # Every branch above always ends with gripper as the last column,
    # in -1=open/+1=close (robosuite Panda convention) -- remap to
    # 0=open/1=closed here in one place, matching obs/gripper_states'
    # convention, if the operator asked action to match obs.
    if schema.gripper_action_match_obs:
        actions[:, -1] = (actions[:, -1] + 1.0) / 2.0

    # ... then strip it here in one place rather than duplicating the
    # flag check in all four branches.
    if not schema.action_include_gripper:
        actions = actions[:, :-1]

    grp.attrs["num_samples"] = n
    if success is not None:
        grp.attrs["success"] = bool(success)
    # Per-episode provenance: which action space this demo's `actions`
    # was computed with, and which obs fields are actually present
    # (readers should not assume the full original LIBERO obs set --
    # --resume lets a file mix schemas episode-to-episode if the
    # operator changed the "사용자 지정" config between sessions).
    grp.attrs["action_space"] = schema.action_space
    grp.attrs["gripper_action_convention"] = "01" if schema.gripper_action_match_obs else "pm1"
    grp.attrs["action_column_names"] = json.dumps(resolved_action_column_names(schema))
    # 이 에피소드의 이미지에 (원본 저장이면 변환 시점에) 적용할/된 정사각
    # 크롭 정렬. buf 의 것을 쓴다 -- 백그라운드 저장 중 writer 쪽 값이
    # 바뀌어도 찍히는 값은 그 에피소드가 실제로 쓰던 것이어야 한다.
    grp.attrs["crop_params"] = json.dumps(buf.crop_params)
    # 어느 스테이션에서 찍었는지. 형식은 v0 에서 고정이라 코드 버전은 남기지
    # 않지만, 스테이션은 하드웨어가 바뀌면 같이 바뀐다 -- 카메라를 교체하거나
    # 두 번째 스테이션이 생기면 프레이밍이 갈리는 지점이 여기다.
    grp.attrs["station"] = load_station().name

    obs = grp.create_group("obs")
    if buf.per_axis:
        # knu-2.0.0: 카메라가 자기 주기로 준 것 전부를 자기 축에 쓴다.
        _write_axes(grp, obs, buf, schema, pool)
    else:
        if schema.save_agentview_rgb:
            write_image_dataset(obs, OBS_AGENTVIEW_RGB,
                                np.stack(buf.agentview_rgb), pool)
        if schema.save_eye_in_hand_rgb:
            write_image_dataset(obs, OBS_EYE_IN_HAND_RGB,
                                np.stack(buf.eye_in_hand_rgb), pool)
    if schema.save_joint_states:
        obs.create_dataset(OBS_JOINT_STATES, data=np.stack(buf.joint_states))
    if schema.save_gripper_states:
        obs.create_dataset(
            OBS_GRIPPER_STATES, data=np.stack(buf.gripper_states)
        )
    if schema.save_ee_states or schema.save_ee_pos or schema.save_ee_ori:
        ee = np.stack(buf.ee_pos_quat)  # (n, 7)
        ee_ori = np.stack([_quat_to_axis_angle(*q[3:7]) for q in ee]).astype(np.float32)
        ee_pos = ee[:, :3].astype(np.float32)
        if schema.save_ee_states:
            ee_states = np.concatenate([ee_pos, ee_ori], axis=1).astype(np.float32)
            obs.create_dataset(OBS_EE_STATES, data=ee_states)
        if schema.save_ee_pos:
            obs.create_dataset(OBS_EE_POS, data=ee_pos)
        if schema.save_ee_ori:
            obs.create_dataset(OBS_EE_ORI, data=ee_ori)
    if schema.save_joint_velocities and buf.joint_velocities:
        obs.create_dataset(
            OBS_JOINT_VELOCITIES, data=np.stack(buf.joint_velocities)
        )
    if schema.save_timestamp and buf.timestamps:
        obs.create_dataset(
            "timestamp", data=np.array(buf.timestamps, dtype=np.float64)
        )
    # 무손실 필수 (#17) -- JPEG 류 손실 압축은 depth 값을 파괴한다.
    # (gzip 도 무손실이라 그대로 유효하다.)
    if schema.save_agentview_depth and buf.agentview_depth:
        write_image_dataset(obs, "agentview_depth",
                            np.stack(buf.agentview_depth), pool)
    if schema.save_eye_in_hand_depth and buf.eye_in_hand_depth:
        write_image_dataset(obs, "eye_in_hand_depth",
                            np.stack(buf.eye_in_hand_depth), pool)
    # Raw teleop command stream -- written whenever the caller supplied it,
    # independent of the schema and of which action space `actions` used:
    # realized-trajectory actions zero out wherever the follower is blocked
    # by contact, and the command is the only record of what the operator
    # was actually asking for there. Tiny (7+1 floats/frame), so never
    # worth a schema toggle. See scripts/convert/derive_commanded_ee_actions.py.
    if len(buf.commanded_joint_positions) == n:
        obs.create_dataset(
            OBS_COMMANDED_JOINT_STATES,
            data=np.stack(buf.commanded_joint_positions),
        )
    if len(buf.commanded_gripper) == n:
        obs.create_dataset(
            OBS_COMMANDED_GRIPPER_STATES,
            data=np.array(buf.commanded_gripper, dtype=np.float32).reshape(-1, 1),
        )
    # 포스·토크 (2026-08-23): franka 가 매 스텝 추정해 주는 값의 20Hz 스냅숏.
    # commanded_* 와 같은 원칙 -- 작고, 실현 궤적이 숨기는 접촉을 담은 유일한
    # 기록이라 스키마 토글 없이 있으면 항상 쓴다. hdf5 원본 전용 (변환기의
    # _CONSUMED_OBS_KEYS 밖 -- LeRobot 산출물에는 들어가지 않는다).
    for key in FT_OBS_KEYS:
        if len(buf.ft[key]) == n:
            obs.create_dataset(key, data=np.stack(buf.ft[key]))
    if buf.per_axis:
        # timing/ 의 내용은 t/ 와 meta/ 로 갈라져 들어갔다 (_write_axes).
        # 남은 control 축 부수값만 meta/control 에 둔다 -- 같은 값을 두 곳에
        # 쓰면 언젠가 갈라진다.
        mg = grp.require_group("meta").require_group(CONTROL_AXIS)
        t0 = float(grp.attrs["t0_wall"])
        for key in (TIMING_ACTION, TIMING_ROBOT_STATE):
            v = buf.timing.get(key)
            if v is not None and len(v) == n:
                mg.create_dataset(key, data=np.asarray(v, dtype=np.float64) - t0)
    else:
        _write_timing(grp, buf.timing, n)

    grp.create_dataset("actions", data=actions)
    grp.create_dataset("rewards", data=np.zeros(n, dtype=np.float32))
    dones = np.zeros(n, dtype=np.float32)
    dones[-1] = 1.0
    grp.create_dataset("dones", data=dones)
    if buf.per_axis:
        # control 축의 모든 데이터셋에 축을 **명시한다** -- 마지막에 한다,
        # 전부 만들어진 뒤여야 빠뜨리지 않는다. 카메라는 _write_axes 가 이미
        # 붙였다. 안 붙이면 소비자(트림 등)가 길이로 추정하게 되고, 그 추정은
        # 축이 갈린 순간부터 틀린다.
        for name in ("actions", "actions_ee", "rewards", "dones"):
            _tag_axis(grp, name)
        for name in list(obs.keys()):
            if "axis" not in obs[name].attrs:
                _tag_axis(obs, name)
    return n


#: 시간축 이름 -> (이미지 데이터셋 이름, 1.3.0 timing 접두사, 크롭 role)
AXIS_CAMERAS = {
    "agent": (OBS_AGENTVIEW_RGB, "agentview", "agent"),
    "wrist": (OBS_EYE_IN_HAND_RGB, "eye_in_hand", "wrist"),
}
CONTROL_AXIS = "control"


def _tag_axis(grp: h5py.Group, name: str, axis: str = CONTROL_AXIS) -> None:
    """이 데이터셋이 어느 시간축에 실리는지 **명시한다**.

    길이로 추정하면 안 된다 -- 축마다 길이가 다른 순간부터 "actions 와 길이가
    같으면 프레임 축" 이라는 추정이 카메라를 조용히 놓친다 (episode_trim 에서
    실제로 그랬다).
    """
    if name in grp:
        grp[name].attrs["axis"] = axis


def _write_axes(grp: h5py.Group, obs: h5py.Group, buf: "LiberoEpisodeBuffer",
                schema: DatasetSchemaConfig, pool: Any) -> None:
    """knu-2.0.0 의 축별 기록: ``t/*``, 카메라 이미지, ``meta/*``.

    control 축(액션·상태)은 호출자가 옛 경로 그대로 쓴다 -- 20 Hz 루프가 한
    tick 에 한 줄씩 모은 것이라 구조가 안 바뀐다. 여기서 하는 일은 **카메라를
    그 격자에서 떼어내는 것**이다.

    시각의 기준은 ``t0_wall`` (control 첫 tick 의 호스트 시각) 이고, 모든
    ``t/`` 는 그로부터의 초다. 절대 epoch 를 float64 로 그대로 두면 유효숫자가
    1 µs 언저리까지 떨어진다.

    카메라 축에는 **장치 시각**(``t_device``, librealsense global_time)을 쓴다.
    도착 시각을 쓰면 기종마다 다른 고정 전송 지연이 축에 섞인다 -- 실측으로
    D455 15.03 ms, D405 8.60 ms 이고 표준편차는 0.1 ms 수준이라 지터가 아니라
    상수다. 도착 시각은 ``meta/<축>/host`` 에 남겨 둔다 (배포는 도착 순서로
    프레임을 고르므로 그 규칙을 재현하려면 필요하다).
    """
    tg = grp.require_group("t")
    mg = grp.require_group("meta")

    t_ctrl = np.asarray(buf.timing.get(TIMING_FRAME, []), dtype=np.float64)
    if t_ctrl.size == 0:
        # 프레임 시각이 없으면 축을 만들 수 없다. 카메라만 쌓고 control 축을
        # 못 쓰는 파일을 만드느니 옛 구조로 떨어진다.
        raise ValueError(
            "축을 나눠 쓰려면 timing/frame 이 필요하다 -- 이 에피소드에는 없다")
    t0 = float(t_ctrl[0])
    grp.attrs["t0_wall"] = t0
    d = tg.create_dataset(CONTROL_AXIS, data=t_ctrl - t0)
    d.attrs["axis"] = CONTROL_AXIS

    for axis, (ds_name, _pre, role) in AXIS_CAMERAS.items():
        frames = buf.capture.get(axis)
        if not frames:
            continue
        if axis == "agent" and not schema.save_agentview_rgb:
            continue
        if axis == "wrist" and not schema.save_eye_in_hand_rgb:
            continue
        imgs = np.stack([buf._process_image(a, role) for _t, a, _m in frames])
        write_image_dataset(obs, ds_name, imgs, pool)
        obs[ds_name].attrs["axis"] = axis

        # 축 자체는 장치 시각. 노드가 안 주면(옛 노드) 도착 시각으로 떨어진다.
        dev = [m.get("t_device", t) for t, _a, m in frames]
        tg.create_dataset(axis, data=np.asarray(dev, dtype=np.float64) - t0)
        ag = mg.require_group(axis)
        ag.create_dataset("host", data=np.asarray(
            [t for t, _a, _m in frames], dtype=np.float64) - t0)
        for key, src in (("frame_no", "frame_no"), ("node_seq", "seq"),
                         ("exposure", "exposure")):
            vals = [m.get(src) for _t, _a, m in frames]
            if any(v is None for v in vals):
                continue          # 안 주는 노드가 있다 -- 없는 값을 지어내지 않는다
            ag.create_dataset(key, data=np.asarray(vals, dtype=np.int64))
        dom = frames[-1][2].get("t_domain")
        if dom:
            ag.attrs["clock"] = str(dom)


def _write_timing(grp: h5py.Group, timing: dict, n: int) -> None:
    """``timing/<name>`` for every column with one value per frame.

    Numbers become float64 (times) or int64 (frame counters); a string column
    (a camera's clock domain) becomes a group attr holding its last value.
    """
    full = {k: v for k, v in timing.items() if len(v) == n}
    if not full:
        return
    tg = grp.create_group(TIMING_GROUP)
    tg.attrs["clock"] = "host time.time() seconds"
    for key, values in full.items():
        if isinstance(values[-1], str):
            tg.attrs[key] = values[-1]
        elif isinstance(values[-1], (int, np.integer)) and not isinstance(values[-1], bool):
            tg.create_dataset(key, data=np.asarray(values, dtype=np.int64))
        else:
            tg.create_dataset(key, data=np.asarray(values, dtype=np.float64))


class NullTaskWriter:
    """A writer that records nothing -- for teleoperating without a dataset.

    Setting a scene up, checking a camera angle, or letting someone try the
    leader arm are all things done far more often than a recording session,
    and all of them used to require inventing a throwaway task name and then
    deleting the .hdf5 it left behind. Worse, that file lands in the data root
    next to the real ones, where the next repack or conversion picks it up.

    This stands in for LiberoTaskWriter so the worker's state machine (which
    touches the writer in a dozen places) needs no branching: the episode
    buffer still fills, so the pose gate, the frame counter and the live view
    all behave exactly as they do in a real session -- only the file is
    missing. Saving is accepted and dropped, which is the honest behaviour for
    a mode whose whole point is that nothing is kept.
    """

    def __init__(self, schema: Optional[DatasetSchemaConfig] = None) -> None:
        self.schema = schema or DatasetSchemaConfig()
        # Not None: the worker emits str(writer.path) on connect, and a literal
        # "None" in the GUI's file field reads as a bug rather than a mode.
        self.path = "(기록 안 함)"
        self._buffer = LiberoEpisodeBuffer(self.schema)

    def record_session_config(self, **kwargs: Any) -> None:
        pass

    @property
    def num_episodes(self) -> int:
        return 0

    def list_episodes(self) -> list[dict]:
        return []

    def delete_episode(self, name: str) -> None:
        pass

    def start_episode(self) -> None:
        self._buffer.clear()

    def add_frame(self, **kwargs: Any) -> None:
        self._buffer.add_frame(**kwargs)

    def discard_episode(self) -> None:
        self._buffer.clear()

    def set_capture(self, axis: str, frames: list) -> None:
        """카메라 프레임 전부를 버퍼에 싣는다 (SceneWriter 와 같은 계약).
        버퍼만 만지므로 저장 스레드를 거치지 않는다."""
        self._buffer.set_capture(axis, frames)

    def detach_buffer(self) -> LiberoEpisodeBuffer:
        buf = self._buffer
        self._buffer = LiberoEpisodeBuffer(self.schema)
        return buf

    def save_episode(self, success: Optional[bool] = None) -> Optional[str]:
        return self.save_buffer(self.detach_buffer(), success=success)

    def save_buffer(self, buf: LiberoEpisodeBuffer, success: Optional[bool] = None,
                    pool: Any = None) -> Optional[str]:
        buf.clear()
        return None

    def close(self) -> None:
        pass

    def __enter__(self) -> "NullTaskWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class LiberoTaskWriter:
    """Owns one ``<task>_demo.hdf5`` file: one file per task, one ``demo_N`` per episode.

    Not safe for concurrent writers on the same file; one collection session
    owns one open writer.
    """

    def __init__(
        self,
        root: Path,
        task_name: str,
        language_instruction: str,
        robot_name: str = "fr3_gello_real",
        resume: bool = False,
        schema: Optional[DatasetSchemaConfig] = None,
        crop_params: Optional[dict] = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        safe_name = task_name.strip().replace(" ", "_")
        self.path = self.root / f"{safe_name}_demo.hdf5"
        self.language_instruction = language_instruction
        self.schema = schema or DatasetSchemaConfig()
        # Connect 시점의 스냅샷. 매 에피소드 attrs 에 그대로 찍힌다 -- 변환이
        # 파일만 보고 조작자가 맞춘 프레이밍을 재현할 수 있도록.
        self.crop_params = crop_params or default_crop_params()
        self._buffer = LiberoEpisodeBuffer(self.schema, self.crop_params)

        if self.path.exists() and not resume:
            raise FileExistsError(
                f"{self.path} already exists; pass resume=True to append episodes."
            )

        self._file = h5py.File(self.path, "a")
        _mark_close_on_exec(self._file)
        self._data = self._file.require_group("data")
        if "env_args" not in self._data.attrs:
            env_args = {
                "env_name": "real_fr3_gello",
                "type": "real_robot",
                "env_kwargs": {
                    "robot": robot_name,
                    "note": (
                        "Real-robot capture -- no simulator/BDDL scene; this "
                        "field exists only so LIBERO-style readers that "
                        "expect it do not KeyError."
                    ),
                },
            }
            self._data.attrs["env_args"] = json.dumps(env_args)
        if "problem_info" not in self._data.attrs:
            problem_info = {
                "language_instruction": f'"{self.language_instruction}"',
                "problem_name": safe_name,
            }
            self._data.attrs["problem_info"] = json.dumps(problem_info)
        if "next_demo_idx" not in self._data.attrs:
            # delete_episode() renumbers to close gaps and resets this attr
            # to the post-renumber count (see renumber_episodes()), so in
            # steady state this is just `num_episodes` -- this fallback
            # only matters for a file that predates renumbering and still
            # has old gaps, where it's still the highest-index-plus-one so
            # a new episode's name can't collide with a surviving one.
            existing = [int(k.split("_")[1]) for k in self._data.keys()]
            self._data.attrs["next_demo_idx"] = max(existing, default=-1) + 1
        self._file.flush()

    def record_session_config(self, **kwargs: Any) -> None:
        """Overwrites the file-level ``session_config`` attr with whatever
        non-camera session settings (reset_pose, grip, enable_wall,
        max_episode_seconds, reset_wait_seconds) this session was started
        with, so a later session can restore them for the same task. Written
        by mstack/collect/worker.py; the wizard GUI's dropdown used to read
        it back, and the workspace UI that replaced it (62cad92) does not yet.
        Always reflects the LATEST session, not the first-ever one: intentionally overwritten every time, since this is
        "how to continue this task", not a history log.
        """
        self._data.attrs["session_config"] = json.dumps(kwargs)
        self._file.flush()

    @property
    def num_episodes(self) -> int:
        return len(self._data.keys())

    def list_episodes(self) -> list[dict]:
        """Current demos sorted by index: ``[{"name", "num_samples", "success"}, ...]``."""
        items = []
        for name in self._data.keys():
            grp = self._data[name]
            success = grp.attrs.get("success")
            items.append(
                {
                    "name": name,
                    "num_samples": int(grp.attrs.get("num_samples", grp["actions"].shape[0])),
                    "success": None if success is None else bool(success),
                }
            )
        items.sort(key=lambda d: int(d["name"].split("_")[1]))
        return items

    def delete_episode(self, name: str) -> None:
        """Removes a ``demo_N`` group, then renumbers the rest to close the
        resulting gap (see :func:`renumber_episodes`).

        HDF5 does not shrink the file on delete -- the freed space is only
        reusable by later writes *within this same file*, not returned to the
        OS. Run ``h5repack`` afterwards if reclaiming disk space matters.
        """
        if name not in self._data:
            raise KeyError(f"{name!r} not found in {self.path}")
        del self._data[name]
        renumber_episodes(self._data)
        self._file.flush()

    def set_episode_success(self, name: str, success: bool) -> None:
        """Re-labels an already-saved episode as success/failure.

        The verdict is worth more a few seconds after the take than during it:
        the operator has stopped moving, the arm is going home, and they can
        actually look at what happened. Only the attribute changes -- frames,
        images and numbering are untouched -- so this stays cheap no matter how
        big the episode was.
        """
        if name not in self._data:
            raise KeyError(f"{name!r} not found in {self.path}")
        self._data[name].attrs["success"] = bool(success)
        self._file.flush()

    def start_episode(self) -> None:
        self._buffer.clear()

    def add_frame(self, **kwargs: Any) -> None:
        """Forwards to :meth:`LiberoEpisodeBuffer.add_frame`."""
        self._buffer.add_frame(**kwargs)

    def discard_episode(self) -> None:
        self._buffer.clear()

    def set_capture(self, axis: str, frames: list) -> None:
        """카메라 프레임 전부를 버퍼에 싣는다 (SceneWriter 와 같은 계약).
        버퍼만 만지므로 저장 스레드를 거치지 않는다."""
        self._buffer.set_capture(axis, frames)

    def detach_buffer(self) -> LiberoEpisodeBuffer:
        """Swap out the filled episode buffer and install a fresh one, so the
        next episode can start recording while the detached buffer is being
        written by a background thread (see mstack.collect.worker.EpisodeSaver)."""
        buf = self._buffer
        self._buffer = LiberoEpisodeBuffer(self.schema, self.crop_params)
        return buf

    def save_episode(self, success: Optional[bool] = None) -> Optional[str]:
        """Synchronous convenience wrapper: detach + save_buffer in one call."""
        return self.save_buffer(self.detach_buffer(), success=success)

    def save_buffer(self, buf: LiberoEpisodeBuffer, success: Optional[bool] = None,
                    pool: Any = None) -> Optional[str]:
        """Commits one (detached) episode buffer as a new ``demo_N`` group.

        h5py is not thread-safe: every file-touching call on this writer
        (save_buffer / delete_episode / list_episodes / close) must be
        serialized onto ONE thread by the caller -- EpisodeSaver owns exactly
        that serialization in the GUI.

        Args:
            buf: the buffer returned by :meth:`detach_buffer`.
            success: operator-labeled outcome (no simulator goal-check exists
                for a real robot). Not a canonical LIBERO field; stored as a
                per-demo attr for downstream filtering. ``None`` if unlabeled.

        Returns the group name, or None if the buffer was empty.
        """
        n = len(buf)
        if n < 2:
            buf.clear()
            return None

        demo_idx = int(self._data.attrs["next_demo_idx"])
        self._data.attrs["next_demo_idx"] = demo_idx + 1
        name = f"demo_{demo_idx}"
        grp = self._data.create_group(name)
        write_episode_payload(grp, buf, self.schema, success=success, pool=pool)

        self._file.flush()
        buf.clear()
        return name

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "LiberoTaskWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------- repack state
# 재압축 직후 파일도 메타데이터 오버헤드로 0.3~0.4%는 남는다(실측). 3%를 넘으면
# 지운 에피소드가 차지하던 자리로 보는 게 안전하다 -- 실측에서 삭제가 있었던
# 파일들은 4.1 / 8.3 / 17.3% 였다.
DEAD_SPACE_RATIO = 0.03


def _stored_bytes(group) -> int:
    """Bytes every dataset in this file actually occupies on disk.

    get_storage_size() is the compressed, on-disk size -- not the logical
    array size -- so this stays meaningful for gzip'd images. Metadata only:
    no chunk is read.
    """
    total = 0
    stack = [group]
    while stack:
        g = stack.pop()
        for key in g:
            item = g[key]
            if isinstance(item, h5py.Group):
                stack.append(item)
            else:
                total += item.id.get_storage_size()
    return total


def hdf5_repack_status(path) -> dict:
    """Does this file still need scripts/convert/repack_hdf5.py?

    Since 2026-09-22 the collector writes images as ``gzip`` level 4 with
    per-frame chunks, so a gzip episode no longer proves a repack ran -- a
    file collected wholly after the switch is all-gzip from birth and already
    in its final form. ``repacked`` therefore answers "is there anything left
    for repack to do": ``lzf`` images mark a pre-switch file that was never
    repacked (repack's remaining job), and anything already gzip needs
    nothing, fresh or repacked. A file written by the current collector must
    come back ``repacked`` -- reporting it as to-do would send every new
    dataset through an old-dataset tool.

    Two signals, because the marker only exists on files repacked after it was
    introduced. The image compressor is the retroactive one and is decisive on
    its own.

    **Every episode is checked, not just the first.** A file that was repacked
    and then collected into again is the common case -- the operator adds a
    few demos to an existing task file. Before the switch that left it
    *mixed* (old gzip, new lzf); now the mixed case is the reverse -- an old
    un-repacked ``lzf`` file with new gzip episodes next to the ``lzf`` ones --
    and the stale marker still names the earlier run either way. Sampling one
    episode (or trusting the marker) reports such a file as finished and
    silently drops it from the repack selection, which is exactly the file
    that still has ``lzf`` episodes in it. So a mixed file counts as not
    repacked, and the marker cannot override that.

    Returns ``{"repacked", "compression", "mixed", "marker", "new_since",
    "size", "episodes", "error"}``; never raises -- an unreadable file comes
    back with ``error`` set so a caller listing a directory can show it
    instead of dying.
    """
    out = {"repacked": False, "compression": None, "mixed": False,
           "marker": None, "new_since": 0, "deleted_since": 0,
           "dead_bytes": 0, "dead_ratio": 0.0, "size": 0, "episodes": 0,
           "error": None}
    try:
        out["size"] = Path(path).stat().st_size
        with h5py.File(path, "r") as f:
            if "data" in f:
                marker_grp = f["data"]        # legacy: 마커도 에피소드도 data/
                container = f["data"]
                episode_names = list(container.keys())
            else:
                # scene-v1: 마커는 metadata 그룹에, 에피소드는 루트에 있다.
                # 에피소드 안쪽 페이로드는 legacy 와 동일해 아래 로직을 공유.
                marker_grp = f["metadata"]
                container = f
                episode_names = [k for k in f.keys() if k.startswith("episode_")]
            out["episodes"] = len(episode_names)
            out["marker"] = marker_grp.attrs.get(REPACK_MARKER_ATTR)
            if isinstance(out["marker"], bytes):
                out["marker"] = out["marker"].decode(errors="replace")
            at_repack = marker_grp.attrs.get(REPACK_COUNT_ATTR)
            if at_repack is not None:
                out["new_since"] = max(0, out["episodes"] - int(at_repack))
                out["deleted_since"] = max(0, int(at_repack) - out["episodes"])
            # Dead space: HDF5 never returns a deleted group's bytes to the OS,
            # it only makes them reusable inside the same file. So a curated
            # file keeps paying for takes that are gone, and nothing about the
            # remaining episodes shows it -- they are all still gzip, the
            # marker is still there, and an episode-count comparison misses
            # the common "delete two, record two more" case entirely.
            # Comparing the file's size against what its datasets actually
            # occupy catches all of it, and costs only metadata reads.
            out["dead_bytes"] = max(0, out["size"] - _stored_bytes(f))
            out["dead_ratio"] = out["dead_bytes"] / out["size"] if out["size"] else 0.0
            comps = set()
            for name in episode_names:
                obs = container[name].get("obs")
                if obs is None:
                    continue
                for key in (OBS_AGENTVIEW_RGB, OBS_EYE_IN_HAND_RGB):
                    ds = obs.get(key)
                    if ds is not None:
                        comps.add(ds.compression)
                        break
            out["mixed"] = len(comps) > 1
            if comps:
                out["compression"] = "+".join(sorted(c or "없음" for c in comps))
        fully_gzip = comps == {"gzip"}
        marked = bool(out["marker"]) and not out["mixed"]
        # 죽은 공간이 크면 압축 방식과 무관하게 재압축 대상이다.
        out["repacked"] = (fully_gzip or marked) and out["dead_ratio"] < DEAD_SPACE_RATIO
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out
