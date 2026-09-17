"""Per-frame timing (knu-1.3.0, 2026-09-17).

  1. the camera client hands out an image and its stamps from one cache entry
  2. the camera node reads frame_no / device time off a frame, and a driver
     without them leaves the keys out instead of raising
  3. the worker maps an observation to timing columns; missing stamps stay missing
  4. the writer stores timing/<name> (float64 times, int64 frame counters, clock
     domain as a group attr) and skips a column that is short
  5. knu-1.3.0 requires the always-produced columns, not the camera-only ones
  6. resuming an old file does not raise its stamp to 1.3.0 when its episodes
     have no timing, and does when they all have it

No robot, no camera: stamps are injected.
"""
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

from mstack.collect.worker import _frame_timing  # noqa: E402
from mstack.comm.camera_client import NodeCamera  # noqa: E402
from mstack.comm.camera_node import _device_stamp  # noqa: E402
from mstack.data.dataset_schema import (  # noqa: E402
    FT_OBS_FIELDS,
    SCHEMA_FIELDS,
    TIMING_OPTIONAL,
    TIMING_REQUIRED,
    schema_required_fields,
)
from mstack.scene.scene_format import SceneMetadata, SceneWriter  # noqa: E402

# ---- 1. stamped read
cam = NodeCamera("TEST")
cam._sub = object()          # "connected" for the cache check; no socket is used
img = np.zeros((4, 4, 3), np.uint8)
now = time.time()
cam._latest["color"] = (now, img, {"ts": now, "seq": 40, "frame_no": 41,
                                   "t_device": now - 0.03,
                                   "t_domain": "global_time", "shape": [4, 4, 3]})
got, stamps = cam.read_latest_stamped(max_age_ms=500)
assert got is img
assert stamps == {"t_host": now, "seq": 40, "frame_no": 41, "t_device": now - 0.03,
                  "t_domain": "global_time"}, stamps
assert cam.read_latest(max_age_ms=500) is img
cam._latest["color"] = (now, img, {"ts": now})
assert cam.read_latest_stamped(max_age_ms=500)[1] == {"t_host": now}
print("1. image and stamps come from the same cache entry; absent stamps stay absent OK")


# ---- 2. node-side stamp extraction
class _Frame:
    def get_frame_number(self):
        return 1234

    def get_timestamp(self):
        return 1_789_900_000_123.5          # ms

    def get_frame_timestamp_domain(self):
        return "timestamp_domain.global_time"


st = _device_stamp(_Frame())
assert st == {"frame_no": 1234, "t_device": 1_789_900_000.1235,
              "t_domain": "global_time"}, st
assert _device_stamp(object()) == {}
print("2. device frame number/time read; a driver without them yields no keys OK")

# ---- 3. observation -> timing columns
obs = {"_state_time": 10.0,
       "_agent_stamps": {"t_host": 9.99, "seq": 6, "frame_no": 7, "t_device": 9.98,
                         "t_domain": "global_time"},
       "_wrist_stamps": {"t_host": 9.97}}
cols = _frame_timing(obs, t_frame=10.01, t_action=10.005)
assert cols == {"frame": 10.01, "action": 10.005, "robot_state": 10.0,
                "agentview_host": 9.99, "agentview_frame_no": 7, "agentview_node_seq": 6,
                "agentview_device": 9.98, "agentview_domain": "global_time",
                "eye_in_hand_host": 9.97}, cols
assert "robot_state" not in _frame_timing({}, 1.0, 1.0)
print("3. worker maps stamps to columns and does not invent missing ones OK")

# ---- 5. version requirements
need = schema_required_fields("knu-1.3.0")["episode_datasets"]
for k in TIMING_REQUIRED:
    assert f"timing/{k}" in need, need
for k in TIMING_OPTIONAL:
    assert f"timing/{k}" not in need, need
assert SCHEMA_FIELDS["knu-1.3.0"]["metadata_attrs"] == SCHEMA_FIELDS["knu-1.2.2"]["metadata_attrs"]
print("5. knu-1.3.0 requires the always-produced columns only OK")

# ---- 4 + 6. writing and resume stamping
TMP = Path(tempfile.mkdtemp(prefix="frame_timing_"))
LAYOUT = {"grid": [3, 3], "placements": {"OBJ-CUP-BLU-01": {"zone": [0, 0]}}}
PROV = dict(payload_mass=0.85, payload_com=[-0.01, 0.0, 0.03],
            reset_pose="libero", reset_qpos=[0.0] * 7, provenance_source="live")


def _episode(w, rng, with_timing: bool, short_device: bool = False):
    for i in range(4):
        timing = None
        if with_timing:
            timing = {k: 100.0 + i * 0.05 for k in TIMING_REQUIRED}
            timing["agentview_frame_no"] = 500 + 2 * i
            timing["agentview_domain"] = "global_time"
            if not (short_device and i == 0):
                timing["agentview_device"] = 99.99 + i * 0.05
        w.add_frame(
            agentview_rgb=rng.integers(0, 255, (48, 64, 3), dtype=np.uint8),
            eye_in_hand_rgb=rng.integers(0, 255, (48, 64, 3), dtype=np.uint8),
            joint_positions=rng.standard_normal(7).astype(np.float32),
            gripper_position=0.5, ee_pos_quat=np.array([.4, 0, .3, 0, 0, 0, 1.]),
            gripper_closed=False,
            commanded_joint_positions=rng.standard_normal(7).astype(np.float32),
            commanded_gripper=0.0,
            ft={k: rng.standard_normal(n).astype(np.float32) for k, n in FT_OBS_FIELDS},
            timing=timing)
    return w.save_buffer(w.detach_buffer(), instruction="pick up the blue cup and "
                         "place it inside the large yellow bowl", instruction_id="I000",
                         success=True, collector="t")


rng = np.random.default_rng(0)
root = TMP / "write"
w = SceneWriter(root=root, metadata=SceneMetadata(
    scene_id="S000", objects=["OBJ-CUP-BLU-01"], layout=LAYOUT, station="t",
    dataset_version="knu-1.3.0", **PROV))
name = _episode(w, rng, with_timing=True, short_device=True)
path = w.path
w.close()
with h5py.File(path, "r") as f:
    tg = f[name]["timing"]
    for k in TIMING_REQUIRED:
        assert tg[k].dtype == np.float64 and tg[k].shape == (4,), k
    assert tg["agentview_frame_no"].dtype == np.int64
    assert list(tg["agentview_frame_no"][()]) == [500, 502, 504, 506]
    assert "agentview_device" not in tg, "a short column must not be written"
    assert tg.attrs["agentview_domain"] == "global_time"
    assert "clock" in tg.attrs
    assert "timing" not in f[name]["obs"]
    # knu-1.3.0 files carry the version once
    assert f["metadata"].attrs["dataset_version"] == "knu-1.3.0"
    assert "schema_version" not in f["metadata"].attrs
print("4. timing/<name> written with dtypes, short column skipped, domain as attr OK")

# 6a. old episodes without timing: no raise
root = TMP / "resume_old"
w = SceneWriter(root=root, metadata=SceneMetadata(
    scene_id="S000", objects=["OBJ-CUP-BLU-01"], layout=LAYOUT, station="t",
    dataset_version="knu-1.2.2", **PROV))
_episode(w, rng, with_timing=False)
w.close()
w2 = SceneWriter(root=root, scene_id="S000", resume=True, session_version="knu-1.3.0",
                 session_payload={"mass": 0.85, "com": [-0.01, 0.0, 0.03]})
assert w2.metadata.dataset_version == "knu-1.2.2", w2.version_note
assert "올리지 못했습니다" in (w2.version_note or ""), w2.version_note
w2.close()
# the doctor does not raise it either: timing cannot be backfilled
from mstack.scene.schema_doctor import fill_and_raise, reachable_version  # noqa: E402

old_path = root / "scene_000.hdf5"
assert reachable_version(old_path) == "knu-1.2.2", reachable_version(old_path)
assert fill_and_raise(old_path) == "knu-1.2.2"

# 6b. every episode has timing: raise
root = TMP / "resume_new"
w = SceneWriter(root=root, metadata=SceneMetadata(
    scene_id="S000", objects=["OBJ-CUP-BLU-01"], layout=LAYOUT, station="t",
    dataset_version="knu-1.2.2", **PROV))
_episode(w, rng, with_timing=True)
w.close()
w3 = SceneWriter(root=root, scene_id="S000", resume=True, session_version="knu-1.3.0",
                 session_payload={"mass": 0.85, "com": [-0.01, 0.0, 0.03]})
assert w3.metadata.dataset_version == "knu-1.3.0", w3.version_note
w3.close()
with h5py.File(root / "scene_000.hdf5", "r") as f:
    a = f["metadata"].attrs
    assert a["dataset_version"] == "knu-1.3.0" and "schema_version" not in a, dict(a)

# the version lives in one attribute; the removed copy is flagged by the checker
import subprocess  # noqa: E402

from mstack.scene.dataset_meta import dataset_schema_version, scene_schema_versions  # noqa: E402

assert scene_schema_versions(root) == {"S000": "knu-1.3.0"}
assert dataset_schema_version(root) == "knu-1.3.0"
with h5py.File(root / "scene_000.hdf5", "a") as f:        # an old copy comes back
    f["metadata"].attrs["schema_version"] = "knu-1.3.0"
chk = subprocess.run([sys.executable, str(Path(WT) / "scripts/check/check_scene_file.py"),
                      str(root / "scene_000.hdf5")], capture_output=True, text=True)
assert "schema_version" in chk.stdout and "폐기된 중복 속성" in chk.stdout, chk.stdout[-800:]
print("7. one version attribute: readers use dataset_version, checker flags the removed copy OK")
print("6. resume and doctor raise to 1.3.0 only when existing episodes carry timing OK")

# ---- 8. stall counting only while recording
# The 100 Hz ramps read cameras for the live view; a 30 fps camera sampled there
# repeats frames as a matter of course and used to log false "stall" warnings.
import os  # noqa: E402
import types  # noqa: E402

os.environ.setdefault("GELLO_NO_PHASE_BUS", "1")
from PyQt6.QtCore import QCoreApplication  # noqa: E402

_app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
from mstack.collect.worker import CollectionWorker, WorkerConfig  # noqa: E402


class _Client:
    def get_observations(self):
        return {"joint_positions": np.zeros(8), "ee_pos_quat": np.zeros(7),
                "joint_velocities": np.zeros(8), "state_time": time.time()}


class _Cam:
    def __init__(self):
        self.seq = 0

    def read_latest_stamped(self, max_age_ms=500):
        return np.zeros((4, 4, 3), np.uint8), {"t_host": time.time(), "seq": self.seq}


w = CollectionWorker(WorkerConfig(task_name="t", language_instruction="t", data_root="/tmp"))
cams = {"agent": _Cam(), "wrist": _Cam()}
w._robot = types.SimpleNamespace(_client=_Client(), cameras=cams)
w._depth_roles = set()
logs = []
w.log_message.connect(logs.append)
w._cam_stale, w._cam_stale_run, w._cam_stale_max_run = {}, {}, {}
for _ in range(10):                       # ramp: same frame over and over
    obs = w._get_obs()
assert obs["agent"].shape == (4, 4, 3) and not logs and not w._cam_stale, (logs, w._cam_stale)
for i in range(6):                        # recording: agent advances, wrist frozen
    cams["agent"].seq = i
    w._get_obs(count_stale=True)
assert "agent" not in w._cam_stale, w._cam_stale
assert w._cam_stale["wrist"] == 5 and w._cam_stale_max_run["wrist"] == 5, w._cam_stale
_app.processEvents()
assert sum("wrist" in m and "연속 동일" in m for m in logs) == 1, logs
print("8. stalls are counted from node seq while recording only, not in ramps OK")

print("test_frame_timing 통과")
