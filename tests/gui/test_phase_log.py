"""Phase log for cycle-time measurement (2026-09-17).

  1. every phase *change* becomes one line with sub-second wall time and the
     session identity; repeating the same phase does not add a line
  2. the end of an episode is logged with its outcome and frame count
  3. the file sits next to the collection history, so MSTACK_HISTORY moves both
  4. a write failure never raises (collection must not stop on a full disk)

No robot, no camera: the worker's hardware is never touched.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

tmp = Path(tempfile.mkdtemp(prefix="phase_log_"))
os.environ["MSTACK_HISTORY"] = str(tmp / "hist" / "collection_history.jsonl")
os.environ["GELLO_NO_PHASE_BUS"] = "1"

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv[:1])

from mstack.collect.worker import CollectionWorker, WorkerConfig  # noqa: E402
from mstack.data.phase_log import append_phase, load_phases, phase_log_path  # noqa: E402

# ---- 3. location
assert phase_log_path() == tmp / "hist" / "collection_phases.jsonl", phase_log_path()
print("3. phase log lives next to the collection history OK")

# ---- 1. transitions
cfg = WorkerConfig(task_name="t", language_instruction="t", data_root="/data/fr3-tabletop",
                   scene_id="S006", collector="tester", run_id="RUN-1")
w = CollectionWorker(cfg)
w._phase_session = "SESS-1"
t0 = time.time()
for ph in ("homing", "homing", "gate", "approach", "recording"):
    w._set_state(ph)
lines = load_phases()
assert [x["phase"] for x in lines] == ["homing", "gate", "approach", "recording"], lines
for x in lines:
    assert (x["run"], x["session"], x["collector"], x["dataset"], x["scene"]) == \
        ("RUN-1", "SESS-1", "tester", "fr3-tabletop", "S006"), x
    assert x["practice"] is False
    assert t0 - 1 < x["t"] < time.time() + 1 and x["t"] != int(x["t"]), x
ts = [x["t"] for x in lines]
assert ts == sorted(ts)
print("1. one line per phase change, fractional wall time, session identity OK")

# ---- 2. episode end
w._pending_success = True
w._log_phase(time.time(), event="episode_end", outcome="save", frames=212,
             success=w._pending_success)
end = load_phases()[-1]
assert (end["event"], end["outcome"], end["frames"], end["success"]) == \
    ("episode_end", "save", 212, True), end
print("2. episode end carries outcome, success and frame count OK")

# ---- 4. failure is silent
blocker = tmp / "file_not_dir"
blocker.write_text("x")
assert append_phase({"phase": "x"}, path=blocker / "sub" / "p.jsonl") is False
print("4. unwritable path returns False without raising OK")

print("test_phase_log 통과")
