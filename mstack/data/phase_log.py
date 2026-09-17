"""Collection phase log -- one JSON line per phase change, with sub-ms wall time.

The workspace log already prints ``[단계] homing`` lines, but at one-second
resolution: enough to reconstruct an incident, not enough to measure a cycle.
The approach ramp is ~0.4 s and homing often under a second, so "how long does
each part of an episode take" needs the transition instants themselves.

Why a separate file instead of the collection history: a history line is one
connect session (and is written when it ends); this is every transition inside
it. They share a directory so both follow ``MSTACK_HISTORY`` -- the dev icon
points that at the collection checkout's history, and the phase log describes
the same people and sessions.

Lines::

    {"run": ..., "session": ..., "collector": ..., "dataset": ..., "scene": ...,
     "practice": false, "t": 1789900000.123456, "phase": "recording"}
    {..., "t": ..., "event": "episode_end", "outcome": "save", "success": true,
     "frames": 212}

``t`` is ``time.time()`` -- the clock the camera node, phase bus and raw robot
logger already stamp with, so the three can be lined up.

Writing never raises: a full disk must not stop a collection.
"""

from __future__ import annotations

import json
from pathlib import Path

from mstack.data.collection_history import history_path

PHASE_LOG_FILENAME = "collection_phases.jsonl"


def phase_log_path() -> Path:
    """Next to the collection history, wherever that is configured."""
    return history_path().parent / PHASE_LOG_FILENAME


def append_phase(entry: dict, path: "Path | None" = None) -> bool:
    """Append one line. Returns False (never raises) when it could not."""
    p = path or phase_log_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return True
    except OSError:
        return False


def load_phases(path: "Path | None" = None) -> list:
    """All lines, oldest first; broken lines are skipped."""
    p = path or phase_log_path()
    out: list = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out
