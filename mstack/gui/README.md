# Collection GUI

A PyQt6 workspace for configuring collection sessions, recording demonstrations,
reviewing episodes and preparing dataset uploads. The camera view stays visible
when the operator switches activities; scene plans and dataset tools share the
same session state.

## Tools and responsibilities

| Tool | Role | Intent |
| --- | --- | --- |
| PyQt6 widgets and signals | Windows, dialogs, status and progress | Keep UI updates on the Qt main thread. |
| QThread workers | Camera preview, episode loading and background jobs | Keep device waits and file reads out of UI callbacks. |
| QProcess and ZeroMQ | Robot/camera node lifecycle and communication | Let nodes own devices while the GUI consumes observations and reports failures. |
| NumPy, OpenCV and h5py | Image display and recorded episode access | Reuse dataset formats and frame alignment for review and playback. |

The collection engine owns teleoperation and episode writing. Widgets use that
engine, data/scene APIs and communication clients; they do not import concrete
robot or camera drivers.

## Code ownership

- [`apps/workspace/launcher/`](../../apps/workspace/launcher/) selects the station,
  dataset and hardware before the workspace opens.
- [`apps/workspace/shell/`](../../apps/workspace/shell/) owns the window layout;
  [`features/`](../../apps/workspace/features/) groups each activity's pages,
  operations and dialogs; [`shared/`](../../apps/workspace/shared/) holds common
  UI helpers and node lifecycle code.
- This directory provides reusable [`widgets/`](widgets/), [`dialogs/`](dialogs/)
  and [`workers.py`](workers.py). String conventions live in [`i18n.py`](i18n.py).
- [`mstack/collect/`](../collect/) owns the recording workflow;
  [`mstack/data/`](../data/) and [`mstack/scene/`](../scene/) own stored data and plans.

## Run

Complete the [environment and station setup](../../README.md#setup), then run
from the repository root in the GUI environment:

```bash
python apps/collect_launcher.py
```

The launcher configures the session and opens the workspace. To reuse the saved
settings without the launcher, run `python apps/collect_workspace.py`.

The GUI can start the robot node using the station's `node.python_path`.
For a manually managed node, run the following in the robot-node environment:

```bash
python scripts/launch/launch_nodes.py --robot fr3
```

Use the [calibration tools](../../scripts/calib/README.md) for leader setup and
[layout reference assets](../../assets/README.md) for scene placement images.

## Verification

From the repository root, using the GUI environment, without hardware or a display:

```bash
bash tests/gui/run_all.sh python
python -m mstack.scene.instruction_grammar
python -m mstack.scene.scene_rules
```

[Workspace screenshot](images/libero_collector_gui.jpg)
