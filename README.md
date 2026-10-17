# manipulation-stack

Teleoperation, demonstration collection, dataset preparation and VLA deployment
for a Franka Research 3 with a GELLO leader and RealSense cameras.

Collection and policy deployment share station configuration, camera roles and
[image preprocessing](mstack/data/crop.py) so inference observations match the
training data.

## Components and tools

| Component | Tools | Purpose |
| --- | --- | --- |
| [Collection GUI](mstack/gui/README.md) | PyQt6, Qt workers | Configure sessions, record demonstrations and review datasets while keeping the camera view visible. |
| [Datasets](docs/dataset-schema.md) and [scenes](configs/collection/plans/README.md) | NumPy, h5py, LeRobot, Hugging Face Hub | Store observations and actions with explicit schemas, organize scene plans, curate and export training data. |
| [Hardware and transport](docs/architecture.md) | pylibfranka / libfranka, Dynamixel SDK, pyrealsense2, ZeroMQ | Connect the arm, leader and cameras through device implementations and separate robot/camera nodes. |
| [Policy inference](docs/policy-server.md) | LeRobot or mamba-embeddingvla backends, HTTP | Run checkpoints on an inference server and return action chunks with their model contract. |
| [Policy control](mstack/robots/policy_control/README.md) | Pinocchio, PIQP (C++), NumPy | Interpret model-specific actions, track targets with constrained QP trajectories, and generate joint-position commands through a common servo. |

## Code organization

```text
mstack/config/       station configuration and shared constants
mstack/core/         robot, agent, camera and environment interfaces
mstack/robots/       robot implementations, including policy_control/
mstack/agents/       leader and policy clients
mstack/cameras/      camera implementations
mstack/hw/           device drivers
mstack/comm/         robot/camera protocols and process communication
mstack/data/         dataset schemas, image processing and formats
mstack/scene/        scene definitions, rules and plans
mstack/collect/      collection engine and episode writing
mstack/gui/          reusable Qt widgets, dialogs and workers
apps/                entry points and workspace UI
configs/             station, scene, collection and policy configuration
scripts/             launch, check, calibration, conversion and analysis tools
```

`config` is a shared dependency; `core` defines hardware interfaces. `data`,
`scene` and `gui` do not import concrete device implementations. `collect`
coordinates hardware for recording. These boundaries are checked by the
[dependency tests](tests/gui/test_layer_rules.py).

## Setup

Use separate Python environments for the GUI/data tools and the robot node:

| Environment | Dependencies |
| --- | --- |
| GUI, data tools and tests | `python -m pip install -r requirements.txt` |
| Robot node | `python -m pip install -r requirements_node.txt`, plus the [patched pylibfranka source build](patches/README.md) |

QP control additionally needs [requirements-policy-qp.txt](requirements-policy-qp.txt)
in a development robot-node environment; follow the
[installation and offline validation instructions](docs/policy-qp.md).

Select a station in [configs/stations/](configs/stations/) with
`GELLO_STATION=<name>`. Put robot addresses, camera serials, interpreter paths
and the policy server URL in its ignored `<name>.local.yaml` overlay, using
[local.example.yaml](configs/stations/local.example.yaml) as a template.

Start with the [GUI guide](mstack/gui/README.md),
[GELLO calibration tools](scripts/calib/README.md), or
[policy server guide](docs/policy-server.md). Run commands from the repository root.

## Development references

- [GUI checks](mstack/gui/README.md#verification) and [QP replay validation](docs/policy-qp.md#installation-and-offline-qualification)
- [Dataset quality metrics](docs/curation-metrics.md)
- [Hardware checks](scripts/check/) and [collection topology](docs/architecture.md)
- [Contributor conventions and safety boundaries](CLAUDE.md)

## License

[MIT](LICENSE). Derived from GELLO; see [NOTICE](NOTICE) for upstream attribution
and [CONTRIBUTORS.md](CONTRIBUTORS.md) for contributors.
