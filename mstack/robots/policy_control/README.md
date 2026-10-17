# Policy control layers

This package turns model-specific VLA action chunks into continuous, constrained
FR3 joint-position commands. Both action types share a QP tracker and servo;
chunk changes update the tracking objective without a separate blending stage.

## Model contract

[`ActionSpec`](../../comm/policy_protocol.py) declares each model's output meaning
and action interval. SmolVLA uses absolute joint positions, preserving its learned
arm posture. EE waypoint models declare their units and rotation convention;
their relative poses use the fixed EE anchor captured with the observation.
Variable chunk arrival rates and inference delays do not change the action grid.

## Four layers

| Layer | Libraries / implementation | Intent |
| --- | --- | --- |
| Chunk Manager | NumPy, SciPy SLERP; [`reference.py`](reference.py) | Resolve absolute joint or relative EE targets, align the action timeline to the observation timestamp, and reject stale chunks. |
| QP Tracker (~100 Hz) | Pinocchio + PIQP's C++ solver, SciPy sparse; [`qp.py`](qp.py), [`kinematics.py`](kinematics.py) | Track joint or EE targets with posture regularization and joint position, velocity, acceleration and jerk constraints. Produce a finite joint trajectory that ends at rest. |
| Common Servo (1 kHz) | NumPy polynomial evaluation; [`servo.py`](servo.py), [`trajectory.py`](trajectory.py) | Evaluate the QP's validated trajectory and preserve position, velocity and acceleration across plan changes. Continue the installed stopping plan if a replacement is late. |
| Robot execution (1 kHz) | pylibfranka / libfranka; [`franka_fr3.py`](../franka_fr3.py) | Send absolute `JointPositions` through the existing JointImpedance controller, read robot feedback and handle errors. |

[`runtime.py`](runtime.py) keeps QP solving in a separate process so the servo
does not wait for optimization. PIQP is the current solver; Ruckig is not added
after the QP because it would alter the trajectory whose limits were checked.
The existing Python servo loop still needs hardware timing qualification.
Posture regularization is not a geometric self-collision guarantee.

See the [detailed design and offline validation workflow](../../../docs/policy-qp.md)
for constraints and failure handling, [model manifests](../../../configs/policies/)
for ActionSpec examples, and [dependencies](../../../requirements-policy-qp.txt)
for the development environment.
