# ActionSpec policy control on dev

Policy deployment defaults to `--controller qp`. SmolVLA retains its seven
absolute joint targets, including the learned redundant posture. Models whose
postprocessed outputs are relative EE waypoints use a separate EE objective.
Both paths produce the same constrained joint trajectory and use the same
1 kHz servo. There is no cross-chunk blending stage.

## Model contract

`GET /info` includes `action_spec` (version 1). The server loads an explicit
`--action-spec FILE`, otherwise `action_spec.json` beside/in the checkpoint.
Only SmolVLA has an implicit default: `joint_absolute`, radians, joint1..joint7,
then gripper 0=open/1=closed. Other models must declare their **postprocessed**
output semantics. Model name and output dimension cannot determine EE units,
coordinate frame, incremental versus fixed-anchor meaning, or gripper encoding.
A model whose own postprocessor already converts to absolute joints must declare
joint actions even if its network internally predicts relative actions.

Examples live in `configs/policies/`:

| Kind | Arm values | Interpretation |
| --- | --- | --- |
| `joint_absolute` | 7 radians | Preserve all seven joint targets |
| `ee_relative_waypoint` | xyz + rotation vector | Relative to the single EE pose at the captured observation |

For waypoints, `p_i = p_anchor + R_anchor * (position_scale * xyz_i)` and
`R_i = R_anchor * Exp(rotation_scale * rotvec_i)`. The anchor does not change at
arrival, at each control tick, or from waypoint to waypoint. Units after scaling
are metres and radians. `gripper_encoding` is `zero_one` or `minus_one_one`;
post-decoding hand values saturate to [0,1]. The normalized example uses .3 m /
1.5 rad and a signed gripper; it is an example, not a default for all EE models.

`action_dt` is the training action interval. `first_action_offset` defaults to
zero and can explicitly represent checkpoints whose first action is one step
in the future. 20/30 Hz **chunk arrivals do not change this grid**. Unknown
fields, versions, conventions, dimensions and nonfinite values are rejected.
Legacy per-tick `ee_delta` has a different meaning and is available only through
the explicit `--controller filter` path. It is never reinterpreted as waypoints.

## Runtime and ownership

1. `mstack/agents/policy_chunk_client.py` captures joints/cameras and the node's
   monotonic state timestamp, sends one asynchronous HTTP request, and submits
   the returned whole chunk. HTTP stalls do not block heartbeats. The observation
   timestamp/anchor/sequence/epoch are retained with each request; new server
   dumps include this context and `policy_info.json`.
2. `reference.py` resolves joint or fixed-anchor EE targets. Within-chunk joint
   positions and EE translations interpolate linearly; EE rotations use SLERP.
   A new chunk replaces the reference objective immediately. Out-of-order,
   expired, future-dated or wrong-session chunks cannot replace valid input.
3. `qp.py` runs in a spawned node-local process, nominally every 10 ms. The sparse
   PIQP C++ solver handles both objectives. Pinocchio supplies the calibrated
   TCP velocity Jacobian for both action kinds, plus FK and the SE(3) error
   Jacobian for EE tracking. Soft posture retention,
   regularization and a linearization trust region avoid Jacobian inversion and
   damp motion near singularities. Joint policies do not acquire a home-posture
   objective or lose their redundant joint target.
4. `servo.py` evaluates the validated joint polynomial once per libfranka command
   tick. `franka_fr3.py` sends `JointPositions` under the existing JointImpedance
   controller. It bypasses the old second-order filter while QP owns the arm.
   Teleoperation/collection and `--controller filter` retain that filter.

QP entry requires the existing command to settle at rest, fresh measured state,
and a measured/commanded joint gap below .05 rad. Both action kinds additionally
check that calibrated FK agrees with libfranka's configured TCP frame (1 cm / .05 rad).
The gripper's existing worker remains separate. Normal LeRobot observation
features are unchanged; only `get_policy_observation()` exposes QP metadata.

While QP owns the arm, direct joint commands are rejected. `hold()` latches this
policy session, cancels future activations, and executes the current validated
trajectory to rest. It never resets velocity/acceleration discontinuously.
`disable_policy(epoch)` is allowed only after latched hold reaches terminal rest.
A new session is required to resume after a fault/hold.

## Constraints and the two clocks

The default horizon is 30 x 10 ms. Decision variables are endpoint q/v/a and
acceleration change, with exact constant-jerk integration. Hard constraints:

- Existing FR3 joint ranges, inset by .01 rad.
- |v| <= 1.5 rad/s, |a| <= 6 rad/s², |jerk| <= 3000 rad/s³ (or lower node limits).
- Conservative piecewise-linear inner bounds on FR3 position-dependent velocity.
- Terminal velocity and acceleration equal zero.
- TCP linear speed norm <= .40 m/s and angular speed norm <= .80 rad/s,
  with independent continuous nonlinear validation as described below.

Cubic position and quadratic velocity Bezier control-point bounds certify each
whole segment, not only its sampled endpoints. Acceleration endpoints and the
constant jerk complete the certificate. FR3 velocity-envelope chords lie below
the concave square-root limit; tangents would be unsafe outer approximations.
The returned trajectory is independently checked after solving and again before
installation. Infeasible, inaccurate, nonfinite or late solutions are rejected.

Reference time is the robot node's monotonic observation clock. Polynomial time
is the count of successfully generated 1 ms commands. A delayed tick advances
only one command interval; it does not jump forward to catch the wall clock.
Plans start on the common 10 ms knot grid, at least 20 command ticks ahead. Their
initial q/v/a must equal the scheduled command at that activation tick. New
plans replace future plans only with this continuity check. Chunk time alignment
therefore remains independent of inference arrival rate and server clock skew.

If a result misses its activation tick, the already installed finite plan remains
valid. A worker failure, excessive measured/commanded position error (.15 rad),
or missing client heartbeat (.5 s) latches hold. An expired chunk stops further
replanning. Every accepted plan ends at rest within .3 s of its own start;
a hold on the active plan finishes in at most its remaining .3 s. This is a
bounded normal stop, not an emergency stop or a shortest-distance braking policy.
The .5 s communication timeout adds to that stopping latency.

The commanded polynomial satisfies the configured limits. Actual robot motion,
impedance tracking, contact forces, actuator torques and scheduler timing need
separate hardware qualification. There is no collision geometry, obstacle model,
or guaranteed self-collision avoidance here. Posture retention and joint limits
help avoid unwanted winding but do not prove the arm cannot collide with itself.
Pinocchio is a kinematics dependency, not a collision-avoidance policy by itself.

## TCP speed caps

The initial defaults are **0.40 m/s** at the configured TCP origin and
**0.80 rad/s** angular speed, for both SmolVLA joint targets and EE waypoints.
Defaults live in `mstack/config/policy_control.py`. Node operators can set
`--policy-ee-linear-velocity` and `--policy-ee-angular-velocity` on
`scripts/launch/launch_nodes.py`; the active values appear in `policy_status()`
as `tcp_speed_limits`. They apply to the QP policy path. Teleoperation and the
explicit legacy filter retain their existing controller.

The QP uses the geometric Jacobian at the TCP origin, not the SE(3) tracking-error
Jacobian. Thirteen pairs of linear inequalities approximate each 3-D Euclidean
speed ball from inside, with 5% headroom, around the nominal joint trajectory.
These frozen-Jacobian rows are a planning surrogate, not the final certificate.

`cartesian.validate_speed()` evaluates the actual Jacobian and bounds speed
between evaluation times using an analytic Cartesian acceleration bound. Link
translation lengths bound first and second derivatives of TCP position; joint
velocity Bezier handles and acceleration endpoints bound their coefficients
over each segment. A midpoint speed plus this acceleration bound times half the
interval covers the whole interval. Inconclusive intervals are bisected at most
five times; exhausted validation is a rejection, not a sampled-only acceptance.
The planner allows one relinearization after a TCP validation failure. The node
independently validates again outside the servo lock. Failed or late candidates
leave the previous validated stopping plan active; there is no servo clipping.
This introduces speed limits, not an additional EE acceleration limit.

The selection used 25 retained legacy policy-server sessions, 156 requests and
7,644 within-chunk joint transitions at a 50 ms action interval. Estimated
within-chunk TCP speeds had p95 values of **0.409 m/s / 0.798 rad/s**. The selected
caps exceed about 95% of those target speeds. Twelve large chunk transitions
were replayed with four candidates:

| Linear / angular cap | Mean joint tracking RMS | Median best-fit alignment lag |
| --- | --- | --- |
| .20 m/s / .50 rad/s | .04318 rad | 137.5 ms |
| .30 m/s / .60 rad/s | .03903 rad | 125 ms |
| **.40 m/s / .80 rad/s** | **.03752 rad** | **115 ms** |
| .50 m/s / 1.00 rad/s | .03705 rad | 110 ms |

The previous joint-only QP had .03689 rad mean tracking RMS on those cases.
The selected cap increased it by about 1.7%, while reducing the reproduced TCP
peaks from .675 m/s / 1.163 rad/s to .379 m/s / .731 rad/s. All four candidates
had zero failed plans or limit violations. Comparisons fix computation latency
to one tick to isolate the cap's effect; alignment lag is a diagnostic fit, not
measured robot response time. Scheduling is exercised separately by the normal
replay and process benchmark.

With measured compute latency, the final 12-case replay produced 3,470 plans
without failures or limit violations. Full update time (reference preparation,
planning and validation) was 6.66 ms median, 10.55 ms p99 and 12.55 ms observed
maximum. Thus 100 Hz remains a nominal rate, not an every-cycle guarantee.
Synthetic variable-arrival replay also verified rejection of nine deliberately
late plans while retaining a valid stopping trajectory.

These logs contain predicted chunks and sparse observations, not dense measured
robot motion or labelled successful demonstrations. Server timing is only an
estimate of capture/activation timing. The defaults are provisional operating
caps; they do not establish an optimal learning speed or contact safety.
Reproduce the audit and candidate comparison with:

```bash
python scripts/analyze/policy_ee_speed.py \
  --log-root /path/to/policy-server/dumps --output /path/to/speed-results \
  --compare-boundaries 12
```

## Why PIQP and no second trajectory generator

The previous ~0.58 s offline outlier came predominantly from native solver
iterations, not Python input preparation. Translating only the wrapper to C++
would not fix that formulation. This implementation uses sparse stage dynamics,
fixed matrix patterns and PIQP's native C++ solver. Earlier compact-form tests
also found PIQP's tail latency more favorable than the tested ProxQP settings.
ProxQP can be benchmarked behind the same planner contract later; its name is
not part of the action protocol.

Ruckig is not cascaded after this QP. A second generator could change the path
that the QP validated. The common servo instead evaluates the same polynomial;
there is no second optimization/filter, clipping step, or cross-chunk fade.
QP failure never relaxes safety constraints to improve EE tracking.

## Installation and offline qualification

Use a dedicated development robot-node environment. Install
`requirements-policy-qp.txt` there, with `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1` set **before starting the node**. The pinned URDF/tinyxml
versions match Pinocchio 3.8's binary ABI. The existing patched pylibfranka build
and realtime configuration remain required; no vendor SDK was changed.

Offline commands (no robot is needed):

```bash
python -m pytest -q tests/control tests/gui/test_policy_safety.py
python scripts/analyze/replay_policy_qp.py \
  --log-root /path/to/policy-server/dumps --output /path/to/replay-results
bash tests/gui/run_all.sh /path/to/dev/python
```

Replay uses the production ActionSpec, reference, QP and servo classes. It selects
large real joint-policy chunk transitions before running the candidate, samples
the command at 1 kHz, checks limits and terminal stopping, and reports timing
percentiles including target preparation and validation. Additional cases use
synthetic EE targets and alternating 20/30 Hz arrivals, out-of-order results and
injected 35 ms compute delays. Legacy server logs cannot recover exact client
capture/activation times; their alignment is estimated from server timestamps.
The raw action data stay outside the repository.

For a later operator-controlled hardware trial, the updated policy server reports
the checkpoint spec and the client uses `--controller qp` (the new default).
An explicit client `--action-spec` permits an old server only when it lacks a spec;
it cannot silently override an explicit server spec. `--inference-hz` limits HTTP
requests, not action playback speed. `--exec-horizon` can truncate a returned
chunk; `--lead-ticks` and `--waypoint` belong to the legacy filter path.

Measured offline timing is not a certified worst-case execution time. The current
1 kHz ActiveControl loop is still Python and requires measured scheduling/jitter
qualification on the actual robot computer. No hardware motion, production
environment update, or merge to main is part of the offline implementation.
