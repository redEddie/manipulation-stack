"""GELLO robot backend for a Franka FR3 controlled through `pylibfranka`.

Unlike upstream GELLO's ``panda.py`` (which targets Facebook's *polymetis*
server and was dropped with the other non-FR3 backends),
this backend talks to the robot directly with the locally built ``pylibfranka``
0.21.2 bindings (libfranka 0.21, FCI protocol v10, FR3 system image 5.10.0).
``patches/README.md`` covers how that binding is produced.

REQUIRED: ``pylibfranka`` must be patched with ``patches/pylibfranka-0.21-gil-release.diff``
and rebuilt.  The stock bindings never release the GIL, so a blocking gripper
call freezes the 1 kHz control thread and the robot aborts with
``communication_constraints_violation`` within a second of connecting.  See
``patches/README.md``.

Design
------
The GELLO teleop stack drives the robot through a synchronous ZMQ REQ/REP
server (:class:`mstack.comm.zmq_core.robot_node.ZMQServerRobot`).  Each
``command_joint_state`` call must return quickly.  ``pylibfranka`` on the other
hand exposes an *active control* interface that has to be serviced at 1 kHz
(``readOnce``/``writeOnce``).  We bridge the two:

* ``command_joint_state`` only updates a shared setpoint (``_desired_q``) and a
  shared gripper target -- it never blocks on the robot.
* A background thread owns the 1 kHz joint-position control loop.  Every tick it
  advances an internal command ``_q_cmd`` toward ``_desired_q`` through a
  critically-damped second-order reference filter saturated in jerk,
  acceleration and velocity.

  The jerk clamp is load-bearing, not a nicety.  libfranka only runs
  ``limitRate()``/``lowpassFilter()`` on the blocking ``Robot::control()`` path
  (``src/control_loop.cpp``); the active-control API we use here sends ``q_c``
  to the robot completely unshaped.  This filter is therefore the *only* thing
  keeping the command inside the reflex limits.  Without the jerk clamp, normal
  ~1 Hz hand motion on the leader saturates the acceleration clamp and flips it
  between +/-a_max on adjacent ticks -- a jerk of 2*a_max/dt = 8000 rad/s^3
  against a ``kMaxJointJerk`` of 5000, which aborts with
  ``joint_motion_generator_acceleration_discontinuity``.
* A second thread services the (blocking) gripper API as a binary
  grasp/release state machine with hysteresis (see ``_gripper_cmd_loop`` for why
  ``move()`` alone cannot hold objects).

Safety
------
* ``read_only=True`` (the default is *False*) connects and streams state but
  never starts motion -- use it to validate the pipeline first.
* Collision-reflex thresholds are 100 N*m (shoulders) / 40 N*m (wrists) and
  100 N cartesian: legitimate teleop contact never fires them (the datasheet
  joint torque limits are NOT a valid choice here -- see FR3_COLLISION_TORQUE),
  while abnormal wrist loads and violent impacts still do.  Joint impedance is
  set on connect.  The robot's hard limits stay the safety floor.
* ``max_joint_velocity``/``max_joint_acceleration`` (1.5 rad/s, 6.0 rad/s^2)
  keep real margin below libfranka's actual per-joint hard limits (2.62,
  10.0 -- see ``rate_limiting.h``/``fr3.urdf``). These were raised from
  1.0/4.0 only after adding the v_max approach taper in ``_control_loop``
  (2026-08-19): without it, hitting the speed cap kinks acceleration
  a_max -> 0 in one 1 ms tick (this loop's dt is a hardcoded constant, not
  measured actual elapsed time), a jerk spike that fired a live
  joint_motion_generator_acceleration_discontinuity abort at 1.5/6.0.
  Don't raise these without re-checking that jerk budget first.
  ``max_joint_jerk`` is a limit, not a tuning knob --
  keep it under ``franka::kMaxJointJerk`` (5000) with margin.
* ``max_setpoint_gap`` (0.9 rad, env ``MSTACK_MAX_SETPOINT_GAP``) refuses a
  setpoint that sits further than that from the filter's current output and
  parks the arm where it is.  This is the layer that catches a dropped or
  runaway leader -- and a policy that emits a nonsense pose.  Until 2026-09-10
  nothing did: the acceleration_discontinuity reflex was killing the control
  loop, which was a side effect, not a safety feature, and raising v/a plus
  fixing the v_max taper removed it.  See ``MAX_SETPOINT_GAP_RAD`` for why the
  limit is on the *gap* rather than on speed, and why 0.9 is a starting value
  meant to come down.

Make sure ``ros2_control_node`` is **not** running: the FCI accepts one client.

Note that ``~/pylibfranka-setup.md`` section 7-2 blames the FCI aborts on the
real-time setup (memlock).  That is wrong.  Nothing in libfranka or pylibfranka
calls ``mlockall()``, so the memlock limit never applied, and ``rtprio`` is
already sufficient -- libfranka sets SCHED_FIFO itself under ``kEnforce``.  The
two real causes were the GIL (see ``patches/README.md``) and the missing jerk
clamp above.
"""

import os
import threading
import time
from typing import Dict, Optional

import numpy as np

from mstack.comm.robot_raw import (
    DEFAULT_RAW_PORT,
    NO_RAW_BUS_ENV,
    RAW_FIELDS,
    RAW_LEN,
    RAW_TOPIC,
    field_slice,
)

from mstack.core.robot import Robot
from mstack.data.dataset_schema import (
    OBS_EE_WRENCH,
    OBS_EE_WRENCH_EE,
    OBS_EXT_JOINT_TORQUES,
    OBS_JOINT_TORQUES,
    ROBOT_EE_POS_QUAT,
    ROBOT_GRIPPER_POSITION,
    ROBOT_JOINT_POSITIONS,
    ROBOT_JOINT_VELOCITIES,
)

#: 관측 키 -> libfranka ``RobotState`` 필드. 로봇이 1kHz 로 이미 계산해 두는
#: 값이라 캡처는 공짜다. 셋 다 (관측 키, 상태 속성) 한 줄로 두는 이유는 필드를
#: 늘릴 때 고칠 자리를 하나로 묶기 위해서다 -- 읽기 루프 2곳, 관측 조립 1곳이
#: 전부 이 표를 돈다.
FT_STATE_ATTRS = (
    # 측정 관절토크 (N*m). 링크 사이 실제 토크 센서 값.
    (OBS_JOINT_TORQUES, "tau_J"),
    # 외력으로 추정된 관절토크 (N*m). 모델이 예측한 토크를 뺀 나머지.
    (OBS_EXT_JOINT_TORQUES, "tau_ext_hat_filtered"),
    # 외력 렌치 [Fx Fy Fz Tx Ty Tz]. 베이스 좌표 O.
    (OBS_EE_WRENCH, "O_F_ext_hat_K"),
    # 같은 렌치를 강성 좌표 K(기본값 = EE 좌표)로. 조작은 손끝 기준이라
    # 베이스 좌표보다 이쪽이 바로 쓰인다.
    (OBS_EE_WRENCH_EE, "K_F_ext_hat_K"),
)

#: 설정점(``_desired_q``)이 필터 출력(``_q_cmd``)에서 이만큼 넘게 떨어지면
#: 그 명령을 받지 않고 팔을 **그 자리에 세운다** (rad, 관절별 최대).
#:
#: 왜 이것이 필요한가. 2026-09-10 이전에 리더를 놓치거나 난폭하게 흔들면
#: libfranka 가 명령을 거부하며 acceleration_discontinuity 반사로 제어 루프를
#: 죽였다 -- 설계된 안전장치가 아니라 부작용이었지만, 실제로 팔을 세우는
#: 유일한 것이었다. v/a 를 1.5/6.0 으로 올리고(2026-08-19) v_max 접근 테이퍼의
#: 이산 항을 채워 넣으면서(2026-09-10) 그 경로가 사라졌다. 추종이 좋아진 만큼
#: 이제는 필터가 **떨어진 리더를 끝까지 충실히 쫓아간다.**
#:
#: 왜 속도가 아니라 간격인가. 필터가 이미 v_max 로 묶여 있어 팔로워는 빠르게
#: 갈 수 없다. 남는 위험은 속도가 아니라 **아무도 의도하지 않은 자세까지
#: 1.5 rad/s 로 꾸준히 밀고 가는 거리**다. 그 양이 바로 이 간격이다.
#:
#: 왜 0.9 인가. 2026-09-10 원시 로그(1 kHz) 실측 최대가 recording 0.532,
#: homing 0.157, gate/reset_wait 0.019 rad 이었다. 0.9 는 그 최대의 1.7배다 --
#: 첫 값이라 넉넉히 잡았고, **줄이는 것이 전제다**. 얼마까지 줄일 수 있는지는
#: ``scripts/analyze/setpoint_gap.py`` 가 같은 로그에서 답한다. 줄일 때는
#: 소스를 고치지 말고 아래 환경변수를 쓰면 된다.
MAX_SETPOINT_GAP_RAD = 0.9

#: ``MAX_SETPOINT_GAP_RAD`` 를 덮어쓴다. 0 이하면 검사를 끈다 (진단용).
#: 환경변수로 두는 이유는 이 값이 **실측으로 내려갈 예정**이기 때문이다 --
#: 후보값을 시험하는 데 커밋도 재빌드도 필요 없어야 한다.
SETPOINT_GAP_ENV = "MSTACK_MAX_SETPOINT_GAP"

# FR3 gripper stroke (m).  Franka Hand opens to ~0.08 m.
MAX_GRIPPER_WIDTH = 0.08

# Normalized leader-trigger value (0=open .. 1=closed) at which the binary
# gripper closes.  The GELLO leader's trigger spring (JointLimitWall) starts its
# exponential squeeze resistance at this same value, so the moment resistance is
# felt under the finger is the moment the hand grasps.
GRIPPER_CLOSE_AT = 0.6


def _setpoint_gap_limit(explicit: Optional[float] = None) -> float:
    """설정점 간격 상한 (rad). 인자 > 환경변수 > 모듈 기본값 순.

    0 이하면 검사를 끈다. 못 읽는 값이면 끄지 않고 기본값으로 간다 --
    오타 하나로 안전 검사가 조용히 사라지면 안 된다.
    """
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(SETPOINT_GAP_ENV)
    if raw is None or raw.strip() == "":
        return MAX_SETPOINT_GAP_RAD
    try:
        return float(raw)
    except ValueError:
        print(f"[FR3] {SETPOINT_GAP_ENV}={raw!r} 를 못 읽었다 -- "
              f"기본값 {MAX_SETPOINT_GAP_RAD} rad 로 간다", flush=True)
        return MAX_SETPOINT_GAP_RAD
# 폭 읽기 주기. 기록이 20Hz라 그보다 빠를 이유가 없고, read_once() 한 번이
# 30ms 안쪽이라 이 주기를 지킬 수 있다(scripts/check/check_gripper_concurrent_read.py).
GRIPPER_READ_HZ = 20.0

# Conservative default joint impedance (N*m/rad), same order as libfranka docs.
DEFAULT_JOINT_IMPEDANCE = [3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0]

# readOnce 간 간격이 이 값을 넘으면 "늦은 틱" 으로 센다. 정상은 1 ms 며,
# 네트워크/GIL 지연으로 한 틱을 놓치면 2 ms 가 된다 -- 1.5 ms 는 그 사이의
# 판별선이다.
LATE_TICK_S = 1.5e-3

# FR3 joint torque limits (N*m), datasheet.  These are the *actuation* limits,
# and they are NOT a way to disable the collision reflex: the reflex compares
# the estimated *external* torque (tau_ext_hat_filtered), which a braced
# contact pushes far beyond what the motor itself can output.  Poking a desk
# puts ~12 N*m on a wrist joint at only ~80 N of contact force (0.15 m lever),
# tripping joint_reflex before the 100 N cartesian threshold -- observed live
# on 2026-07-17.  Kept for reference; do not use as collision thresholds.
FR3_MAX_JOINT_TORQUE = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]

# Collision-reflex torque thresholds (N*m) actually used.  Sized from the
# 100 N cartesian force reflex, which fires first in any sustained tool
# contact and so caps the joint torques legitimate contact can produce:
# * J1-J4: 100.  Full-reach contact at the 100 N force cap puts up to
#   ~0.9 m x 100 N = 90 N*m on the shoulder -- the 87 N*m datasheet value
#   would nuisance-trip exactly at the force cap, so sit just above it.
# * J5-J7: 40.  The same 100 N at the hand acts on <= ~0.25 m of wrist
#   lever, i.e. <= ~25 N*m during any legitimate contact, so 40 never fires
#   in teleop -- but it still catches abnormal loads (snags, prying with a
#   long tool, the arm bracing its own wrist against an edge) at ~3x the
#   wrist's 12 N*m actuation limit, instead of giving the wrist structure
#   no watchdog at all as a flat 100 would.
# The non-tunable safety layers (Watchman limits, torque-sensor range
# faults, the hard joint limits) remain active regardless of these values.
FR3_COLLISION_TORQUE = [100.0, 100.0, 100.0, 100.0, 40.0, 40.0, 40.0]

# FR3 joint position limits (rad), same values as dsfranka's
# cpp/bridge/robot_limits.hpp.  Note J4 is never positive and J6 never reaches
# 0: the GELLO leader turns through both regions freely, so a leader pose does
# not imply a reachable FR3 pose.
#
# These are data, not enforcement.  The follower deliberately does NOT clamp
# commands to them -- silently overriding the leader decouples the two arms and
# leaves the operator with a dead zone.  They exist so the *leader* can be given
# a physical wall at the same place (scripts/check/gello_joint_limit_wall.py).
FR3_Q_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_Q_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])

# Named reset (home) poses, mirroring dsfranka's configs/teleop.yaml
# ``home.presets``.  dsfranka still spells these ``franka_ready`` / ``dsfranka``;
# the names here are the ones both repos are converging on.  All four are inside
# the FR3's joint limits.
#: 링크 축을 중심으로 도는 관절 (0-based). 팔꿈치처럼 굽히는 관절(J2/J4/J6)과
#: 달리 이쪽은 제한 없이 돌 수 있어서, 크게 어긋나 있으면 정렬이 케이블을
#: 감는 방향으로 갈 수 있다. 그래서 "정렬 중 범위 이탈" 판정은 이 관절들만
#: 본다 (2026-09-01 사용자 결정) -- 굽힘 관절이 멀리 있는 것은 그냥 자세가
#: 다른 것이라 정렬을 멈출 이유가 없다.
FR3_ROLL_JOINTS = (0, 2, 4, 6)   # J1, J3, J5, J7

FR3_RESET_POSES = {
    # Franka's built-in "ready" pose -- what the arm boots to, and where Desk's
    # "move to start" parks it.  q4 = -3pi/4.
    "fr3_ready": np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]),
    # menagerie fr3_hand / panda "home" keyframe -- matches the sim model.
    # q4 = -pi/2.
    "panda": np.array([0.0, 0.0, 0.0, -1.570796, 0.0, 1.570796, 0.785398]),
    # LIBERO's init_qpos.  The benchmark OVERRIDES the robosuite default below
    # (libero/envs/robots/mounted_panda.py); the two are ~41 deg apart at J6, so
    # do not substitute one for the other.
    "libero": np.array(
        [0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, 0.785398]
    ),
    # robosuite's default Panda init_qpos -- NOT what LIBERO uses; kept for
    # reference so the two are not confused again.
    "robosuite": np.array([0.0, 0.196350, 0.0, -2.617994, 0.0, 2.941593, 0.785398]),
}
DEFAULT_RESET_POSE = "panda"


class _RawPublisher:
    """1 kHz 원시 상태를 PUB 으로 흘린다. 실패하면 스스로 꺼진다.

    제어 루프 안에서 불리므로 규칙이 엄격하다:

    * **할당하지 않는다** -- 버퍼는 미리 잡고 슬라이스에 대입만 한다.
    * **블로킹하지 않는다** -- NOBLOCK, HWM 을 넘으면 조용히 버린다.
    * **던지지 않는다** -- 한 번이라도 실패하면 영구히 끄고 루프는 계속 돈다.

    진단이 제어 루프를 죽이는 일은 없어야 한다. 이 저장소가 루프를 죽여 본
    원인 두 가지 중 하나가 GIL 이었다 (모듈 독스트링 참고).
    """

    def __init__(self, port: int) -> None:
        self._sock = None
        self._buf = np.zeros(RAW_LEN, dtype=np.float64)
        # 슬라이스를 미리 잡아 둔다 -- field_slice() 는 이름을 선형 탐색하므로
        # 틱마다 부르면 7 µs 가 나온다 (실측). 미리 잡으면 1 µs 대다.
        self._sl = tuple(field_slice(f) for f in RAW_FIELDS)
        if os.environ.get(NO_RAW_BUS_ENV) == "1":
            return
        try:
            import zmq

            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.PUB)
            # 구독자가 밀리면 버린다. 1 kHz 라 밀리면 금방 쌓인다.
            sock.setsockopt(zmq.SNDHWM, 4000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(f"tcp://127.0.0.1:{port}")
            self._sock = sock
            self._nb = zmq.NOBLOCK
        except Exception as e:  # noqa: BLE001
            print(f"[FR3] 원시 상태 발행 비활성 ({type(e).__name__}: {e})", flush=True)
            self._sock = None

    def send(self, state, q_des=None, q_cmd=None) -> None:
        """한 틱을 발행한다.

        ``q_cmd`` 는 **직전 틱에 로봇으로 쓴 값**이다 -- 이 함수는 필터가
        돌기 전에 불리므로, 지금의 ``state`` 는 바로 그 명령에 대한 로봇의
        응답이다. 둘을 짝지어 보면 "무엇을 시켰고 무엇이 나왔나" 가 된다.
        """
        if self._sock is None:
            return
        try:
            b = self._buf
            sq, sqd, sdq, st, sdt, sdes, scmd = self._sl
            b[0] = time.time()
            b[sq] = state.q
            b[sqd] = state.q_d
            b[sdq] = state.dq
            b[st] = state.tau_J
            b[sdt] = state.dtau_J
            if q_des is not None:
                b[sdes] = q_des
            if q_cmd is not None:
                b[scmd] = q_cmd
            self._sock.send_multipart([RAW_TOPIC, b], flags=self._nb, copy=False)
        except Exception:  # noqa: BLE001 -- 한 번 실패하면 끈다
            self._sock = None

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001
                pass
            self._sock = None


class FrankaFR3Robot(Robot):
    """Franka FR3 backend driven by ``pylibfranka``.

    Args:
        robot_ip: FCI address of the robot (default ``172.16.0.2``).
        use_gripper: expose / drive the Franka Hand.
        read_only: connect and stream state but never command motion.
        enforce_rt: pass ``RealtimeConfig.kEnforce`` (True) or ``kIgnore``.
        max_joint_velocity: reference-filter velocity saturation (rad/s).
        max_joint_acceleration: reference-filter acceleration saturation (rad/s^2).
        filter_wn: natural frequency of the reference filter (rad/s); higher =
            more responsive, lower = smoother.
        home_gripper: run a blocking gripper ``homing()`` on connect (it moves!).
    """

    def __init__(
        self,
        robot_ip: str = "172.16.0.2",
        use_gripper: bool = True,
        read_only: bool = False,
        enforce_rt: bool = True,
        # 이력:
        # - 2026-07-28: 1.5/6.0 으로 올렸다가 joint_motion_generator_
        #   acceleration_discontinuity 반사로 1.0/4.0 으로 되돌림. 원인은 제어
        #   루프가 dt 를 1ms 상수로 가정해, 틱이 늘어지는 순간(ZMQ 스레드의
        #   GIL 등) 필터 내부 상태(q_cmd/qd_cmd)와 실제로 보낸 것이 어긋나고
        #   그 오차가 v_max/a_max 가 클수록 커졌기 때문.
        # - 2026-08-19 재검토: 오프라인 시뮬로 보면 반사의 유력한 원인은 dt 가
        #   아니라 속도 상한에 닿는 순간의 가속 꺾임(a_max -> 0 in 1ms = 저크
        #   a_max*1000; a_max=6 이면 J2 한계 3750 초과)이었다. 그래서 dt 는
        #   상수 그대로 두고, v_max 접근 테이퍼(아래 _control_loop)를 넣어
        #   저크를 j_max 안에 가둔 뒤 1.5/6.0 을 다시 적용. 텔레옵이 답답하고
        #   J6 같은 손목이 먼저 상한에 걸려 경로가 휘던 느낌이 이유. 문제가
        #   보이면 이 커밋 하나를 revert 하면 이전 동작(1.0/4.0, wn 10).
        max_joint_velocity: float = 1.5,
        max_joint_acceleration: float = 6.0,
        # Below franka::kMaxJointJerk (5000) with margin. The ActiveControl API
        # does NOT run libfranka's limitRate()/lowpassFilter() -- only the
        # blocking Robot::control() path does -- so this filter is the only
        # thing keeping the command inside the robot's reflex limits.
        max_joint_jerk: float = 3000.0,
        # 2026-08-19: 10 -> 14. 리더를 따라잡는 반응성(임계감쇠라 오버슈트
        # 없음). 더 올리면 20Hz 리더 계단을 그대로 실어 손목이 거칠어진다.
        filter_wn: float = 14.0,
        home_gripper: bool = False,
        collision_torque: Optional[list] = None,  # None -> FR3_COLLISION_TORQUE
        collision_force: float = 100.0,
        # None -> 환경변수 MSTACK_MAX_SETPOINT_GAP -> MAX_SETPOINT_GAP_RAD.
        max_setpoint_gap: Optional[float] = None,
    ):
        import pylibfranka as pf

        self._pf = pf
        self._read_only = read_only
        self._use_gripper = use_gripper
        self._v_max = float(max_joint_velocity)
        self._a_max = float(max_joint_acceleration)
        self._j_max = float(max_joint_jerk)
        # critically damped: kd = 2*wn, kp = wn^2
        self._kp = float(filter_wn) ** 2
        self._kd = 2.0 * float(filter_wn)
        self._dt = 1e-3  # FCI control period (1 kHz)
        self._gap_max = _setpoint_gap_limit(max_setpoint_gap)
        # 실측 최고 간격. 상한을 얼마까지 내릴 수 있는지는 결국 이 숫자가
        # 정한다 -- 그래서 세어 두고 종료할 때 찍는다.
        self._gap_peak = 0.0
        self._gap_stops = 0
        self._gap_stopped = False

        rt = pf.RealtimeConfig.kEnforce if enforce_rt else pf.RealtimeConfig.kIgnore
        print(f"[FR3] connecting to {robot_ip} (realtime={'enforce' if enforce_rt else 'ignore'})")
        self.robot = pf.Robot(robot_ip, rt)

        # Clear a leftover reflex/error state (e.g. joint_reflex from the last
        # run) so a plain restart works without touching Desk.  Harmless when
        # the robot is already Idle; fails (and only prints) when e.g. the
        # user stop is pressed -- read-only streaming still works then.
        try:
            self.robot.automatic_error_recovery()
        except Exception as e:  # noqa: BLE001
            print(f"[FR3] automatic error recovery failed: {e}")

        # Collision-reflex thresholds.  NOT the datasheet joint torque limits,
        # whose 12 N*m wrist values trip joint_reflex on mere desk contact (the
        # reflex watches estimated *external* torque; see FR3_COLLISION_TORQUE
        # for how these values are sized against the 100 N force reflex).  The
        # gripper must be able to touch the ground during teleop.  lower==upper
        # (contact report == reflex); we only care about the reflex level here.
        torque_thresh = list(collision_torque or FR3_COLLISION_TORQUE)
        self.robot.set_collision_behavior(
            torque_thresh,
            torque_thresh,
            [collision_force] * 6,
            [collision_force] * 6,
        )
        self.robot.set_joint_impedance(DEFAULT_JOINT_IMPEDANCE)

        st = self.robot.read_once()
        q0 = np.asarray(st.q, dtype=float)
        print(f"[FR3] connected. q = {np.round(q0, 3)}  mode = {st.robot_mode}")
        # 어떤 상한이 실제로 걸렸는지 찍는다. 환경변수를 걸어 놓고 그것이
        # 노드까지 안 갔는지 모르는 상태가 제일 나쁘다.
        if not read_only:
            print(
                f"[FR3] 설정점 간격 상한 {self._gap_max:.2f} rad "
                f"(내리려면 {SETPOINT_GAP_ENV}=<rad>)"
                if self._gap_max > 0.0 else
                f"[FR3] ** 설정점 간격 검사가 꺼져 있다 ({SETPOINT_GAP_ENV}={self._gap_max}) **",
                flush=True)

        # Shared state (guarded by _lock).
        self._lock = threading.Lock()
        self._q = q0.copy()            # latest measured joint positions
        self._dq = np.zeros(7)         # latest measured joint velocities
        self._desired_q = q0.copy()    # setpoint from command_joint_state
        self._q_cmd = q0.copy()        # filtered command sent to the robot
        self._qd_cmd = np.zeros(7)     # filter velocity state
        self._ee_pose = np.asarray(st.O_T_EE, dtype=float)
        # 포스·토크 (2026-08-23): franka 가 1kHz 로 이미 추정해 주는 값이라
        # 캡처는 공짜다. pylibfranka 빌드가 필드를 노출하는지 첫 상태에서
        # 한 번만 확인하고, 없으면 관측에서 키를 빼서 상류(add_frame)가
        # 기록을 생략하게 한다 -- 0 으로 채워 "측정된 무접촉"처럼 보이게
        # 하는 것이 최악이므로 조용한 0 채움은 하지 않는다.
        # 부하 모델. **정적이라 obs 가 아니라 파일 메타에 한 번 적는다.**
        # 이 값이 틀리면 미신고 질량이 통째로 외력 추정에 섞이는데(2026-09-06
        # 실측: 120 g 을 빼자 500 g 추가 613 g 으로 읽혔다), 파일에 안 남으면
        # 나중에 그 파일이 어떤 부하 설정으로 찍혔는지 알 방법이 없다.
        self._payload = {
            "mass": float(np.asarray(st.m_total, dtype=float).ravel()[0]),
            "com": np.asarray(st.F_x_Ctotal, dtype=float).ravel().tolist(),
        } if hasattr(st, "m_total") else {}
        if self._payload:
            print(f"[FR3] payload {self._payload['mass'] * 1000:.0f} g, "
                  f"com {np.round(self._payload['com'], 4).tolist()}")

        self._has_ft = all(hasattr(st, a) for _, a in FT_STATE_ATTRS)
        self._ft: Dict[str, np.ndarray] = {}
        if self._has_ft:
            self._read_ft(st)
        else:
            missing = [a for _, a in FT_STATE_ATTRS if not hasattr(st, a)]
            print(f"[FR3] robot state 에 포스·토크 필드가 없습니다: {missing} "
                  "-- 포스·토크·접촉 관측은 기록되지 않습니다 (knu-1.0.0 로 기록됨)")
        self._success_rate = 1.0
        # 틱 지연 계측 (2026-09-07): reflex 가 떴을 때 "틱이 늦었나" 를 로그로
        # 판별하기 위함. readOnce 간 간격을 재고, LATE_TICK_S 초과면 유실로 센다.
        # 비용은 틱당 monotonic() 두 번뿐이라 1 kHz 루프에 무시할 수준이다.
        self._max_tick_gap = 0.0
        self._late_ticks = 0
        self._control_error: Optional[str] = None
        self._stop = threading.Event()

        # Gripper.
        self._gripper = None
        self._gripper_target = 0.0     # normalized 1=closed, 0=open (GELLO convention)
        self._gripper_state_width = MAX_GRIPPER_WIDTH
        if use_gripper:
            self._gripper = pf.Gripper(robot_ip)
            if home_gripper:
                print("[FR3] homing gripper (this moves the fingers)...")
                self._gripper.homing()
            gs = self._gripper.read_once()
            self._gripper_state_width = float(gs.width)
            self._gripper_target = 1.0 - gs.width / MAX_GRIPPER_WIDTH

        # Background threads.
        self._control_thread: Optional[threading.Thread] = None
        self._gripper_thread: Optional[threading.Thread] = None
        self._gripper_read_thread: Optional[threading.Thread] = None
        # 읽기 스레드는 두 모드 모두에서 돈다. read_only 는 "상태만 스트리밍"인데
        # 그리퍼 폭만 갱신되지 않아, 관측의 8번째 열이 연결 시점 값에 멈춰 있었다.
        if use_gripper:
            self._gripper_read_thread = threading.Thread(
                target=self._gripper_read_loop, daemon=True
            )
            self._gripper_read_thread.start()
        if read_only:
            # Stream measured state only; never command motion.
            self._control_thread = threading.Thread(
                target=self._read_only_loop, daemon=True
            )
            self._control_thread.start()
            print("[FR3] read-only mode: streaming state, NO motion will be commanded.")
        else:
            self._control_thread = threading.Thread(
                target=self._control_loop, daemon=True
            )
            self._control_thread.start()
            if use_gripper:
                self._gripper_thread = threading.Thread(
                    target=self._gripper_cmd_loop, daemon=True
                )
                self._gripper_thread.start()
            # Give the control loop a moment to establish the 1 kHz stream.
            time.sleep(0.2)
            if self._control_error is not None:
                raise RuntimeError(f"[FR3] control loop failed to start: {self._control_error}")

    # ------------------------------------------------------------------ Robot
    def num_dofs(self) -> int:
        return 8 if self._use_gripper else 7

    def get_joint_state(self) -> np.ndarray:
        with self._lock:
            q = self._q.copy()
            gripper_norm = 1.0 - self._gripper_state_width / MAX_GRIPPER_WIDTH
        if self._use_gripper:
            return np.append(q, gripper_norm)
        return q
    
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """설정점을 받는다. 필터 출력에서 너무 먼 명령은 받지 않고 팔을 세운다.

        여기가 유일한 입구라 검사도 여기 하나로 족하다. 명령 사이에는 간격이
        늘어날 수 없다 -- 필터는 항상 ``_desired_q`` **쪽으로만** 움직이므로
        간격은 단조 비증가다. 그래서 1 kHz 루프에는 이 검사가 없어도 된다.

        멈추는 방법은 ``_desired_q`` 를 지금의 필터 출력으로 래치하는 것이다.
        속도 상태(``_qd_cmd``)는 **건드리지 않는다** -- 0 으로 꽂으면 그
        자체가 거대한 저크이고, 그것이 바로 우리가 없애려던 반사다. 필터가
        자기 저크·가속 한계 안에서 스스로 선다 (v_max/a_max 로 0.25초,
        0.19 rad).

        단순히 명령을 무시하는 것으로는 부족하다. 그러면 직전 설정점이 남아
        팔이 거기까지 계속 간다 -- 리더가 이미 떨어진 뒤라면 그 목표가 바로
        잘못된 자세다.

        걸린 상태는 따로 래치하지 않는다. 리더가 멀리 있는 동안에는 다음
        명령도 같은 이유로 걸려 팔이 계속 서 있고, 조작자가 리더를 상한 안으로
        되가져오면 그대로 이어진다. 에피소드를 끊고 게이트를 다시 태우는 것은
        리더의 의미를 아는 **워커**의 일이지 노드의 일이 아니다 -- 노드는
        에피소드를 모른다.

        VLA 배포에도 같은 검사가 걸린다 (``lerobot_plugin`` 이 이 메서드로
        내려온다). 엉뚱한 자세를 뱉는 정책도 리더를 놓친 것과 똑같이 위험하다.
        """
        joint_state = np.asarray(joint_state, dtype=float)
        q_des = joint_state[:7]
        note = ""

        with self._lock:
            # read_only 는 제어 루프가 없어 _q_cmd 가 안 움직인다. 검사하면
            # 첫 명령부터 영원히 걸린다.
            if self._gap_max > 0.0 and not self._read_only:
                d = np.abs(q_des - self._q_cmd)
                gap = float(d.max())
                self._gap_peak = max(self._gap_peak, gap)
                if gap > self._gap_max:
                    # 여기서 팔이 선다: 지금 필터가 내보내는 자리를 목표로
                    # 삼는다. 속도 상태는 그대로 두어 필터가 스스로 감속한다.
                    q_des = self._q_cmd.copy()
                    self._gap_stops += 1
                    if not self._gap_stopped:
                        self._gap_stopped = True
                        note = (f"[FR3] 안전 정지: 설정점이 J{int(d.argmax()) + 1} 에서 "
                                f"{gap:.3f} rad 떨어졌다 (상한 {self._gap_max:.2f} rad). "
                                f"리더를 상한 안으로 되가져오면 이어진다.")
                elif self._gap_stopped:
                    self._gap_stopped = False
                    note = f"[FR3] 안전 정지 해제 (간격 {gap:.3f} rad)"
            self._desired_q = q_des.copy()
            # 그리퍼는 계속 리더를 따른다. 위험한 것은 팔의 질량과 도달거리이지
            # 손가락이 아니고, 세워 둔 동안 물체를 놓지도 못하게 하면 곤란하다.
            if self._use_gripper and len(joint_state) >= 8:
                self._gripper_target = float(np.clip(joint_state[7], 0.0, 1.0))

        # 락 밖에서 찍는다 -- print 는 1 kHz 루프도 기다리는 락이다.
        if note:
            print(note, flush=True)

    def get_observations(self) -> Dict[str, np.ndarray]:
        # If the 1kHz control thread has died (e.g. a reflex abort), q/dq/
        # pose just stop updating forever -- silently returning that frozen
        # snapshot makes the robot look "stopped" to every caller with no
        # error anywhere (this is how a dead control loop turned into a
        # silent GUI freeze instead of a visible failure). Surface it
        # instead: raise here so it becomes a request-level error over ZMQ
        # (see zmq_core/robot_node.py), which the client already knows how
        # to catch and react to.
        if self._control_error is not None:
            raise RuntimeError(f"FR3 control loop is dead: {self._control_error}")
        with self._lock:
            q = self._q.copy()
            dq = self._dq.copy()
            pose = self._ee_pose.copy()
            gripper_norm = 1.0 - self._gripper_state_width / MAX_GRIPPER_WIDTH
            ft = {k: v.copy() for k, v in self._ft.items()}
        if self._use_gripper:
            pos = np.append(q, gripper_norm)
            vel = np.append(dq, 0.0)
        else:
            pos, vel = q, dq
        out = {
            ROBOT_JOINT_POSITIONS: pos,
            ROBOT_JOINT_VELOCITIES: vel,
            ROBOT_EE_POS_QUAT: self._pose_to_pos_quat(pose),
            ROBOT_GRIPPER_POSITION: np.array(gripper_norm),
        }
        # 포스·토크: 필드를 노출하는 pylibfranka 빌드에서만 키가 존재한다.
        # 소비자(mstack.collect.worker._get_obs)는 .get() 으로 읽으므로 키 부재는
        # "기록 안 함"이지 오류가 아니다.
        out.update(ft)
        return out

    def payload(self) -> dict:
        """부하 모델 (질량 kg, 플랜지 기준 무게중심 m). 연결 때 한 번 읽은 값."""
        return dict(self._payload)

    def _read_ft(self, st) -> None:
        """``self._ft`` 를 갱신한다. 호출자가 ``self._lock`` 을 쥐고 있어야 한다."""
        for key, attr in FT_STATE_ATTRS:
            self._ft[key] = np.asarray(getattr(st, attr), dtype=float)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _pose_to_pos_quat(o_t_ee: np.ndarray) -> np.ndarray:
        """Convert a column-major 4x4 flat pose into [x, y, z, qx, qy, qz, qw]."""
        T = np.asarray(o_t_ee, dtype=float).reshape(4, 4, order="F")
        pos = T[:3, 3]
        R = T[:3, :3]
        # Robust rotation-matrix -> quaternion (Shepperd's method, simple form).
        tr = np.trace(R)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return np.array([pos[0], pos[1], pos[2], qx, qy, qz, qw])

    # ------------------------------------------------------------------ loops
    def _read_only_loop(self) -> None:
        while not self._stop.is_set():
            try:
                st = self.robot.read_once()
                with self._lock:
                    self._q = np.asarray(st.q, dtype=float)
                    self._dq = np.asarray(st.dq, dtype=float)
                    self._ee_pose = np.asarray(st.O_T_EE, dtype=float)
                    self._success_rate = float(st.control_command_success_rate)
                    if self._has_ft:
                        self._read_ft(st)
            except Exception as e:  # noqa: BLE001
                self._control_error = str(e)
                print(f"[FR3] read-only loop error: {e}")
                break
            time.sleep(0.001)

    def _control_loop(self) -> None:
        pf = self._pf
        raw_pub = None
        try:
            ctrl = self.robot.start_joint_position_control(
                pf.ControllerMode.JointImpedance
            )
            # First tick: latch the command to the robot's own desired q_d, not
            # the measured q.  The robot-side motion generator differentiates
            # the incoming command stream starting from q_d, so any q-vs-q_d
            # gap becomes a phantom velocity step: 0.011 rad (observed after
            # hand-guiding on system 5.10.0, where q_d resyncs lazily) reads as
            # 11 rad/s in one tick and aborts with joint_motion_generator_
            # velocity/acceleration_discontinuity before teleop even starts.
            # Same convention as libfranka's own examples (q_d start pose).
            state, _ = ctrl.readOnce()
            with self._lock:
                self._q = np.asarray(state.q, dtype=float)
                q_d = np.asarray(state.q_d, dtype=float)
                gap = float(np.abs(q_d - self._q).max())
                if gap > 0.05:
                    # Impedance (3000 N*m/rad) would yank the arm toward a
                    # stale q_d.  Refuse instead; recovery/brake cycle resyncs.
                    raise RuntimeError(
                        f"stale desired pose: max|q - q_d| = {gap:.3f} rad; "
                        "run error recovery or re-open the brakes, then relaunch"
                    )
                self._q_cmd = q_d.copy()
                self._qd_cmd = np.zeros(7)
                self._desired_q = q_d.copy()
                q_cmd = self._q_cmd.copy()
                qd_cmd = self._qd_cmd.copy()
            cmd = pf.JointPositions(list(q_cmd))
            ctrl.writeOnce(cmd)

            # dt 는 상수 1ms 로 둔다 (2026-08-19 검토). 틱을 하나 놓치면 로봇
            # 타임라인에서는 명령이 한 틱 정지했다가 재개되는데, 실측 dt 로
            # '2ms 어치' 를 한 번에 보내면 그 재개 점프가 두 배가 되어 가속
            # 불연속을 오히려 키운다. 상수 dt 는 궤적이 1ms 늦어질 뿐 점프가
            # 작다 -- 로봇이 보는 것은 명령 스트림이지 필터의 시간 인식이
            # 아니다.
            dt = self._dt
            acc_prev = np.zeros(7)
            t_prev = time.monotonic()
            last_late_log = 0.0
            # 진단용 1 kHz 원시 상태 발행. 실패해도 루프는 그대로 돈다.
            raw_pub = _RawPublisher(DEFAULT_RAW_PORT)
            while not self._stop.is_set():
                state, _ = ctrl.readOnce()
                t_now = time.monotonic()
                gap = t_now - t_prev
                t_prev = t_now
                if gap > self._max_tick_gap:
                    self._max_tick_gap = gap
                if gap > LATE_TICK_S:
                    self._late_ticks += 1
                    # 유실이 있을 때만, 초당 1 번까지 -- 스팸 방지.
                    if t_now - last_late_log >= 1.0:
                        last_late_log = t_now
                        print(f"[FR3] late tick: gap {gap * 1e3:.1f} ms "
                              f"(누적 {self._late_ticks}회, "
                              f"success_rate={self._success_rate:.4f})",
                              flush=True)
                with self._lock:
                    target = self._desired_q.copy()
                    self._q = np.asarray(state.q, dtype=float)
                    self._dq = np.asarray(state.dq, dtype=float)
                    self._ee_pose = np.asarray(state.O_T_EE, dtype=float)
                    self._success_rate = float(state.control_command_success_rate)
                    if self._has_ft:
                        self._read_ft(state)
                # 목표를 읽은 뒤에 발행한다 -- 필터의 입력과 출력이 같은
                # 틱에서 맞물려야 나중에 맞대 볼 수 있다.
                raw_pub.send(state, target, q_cmd)

                # Critically-damped second-order reference filter, saturated in
                # jerk, acceleration and velocity -> smooth, bounded command.
                #
                # The jerk clamp is what keeps the robot from tripping
                # joint_motion_generator_acceleration_discontinuity: the leader
                # setpoint is a noisy 100 Hz staircase, so an unclamped acc
                # flips between +/-a_max on adjacent ticks, which is a jerk of
                # 2*a_max/dt -- far past franka::kMaxJointJerk.
                err = target - q_cmd
                acc_target = np.clip(
                    self._kp * err - self._kd * qd_cmd, -self._a_max, self._a_max
                )
                # 속도 상한에 닿기 전에 가속을 미리 줄인다. 없으면 v_max 에
                # 닿는 순간 가속이 한 틱에 꺾여 저크가 로봇 한계를 넘는다.
                #
                # 연속시간 공식 sqrt(2 j gap) 은 **두 가지를 빠뜨린다** (2026-09-10
                # 실측으로 확인). 조작자가 일부러 크게 움직여 반사를 유도한
                # 로그에서 저크 5105 rad/s^3 이 나왔고 (로봇 한계 5000), 원인이
                # 이것이었다:
                #
                # 1. **이산 보정.** 가속을 a 에서 0 으로 저크 j 로 내리는 데
                #    a/(j dt) 틱이 걸리고, 그동안 속도가 a^2/(2j) 가 아니라
                #    a^2/(2j) + a dt/2 만큼 는다.
                # 2. **당해 틱 소비.** 지금 틱에도 가속 a 가 속도를 a dt 만큼
                #    올린다. 그 몫을 gap 에서 먼저 빼야 한다.
                #
                # 둘을 넣어 a^2/(2j) + a dt/2 <= gap - |a_prev| dt 를 a 에 대해
                # 풀면 아래 식이다. 실측 재생: 5105 -> 3000 (초과 0틱), 가속
                # 상한 6.0 은 그대로라 성능 손실이 없다. 스파이크가 없던 창은
                # 그대로 3000 이다.
                #
                # 옛 식이 늦게 걸린 이유: gap < a_max^2/(2j) = 0.006 rad/s 에서만
                # 개입하는데, 6 rad/s^2 로 달려오면 그 구간이 한 틱뿐이고 저크
                # 클램프는 한 틱에 j dt = 3 밖에 못 바꾼다.
                lag = np.abs(acc_prev) * dt
                gap_up = np.maximum(self._v_max - qd_cmd - lag, 0.0)
                gap_dn = np.maximum(self._v_max + qd_cmd - lag, 0.0)
                jdt = self._j_max * dt
                acc_up = (-jdt + np.sqrt(jdt * jdt + 8.0 * self._j_max * gap_up)) / 2.0
                acc_dn = (-jdt + np.sqrt(jdt * jdt + 8.0 * self._j_max * gap_dn)) / 2.0
                acc_target = np.clip(acc_target, -acc_dn, acc_up)
                dacc_max = self._j_max * dt
                acc = np.clip(acc_target, acc_prev - dacc_max, acc_prev + dacc_max)

                qd_new = np.clip(qd_cmd + acc * dt, -self._v_max, self._v_max)
                # Feed back the acceleration that actually survived the velocity
                # clamp; using the pre-clamp value would let acc_prev drift away
                # from reality and spike the moment the clamp releases.
                acc_prev = (qd_new - qd_cmd) / dt
                qd_cmd = qd_new
                q_cmd = q_cmd + qd_cmd * dt

                cmd = pf.JointPositions(list(q_cmd))
                if self._stop.is_set():
                    cmd.motion_finished = True
                ctrl.writeOnce(cmd)

                with self._lock:
                    self._q_cmd = q_cmd.copy()
                    self._qd_cmd = qd_cmd.copy()
        except Exception as e:  # noqa: BLE001
            # 이 줄이 반사의 이름을 가진 원본이다 (libfranka 의 abort 메시지:
            # "Move command aborted: motion aborted by reflex! [...]").
            # flush 가 없으면 파이프로 넘어갈 때 블록 버퍼에 갇히는데, 이
            # 프로세스는 그 뒤로 아무것도 안 찍을 수도 있어서 그대로 사라진다
            # -- 상류 GUI 에서 "반사 종류가 안 보인다" 였던 이유의 절반이다
            # (나머지 절반은 worker 가 예외를 버린 것, 2026-09-06).
            # 틱 계측을 꼬리에 단다: success_rate 가 1.0 이고 late tick 이 0
            # 이면 틱 지연설은 기각이고 다른 원인을 찾아야 한다 (2026-09-07).
            self._control_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            self._control_error += (
                f" [tick: max_gap={self._max_tick_gap * 1e3:.1f} ms, "
                f"late={self._late_ticks}, success_rate={self._success_rate:.4f}]"
            )
            print(f"[FR3] CONTROL LOOP ABORTED: {self._control_error}", flush=True)
        finally:
            # 포트를 놓아 준다 -- 루프가 다시 서면 같은 포트에 다시 바인드해야
            # 하는데, 남아 있으면 두 번째 바인드가 실패해 발행이 조용히 꺼진다.
            if raw_pub is not None:
                raw_pub.close()

    def _gripper_read_loop(self) -> None:
        """Samples the measured finger width, and does nothing else.

        This is a separate thread from the one that commands the hand, and that
        separation is the whole point. ``grasp``/``move`` block until the
        fingers stop -- 1.39 s for a full stroke, measured -- so a thread that
        both commands and reads stops reading for the entire motion. What
        reached the .hdf5 files was therefore not a width trajectory but a
        frozen value that jumped once: over 254 episodes the reading held for 37
        frames (1.85 s at 20 Hz) after the close command and then moved 100% of
        its range in a *single* frame. The delay is real physics and a policy
        has to learn it; a step function standing in for the ramp is not.

        Two threads sharing one ``franka::Gripper`` is not something libfranka
        documents as safe, so it was measured before being relied on
        (``scripts/check/check_gripper_concurrent_read.py``): 0 exceptions and a clean
        ramp through both a grasp and a move, at the full 20 Hz.

        Exceptions are swallowed rather than latched into ``_control_error``:
        a missed width sample is a gap in an observation channel, not a reason
        to tear down a running session.
        """
        period = 1.0 / GRIPPER_READ_HZ
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                gs = self._gripper.read_once()
                with self._lock:
                    self._gripper_state_width = float(gs.width)
            except Exception:  # noqa: BLE001
                pass
            # sleep 이 아니라 wait: stop() 이 걸리면 주기를 기다리지 않고 나간다.
            self._stop.wait(max(0.0, period - (time.monotonic() - t0)))

    def _gripper_cmd_loop(self) -> None:
        """Drive the Franka Hand as a binary grasp/release state machine.

        ``move(width, speed)`` must not be used to close on objects: when the
        fingers hit an object short of the target width, the command aborts
        *without holding force* and the fingers unload and back off a little --
        the "touches the object, then reopens" failure.  The only call that
        holds force is ``grasp(width, speed, force, eps_in, eps_out)``, and it
        self-releases after ~1 s if the final width lands outside
        ``[width - eps_in, width + eps_out]``, so the window must span the
        whole stroke.  (franka_ros#130, libfranka#93.)

        The hand cannot servo width continuously anyway (~0.8 s per stroke), so
        the leader trigger (0=open .. 1=closed) is discretized with a
        hysteresis: closing edge at >= ``close_at`` issues
        ``grasp(0.0, ..., eps=0.08)`` -- success at any object width, held at
        ``grasp_force`` -- and the opening edge at <= ``open_at`` issues
        ``move`` back to full open.  Do NOT narrow the epsilons (see above).

        Commands are issued on state edges only and block this thread for up to
        a stroke.  That is fine: the GIL-release patch keeps the 1 kHz control
        thread running, and a pending edge is acted on as soon as the in-flight
        command returns.  On an exception the state is left unchanged, so the
        edge is retried while the trigger still demands it.
        """
        close_at = GRIPPER_CLOSE_AT  # trigger crossing that closes the hand ...
        open_at = 0.2      # ... and the one that reopens it (hysteresis)
        speed = 0.1        # m/s, Franka Hand max
        grasp_force = 40.0  # N holding force
        eps = 0.08         # m; success window must span the full stroke

        with self._lock:
            closed = self._gripper_target > 0.5  # match the hand's startup state
        while not self._stop.is_set():
            with self._lock:
                target = self._gripper_target
            try:
                if not closed and target >= close_at:
                    ok = self._gripper.grasp(
                        0.0, speed, grasp_force,
                        epsilon_inner=eps, epsilon_outer=eps,
                    )
                    closed = True
                    if not ok:
                        # Unreachable with a full-stroke window; if it fires,
                        # the epsilons no longer cover the stroke.
                        print("[FR3] gripper grasp reported failure")
                elif closed and target <= open_at:
                    self._gripper.move(MAX_GRIPPER_WIDTH, speed)
                    closed = False
            except Exception as e:  # noqa: BLE001
                print(f"[FR3] gripper {'grasp' if not closed else 'move'} failed: {e}")
            self._stop.wait(0.05)

    # ------------------------------------------------------------------ misc
    @property
    def control_command_success_rate(self) -> float:
        with self._lock:
            return self._success_rate

    def stop(self) -> None:
        self._report_gap_peak()
        self._stop.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=1.0)
        if self._gripper_thread is not None:
            self._gripper_thread.join(timeout=1.0)
        if self._gripper_read_thread is not None:
            self._gripper_read_thread.join(timeout=1.0)
        try:
            self.robot.stop()
        except Exception:  # noqa: BLE001
            pass

    def _report_gap_peak(self) -> None:
        """이번 실행에서 설정점 간격이 실제로 얼마나 벌어졌는지 남긴다.

        상한을 얼마까지 내릴 수 있는지 정하는 것은 결국 이 숫자다. 상한이
        0.9 로 시작한 것은 실측 최대 0.532 의 1.7배라는 이유뿐이고,
        **내리는 것이 전제**다 -- 그러려면 매 세션의 최고값이 보여야 한다.
        1 kHz 원시 로그에서 단계별로 더 자세히 보려면
        ``scripts/analyze/setpoint_gap.py`` 를 쓴다.
        """
        if self._read_only or self._gap_max <= 0.0 or self._gap_peak <= 0.0:
            return
        headroom = self._gap_max - self._gap_peak
        msg = (f"[FR3] 설정점 간격 최고 {self._gap_peak:.3f} rad "
               f"(상한 {self._gap_max:.2f}, 여유 {headroom:.3f})")
        if self._gap_stops:
            msg += f"  안전 정지 {self._gap_stops}회"
        print(msg, flush=True)

    def __del__(self):
        try:
            self.stop()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    # Read-only smoke test (no motion).
    import tyro

    def main(robot_ip: str = "172.16.0.2", use_gripper: bool = True):
        r = FrankaFR3Robot(robot_ip=robot_ip, use_gripper=use_gripper, read_only=True)
        try:
            while True:
                print("q =", np.round(r.get_joint_state(), 3),
                      " success=", round(r.control_command_success_rate, 3))
                time.sleep(0.5)
        except KeyboardInterrupt:
            r.stop()

    tyro.cli(main)
