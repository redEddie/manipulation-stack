"""수집된 에피소드를 실로봇에서 재생한다 (검수·재현용).

기록된 ``commanded_joint_states``(텔레옵 때 리더가 보낸 명령 그 자체)를
같은 주기로 다시 보낸다 -- 정책 없이 "데이터가 로봇 위에서 재현되는가"를
확인하는 가장 직접적인 방법이다. legacy(``data/demo_N``)와
scene(``episode_NNN``) 파일을 모두 지원한다.

안전장치:
- 시작 전 에피소드의 첫 명령 포즈까지 저속 램프 후 Enter 확인
- 매 틱 |목표-실측| 클램프(MAX_STEP_RAD) -- 실제 속도 제한은 로봇 노드의
  레퍼런스 필터가 한다 (fr3_policy_client 와 동일한 이중 구조)
- ``--speed 0.5`` 처럼 감속 재생 가능 (첫 재생은 0.5 권장)
- ``--dry-run`` 은 로봇 없이 궤적 통계만 출력

전제: 로봇 노드가 떠 있어야 한다 (GUI '노드 시작' 또는
``(pylibfranka-venv) python scripts/launch/launch_nodes.py --robot fr3``).

사용:
    (lerobot-venv) python scripts/analyze/replay_episode.py \
        ~/libero_datasets/scene_000.hdf5 episode_000 --speed 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mstack.config.station import load_station  # noqa: E402
from mstack.data.dataset_schema import (  # noqa: E402
    OBS_COMMANDED_GRIPPER_STATES,
    OBS_COMMANDED_JOINT_STATES,
    OBS_GRIPPER_STATES,
    OBS_JOINT_STATES,
)
from mstack.data.frame_table import frame_table  # noqa: E402

STATION = load_station()
RAMP_STEP = 0.05      # rad/tick 램프 (수집기·정책 클라이언트와 동일 상수)
MAX_STEP_RAD = 0.50   # 재생 틱당 명령-실측 괴리 상한 (정책 클라이언트와 동일)


def _episode_names(f: h5py.File) -> list:
    root = f["data"] if "data" in f else f
    return sorted(k for k in root if k.startswith(("episode_", "demo_")))


def load_trajectory(path: Path, episode: str) -> dict:
    """(T,7) 관절 명령 + (T,) 그리퍼 명령 + 부가정보. 두 포맷 공통.

    파일 잠금(수집/공간 회수 중)과 없는 에피소드 이름은 traceback 대신
    원인과 다음 행동이 보이는 SystemExit 로 끝낸다.
    """
    try:
        h = h5py.File(path, "r")
    except BlockingIOError:
        raise SystemExit(
            f"[replay] {path.name} 이 사용 중입니다 (수집 세션이나 공간 회수가 "
            "잠그고 있음). 끝난 뒤 다시 시도하세요.") from None
    except OSError as e:
        raise SystemExit(f"[replay] 파일을 열지 못했습니다: {path} ({e})") from None
    with h as f:
        if episode not in f and not ("data" in f and episode in f["data"]):
            names = _episode_names(f)
            shown = ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")
            raise SystemExit(
                f"[replay] {path.name} 에 {episode!r} 가 없습니다.\n"
                f"  있는 에피소드({len(names)}개): {shown}")
        grp = f[episode] if episode in f else f["data"][episode]
        # 이 경로는 실제 로봇을 움직인다 -- 두 레이아웃에서 같은 행이
        # 나오는 게 안전 전제다. 명령 계열과 폴백 계열을 한 번에
        # 물어봐, 표에 실린 것들 중에서 쓸 계열을 고른다
        # (없는 obs 는 표에 안 실린다).
        #
        # rate 는 주지 않는다. 주면 그 주파수로 기록된 파일만 읽히고
        # 나머지는 거부된다. 재생 주파수는 --fps 가 따로 정하고, 여기서
        # 필요한 것은 기록된 명령 계열 그대로다.
        ft = frame_table(grp, keys=[
            OBS_COMMANDED_JOINT_STATES, OBS_COMMANDED_GRIPPER_STATES,
            OBS_JOINT_STATES, OBS_GRIPPER_STATES])
        if OBS_COMMANDED_JOINT_STATES in ft.obs:
            q = ft.obs[OBS_COMMANDED_JOINT_STATES]
            src = "commanded_joint_states"
        else:
            # 아주 옛 파일 폴백 -- 측정치 재생은 명령 재생보다 부드럽지 않다
            q = ft.obs[OBS_JOINT_STATES]
            src = "joint_states (폴백)"
        if OBS_COMMANDED_GRIPPER_STATES in ft.obs:
            g = ft.obs[OBS_COMMANDED_GRIPPER_STATES][:, 0]
        else:
            g = ft.obs[OBS_GRIPPER_STATES][:, 0]
        instr = grp.attrs.get("instruction")
        if instr is None:
            info = f["data"].attrs.get("problem_info") if "data" in f else None
            instr = str(info)[:60] if info else "(없음)"
        # 재생 주기도 파일에서 온다 -- t/control (없으면 t/agent) 의 실측 주기.
        # station fps 는 지금 찍을 기록 주기 설정이라 이 파일이 찍힌 주기가
        # 아니다. 축 시각이 없는 옛 파일만 20 Hz 로 본다.
        play_hz = None
        tg = grp.get("t")
        if tg is not None:
            for axis in ("control", "agent"):
                if axis in tg:
                    tt = tg[axis][:]
                    if len(tt) >= 2 and tt[-1] > tt[0]:
                        play_hz = float((len(tt) - 1) / (tt[-1] - tt[0]))
                        break
    return {"q": np.asarray(q, dtype=float), "grip": np.asarray(g, dtype=float),
            "source": src, "instruction": str(instr),
            "fps": 20.0 if play_hz is None else play_hz,
            "uid": None}


def describe(traj: dict, fps: float) -> None:
    q = traj["q"]
    dq = np.abs(np.diff(q, axis=0))
    print(f"  프레임      : {len(q)} ({len(q) / fps:.1f}s @ {fps:g}Hz)")
    print(f"  명령 출처   : {traj['source']}")
    print(f"  instruction : {traj['instruction']}")
    print(f"  틱당 |dq| 최대: {dq.max():.3f} rad (p99 {np.percentile(dq, 99):.3f})")
    print(f"  그리퍼      : {traj['grip'].min():.2f}..{traj['grip'].max():.2f} "
          f"(전환 {int((np.abs(np.diff(traj['grip'] > 0.5)) > 0).sum())}회)")
    print(f"  시작 포즈   : {np.round(traj['q'][0], 3)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hdf5", type=Path)
    ap.add_argument("episode", help="episode_000 (scene) 또는 demo_0 (legacy)")
    ap.add_argument("--fps", type=float, default=None,
                    help="재생 주기 (기본: 파일의 t/control 실측 주기, "
                         "없으면 t/agent, 그것도 없으면 20)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="재생 배속 (0.5 = 절반 속도, 첫 재생 권장)")
    ap.add_argument("--dry-run", action="store_true", help="로봇 없이 궤적 통계만")
    ap.add_argument("--yes", action="store_true", help="시작 확인 프롬프트 생략")
    args = ap.parse_args()

    if not 0.1 <= args.speed <= 1.0:
        raise SystemExit("--speed 는 0.1~1.0 (기록보다 빠른 재생은 지원하지 않는다)")

    traj = load_trajectory(args.hdf5, args.episode)
    # 명시 --fps 가 없으면 파일의 실측 주기로 재생한다. 120 Hz 찍힌 파일을
    # station 설정(기록 주기)으로 재생하면 6배 빠르게 나간다.
    fps = args.fps if args.fps is not None else traj["fps"]
    print(f"[replay] {args.hdf5.name} / {args.episode}")
    describe(traj, fps)
    if args.dry_run:
        print("[replay] dry-run 종료 (로봇 미접촉)")
        return

    from mstack.agents.lerobot_plugin import JOINT_KEYS, FR3ZMQRobot, FR3ZMQRobotConfig

    robot = FR3ZMQRobot(FR3ZMQRobotConfig(
        id="fr3", host=STATION.node.host, port=STATION.node.port, cameras={}))
    try:
        robot.connect()
    except Exception as e:
        raise SystemExit(
            f"[replay] 로봇 노드에 연결하지 못했습니다 "
            f"({STATION.node.host}:{STATION.node.port}, {type(e).__name__}).\n"
            "  노드를 먼저 띄우세요 -- GUI '노드 시작' 또는\n"
            "  (pylibfranka-venv) python scripts/launch/launch_nodes.py --robot fr3")

    def joints(obs) -> np.ndarray:
        return np.array([obs[k] for k in JOINT_KEYS[:7]])

    def command(q7: np.ndarray, grip: float) -> None:
        robot.send_action(dict(zip(JOINT_KEYS, np.append(q7, grip).tolist())))

    try:
        # ── 첫 명령 포즈로 저속 램프 (그리퍼는 에피소드 시작값으로) ──
        q0, g0 = traj["q"][0], float(np.clip(traj["grip"][0], 0, 1))
        print("[replay] 시작 포즈로 램프 중 ...")
        for _ in range(600):
            q = joints(robot.get_observation())
            d = q0 - q
            if np.abs(d).max() < 0.02:
                break
            command(q + np.clip(d, -RAMP_STEP, RAMP_STEP), g0)
            time.sleep(1.0 / fps)
        else:
            raise SystemExit("램프가 수렴하지 않았다 -- 로봇/노드 상태 확인")
        print("[replay] 시작 포즈 도착.")
        if not args.yes:
            input(f"Enter 를 누르면 재생을 시작합니다 "
                  f"({len(traj['q'])}프레임, {args.speed:g}x) ... ")

        dt = 1.0 / (fps * args.speed)
        t_next = time.monotonic()
        for t in range(len(traj["q"])):
            q_meas = joints(robot.get_observation())
            target = traj["q"][t]
            q_tgt = q_meas + np.clip(target - q_meas, -MAX_STEP_RAD, MAX_STEP_RAD)
            command(q_tgt, float(np.clip(traj["grip"][t], 0, 1)))
            t_next += dt
            sleep_s = t_next - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
        print(f"[replay] 완료 ({len(traj['q'])}프레임). 마지막 포즈에서 정지 상태 유지.")
        print("[replay] 홈 복귀는 GUI 나 정책 클라이언트의 램프를 사용하세요.")
    except KeyboardInterrupt:
        print("\n[replay] 중단 -- 현재 포즈 유지.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
