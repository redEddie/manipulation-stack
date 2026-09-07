import ctypes
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

# 이 스크립트가 속한 체크아웃의 mstack 을 쓴다. 체크아웃이 여럿이면
# (수집용 / 개발용) sys.path 에 남의 것이 먼저 걸릴 수 있고, 그러면 여기
# 코드를 실행해도 라이브러리는 저쪽 것이 import 된다 -- 2026-08-31 실제
# 사고. 그때는 패키지 이름이 gello 였고, venv 에 남은 editable 설치가
# deploy 워크트리를 가리키고 있었다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstack.core.robot import PrintRobot  # noqa: E402
from mstack.config.station import load_station  # noqa: E402
from mstack.comm.zmq_core.robot_node import ZMQServerRobot  # noqa: E402

_PR_SET_PDEATHSIG = 1


def die_with_parent(sig: int = signal.SIGTERM) -> None:
    """Ask the kernel to signal this process when its parent goes away.

    The GUI stops this node in closeEvent, but that only runs on an orderly
    quit. A hard exit (PyQt aborts the process on an unhandled exception in a
    slot) leaves the node running and holding the robot's FCI connection, so
    the next GUI cannot start one -- and nothing on screen explains why.
    PDEATHSIG survives execve and does not depend on the parent running any
    cleanup code, which is exactly the case that was failing.

    Opt-in via --die-with-parent so a node started by hand in a terminal is
    unaffected.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0)
    except Exception as e:  # noqa: BLE001
        print(f"[node] PDEATHSIG 설정 실패 (계속 진행): {e}", flush=True)


_STATION = load_station()


@dataclass
class Args:
    """로봇 노드 인자.

    robot_port/hostname/robot_ip 의 기본값은 스테이션 설정에서 온다
    (configs/stations/<이름>.yaml, GELLO_STATION 으로 선택). CLI 인자가 이긴다.
    """

    robot: str = "fr3"
    # ZMQ REP 소켓이 열릴 포트
    robot_port: int = _STATION.node.port
    # ZMQ REP 소켓이 바인드할 주소
    hostname: str = _STATION.node.host
    # GUI가 켜줄 때 붙인다. 부모가 죽으면 커널이 이 프로세스를 종료시킨다.
    die_with_parent: bool = False
    # 로봇 팔의 IP (정책 서버 주소가 아니다). FR3 의 FCI 주소이며,
    # 기본값은 스테이션 파일에서 온다.
    robot_ip: str = _STATION.robot.ip
    # FR3 (pylibfranka) hardware options; only used when robot == "fr3".
    fr3_read_only: bool = False
    fr3_use_gripper: bool = True
    fr3_enforce_rt: bool = True


def launch_robot_server(args: Args):
    """스테이션이 고른 로봇으로 ZMQ REP 노드를 연다.

    실물 FR3 와, 로봇 없이 GUI/수집 로직만 돌릴 때 쓰는 PrintRobot 만 남아
    있다. 상류 GELLO 저장소가 지원하던 xArm/UR/Panda/YAM 과 MuJoCo 시뮬
    서버는 이 스테이션에서 쓰이지 않아 제거했다 -- 필요하면 상류를 보라.
    """
    port = args.robot_port
    if args.robot == "fr3":
        # Real FR3 via pylibfranka (see mstack/robots/franka_fr3.py).
        if not args.robot_ip:
            raise SystemExit(
                "로봇 IP 가 비어 있습니다. FCI 주소를 주세요:\n"
                "  --robot-ip <주소>\n"
                f"  또는 configs/stations/{_STATION.name}.local.yaml 의 robot.ip\n"
                "     (git 이 무시하는 로컬 파일입니다 -- 옆의 "
                f"{_STATION.name}.yaml 주석 참고)\n"
                "Franka 출고 기본값은 172.16.0.2 입니다.")
        from mstack.robots.franka_fr3 import FrankaFR3Robot

        robot = FrankaFR3Robot(
            robot_ip=args.robot_ip,
            use_gripper=args.fr3_use_gripper,
            read_only=args.fr3_read_only,
            enforce_rt=args.fr3_enforce_rt,
        )
    elif args.robot in ("none", "print"):
        robot = PrintRobot(8)
    else:
        raise NotImplementedError(
            f"Robot {args.robot} not implemented, choose one of: fr3, none/print"
        )
    server = ZMQServerRobot(robot, port=port, host=args.hostname)
    print(f"Starting robot server on port {port}")
    server.serve()


def main(args):
    if args.die_with_parent:
        die_with_parent()
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
