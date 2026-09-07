"""Live leader/follower joint monitor -- match the GELLO to the robot by hand.

Reads the GELLO (leader) and the robot (follower) and redraws a per-joint
delta table in place until every joint is under --threshold, so you can
physically move the leader arm into the pose that the collection worker's
start gate checks. No motion is commanded; the robot is only read over ZMQ.

Usage:
    # terminal 1
    python scripts/launch/launch_nodes.py --robot fr3
    # terminal 2
    python scripts/calib/gello_match_pose.py
    # move the red joints until everything is green, then Ctrl+C

The Dynamixel serial port is exclusive: stop this script (Ctrl+C) before
starting a collection session in the workspace GUI.
"""

import glob
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro
from termcolor import colored

import sys
from pathlib import Path

# 이 스크립트가 속한 체크아웃의 mstack 을 쓴다. 체크아웃이 여럿이면
# (수집용 / 개발용) sys.path 에 남의 것이 먼저 걸릴 수 있고, 그러면 여기
# 코드를 실행해도 라이브러리는 저쪽 것이 import 된다 -- 2026-08-31 실제
# 사고. 그때는 패키지 이름이 gello 였고, venv 에 남은 editable 설치가
# deploy 워크트리를 가리키고 있었다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstack.agents.gello_agent import GelloAgent  # noqa: E402
from mstack.comm.zmq_core.robot_node import ZMQClientRobot  # noqa: E402


@dataclass
class Args:
    gello_port: Optional[str] = None
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    hz: float = 10.0
    threshold: float = 0.5
    """Per-joint pass mark in radians; keep equal to MATCH_GATE_RAD
    (mstack/config/constants.py), which the collection worker's gate uses."""
    grip: str = "right"
    """Which hand holds the GELLO handle ("right" or "left").  Pass the same
    --grip the session will use, or the handle-roll joint shown here will
    be 180deg off from what the session's start gate sees."""


def main(args: Args) -> None:
    gello_port = args.gello_port
    if gello_port is None:
        usb_ports = glob.glob("/dev/serial/by-id/*")
        if not usb_ports:
            raise RuntimeError("No GELLO found, please specify --gello-port")
        gello_port = usb_ports[0]
        print(f"using GELLO port {gello_port}")

    # Same construction as the collection worker (PORT_CONFIG_MAP, no start_joints, same
    # --grip), so the deltas shown here are exactly what its start gate will
    # see.
    print(f"grip: {args.grip}")
    agent = GelloAgent(port=gello_port, grip=args.grip)
    driver = getattr(agent._robot, "_driver", None)
    if getattr(driver, "_is_fake", False):
        raise RuntimeError(
            "Dynamixel driver fell back to the fake driver. "
            "Check the port permissions and cabling."
        )

    robot = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    print("waiting for robot state (is launch_nodes.py running?)...")

    dt = 1.0 / args.hz
    n_lines = 0
    try:
        while True:
            t0 = time.time()
            leader = np.asarray(agent.act({}))
            follower = np.asarray(robot.get_joint_state())
            n = min(len(leader), len(follower))

            lines = []
            all_ok = True
            for i in range(n):
                name = f"J{i + 1}" if i < 7 else "grip"
                delta = leader[i] - follower[i]
                ok = abs(delta) <= args.threshold
                all_ok &= ok
                mark = colored("OK", "green") if ok else colored("!!", "red", attrs=["bold"])
                lines.append(
                    f"  {name:4s} leader {leader[i]:+7.3f}  follower {follower[i]:+7.3f}"
                    f"  delta {delta:+6.3f}  [{mark}]"
                )
            if all_ok:
                lines.append(
                    colored(
                        f"  MATCHED: all deltas <= {args.threshold} rad -- "
                        "Ctrl+C and start a collection session",
                        "green",
                        attrs=["bold"],
                    )
                )
            else:
                lines.append(
                    colored(f"  move the red joints (gate: {args.threshold} rad)", "yellow")
                )

            out = "\n".join("\x1b[K" + line for line in lines)
            if n_lines:
                print(f"\x1b[{n_lines}F{out}")
            else:
                print(out)
            n_lines = len(lines)

            rest = dt - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)
    except KeyboardInterrupt:
        print("\nstopped, serial port released")


if __name__ == "__main__":
    main(tyro.cli(Args))
