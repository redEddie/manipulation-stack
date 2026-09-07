from dataclasses import dataclass
from multiprocessing import Process

import tyro

import sys
from pathlib import Path

# 이 스크립트가 속한 체크아웃의 mstack 을 쓴다. 체크아웃이 여럿이면
# (수집용 / 개발용) sys.path 에 남의 것이 먼저 걸릴 수 있고, 그러면 여기
# 코드를 실행해도 라이브러리는 저쪽 것이 import 된다 -- 2026-08-31 실제
# 사고. 그때는 패키지 이름이 gello 였고, venv 에 남은 editable 설치가
# deploy 워크트리를 가리키고 있었다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstack.cameras.realsense_camera import RealSenseCamera, get_device_ids  # noqa: E402
from mstack.comm.zmq_core.camera_node import ZMQServerCamera  # noqa: E402


@dataclass
class Args:
    # 노드가 바인드할 주소. 다른 기계에서 붙으려면 그 인터페이스 주소를 준다.
    hostname: str = "127.0.0.1"


def launch_server(port: int, camera_id: int, args: Args):
    camera = RealSenseCamera(camera_id)
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera server on port {port}")
    server.serve()


def main(args):
    ids = get_device_ids()
    camera_port = 5000
    camera_servers = []
    for camera_id in ids:
        # start a python process for each camera
        print(f"Launching camera {camera_id} on port {camera_port}")
        camera_servers.append(
            Process(target=launch_server, args=(camera_port, camera_id, args))
        )
        camera_port += 1

    for server in camera_servers:
        server.start()


if __name__ == "__main__":
    main(tyro.cli(Args))
