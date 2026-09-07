from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tyro

import sys
from pathlib import Path

# 이 스크립트가 속한 체크아웃의 mstack 을 쓴다. 체크아웃이 여럿이면
# (수집용 / 개발용) sys.path 에 남의 것이 먼저 걸릴 수 있고, 그러면 여기
# 코드를 실행해도 라이브러리는 저쪽 것이 import 된다 -- 2026-08-31 실제
# 사고. 그때는 패키지 이름이 gello 였고, venv 에 남은 editable 설치가
# deploy 워크트리를 가리키고 있었다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstack.comm.zmq_core.camera_node import ZMQClientCamera  # noqa: E402


@dataclass
class Args:
    ports: Tuple[int, ...] = (5000, 5001)
    hostname: str = "127.0.0.1"


def main(args):
    cameras = []
    import cv2

    images_display_names = []
    for port in args.ports:
        cameras.append(ZMQClientCamera(port=port, host=args.hostname))
        images_display_names.append(f"image_{port}")
        cv2.namedWindow(images_display_names[-1], cv2.WINDOW_NORMAL)

    while True:
        for display_name, camera in zip(images_display_names, cameras):
            image, depth = camera.read()
            stacked_depth = np.dstack([depth, depth, depth]).astype(np.uint8)
            image_depth = cv2.hconcat([image[:, :, ::-1], stacked_depth])
            cv2.imshow(display_name, image_depth)
            cv2.waitKey(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
