"""카메라 프레임을 **하나도 안 버리고** 모으는가 (C6).

지금 수집 루프는 20 Hz 로 돌며 `read_latest` 로 최신 한 장만 집어간다. 카메라는
30 fps 라 프레임의 33% 가 버려진다 (S023 실측: 저장된 이웃 프레임의 frame_no
증가량이 1:2). 그런데 **드레인 스레드는 이미 그 전부를 소켓에서 꺼내고 있었다** --
최신만 남기고 버렸을 뿐이다. 그래서 "버리지 않는다" 가 곧 구현이다.

여기서 확인하는 것:

1. 수집을 켠 동안 온 color 프레임이 **전부** 쌓인다 (`frame_no` 에 구멍이 없다)
2. 20 Hz 로 `read_latest` 를 부르는 것과 **무관하게** 쌓인다 -- 루프는 미리보기와
   제어에 최신 한 장만 쓰고, 기록은 수집 버퍼를 쓴다
3. 상한을 넘으면 **조용히 덮어쓰지 않고 센다** -- 멈춘 세션이 RAM 을 먹으면 안 되고,
   얼마나 잃었는지 모르는 것이 더 나쁘다
4. 끄면 다시 안 쌓인다

가짜 노드를 서브프로세스로 띄워 **실제 ZMQ 경로**로 본다. 여기서 중요한 것이
RCVHWM=12 (30 fps 에서 0.4 초치) 다 -- 드레인이 그보다 오래 막히면 ZMQ 가 버리고,
지금까지는 최신만 쓰니 상관없었지만 C6 에서는 그것이 곧 데이터 손실이다.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

from mstack.comm.camera_client import NodeCamera, node_ping  # noqa: E402

PUB, CTL = 16031, 16032        # 다른 카메라 테스트와 겹치지 않게

node = subprocess.Popen(
    [sys.executable, "-m", "mstack.comm.camera_node",
     "--cam", "FAKE-A",
     "--pub-port", str(PUB), "--ctl-port", str(CTL), "--fake"],
    cwd=WT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
try:
    info = None
    for _ in range(40):
        info = node_ping(ctl_port=CTL, timeout_ms=500)
        if info:
            break
        time.sleep(0.2)
    assert info and info["ok"], info

    cam = NodeCamera("FAKE-A", pub_port=PUB, ctl_port=CTL)
    cam.connect()
    assert not cam.capturing

    # ---------------------------------------------- 1. 전부 쌓이는가
    cam.start_capture(max_frames=10_000)
    assert cam.capturing
    time.sleep(1.2)
    got, dropped = cam.stop_capture()
    assert not cam.capturing
    assert dropped == 0, dropped
    assert len(got) >= 10, f"1.2초 동안 {len(got)}장뿐 -- 노드가 안 보내고 있다"

    fn = np.array([m["frame_no"] for _, _, m in got])
    gaps = np.diff(fn)
    assert (gaps == 1).all(), (
        f"frame_no 에 구멍이 있다: 증가량 {sorted(set(gaps.tolist()))} -- "
        "드레인이 막혔거나 RCVHWM 에서 버려졌다. 최신만 쓰던 시절엔 무해했지만 "
        "이제는 잃어버린 프레임이다")
    print(f"1. {len(got)}장 연속 수집, frame_no 구멍 0 OK "
          f"(증가량 전부 1, {fn[0]}~{fn[-1]})")

    # ---------------------------------------------- 2. read_latest 와 무관
    # 20 Hz 루프가 최신만 집어가는 동안에도 전부 쌓여야 한다.
    cam.start_capture(max_frames=10_000)
    polled = []
    t_end = time.time() + 1.2
    while time.time() < t_end:
        polled.append(cam.read_latest(max_age_ms=500))
        time.sleep(0.05)                       # 20 Hz
    got2, _ = cam.stop_capture()
    fn2 = np.array([m["frame_no"] for _, _, m in got2])
    assert (np.diff(fn2) == 1).all(), "폴링 중에 프레임이 새어나갔다"
    assert len(got2) > len(polled), (
        f"수집 {len(got2)}장 <= 폴링 {len(polled)}회 -- 20 Hz 로 집어가는 것보다 "
        "많이 쌓여야 의미가 있다 (카메라가 더 빠르다)")
    print(f"2. 20 Hz 폴링 {len(polled)}회 중에도 {len(got2)}장 전부 쌓인다 OK "
          f"(폴링 대비 {len(got2) / len(polled):.2f}배)")

    # ---------------------------------------------- 3. 상한은 세고 멈춘다
    cam.start_capture(max_frames=5)
    time.sleep(0.8)
    got3, dropped3 = cam.stop_capture()
    assert len(got3) == 5, len(got3)
    assert dropped3 > 0, (
        "상한을 넘겼는데 버린 수가 0 이다 -- 조용히 덮어썼거나 무한정 쌓고 있다")
    print(f"3. 상한 5장에서 멈추고 {dropped3}장을 버렸다고 **센다** OK")

    # ---------------------------------------------- 4. 끄면 안 쌓인다
    time.sleep(0.4)
    got4, _ = cam.stop_capture()
    assert got4 == [], f"수집을 껐는데 {len(got4)}장이 쌓였다"
    # 그래도 최신 프레임은 계속 온다 (미리보기가 죽으면 안 된다)
    img = cam.read_latest(max_age_ms=500)
    assert img.ndim == 3, img.shape
    print("4. 끄면 안 쌓이고, 최신 프레임은 계속 온다 OK")

    cam.disconnect()
    print("\n카메라 전 프레임 수집 인수 통과")
finally:
    node.terminate()
    try:
        node.wait(timeout=5)
    except subprocess.TimeoutExpired:
        node.kill()
