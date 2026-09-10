"""1 kHz 로봇 원시 상태 로거 -- 별도 프로세스에서 구독해 파일로 남긴다.

    python -m mstack.comm.robot_raw_logger [--die-with-parent]

**왜 별도 프로세스인가.** 파이썬 GIL 전환 간격은 기본 5 ms 다. 쓰기를 로봇
노드 안의 스레드로 두면 그 스레드가 순수 파이썬 구간을 도는 동안 1 kHz 제어
루프가 최대 5 ms 멈춘다 -- 틱 5개 분량이고 late tick 판정선은 1.5 ms 다.
실측으로도 파이썬 작업 스레드를 붙이면 루프의 **모든** 틱이 늦었다 (486/486).
프로세스를 나누면 GIL 이 아예 분리되어, 여기서 무엇을 하든 제어 루프에 닿지
않는다. 노드가 내는 비용은 발행 5.9 µs/tick -- 1 ms 예산의 0.6% 뿐이다.

**무엇을 푸는가.** 에피소드 HDF5 는 20 Hz 라 나이퀴스트가 10 Hz 다. 손으로
느끼는 진동(대개 20~100 Hz)은 원리적으로 거기 안 잡힌다. 이 로거는 1 kHz 를
그대로 받아 500 Hz 까지 남긴다.

**창(window).** 항상 다 쓰면 하루에 수십 GB 가 된다. 그래서 단계 표지
(:mod:`mstack.comm.phase_bus`)를 함께 구독해, ``TRIGGER_PHASE`` 로 들어갈 때
창을 열고 ``WINDOW_S`` 초 뒤에 닫는다. 기본값이 ``homing`` 인 이유는
2026-09-10 조작자 보고가 "녹화 종료 후 homing 첫 스텝" 이었기 때문이다 --
homing 은 한 바퀴의 첫 단계라, 여기서 열면 homing -> gate -> approach ->
recording 이 한 파일에 통째로 담긴다. 창이 열려 있는 동안의 새 트리거는
무시한다.

**실패 방향.** 이 프로세스가 죽어도 수집은 그대로 돈다. PUB 은 구독자가
없어도 블로킹하지 않는다 -- 진단이 수집을 멈추는 일은 없어야 한다.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import zmq

from mstack.comm.phase_bus import DEFAULT_PHASE_PORT, PHASE_TOPIC
from mstack.comm.robot_raw import (
    DEFAULT_RAW_PORT,
    DOF,
    RAW_FIELDS,
    RAW_LEN,
    RAW_TOPIC,
    field_slice,
)
from mstack.config.paths import state_dir

#: 창을 여는 단계. homing 이 한 바퀴의 첫 단계라 여기서 열면 한 사이클이
#: 통째로 담긴다. gate(정렬 시작)로 바꾸면 기록 구간 위주가 된다.
TRIGGER_PHASE = "homing"

#: 창 길이(초). 실측 근거: 2026-09-09 세션에서 제어 루프 중단 21건 중 20건이
#: 직전 정렬 시작으로부터 30초 안에 났고, 정렬 시작->에피소드 종료는 p99 가
#: 29초 max 39초였다. 45초면 한 바퀴와 그 뒤 homing 까지 덮는다.
WINDOW_S = 45.0

#: 남길 파일 수. 파일당 약 6.4 MB (float32) 이므로 20개면 130 MB 안쪽.
KEEP_FILES = 20

_PR_SET_PDEATHSIG = 1


def die_with_parent(sig: int = signal.SIGTERM) -> None:
    """부모가 사라지면 커널이 이 프로세스를 정리하게 한다.

    로봇/카메라 노드와 같은 계약. GUI 가 슬롯 예외로 즉사하면 이 프로세스가
    남아 포트를 쥔 채 다음 실행을 막는다.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0)
    except Exception as e:  # noqa: BLE001
        print(f"[raw-log] PDEATHSIG 설정 실패 (계속 진행): {e}", flush=True)


def out_dir() -> Path:
    """원시 로그가 쌓이는 곳. 상태 파일들과 섞이지 않게 하위 디렉터리로 둔다."""
    d = state_dir() / "robot_raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rotate(d: Path, keep: int = KEEP_FILES) -> None:
    """오래된 파일부터 지워 ``keep`` 개만 남긴다."""
    files = sorted(d.glob("raw_*.npz"), key=lambda p: p.stat().st_mtime)
    for p in files[:-keep] if len(files) > keep else []:
        try:
            p.unlink()
        except OSError:
            pass


def _write_window(d: Path, buf: np.ndarray, n: int, phases: list) -> Path:
    """창 하나를 ``.npz`` 로 쓴다. 시각만 float64, 나머지는 float32.

    자기 설명적으로 둔다 -- 필드 이름이 곧 배열 이름이라, 나중에 읽는 쪽이
    이 모듈을 몰라도 ``np.load`` 만으로 무엇인지 알 수 있다.
    """
    rows = buf[:n]
    arrays = {"t": rows[:, 0].astype(np.float64)}
    for f in RAW_FIELDS:
        arrays[f] = rows[:, field_slice(f)].astype(np.float32)
    arrays["phases"] = np.array(json.dumps(phases, ensure_ascii=False))
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(rows[0, 0] if n else time.time()))
    path = d / f"raw_{stamp}.npz"
    np.savez_compressed(path, **arrays)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-port", type=int, default=DEFAULT_RAW_PORT)
    ap.add_argument("--phase-port", type=int, default=DEFAULT_PHASE_PORT)
    ap.add_argument("--window", type=float, default=WINDOW_S)
    ap.add_argument("--keep", type=int, default=KEEP_FILES)
    ap.add_argument("--trigger", default=TRIGGER_PHASE,
                    help="이 단계로 들어가면 창을 연다 (기본 homing)")
    ap.add_argument("--die-with-parent", action="store_true")
    args = ap.parse_args(argv)

    if args.die_with_parent:
        die_with_parent()

    d = out_dir()
    ctx = zmq.Context.instance()
    raw = ctx.socket(zmq.SUB)
    raw.connect(f"tcp://127.0.0.1:{args.raw_port}")
    raw.setsockopt(zmq.SUBSCRIBE, RAW_TOPIC)
    # 창이 닫혀 있을 때 쌓이지 않게. 열려 있을 때는 폴러가 바로바로 비운다.
    raw.setsockopt(zmq.RCVHWM, 4000)
    ph = ctx.socket(zmq.SUB)
    ph.connect(f"tcp://127.0.0.1:{args.phase_port}")
    ph.setsockopt(zmq.SUBSCRIBE, PHASE_TOPIC)

    poller = zmq.Poller()
    poller.register(raw, zmq.POLLIN)
    poller.register(ph, zmq.POLLIN)

    cap = int(args.window * 1100) + 1000        # 1 kHz + 여유
    buf = np.zeros((cap, RAW_LEN), dtype=np.float64)
    n = 0
    open_ = False                                # 창이 열려 있나 (n 과 분리 --
    window_end = 0.0                             # n 을 쓰면 가짜 첫 행이 생긴다)
    phases: list = []
    prev_phase = ""

    print(f"[raw-log] 구독 시작: raw:{args.raw_port} phase:{args.phase_port} "
          f"-> {d}  (창 {args.window:.0f}s, 트리거 '{args.trigger}', {args.keep}개 유지)",
          flush=True)

    try:
        while True:
            for sock, _ in poller.poll(timeout=500):
                if sock is ph:
                    try:
                        _, body = ph.recv_multipart()
                        msg = json.loads(body)
                    except Exception:  # noqa: BLE001
                        continue
                    phase = msg.get("phase", "")
                    if phase != prev_phase:
                        prev_phase = phase
                        if open_:
                            phases.append({"phase": phase, "t": msg.get("t")})
                        elif phase == args.trigger:
                            open_ = True
                            n = 0
                            window_end = time.time() + args.window
                            phases = [{"phase": phase, "t": msg.get("t")}]
                elif sock is raw:
                    try:
                        _, payload = raw.recv_multipart()
                    except Exception:  # noqa: BLE001
                        continue
                    if not open_ or n >= cap:
                        continue
                    buf[n] = np.frombuffer(payload, dtype=np.float64)
                    n += 1

            if open_ and (time.time() >= window_end or n >= cap):
                if n == 0:
                    # 노드가 안 떠 있으면 표지만 오고 원시 틱은 없다. 빈 파일을
                    # 만들지 않는다 -- 20개 롤링을 빈 파일로 채우면 안 된다.
                    open_ = False
                    phases = []
                    continue
                try:
                    p = _write_window(d, buf, n, phases)
                    _rotate(d, args.keep)
                    print(f"[raw-log] {p.name}  {n} 틱  "
                          f"{p.stat().st_size / 1024 / 1024:.1f} MB", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[raw-log] 쓰기 실패 ({type(e).__name__}: {e})", flush=True)
                open_ = False
                n = 0
                phases = []
    except KeyboardInterrupt:
        pass
    finally:
        raw.close(linger=0)
        ph.close(linger=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
