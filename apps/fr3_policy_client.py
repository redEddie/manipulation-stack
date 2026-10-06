"""FR3 policy client — runs on the FR3 controller computer (lerobot-venv).

Streams observations to the GPU policy server (apps/policy_server.py, or a
legacy mamba-embeddingvla real_deploy/fr3_policy_server.py) and executes
absolute joint-angle actions on the robot. The only heavy compute here is IK
(analytic, ~4 ms) for EE ckpts.

What the checkpoint expects -- control rate, image size, chunk length, action
type -- comes from the server's GET /info (mstack/comm/policy_protocol.py), not
from constants here. A legacy server without /info gets LEGACY_INFO (20 Hz,
256^2, 10-step chunks), which is what it was built for.

Data path per replan cycle (e.g. 0.40 s = 8 executed steps @ 20 Hz):
  FR3ZMQRobot.get_observation()             joints 7 rad + gripper 0..1, 2x 640x480 RGB
    -> resize_rgb (libero_format)           square-crop -> info.image_size^2, crop_params 적용
    -> base64 JSON POST /predict            ~0.15-0.4 MB/request, on a background thread
    <- {"actions": [[dim] x chunk]}         dim=7 EE-delta (ee ckpt) or 8 joint (joint ckpt)
  then every control tick (main thread, no network):
    q_meas = get_observation()              latest measured joints
    -> raw_to_joint (fr3_kinematics)        ee: re-anchor delta to q_meas + IK -> joint8
       (joint ckpt: pass chunk[idx] through)
    -> send_action at 20 Hz                 with per-step |dq| safety clamp

The per-tick re-anchor to the LATEST measured pose matches the training label
(each frame's EE-delta is relative to that frame's measured pose), so tracking
lag never accumulates. IK stays in the CONTROL thread (fresh q_meas); /predict
stays on the background thread. This is the client half of the server's
/predict + /step split — ee_step_to_joint is byte-identical to the server's /step.

Chunk boundaries do not stall the arm.  Inference for the next chunk is fired
CHUNK_LEAD ticks before the current one runs out, and the arriving chunk's
leading indices -- whose timestamps have already passed -- are dropped rather
than commanded.  See CHUNK_LEAD.

Prerequisites:
  * robot node running:  (pylibfranka-venv) python scripts/launch/launch_nodes.py --robot fr3
  * policy server running on the GPU machine
    (--server, or MSTACK_POLICY_SERVER, or policy.url in the station file)

Comm test WITHOUT the robot (synthetic obs, checks server round-trip + latency):
  python apps/fr3_policy_client.py --dry-run

Real run:
  python apps/fr3_policy_client.py [--instruction "..."] [--max-seconds 30]

waypoint ckpt (chunk anchored at the observation pose, not per-step):
  python apps/fr3_policy_client.py --waypoint
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

# 리포 루트를 넣는다 -- apps/ 를 넣으면 mstack 을 못 찾는다. 예전에는 venv 의
# editable 설치가 우연히 메워 주고 있었다 (패키지 개명 때 드러났다).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mstack.robots.fr3_kinematics import (  # noqa: E402  (mamba real_deploy copy)
    POS_MAX_WP,
    ROT_MAX_WP,
    compute_proprio_single,
    ee_chunk_to_joint_chunk,
    ee_step_to_joint,
    fk,
)

from mstack.comm.policy_protocol import (  # noqa: E402
    EE_DELTA,
    LEGACY_INFO,
    STATE_KEY,
    PolicyInfo,
    encode_image,
    image_key,
)
from mstack.config.station import load_station

# 로봇/카메라/주파수는 스테이션 설정에서 온다 -- 수집 GUI 와 같은 파일을 읽으므로
# "수집 GUI와 동일" 이 주석이 아니라 구조로 보장된다.
# configs/stations/<이름>.yaml, GELLO_STATION 으로 선택.
STATION = load_station()

# ───────────────────────── CONFIG (edit me) ─────────────────────────
# 정책 서버 주소. 이 저장소는 공개라 기관 호스트를 박아 두지 않는다 -- 우선순위는
# --server > MSTACK_POLICY_SERVER > configs/stations/<이름>.yaml 의 policy.url 이고,
# 셋 다 비면 시작할 때 무엇을 채워야 하는지 알려주고 멈춘다.
SERVER_URL = STATION.policy.resolved_url
ROBOT_PORT = STATION.node.port             # launch_nodes.py ZMQ port
HOSTNAME = STATION.node.host
AGENT_CAMERA_SERIAL = STATION.camera("agent").serial
WRIST_CAMERA_SERIAL = STATION.camera("wrist").serial
# 추론 주기(Hz). **정책이 학습된 주기이지, 이 리그가 지금 기록하는 주기가
# 아니다.** 예전에는 STATION.fps 를 읽었고 둘이 같은 20 이라 맞았지만, 그
# 값은 station 설정의 recording.fps -- 다음 데이터세트를 어느 주기로 찍을지다.
# 그것을 120 으로 올리는 순간 20 Hz 로 학습된 정책이 120 Hz 로 실행된다:
# 정책은 "청크 인덱스 i = 관측시각 + i·dt" 로 학습돼 있으므로 dt 가 6분의 1이
# 되면 같은 청크를 6배 빠르게 쏘고, 아래 홈 복귀 램프도 RAMP_STEP 을 6배
# 자주 더해 6 rad/s 로 달리면서 상한 600틱이 5초에서 0.83초로 줄어 수렴 전에
# 예외로 죽는다.
#
# 그래서 여기는 **정책 쪽 사실**이다. The rate now comes from the server's
# GET /info (the training dataset's fps), so a 15 Hz or 30 Hz checkpoint runs at
# its own rate without editing this file. There is no FPS constant any more.
# 청크 경계 정지를 없애는 겹치기 실행. 정책은 "인덱스 i = 관측시각 + i·dt"로 학습돼
# 있으므로, 그 약속을 벽시계와 맞추기만 하면 된다 — 재학습 불필요한 클라이언트 장부 정리다.
#   청크 끝 K틱 전에 관측을 떠서 백그라운드 추론 → 도착한 청크의 앞 K개는 버리고 K번째부터 실행
# 앞 K개를 버리는 게 핵심이다. 미리 뜬 관측으로 만든 인덱스 0..K-1은 *이미 지나간 시각*의
# 목표라, 그대로 쏘면 과거 목표를 명령하는 꼴이고 이는 데이터 쪽에서 잡아낸 후퇴(지연)
# 메커니즘을 타이밍 층에서 그대로 재현한다.
# K·50ms가 추론 예산이다 (2026-08-04 실측: 평균 34.9 / p95 64.4 / 최대 98.8 ms):
#   K=2 → 100ms 예산(p95 커버), 10개 중 8개 실행, 재계획 0.40s
#   K=3 → 150ms 예산(최대치 커버), 7개 실행, 0.35s
# 비용은 반응 지연 — 정책이 K틱 오래된 장면을 보므로 돌발 상황에 그만큼 늦게 반응한다.
# K=0이면 예전 순차 동작(경계에서 팔이 89ms 정지 → 0.5s 주기 2Hz 흔들림)으로 돌아간다.
CHUNK_LEAD = 2
RESET_POSE = "libero"                      # FR3_RESET_POSES key (수집 세션과 동일해야 함)
# 학습된 4개 태스크 (다른 문장을 주면 분포 밖 — 2026-08-03 수집분 117 에피소드):
#   pick up the {blue|white} cup and place it on the {blue|yellow} bowl
DEFAULT_INSTRUCTION = "pick up the white cup and place it inside the large yellow bowl"
# 홈 복귀 램프. **속도로 적고 tick 당 이동은 파생시킨다** -- 예전에는
# 0.05 rad/tick 이라고 적혀 있었고 20 Hz 에서 1.0 rad/s 였다. 주기가 바뀌면
# 같은 상수가 그 배수만큼 빠른 램프가 된다 (수집기가 approach_speed 를
# 속도로 두는 것과 같은 이유 -- mstack/config/station.py 의 ControlSpec).
HOME_SPEED_RAD_S = 1.0
HOME_TIMEOUT_S = 30.0                      # 램프가 이 안에 수렴하지 않으면 중단
# 안전 클램프: 스텝당 "명령 목표 - 측정 위치" 최대 괴리.
# 이건 속도 제한이 아니다 — 실제 속도/가속/저크 제한은 로봇 노드의 레퍼런스 필터
# (v_max 1.0 rad/s, a_max 4.0 rad/s^2, 1 kHz)가 하고, 이 값과 무관하게 항상 건다.
# 여기서 하는 일은 "명령이 실측보다 얼마나 앞서 나갈 수 있는가"의 상한이며,
# 리더 명령 액션 공간에서는 그 앞섬 자체가 신호다(리더가 팔로워를 끌고 가는 힘).
# 학습 데이터 실측: p95 0.254 / p99 0.395 / p99.9 0.707 rad.
# 0.15이면 프레임의 20%가 잘려나가 — 우리가 고친 지연 버그를 클라이언트에서 재현한다.
MAX_STEP_RAD = 0.50                        # p99.5(0.469) 통과, 폭주 액션은 여전히 차단
GRIPPER_OPEN = 0.0
# ─────────────────────────────────────────────────────────────────────


CLIENT_CAMERAS = ("agent", "wrist")        # roles this client can capture


def fetch_info(url: str) -> PolicyInfo | None:
    """GET /info. None means a legacy server: use LEGACY_INFO, and read the action
    type from the first chunk (7 = EE-delta, 8 = joint-absolute).

    Refuses -- before the robot is touched -- a checkpoint this client cannot
    feed: a camera it does not have, or a state vector of another length."""
    r = requests.get(f"{url}/info", timeout=60)
    # The mamba real_deploy server has no do_GET at all, so http.server answers 501.
    if r.status_code in (404, 405, 501):
        print(f"[client] {url} has no /info (legacy server): assuming {LEGACY_INFO}")
        return None
    r.raise_for_status()
    info = PolicyInfo.from_dict(r.json())
    unknown = set(info.cameras) - set(CLIENT_CAMERAS)
    if unknown:
        raise SystemExit(f"checkpoint wants cameras {sorted(unknown)}; this client has {CLIENT_CAMERAS}")
    if info.state_dim != 8:
        raise SystemExit(f"checkpoint wants a {info.state_dim}-dim state; this client sends 8")
    print(f"[client] policy {info.policy}: {info.fps} Hz, {info.image_size}^2 {info.cameras}, "
          f"{info.action_type}[{info.action_dim}] x {info.chunk_size}")
    return info


def raw_to_joint(d: np.ndarray, q_meas: np.ndarray, ee_mode: bool) -> np.ndarray:
    """One raw /predict chunk action -> absolute joint action [8] (client-side /step).

    ee: re-anchor the EE-delta to the LATEST measured pose + IK. joint: passthrough.
    Identical to the server's step(); the difference is q_meas is fresh here."""
    return ee_step_to_joint(d, q_meas[:7]) if ee_mode else np.asarray(d, dtype=float)


def dry_run(url: str, instruction: str, n: int = 5) -> None:
    """Server round-trip test with synthetic obs — robot/cameras NOT required.

    Exercises the full client path: /info, /predict, then local step on each chunk row.
    For a check against recorded episodes, use scripts/check/check_policy_server.py."""
    info = fetch_info(url)
    size = info.image_size if info else LEGACY_INFO["image_size"]
    cams = info.cameras if info else LEGACY_INFO["cameras"]
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    state = np.array([0.0, -0.161, 0.0, -2.445, 0.0, 2.227, 0.785, 0.0])  # libero reset
    print(f"[dry-run] POST {url}/reset ... (서버가 없으면 여기서 최대 60초 대기)")
    r = requests.post(f"{url}/reset", json={"instruction": instruction}, timeout=60)
    r.raise_for_status()
    print(f"[dry-run] /reset ok: {r.json()}")
    fps = info.fps if info else LEGACY_INFO["fps"]
    rtt = []
    for i in range(n):
        payload = {STATE_KEY: state.tolist(), **{image_key(c): encode_image(img) for c in cams}}
        t0 = time.perf_counter()
        r = requests.post(f"{url}/predict", json=payload, timeout=60)
        r.raise_for_status()
        ms = (time.perf_counter() - t0) * 1000
        rtt.append(ms)
        chunk = np.asarray(r.json()["actions"], dtype=float)  # [K, dim]
        ee_mode = chunk.shape[1] == 7
        a0 = raw_to_joint(chunk[0], state, ee_mode)  # local step, q_meas = reset pose
        mode = "EE-delta+local-step" if ee_mode else "joint-absolute"
        print(f"[dry-run] /predict #{i}: {chunk.shape[0]}x{chunk.shape[1]} ({mode}), "
              f"round-trip {ms:.1f} ms | step->joints(rad)={np.round(a0[:7], 3)} grip={a0[7]:.2f}")
    need = int(np.ceil(max(rtt) * fps / 1000))
    print(f"[dry-run] OK — comm path + local step verified. Worst round-trip {max(rtt):.0f} ms "
          f"needs --lead-ticks >= {need} at {fps} Hz (default {CHUNK_LEAD}).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=SERVER_URL,
                    help="정책 서버 주소. 생략하면 MSTACK_POLICY_SERVER, 그 다음 "
                         "스테이션 파일의 policy.url 을 쓴다.")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--dry-run", action="store_true",
                    help="server comm test with synthetic obs (no robot needed)")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--exec-horizon", type=int, default=None,
                    help="청크 중 쓸 최대 개수. Default: the server's chunk_size from /info "
                         "(긴 청크를 더 자주 재계획하고 싶을 때만 줄인다)")
    ap.add_argument("--lead-ticks", type=int, default=CHUNK_LEAD, metavar="K",
                    help="청크 끝 K틱 전에 미리 추론하고 도착 청크의 앞 K개를 버린다 "
                         "(0=예전 순차 동작, 경계 정지 발생)")
    ap.add_argument("--waypoint", action="store_true",
                    help="waypoint ckpt: 청크를 관측 시점 pose 를 앵커로 관절 청크로 "
                         "한 번에 변환한다 (이후 타임스탬프 인덱싱으로 절대 목표 추종).")
    ap.add_argument("--proprio", action="store_true",
                    help="UMI proprioception을 계산해 /predict에 첨부 (use_eef_proprio 모델).")
    args = ap.parse_args()
    if not args.server:
        ap.error(
            "정책 서버 주소가 없다. 다음 중 하나로 준다:\n"
            "  --server http://<host>:<port>\n"
            "  MSTACK_POLICY_SERVER=http://<host>:<port>\n"
            f"  configs/stations/{STATION.name}.yaml 의 policy.url")

    # 첫 출력까지 조용한 구간(카메라/로봇/서버 연결)이 길다 — 시작 즉시 설정을 보여준다.
    print(f"[client] server {args.server} | instruction {args.instruction!r}")
    print(f"[client] exec-horizon {args.exec_horizon or 'auto (from /info)'}, lead-ticks {args.lead_ticks}, "
          f"max-seconds {args.max_seconds}" + (" (dry-run)" if args.dry_run else ""))

    if args.dry_run:
        dry_run(args.server, args.instruction)
        return

    # Everything rate- and shape-dependent below comes from the checkpoint, asked
    # for before the robot is connected so a mismatch never moves the arm.
    info = fetch_info(args.server)
    fps = info.fps if info else LEGACY_INFO["fps"]
    image_size = info.image_size if info else LEGACY_INFO["image_size"]
    exec_horizon = args.exec_horizon or (info.chunk_size if info else LEGACY_INFO["chunk_size"])
    cameras = info.cameras if info else LEGACY_INFO["cameras"]
    ramp_step = HOME_SPEED_RAD_S / fps   # rad/tick
    print(f"[client] running at {fps} Hz, sending {image_size}^2, exec-horizon {exec_horizon}")

    from lerobot.cameras.realsense import RealSenseCameraConfig

    from mstack.agents.lerobot_plugin import JOINT_KEYS, FR3ZMQRobot, FR3ZMQRobotConfig
    from mstack.data.crop import load_crop_params, resize_rgb
    from mstack.robots.franka_fr3 import FR3_RESET_POSES

    # 정책 입력의 프레이밍은 학습 데이터와 같아야 한다. 수집기가 쓰는 것과
    # 같은 crop_params.json 을 읽는다 -- 예전에는 resize_rgb 를 인자 없이
    # 불러서 zoom=1.0/x=0 고정이었고, 그건 agent zoom 1.2 / wrist +31px 로
    # 찍힌 데이터와 프레이밍이 어긋난다.
    #
    # 체크포인트가 *다른* 크롭으로 학습됐다면 그 값을 써야 한다. 각
    # 에피소드의 실제 값은 hdf5 의 attrs["crop_params"] 에 남아 있다.
    crop = load_crop_params()
    print(f"[crop] agent={crop['agent']} wrist={crop['wrist']}", flush=True)

    def _crop_resize(img, role: str):
        p = crop[role]
        return resize_rgb(img, size=image_size, zoom=p["zoom"], x_shift=p["x"], y_shift=p["y"])

    print(f"[client] connecting robot(ZMQ {HOSTNAME}:{ROBOT_PORT}) + RealSense "
          f"agent={AGENT_CAMERA_SERIAL} wrist={WRIST_CAMERA_SERIAL} ... "
          f"(카메라가 안 붙어 있으면 여기서 멈춘다)")
    robot = FR3ZMQRobot(FR3ZMQRobotConfig(
        id="fr3", host=HOSTNAME, port=ROBOT_PORT,
        cameras={
            role: RealSenseCameraConfig(
                serial_number_or_name=serial,
                fps=STATION.camera(role).fps,
                width=STATION.camera(role).width,
                height=STATION.camera(role).height)
            for role, serial in (("agent", AGENT_CAMERA_SERIAL),
                                 ("wrist", WRIST_CAMERA_SERIAL))
        }))
    robot.connect()
    print("[client] robot + cameras connected.")
    reset_q = FR3_RESET_POSES[RESET_POSE]

    def joints(obs) -> np.ndarray:
        return np.array([obs[k] for k in JOINT_KEYS[:7]])

    def command(q7: np.ndarray, grip: float) -> None:
        robot.send_action(dict(zip(JOINT_KEYS, np.append(q7, grip).tolist())))

    session = requests.Session()   # keep-alive — 매 요청 TCP 핸드셰이크 제거
    predict_ms: list[float] = []

    def predict(obs: dict) -> np.ndarray:
        """POST /predict. 워커 스레드에서 돈다 — ZMQ 소켓도 카메라도 만지지 않는다.

        정책 raw 청크만 받는다(EE-delta[7] or joint[8]). delta->joint 재앵커+IK는
        제어 스레드가 매 틱 fresh q_meas로 하므로 여기서 하지 않는다.
        관측 읽기(get_observation)는 반드시 제어 스레드에 남겨야 한다: ZMQ 소켓은
        스레드 안전하지 않다. 여기서는 이미 읽어둔 obs만 다룬다.
        resize+base64(실측 4.2ms)도 여기서 하므로 제어 틱 예산 50ms를 쓰지 않는다.
        obs의 이미지는 카메라 스레드가 매 프레임 새 배열로 갈아끼우므로(rebind이지
        in-place 쓰기가 아니다) 참조를 그대로 넘겨도 프레임이 찢어지지 않는다.
        """
        payload = {
            STATE_KEY: [float(obs[k]) for k in JOINT_KEYS],
            **{image_key(c): encode_image(_crop_resize(obs[c], c)) for c in cameras},
        }
        if args.proprio:
            q_cur = np.array([obs[k] for k in JOINT_KEYS[:7]], dtype=float)
            grip_cur = float(obs[JOINT_KEYS[7]])
            q_prev, grip_prev = q_hist[-2] if len(q_hist) >= 2 else (q_cur, grip_cur)
            payload["proprio"] = compute_proprio_single(
                q_cur, q_prev, start_q, grip_cur, grip_prev).tolist()
        t0 = time.perf_counter()
        r = session.post(f"{args.server}/predict", json=payload, timeout=60)
        r.raise_for_status()
        predict_ms.append((time.perf_counter() - t0) * 1000)
        chunk = np.asarray(r.json()["actions"], dtype=float)  # [K, dim] raw
        if args.waypoint and chunk.shape[1] == 7:
            # waypoint 규약: 청크 전체가 **이 관측의 pose** 를 기준으로 표현돼
            # 있다. 앵커가 스텝마다 움직이지 않으므로 여기(워커 스레드)서 한 번만
            # IK 로 [K,8] joint-absolute 로 바꿔 두면 된다 -- 그 뒤 제어 루프는
            # passthrough 가 되고(아래 ee_mode 가 False), 지연으로 스텝을 건너뛰어도
            # 절대 목표라 변위가 유실되지 않는다.
            q_anchor = np.array([float(obs[k]) for k in JOINT_KEYS[:7]])
            chunk = ee_chunk_to_joint_chunk(chunk, q_anchor,
                                            pos_max=POS_MAX_WP, rot_max=ROT_MAX_WP)
        return chunk

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predict")

    try:
        # ── 홈 복귀 램프 (수집기 _ramp_to와 동일 상수) ──
        print(f"[client] ramping to reset pose '{RESET_POSE}' ...")
        for _ in range(int(HOME_TIMEOUT_S * fps)):
            obs = robot.get_observation()
            q = joints(obs)
            d = reset_q - q
            if np.abs(d).max() < 0.02:
                break
            command(q + np.clip(d, -ramp_step, ramp_step), GRIPPER_OPEN)
            time.sleep(1.0 / fps)
        else:
            raise RuntimeError("reset ramp did not converge")
        print("[client] at reset pose.")
        from collections import deque
        q_hist = deque(maxlen=8)                       # (q7, grip) 20Hz 히스토리 (proprio 0.1s전)
        start_q = joints(robot.get_observation()) if args.proprio else None

        print(f"[client] POST {args.server}/reset ... (서버가 없으면 여기서 최대 60초 대기)")
        r = requests.post(f"{args.server}/reset",
                          json={"instruction": args.instruction}, timeout=60)
        r.raise_for_status()
        print(f"[client] /reset ok: {r.json()['instruction']!r}")

        dt = 1.0 / fps
        K = max(0, args.lead_ticks)
        pending = None      # 진행 중인 백그라운드 추론 (Future)
        t_obs = 0.0         # pending을 만든 관측을 뜬 시각 — 청크 인덱스 0의 기준 시각
        n_replans = n_late = 0

        # 부트스트랩: 첫 청크는 겹칠 이전 청크가 없어 블로킹으로 받는다.
        # 팔이 홈 자세에서 정지 중이라 이 한 번의 정지는 무해하다.
        chunk = predict(robot.get_observation())
        ee_mode = chunk.shape[1] == 7   # 7=EE-delta(클라 재앵커+IK), 8=joint(passthrough)
        if info and not args.waypoint and ee_mode != (info.action_type == EE_DELTA):
            raise RuntimeError(f"/info says {info.action_type}[{info.action_dim}] but the chunk "
                               f"has {chunk.shape[1]} columns -- refusing to guess")
        idx = 0
        n_replans = 1
        if K > 0 and predict_ms[-1] > K * 1000 / fps:
            # The bootstrap call includes connection setup, so this is pessimistic;
            # but a lead shorter than inference stalls the arm at every chunk boundary.
            print(f"[client] 주의: /predict {predict_ms[-1]:.0f} ms > lead budget {K*1000//fps} ms "
                  f"-- consider --lead-ticks {int(np.ceil(predict_ms[-1] * fps / 1000))}")
        if len(chunk) < exec_horizon:
            # 서버 청크가 요청보다 짧으면 슬라이스가 조용히 잘린다 -- 실제 재계획
            # 주기가 의도와 달라지므로 한 번은 눈에 보이게 알린다.
            print(f"[client] 주의: 서버 청크 {len(chunk)}개 < exec-horizon "
                  f"{exec_horizon} — 실제로 쓰는 건 {len(chunk)}개")
        print(f"[client] chunk {chunk.shape} mode={'ee' if ee_mode else 'joint'}, "
              f"lead {K}틱(추론 예산 {K*1000//fps} ms), "
              f"재계획 {(min(len(chunk), exec_horizon) - K) / fps:.2f}s")

        deadline = time.monotonic() + args.max_seconds
        t_next = time.monotonic()
        while time.monotonic() < deadline:
            horizon = min(len(chunk), exec_horizon)

            # (1) 청크 소진 → 교체. 관측시각으로부터 실제로 흐른 틱 수만큼 앞을 버린다.
            #     nominal이면 skip == K지만, 추론이 예산을 넘겼으면 그만큼 더 버린다 —
            #     "인덱스 i = 관측시각 + i·dt"를 벽시계에 맞추는 것이 규칙이고 K는 그 결과일 뿐.
            if idx >= horizon:
                if pending is None:                 # K=0 또는 예외 경로 — 예전 순차 동작
                    t_obs = time.monotonic()
                    chunk = predict(robot.get_observation())
                else:
                    if not pending.done():
                        n_late += 1                 # 예산 초과 — 여기서 경계 정지가 난다
                    chunk = pending.result()
                    pending = None
                n_replans += 1
                horizon = min(len(chunk), exec_horizon)
                skip = round((time.monotonic() - t_obs) / dt)
                if skip >= horizon:
                    print(f"[client] 경고: 추론({predict_ms[-1]:.0f} ms)이 청크 길이를 넘겼다 "
                          f"(skip {skip} >= {horizon}) — 마지막 액션만 실행한다. "
                          f"--lead-ticks를 올리거나 --exec-horizon을 늘려라.")
                idx = min(max(skip, 0), horizon - 1)
                if n_replans % 4 == 1:
                    print(f"[client] replan #{n_replans}: predict {predict_ms[-1]:.0f} ms, "
                          f"chunk {chunk.shape}, skip {skip}")

            # (2) 끝 K틱 전 → 다음 추론을 미리 띄운다. 추론이 도는 동안 남은 K틱이
            #     계속 팔을 먹이므로 경계에서 명령이 끊기지 않는다.
            if pending is None and K > 0 and idx >= horizon - K:
                t_obs = time.monotonic()
                obs = robot.get_observation()       # ZMQ는 제어 스레드에서만
                pending = pool.submit(predict, obs)

            # (3) 한 틱 실행 — 매 틱 최신 측정 pose에 재앵커(ee) / passthrough(joint)
            _o = robot.get_observation()
            q_meas = joints(_o)
            if args.proprio:
                q_hist.append((q_meas, float(_o[JOINT_KEYS[7]])))
            a = raw_to_joint(chunk[idx], q_meas, ee_mode)   # 클라측 /step (fresh anchor)
            idx += 1
            # 안전 클램프: 목표가 측정치에서 MAX_STEP_RAD 이상 벗어나지 않게
            q_tgt = q_meas + np.clip(a[:7] - q_meas, -MAX_STEP_RAD, MAX_STEP_RAD)
            command(q_tgt, float(np.clip(a[7], 0.0, 1.0)))

            t_next += dt
            sleep_s = t_next - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif sleep_s < -dt:
                t_next = time.monotonic()   # 크게 밀림 — 몰아치기 방지로 틱 시계 재동기

        print(f"[client] done ({n_replans} replans, 예산 초과 {n_late}회).")
        if predict_ms:
            p50, p95 = np.percentile(predict_ms, [50, 95])
            print(f"[client] /predict {p50:.0f} / p95 {p95:.0f} / max {max(predict_ms):.0f} ms "
                  f"(예산 {K*1000//fps} ms)")
    except KeyboardInterrupt:
        print("\n[client] interrupted.")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        robot.disconnect()
        print("[client] disconnected.")


if __name__ == "__main__":
    main()
