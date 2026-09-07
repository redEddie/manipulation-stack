# 정책 클라이언트 (`apps/fr3_policy_client.py`)

학습된 VLA 체크포인트를 FR3 위에서 실행한다. GPU 머신의 정책 서버에 관측을
보내고 액션 청크를 받아 20 Hz 로 팔에 흘린다. 이 컴퓨터에서 도는 무거운 연산은
EE 체크포인트용 해석적 IK(~4 ms)뿐이다.

수집기와 **같은 전처리 함수**(`mstack.data.crop.resize_rgb`)를 쓴다. 학습 데이터를
쓸 때 부르는 그 함수다 — 두 경로가 갈리면 정책은 분포 밖 픽셀을 보게 되고, 증상은
"이유 없이 성능이 낮다" 로만 나타난다. 그래서 수집기와 정책 클라이언트가 한
저장소에 있다.

## 서버 주소

공개 저장소라 소스에 호스트를 박지 않는다. 우선순위:

1. `--server http://<host>:<port>`
2. 환경변수 `MSTACK_POLICY_SERVER`
3. `configs/stations/<이름>.yaml` 의 `policy.url`

셋 다 비어 있으면 시작할 때 무엇을 채워야 하는지 알려주고 멈춘다.

## 프로토콜 (HTTP POST JSON)

- `POST /reset {"instruction": str}` — 에피소드 시작 시 1회 (텍스트 인코딩 + 상태 초기화)
- `POST /predict {observation}` → `{"actions": [[dim] × 10]}`
  - `dim` 은 체크포인트 종류에 따라 7 (EE-delta) 또는 8 (절대 관절각)
  - `observation.state`: `[8]` — 관절 7 rad + 그리퍼 0..1 (`get_observation()` 그대로)
  - `observation.images.agent` / `.wrist`:
    `{"base64", "shape":[256,256,3], "dtype":"uint8"}`
    — **수집과 동일한 `resize_rgb`(정사각 크롭 → 256², INTER_AREA)를 클라이언트에서
    적용한 뒤** raw base64. 이 전처리를 생략하면 안 된다.

요청당 약 0.4 MB이며, `/predict` 는 백그라운드 스레드에서 돈다. IK 는 제어
스레드에 남는다 — 매 틱 최신 측정 관절각에 앵커를 다시 잡아야 추종 지연이
누적되지 않기 때문이다(학습 라벨이 프레임별 측정 pose 기준의 EE-delta 라 그렇다).

## 실행 순서 (FR3 컨트롤러 컴퓨터)

```bash
# 0) 통신 테스트 — 로봇도 카메라도 필요 없다 (합성 관측으로 서버 왕복만 확인)
(lerobot-venv) python apps/fr3_policy_client.py --dry-run

# 1) 로봇 노드. robot_ip 기본값은 FR3 의 FCI 주소이지 정책 서버가 아니다.
(pylibfranka-venv) python scripts/launch/launch_nodes.py --robot fr3

# 2) 클라이언트
(lerobot-venv) python apps/fr3_policy_client.py \
    --instruction "pick up the white cup and place it on the yellow bowl" \
    --max-seconds 60
```

`--instruction` 은 학습된 문장이어야 한다 (그 외는 분포 밖):
`pick up the {blue|white} cup and place it on the {blue|yellow} bowl`.

`waypoint` 체크포인트(청크를 관측 시점 pose 에 앵커)는 `--waypoint` 를 붙인다.

## 안전장치

시작할 때 수집기와 동일한 램프로 `libero` 리셋 자세에 복귀한 뒤 시작하고, 매 스텝
목표 관절각을 측정치 ±`MAX_STEP_RAD`(0.50)로 클램프한다. Ctrl-C 안전 종료.

이 클램프는 **속도 제한이 아니다.** 속도·가속·저크 한계는 로봇 노드의 레퍼런스
필터(1 kHz, v_max 1.0 rad/s)가 이 값과 무관하게 항상 건다. 여기서 정하는 것은
"명령이 실측보다 얼마나 앞서 나갈 수 있는가" 이고, 리더 명령 액션 공간에서는 그
앞섬 자체가 팔로워를 끌고 가는 신호다. 수집 117 에피소드 실측이 p95 0.254 /
p99 0.395 rad 라, 예전 값 0.15 로는 프레임의 20%가 잘려 지연 버그를
클라이언트에서 재현하게 된다.

## 청크 경계 정지

청크 10스텝을 다 쏜 뒤에야 추론하던 순차 구조에서는 경계마다 39 ms(추론 34.9 +
인코딩 4.2) 동안 새 명령이 없었고, 0.5 s 주기의 약한 2 Hz 흔들림으로 나타났다.

지금은 해결돼 있다 (`CHUNK_LEAD`, 기본 2틱): 청크가 끝나기 K틱 전에 미리 추론을
쏘고, 도착한 청크에서 이미 시각이 지난 앞쪽 K개는 버린다. 그대로 쏘면 후퇴가
되기 때문이다. 재학습은 필요 없었다. `--lead-ticks 0` 으로 예전 동작을 재현할 수
있다.

## 명령(commanded) 스트림

기존 action space 4종은 전부 팔로워의 **실현된(achieved)** 궤적에서 사후
재구성된다. 접촉 구간에서는 리더가 계속 밀어도 실현 delta 가 ~0 으로 붕괴하므로
조작자의 힘 의도가 궤적에 남지 않는다 — 자유공간 재생은 되는데 접촉 작업 재생이
실패하는 원인이다.

- **수집 시**: 매 프레임 GELLO 리더 명령을 스키마와 무관하게 항상 저장한다 —
  `obs/commanded_joint_states` (T,7 rad), `obs/commanded_gripper_states`
  (T,1, 0=열림..1=닫힘). 기존 `actions` 계산은 그대로다 (하위호환).
- **수집 후**: `scripts/convert/derive_commanded_ee_actions.py <task>_demo.hdf5` 가
  FR3 FK(numpy MDH, 의존성 없음)로 commanded EE pose 를 계산해 commanded delta
  액션 2종을 추가한다:
  - `actions_ee` (T,7): 프레임 t 의 팔로워 EE 좌표계 기준 명령 delta
    (LIBERO `actions_ee` 소비자와 규약 일치 — pos/rot 을 R_t^T 로 회전,
    0.05 m / 0.5 rad 정규화, [-1,1] clip)
  - `actions_world_cmd` (T,7): 같은 값의 world-frame 버전
  - flange→EE 변환은 가정하지 않고 파일별로 자가 캘리브레이션한 뒤
    (FK(joint_states) vs 기록된 `ee_pos_quat` 평균) 잔차를 검증한다 — median
    잔차가 `--max-fk-residual-mm`(기본 5 mm)를 넘으면 중단. `--dry-run` 으로
    통계만 볼 수 있고, 기존 데이터셋은 수정하지 않는다.
- **LeRobot 변환/시각화**: `convert_libero_to_lerobot.py` 가 위 필드를 자동
  감지해 함께 내보낸다 — `observation.commanded_state` (차원 이름을
  `observation.state` 와 같이 `joint1.pos`..`gripper.pos` 로 통일해서, 시각화에서
  실측과 명령이 같은 이름으로 나란히 비교된다), `actions_ee` 가 유도돼 있으면
  `action_ee` 피처로 추가한다. 일부 파일만 유도된 채 섞어 변환하면 스키마 불일치
  에러로 사전에 막는다.
