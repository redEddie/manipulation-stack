"""User-configurable LIBERO dataset schema: which action space to compute
``actions`` from, and which observation fields actually get written.

Persisted as JSON so a custom configuration chosen in the GUI (see
mstack.gui.dialogs.DatasetSchemaDialog) survives across restarts.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from mstack.config.paths import state_dir

ACTION_SPACE_EE_DELTA = "ee_delta"
ACTION_SPACE_EE_ABSOLUTE = "ee_absolute"
ACTION_SPACE_JOINT_DELTA = "joint_delta"
ACTION_SPACE_JOINT_ABSOLUTE = "joint_absolute"
ACTION_SPACES = (
    ACTION_SPACE_EE_DELTA,
    ACTION_SPACE_EE_ABSOLUTE,
    ACTION_SPACE_JOINT_DELTA,
    ACTION_SPACE_JOINT_ABSOLUTE,
)

ACTION_SPACE_LABELS = {
    ACTION_SPACE_EE_DELTA: "EE-delta (LIBERO 기본)",
    ACTION_SPACE_EE_ABSOLUTE: "EE-pose absolute (절대 목표 pose)",
    ACTION_SPACE_JOINT_DELTA: "Joint-angle delta (변화량)",
    ACTION_SPACE_JOINT_ABSOLUTE: "Joint-angle absolute (리더 명령, ACT 규약)",
}

DEFAULT_CONFIG_PATH = state_dir() / "dataset_schema.json"


# --------------------------------------------------------- dataset schema 버전
# 우리 데이터셋 형식의 버전 (GitHub issue #41). SemVer 를 쓰되 접두사로
# 출처를 밝힌다: knu-MAJOR.MINOR.PATCH.
#
#   MINOR = 필드 '추가'만. 옛 리더가 새 파일을 열어도 자기가 아는 필드는
#           그대로 있으니 읽힌다(전방 호환), 새 리더가 옛 파일을 열면
#           추가 필드만 없다(후방 호환). 예: 조인트 토크 추가 -> knu-1.1.0.
#   MAJOR = 기존 필드의 의미·단위·이름 변경이나 삭제. 리더는 자기 MAJOR
#           안의 모든 MINOR 를 읽을 수 있어야 한다.
#   PATCH = 데이터에 영향 없는 명세 문서 수정.
#
# 명세 문서는 docs/dataset-schema.md 이고, 이 상수들이 그 문서의 코드 쪽
# 정본이다 (문서와 어긋나면 검증기가 잡는다).
#: 지금 쓰는(기록하는) 버전. 읽기는 같은 MAJOR 안에서 위아래 모두 된다
#: (schema_is_readable 참조).
#: LeRobot 내보내기 기본 주기 (Hz). **기록 주기(station 의 recording.fps)와
#: 다른 값이다.** 기록 주기는 "지금부터 무엇을 찍을까" 이고, 이것은 "이미 찍은
#: 것을 어느 주기로 학습에 낼까" 다. 둘을 한 값으로 묶어 두면 기록을 120 Hz 로
#: 올리는 순간 30 fps 카메라 이미지가 행마다 네 번 중복된 영상이 여섯 배
#: 용량으로 나간다.
#:
#: 20 인 이유: frame_table 이 control_hz 를 정수로 나누는 주기만 받는데 20 은
#: 새 파일(120/20=6)과 옛 파일(20/20=1)을 모두 나눈다 (30 은 옛 파일에서
#: 거부된다). 그리고 지금 배포 정책이 학습된 주기다.
#: 데이터세트별 선택은 내보내기 프로파일(E3)이 맡는다.
DEFAULT_EXPORT_FPS = 20

#: 화면이 내미는 내보내기 주기 선택지. **파일의 control_hz 를 정수로 나누는
#: 값만** 쓸 수 있다 (frame_table 이 그렇지 않으면 거부한다): 120 Hz 파일은
#: 20(x6)·30(x4) 둘 다 되고, 20 Hz 로 찍힌 옛 파일은 20 만 된다.
#: 30 은 "사진마다 한 행" 이고, 20 은 지금 배포 정책이 학습된 주기다.
#: 목록일 뿐 제한은 아니다 -- 칸은 편집 가능이라 다른 값도 칠 수 있다.
EXPORT_FPS_CHOICES = (20, 30)

SCHEMA_VERSION = "knu-2.0.0"

# --------------------------------------------------------- observation/dataset keys
# Robot observation keys (returned by Robot.get_observations / RobotEnv.get_obs).
ROBOT_JOINT_POSITIONS = "joint_positions"
ROBOT_JOINT_VELOCITIES = "joint_velocities"
ROBOT_EE_POS_QUAT = "ee_pos_quat"
ROBOT_GRIPPER_POSITION = "gripper_position"
#: Host wall time (``time.time()``) at which the 1 kHz loop read the state the
#: other keys came from. Lets a recording measure how old the robot half of a
#: frame is -- see TIMING_* below.
ROBOT_STATE_TIME = "state_time"

# Dataset observation keys (stored under episode/obs in HDF5).
OBS_AGENTVIEW_RGB = "agentview_rgb"
OBS_EYE_IN_HAND_RGB = "eye_in_hand_rgb"
OBS_JOINT_STATES = "joint_states"
OBS_COMMANDED_JOINT_STATES = "commanded_joint_states"
OBS_GRIPPER_STATES = "gripper_states"
OBS_COMMANDED_GRIPPER_STATES = "commanded_gripper_states"
OBS_EE_POS_QUAT = "ee_pos_quat"
OBS_EE_STATES = "ee_states"
OBS_EE_POS = "ee_pos"
OBS_EE_ORI = "ee_ori"
OBS_JOINT_VELOCITIES = "joint_velocities"
# 포스·토크 (knu-1.1.1). FR3 의 robot state 에서 그대로 온다 -- 그 필드가 없는
# 펌웨어/바인딩에서는 기록되지 않는다.
#
# 넷만 남은 것은 후보 여덟을 로봇에 붙여 하나씩 재 본 결과다 (2026-09-06).
# "필드가 존재한다"와 "값이 온다"와 "그 값에 정보가 있다"는 서로 다르고,
# 앞의 둘만 보고 knu-1.1.0 에 넣었다가 그 버전을 통째로 폐기했다.
OBS_JOINT_TORQUES = "joint_torques"
OBS_EXT_JOINT_TORQUES = "ext_joint_torques"
OBS_EE_WRENCH = "ee_wrench"
OBS_EE_WRENCH_EE = "ee_wrench_ee"

#: 탈락한 후보와 그 이유. 다시 넣자는 이야기가 나올 때 같은 측정을 반복하지
#: 않도록 남긴다 -- 전부 실측이다.
#:
#: * ``tau_J_d`` (명령 관절토크): **항상 0**. start_joint_position_control 로
#:   위치를 보내므로 libfranka 가 채울 명령 토크가 없다. 2090 프레임 전부 0.
#: * ``dtau_J`` (관절토크 미분): 항상 0 은 아니지만 **노이즈가 지배**한다.
#:   정지 상태 최대 242 N*m/s, 손으로 흔든 상태 255 N*m/s -- 5% 차이라 운동
#:   여부조차 구분하지 못한다. 1kHz 미분이라 20Hz 차분으로 못 만드는 값인 것은
#:   맞지만, 만들어 봐야 노이즈를 띄엄띄엄 찍은 난수다.
#: * ``joint_contact``/``cartesian_contact``: set_collision_behavior 의 lower 를
#:   upper(리플렉스)와 같게 두고 있어 **항상 0**이다. 게다가 문턱을 낮춰 살려도
#:   ext_joint_torques 를 오프라인에서 자르면 같은 것을 얻는다 -- 수집 시점에
#:   문턱을 굽는 대신 연속값을 남기는 쪽이 되돌릴 수 있다.
#: * ``is_grasped`` (그리퍼): 파지 여부는 gripper_states 폭이 이미 뚜렷하게
#:   가른다 (파지 0.44~0.46 vs 빈 손 0.75~0.77, 실측).
FT_OBS_REJECTED = ("tau_J_d", "dtau_J", "joint_contact", "cartesian_contact")

#: 위 넷을 한 묶음으로. 로봇 -> 수집 워커 -> 기록기가 모두 이 순서로 돌며,
#: 필드를 늘릴 때 고칠 곳이 여기 하나가 되도록 한다. 값은 (키, 원소 수).
FT_OBS_FIELDS = (
    (OBS_JOINT_TORQUES, 7),
    (OBS_EXT_JOINT_TORQUES, 7),
    (OBS_EE_WRENCH, 6),
    (OBS_EE_WRENCH_EE, 6),
)
FT_OBS_KEYS = tuple(k for k, _ in FT_OBS_FIELDS)

# --------------------------------------------------------- per-frame timing (knu-1.3.0)
#: Group under each episode: ``episode_NNN/timing/<name>``, one value per frame.
#: Not under ``obs`` -- these describe the recording, not what a policy observes.
#:
#: All times are host wall clock (``time.time()``, seconds) -- the clock the
#: camera node, phase bus and raw robot logger already use, so they line up.
#: What each says (the difference between two of them is the measurement):
#:
#: * ``frame``        the recording loop finished reading this frame's observation
#: * ``action``       the command paired with this frame was accepted by the robot node
#: * ``robot_state``  the 1 kHz loop read the joint state stored in this frame
#: * ``<cam>_host``   the camera node received the image stored in this frame
#: * ``<cam>_device`` the camera's own timestamp for it (clock: group attr ``<cam>_domain``)
#: * ``<cam>_frame_no`` the camera's frame counter (frames the device produced)
#: * ``<cam>_node_seq`` the node's counter (frames the node received); growth of
#:   frame_no - node_seq is frames lost between device and node, and an unchanged
#:   node_seq between two recorded frames is the same image recorded twice
TIMING_GROUP = "timing"
TIMING_FRAME = "frame"
TIMING_ACTION = "action"
TIMING_ROBOT_STATE = "robot_state"
#: camera roles as they appear in dataset names (obs/agentview_rgb -> agentview)
TIMING_CAMERAS = ("agentview", "eye_in_hand")
#: written whenever the recording supplies them; required by knu-1.3.0
TIMING_REQUIRED = (TIMING_FRAME, TIMING_ACTION, TIMING_ROBOT_STATE) + tuple(
    f"{c}_host" for c in TIMING_CAMERAS)
#: written when the camera driver gives them; never required (a simulated or
#: older camera node has no device clock)
TIMING_OPTIONAL = tuple(f"{c}_{k}" for c in TIMING_CAMERAS
                        for k in ("device", "frame_no", "node_seq"))

# HDF5 repack markers (used by libero_format.py, dataset_sync.py, repack_hdf5.py).
REPACK_MARKER_ATTR = "repacked"
REPACK_COUNT_ATTR = "repacked_episodes"

#: 버전 문자열이 없던 시절의 표기 -> 현재 버전. 기존 파일(scene_000~014)은
#: 전부 ``dataset_version="scene-v1"`` 이고 필드 구성이 knu-1.0.0 과 완전히
#: 같아, 소급 기록 없이 별칭으로만 해석한다.
SCHEMA_VERSION_ALIASES = {
    # 2026-08-31 이전 파일의 표기. 대부분은 아래 stamp 스크립트로 실제
    # knu-1.0.0 을 써 넣었지만, Hub 사본·백업·old_data 처럼 손대지 않은
    # 사본이 남아 있으므로 별칭은 영구히 유지한다.
    "scene-v1": "knu-1.0.0",
    "": "knu-1.0.0",      # 아주 초기 파일: 표기 자체가 없다
}

#: 버전별 필수 필드. 검증기(scripts/check/check_scene_file.py)가 이걸 본다.
#: 새 MINOR 를 추가할 때는 이전 항목을 고치지 말고 새 키를 넣는다 -- 옛
#: 파일을 옛 규칙으로 계속 검사할 수 있어야 한다.
SCHEMA_FIELDS = {
    "knu-1.0.0": {
        # episode 그룹 바로 아래
        "episode_datasets": ("actions", "dones", "rewards"),
        # episode/obs 아래. depth 는 여기 없다 -- 카메라 드라이버가 아직
        # depth 읽기를 지원하지 않아 수집 자체가 꺼져 있다(_FIXED 참조).
        # 되살아나면 필드 '추가'이므로 knu-1.1.0 이다.
        "obs_datasets": (
            OBS_AGENTVIEW_RGB, OBS_EYE_IN_HAND_RGB,
            OBS_JOINT_STATES, OBS_COMMANDED_JOINT_STATES,
            OBS_GRIPPER_STATES, OBS_COMMANDED_GRIPPER_STATES,
            OBS_EE_STATES, OBS_EE_POS, OBS_EE_ORI,
        ),
        # episode 그룹 attrs
        "episode_attrs": (
            "instruction", "instruction_id", "episode_id", "episode_uid",
            "num_samples", "success", "quality_status", "scene_id",
            "slot_episode_idx", "collector", "station", "timestamp",
            "action_space", "action_column_names",
            "gripper_action_convention", "crop_params",
        ),
        # metadata 그룹 attrs
        "metadata_attrs": (
            "scene_id", "objects", "layout", "description", "station",
            "dataset_version", "created", "next_episode_idx",
        ),
    },
}

#: knu-1.1.0 -- **폐기**. 검증기는 알아야 하지만 새로 찍어서는 안 된다.
#:
#: 7종을 넣었는데 그중 셋(desired_joint_torques, joint_contact,
#: cartesian_contact)이 실제로는 **모든 프레임에서 0** 이었다. 필드가 있다는
#: 것만 보고 넣었고, 값이 오는지는 로봇에 붙어 확인하지 않았다 (2026-09-06
#: 실측으로 드러남). 이유는 FT_OBS_REJECTED 주석에 있다.
#:
#: 그래서 정의는 그대로 남긴다 -- 이 버전으로 찍힌 파일이 검증을 통과해야
#: 하기 때문이다. 대신 런처의 버전 선택지에서 빼서 새 파일이 이 버전을 달지
#: 못하게 한다 (apps/workspace/launcher/pages.py 의 SCHEMA_PICKABLE).
#:
#: **이 버전 파일의 세 필드를 쓰지 마세요.** 값이 0인 것은 "접촉이 없었다"가
#: 아니라 "측정되지 않았다" 이다.
_KNU_110_OBS = ("joint_torques", "ext_joint_torques", "desired_joint_torques",
                "ee_wrench", "ee_wrench_ee", "joint_contact", "cartesian_contact")
SCHEMA_FIELDS["knu-1.1.0"] = {
    **SCHEMA_FIELDS["knu-1.0.0"],
    "obs_datasets": SCHEMA_FIELDS["knu-1.0.0"]["obs_datasets"] + _KNU_110_OBS,
}

#: knu-1.1.1 = knu-1.0.0 + 포스·토크 4종 (2026-09-06).
#:
#: 1.1.0 에서 죽은 필드 셋을 뺀 것이다. 필드 제거라 MINOR 규칙("추가만") 밖
#: 이지만, 1.1.0 으로 찍힌 파일이 최종 데이터셋에 하나도 남지 않으므로
#: (S015 재수집) 실질적인 하위 호환 문제는 없다. PATCH 를 올려 "1.1 계열인데
#: 고쳐진 것"임을 나타내고, 1.1.0 은 위처럼 폐기 표시한다.
#:
#: 남은 넷은 전부 실측으로 검증됐다: 0.5 kg 추를 502 g 으로 읽고, Desk 에서
#: 뺀 120 g 을 111 g 으로 읽어냈다 (정지 구간 기준, +-50 g).
SCHEMA_FIELDS["knu-1.1.1"] = {
    **SCHEMA_FIELDS["knu-1.0.0"],
    "obs_datasets": SCHEMA_FIELDS["knu-1.0.0"]["obs_datasets"] + FT_OBS_KEYS,
}

#: 기록 시점의 로봇 부하 모델. metadata 그룹 attrs 로 한 번만 적는다.
#:
#: 정적 값이라 프레임마다 실을 이유가 없고, 그렇다고 안 적으면 **그 파일의
#: 절대 힘값을 해석할 수 없다**. 미신고 질량은 그대로 외력 추정에 섞이기
#: 때문이다 -- 2026-09-06 실측: Desk 에서 120 g 을 빼자 같은 0.5 kg 추가
#: 613 g 으로 읽혔다. 그때 우리는 "Desk 가 맞았을 것"이라고 믿는 것 말고
#: 확인할 방법이 없었고, 그것이 이 필드를 만든 이유다.
META_PAYLOAD_MASS = "payload_mass"      # kg, m_total
META_PAYLOAD_COM = "payload_com"        # m, F_x_Ctotal (플랜지 기준 x,y,z)

#: knu-1.2.0 = knu-1.1.1 + 부하 모델 메타 2종 (2026-09-06).
#:
#: obs 는 1.1.1 과 같다 -- 늘어난 것은 metadata attrs 뿐이라 프레임 데이터는
#: 한 바이트도 커지지 않는다. 그래도 MINOR 인 이유는 "필드 추가"이기
#: 때문이고, 옛 파일은 이 attrs 가 없으므로 1.1.1 규칙으로 계속 검사된다.
#: 기록 시점의 리셋 자세. metadata 그룹 attrs 로 한 번만 적는다 (knu-1.2.1).
#:
#: 팔이 **어디서 출발했는가**는 궤적을 읽는 데 필요한데, 지금까지 파일에는
#: station 이름만 있었다 (knu-eng7). 이름에서 자세를 찾으려면 그 시점의
#: configs/stations/*.yaml 과 FR3_RESET_POSES 를 알아야 하고, 둘 다 나중에
#: 바뀔 수 있다 -- 그러면 옛 파일에 지금 값을 갖다 붙여 읽게 된다. payload 를
#: 파일에 적기로 한 것과 같은 이유다 (2026-09-07 사용자 결정).
#:
#: **이름과 값을 함께 적는다.** 이름만 적으면 FR3_RESET_POSES["libero"] 가
#: 바뀔 때 같은 구멍이 다시 생기고, 값만 적으면 사람이 그것이 무엇인지
#: 알아보지 못한다.
META_RESET_POSE = "reset_pose"          # 별칭 (libero / fr3_ready / panda)
META_RESET_QPOS = "reset_qpos"          # rad, 7관절 절대값 (JSON 목록)

SCHEMA_FIELDS["knu-1.2.0"] = {
    **SCHEMA_FIELDS["knu-1.1.1"],
    "metadata_attrs": SCHEMA_FIELDS["knu-1.1.1"]["metadata_attrs"] + (
        META_PAYLOAD_MASS, META_PAYLOAD_COM,
    ),
}


#: 리셋 자세 추가. MINOR 가 아니라 PATCH 인 이유: 궤적을 읽는 데 도움이 되는
#: 부가 정보이지 관측 자체가 늘어난 것이 아니다 (knu-1.1.1 이 힘·토크 관측을
#: PATCH 로 더한 선례와 같은 결). 2026-09-07 사용자 결정.
SCHEMA_FIELDS["knu-1.2.1"] = {
    **SCHEMA_FIELDS["knu-1.2.0"],
    "metadata_attrs": SCHEMA_FIELDS["knu-1.2.0"]["metadata_attrs"] + (
        META_RESET_POSE, META_RESET_QPOS,
    ),
}


#: 이 파일을 **무엇이 만들었는가** (knu-1.2.2). metadata 그룹 attrs.
#:
#: 물리 셋업(station·payload·reset)은 적어 왔는데 그것을 돌린 소프트웨어는
#: 어디에도 없었다 -- 정책이 이상하게 움직일 때 "이 데이터는 어느 컨트롤러,
#: 어느 수집기 코드에서 나왔나" 가 첫 질문인데 답이 사람 기억뿐이었다
#: (조작자, 2026-09-13). 실제로 이 데이터셋 안에는 제어 상수를 여러 번 바꾼
#: 전후가 섞여 있다 (v_max 1.0→1.5, 저크 클램프 도입, 리더 드롭 가드).
#:
#: 넷 다 **자동으로 읽은 값**이고, 못 읽으면 적지 않는다 (0 이나 "?" 를
#: 적으면 측정한 것처럼 읽힌다 -- knu-1.1.0 의 0 으로 찬 힘 필드가 그
#: 교훈이다). 마지막 하나는 그 값들이 언제 어떻게 들어왔는지를 말한다.
META_COLLECTOR_COMMIT = "collector_commit"      # 수집기 저장소의 git 커밋
META_PYLIBFRANKA_VERSION = "pylibfranka_version"  # 예: 0.21.2 (libfranka 0.21)
META_FR3_SYSTEM_VERSION = "fr3_system_version"  # FR3 시스템 이미지, 예: 5.10.0
#: "live" = 수집하면서 적었다 / "backfilled <날짜>" = 나중에 채웠다.
#: 채워 넣은 값은 **그 시점의 사실이 아닐 수 있다** -- 그것을 구분하지 않으면
#: 나중에 이 필드 전체를 믿을 수 없게 된다.
META_PROVENANCE_SOURCE = "provenance_source"
#: 달려 있던 그리퍼 (knu-2.0.0). 이름과, 그 최대 벌림 (m).
#:
#: **그리퍼 열은 0~1 정규화값이다** -- 실측으로 obs/gripper_states 가
#: 0.0026~0.9624 이고 actions 의 8번째 열도 0/1 이다. 그 값을 미터로 되돌릴
#: 방법이 파일 안에 없으면, 다른 그리퍼로 찍은 데이터와 섞는 순간 같은 0.5
#: 가 다른 폭을 뜻하게 된다. 부하 모델(무게)·리셋 자세(출발점)와 같은
#: 부류다: 팔 밖의 물리 조건이라 관측이 아니라 파일 메타에 한 번 적는다.
META_GRIPPER = "gripper"
META_GRIPPER_MAX_WIDTH = "gripper_max_width"

#: 버전이 **요구하는 것은 하나뿐**이다: 이 값들이 어디서 왔는지.
#:
#: 커밋과 로봇 판번호를 요구하지 않는 이유가 각각 있다. 로봇 판번호는 팔이
#: 꺼진 세션이나 시뮬레이터에서는 못 읽는다. 커밋은 **옛 파일에서 복구할 수
#: 없다** -- 닥터가 뒤늦게 채울 때 쓸 수 있는 정직한 출처가 없고(다른 scene 의
#: 커밋은 그 파일의 커밋이 아니다), 요구에 넣으면 이미 찍힌 파일들은 영원히
#: 이 버전에 못 올라간다. 그래서 약속은 "**이 파일은 자기 출처가 어디서
#: 왔는지 말한다**" 하나이고, live / backfilled 구분이 그 약속을 지킨다.
_PROVENANCE_REQUIRED = (META_PROVENANCE_SOURCE,)

#: 있으면 적는 것들 (검증이 요구하지는 않는다).
#: (2026-09-14 까지는 Desk 응답의 나머지 두 줄을 fr3_system_build 로 적었다.
#: 뜻이 확인되지 않은 값이라 뺐다 -- 한 파일에도 적히기 전이었다.)
_PROVENANCE_OPTIONAL = (META_COLLECTOR_COMMIT, META_PYLIBFRANKA_VERSION,
                        META_FR3_SYSTEM_VERSION)

#: 세 갈래 모두에 provenance 를 더한 PATCH 판 (2026-09-13 조작자 결정).
#: 갈래가 셋인 이유는 지금 데이터셋에 셋이 다 살아 있기 때문이다 --
#: fr3-tabletop 에 1.0.0 이 14개, 1.1.1 이 1개, 1.2.1 이 12개.
#: 각자 자기 갈래에서 한 칸만 올라간다: 없던 관측이 생기는 것이 아니라
#: **누가 찍었는지**가 붙는 것이라 PATCH 다 (1.2.1 의 선례와 같은 결).
for _base, _new in (("knu-1.0.0", "knu-1.0.1"),
                    ("knu-1.1.1", "knu-1.1.2"),
                    ("knu-1.2.1", "knu-1.2.2")):
    SCHEMA_FIELDS[_new] = {
        **SCHEMA_FIELDS[_base],
        "metadata_attrs": SCHEMA_FIELDS[_base]["metadata_attrs"] + _PROVENANCE_REQUIRED,
    }


#: knu-1.3.0 = knu-1.2.2 + per-frame timing (2026-09-17).
#:
#: MINOR: new per-frame datasets, nothing existing changes. Required are the
#: stamps the collection stack always produces; the camera's device clock and
#: frame counter are recorded when available but not required. Old files cannot
#: be raised to this version -- timing cannot be recovered after the fact.
SCHEMA_FIELDS["knu-1.3.0"] = {
    **SCHEMA_FIELDS["knu-1.2.2"],
    "episode_datasets": SCHEMA_FIELDS["knu-1.2.2"]["episode_datasets"] + tuple(
        f"{TIMING_GROUP}/{k}" for k in TIMING_REQUIRED),
}

# --------------------------------------------------------- knu-2.0.0
# **MAJOR 다.** 한 행 = 한 프레임이라는 전제를 버린다.
#
# 1.3.0 까지는 20 Hz 루프가 모든 계열을 한 tick 에 같이 긁어 한 줄로 썼다.
# 그 대가가 컸다: 30 fps 카메라의 33% 와 명령의 5분의 4가 버려지고, 두 카메라가
# 자유 구동인데도 같은 행에 있어 동시각인 척했다.
#
# 2.0.0 은 소스마다 자기 시간축에 쓴다:
#
#     t/control   기록 tick          actions, obs/*, rewards, dones
#     t/command   명령 tick          command/joint_positions, command/gripper
#     t/agent     D455 가 준 시각     obs/agentview_rgb
#     t/wrist     D405 가 준 시각     obs/eye_in_hand_rgb
#
# 모든 데이터셋에 ``attrs["axis"]`` 가 붙어 **어느 축인지 명시된다** -- 길이로
# 추정하면 축이 갈린 순간부터 틀린다. 카메라 축의 시각은 **장치 시각**이고
# 도착 시각은 ``meta/<축>/host`` 에 있다 (기종마다 다른 고정 전송 지연을
# 축에 섞지 않으려고. 실측 D455 15.03 ms, D405 8.60 ms, 표준편차 0.1 ms).
#
# ``timing/`` 은 없다 -- 그 내용이 ``t/`` 와 ``meta/`` 로 갈라져 들어갔고,
# 같은 값을 두 곳에 두면 언젠가 갈라진다.
#
# 읽는 쪽은 ``mstack.data.frame_table.frame_table()`` 하나로 두 레이아웃을
# 모두 표로 받는다 -- 소비자는 어느 버전인지 몰라도 된다.
#: 축이 갈린 파일이 반드시 갖는 것. 카메라 축은 **필수가 아니다** --
#: capture 에 실패한 카메라는 control 축에 남고, 그것도 정상이다.
AXIS_GROUP = "t"
META_GROUP = "meta"
AXIS_CONTROL = "control"
SCHEMA_FIELDS["knu-2.0.0"] = {
    **SCHEMA_FIELDS["knu-1.3.0"],
    # timing/* 는 빠지고 t/control 이 들어온다. 이 교체가 MAJOR 인 이유다.
    # timing/* 는 빠지고 t/control 이 들어온다. 1.3.0 이 요구하던
    # timing/robot_state 는 meta/control/robot_state 로 자리만 옮긴다 --
    # "각 프레임의 로봇 상태를 언제 읽었나" 는 보장이 약해지면 안 된다.
    "episode_datasets": SCHEMA_FIELDS["knu-1.2.2"]["episode_datasets"] + (
        f"{AXIS_GROUP}/{AXIS_CONTROL}",
        f"{META_GROUP}/{AXIS_CONTROL}/{TIMING_ROBOT_STATE}",),
    # 그리퍼 (2026-09-23). **버전을 따로 올리지 않았다** -- 2.0.0 은 아직
    # 개발 중이고 밖으로 나간 파일이 없다. 출시된 버전에 필드를 더하는
    # 것이었다면 MINOR 를 올려야 하지만(그래야 옛 리더가 안 깨진다), 아직
    # 아무도 그 약속에 기대고 있지 않을 때는 그 버전 안에서 고치는 것이
    # 맞다 (조작자, 2026-09-24).
    "metadata_attrs": SCHEMA_FIELDS["knu-1.3.0"]["metadata_attrs"] + (
        META_GRIPPER,),
}


#: A scene file's version lives in one attribute: ``metadata.attrs["dataset_version"]``.
#:
#: Until 2026-09-17 the same value was also written as ``schema_version`` (#41,
#: to match the LeRobot info.json key). Four writers had to keep the copies
#: equal, the doctor updated only one, and 27 files disagreed -- which also
#: skewed the launcher, because it read the copy. The copy was removed from the
#: code and from every file with knu-1.3.0; check_scene_file flags it if an old
#: copy of a file brings it back.
REMOVED_VERSION_ATTR = "schema_version"


def schema_version_key(value) -> tuple:
    """비교용 (MAJOR, MINOR, PATCH). 별칭은 풀고, 못 읽으면 (-1,-1,-1).

    버전 문자열을 문자로 비교하면 knu-1.10.0 이 knu-1.9.0 보다 작아진다.
    비교가 필요한 곳(이어찍기에서 도장을 올릴지 판단할 때)이 생겨 여기 둔다.
    """
    v = normalize_schema_version(value)
    body = v.split("-", 1)[1] if "-" in v else v
    try:
        parts = tuple(int(x) for x in body.split("."))
    except ValueError:
        return (-1, -1, -1)
    return parts + (0,) * (3 - len(parts)) if len(parts) < 3 else parts[:3]


def normalize_schema_version(value) -> str:
    """파일에 적힌 버전 표기를 정본 형태로. 별칭은 풀고, 나머지는 그대로.

    파일이 정본이므로 모르는 표기라고 던지지 않는다 -- 그대로 돌려주고
    판단은 호출자(검증기)에게 맡긴다. 여기서 죽으면 옛 파일을 열지 못하게
    되는데, 그게 버저닝을 도입한 이유와 정반대다.
    """
    s = str(value or "").strip()
    return SCHEMA_VERSION_ALIASES.get(s, s)


def parse_schema_version(value) -> "tuple[int, int, int] | None":
    """``knu-1.0.0`` -> ``(1, 0, 0)``. 형식이 아니면 None."""
    s = normalize_schema_version(value)
    if not s.startswith("knu-"):
        return None
    parts = s[4:].split(".")
    if len(parts) != 3:
        return None
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return None


#: 어느 MAJOR 리더가 어느 MAJOR 파일들을 읽는가.
#:
#: MINOR 는 언제나 위아래 모두 읽힌다 (필드 추가만 하기로 했으므로, 위 버전
#: 파일에는 모르는 필드가 더 있을 뿐이고 아래 버전 파일에는 나중에 생긴
#: 필드가 없을 뿐이다). MAJOR 는 명시해야 한다.
#:
#: **2 가 1 을 읽는다.** ``mstack.data.frame_table.frame_table()`` 이 두
#: 레이아웃을 모두 표로 돌려주기 때문이다 -- 소비자는 어느 버전인지 몰라도
#: 된다. 이 줄이 없으면 이미 모은 2,434 에피소드가 전부 "이 코드가 읽을 수
#: 없는 버전" 으로 보고된다.
#:
#: 반대(1 이 2 를 읽는 것)는 **False 로 둔다.** 옛 체크아웃에는 shim 이
#: 없고, 축이 갈린 파일을 한 행 = 한 프레임으로 읽으면 길이가 다른 계열을
#: 같은 인덱스로 집게 된다. 못 읽는 것이 사실이다.
READS_MAJORS = {
    1: (1,),
    2: (1, 2),
}


def schema_is_readable(value, reader: str = SCHEMA_VERSION) -> bool:
    """이 리더가 그 파일을 읽을 수 있는가 (READS_MAJORS 참조)."""
    a, b = parse_schema_version(value), parse_schema_version(reader)
    if not (a and b):
        return False
    return a[0] in READS_MAJORS.get(b[0], (b[0],))


def schema_required_fields(value) -> "dict | None":
    """그 버전이 요구하는 필드 목록. 모르는 버전이면 None."""
    return SCHEMA_FIELDS.get(normalize_schema_version(value))


@dataclass
class DatasetSchemaConfig:
    """What gets written to the HDF5. Every field defaults to LIBERO's
    original fixed schema, so a bare ``DatasetSchemaConfig()`` reproduces it.

    There used to be a ``use_default`` flag that overrode every other field
    at write time. It was removed: it silently discarded the operator's
    ``action_space`` choice (a session set to ``joint_absolute`` would write
    ``ee_delta`` instead, with no error), which is exactly the kind of
    invisible mismatch this schema exists to prevent. Old saved JSON that
    still carries the key is simply ignored by :meth:`from_json`.
    """

    # ---- 아래 다섯은 GUI 에서 고를 수 없는 고정값이다 ----
    # 액션 구조가 파일마다 갈리면 한 데이터셋 안에 조용히 호환되지 않는 파일이
    # 섞이고, 그걸 잡아주는 장치가 지금 없다(issue #12). 관측 필드는 더하거나
    # 빼도 파일끼리 호환되므로 계속 고를 수 있게 둔다.
    action_space: str = ACTION_SPACE_JOINT_ABSOLUTE
    # LIBERO's original actions always end with a gripper component (see
    # libero_format.py's compute_delta_action) -- True keeps that. Off drops
    # the trailing gripper dimension from `actions` for every action space
    # (e.g. a policy that controls the gripper separately from arm motion).
    action_include_gripper: bool = True   # 고정
    # 고정 On: action 의 그리퍼를 observation 의 gripper_states 와 같은
    # 0=open/1=close 로 쓴다. robosuite 의 -1/+1 대신 이걸 쓰는 이유는 열 이름과
    # 마찬가지로 Hugging Face 뷰어에서 obs 와 action 을 짝지어 보기 위해서다.
    #
    # 다만 "같은 인코딩"이지 "같은 신호"는 아니다. action 은 리더 트리거를
    # 이진화한 값이라 0 또는 1 뿐이고, observation 은 실제 핑거 폭이라 명령 후
    # ~0.3초 뒤부터 8~10Hz 로 0..1 사이를 연속으로 지난다(mstack/data/gripper_synth.py).
    # 그 간격이 정책이 학습해야 할 그리퍼 지연이다.
    gripper_action_match_obs: bool = True

    # 정사각 크롭 후 리사이즈할 한 변(px). 기본 None = 크롭도 리사이즈도 하지
    # 않고 카메라가 준 프레임을 그대로 쓴다(현재 640x480).
    #
    # .hdf5 는 원본 보관소라 최대한 남기고, 줄이는 것은 LeRobot 변환에서 한다
    # (scripts/convert/convert_libero_to_lerobot.py --image-size). 그래야 학습 해상도를
    # 바꿀 때 다시 찍지 않아도 된다. 대신 정사각이 아니므로, 무엇이 크롭되어
    # 살아남는지는 Live 탭의 정사각 가이드로 본다.
    image_size: int | None = None

    save_agentview_rgb: bool = True
    save_eye_in_hand_rgb: bool = True
    save_joint_states: bool = True
    save_gripper_states: bool = True
    save_ee_states: bool = True
    save_ee_pos: bool = True
    save_ee_ori: bool = True

    # Off by default: not part of LIBERO's original schema, computed for
    # free from data the control loop already produces (see
    # mstack/collect/worker.py's _get_obs).
    save_joint_velocities: bool = False
    save_timestamp: bool = False

    # Off by default (issue #17): depth 는 카메라 ASIC 이 이미 계산하는 값이라
    # 호스트 연산은 공짜지만 데이터는 이미지급이다 -- 캠당 640x480 uint16 로
    # 에피소드당 수십 MB, USB 대역 +~176Mbps. 무손실(gzip)로 원본 해상도
    # 그대로 저장하고 crop/resize 는 하지 않는다 (RGB-depth 픽셀 대응은
    # D455 에서 원래 안 맞으므로, 원시 저장 + 필요할 때 후처리가 일관적).
    # LeRobot 변환은 depth 를 무시한다 -- HDF5 원본 보관소에만 남는다.
    save_agentview_depth: bool = False
    save_eye_in_hand_depth: bool = False

    # 고정 빈 dict = 내장 이름(joint1.pos .. joint7.pos, gripper.pos) 사용.
    # 내장 이름이 곧 observation 의 열 이름이라, Hugging Face 뷰어가 obs 와
    # action 을 같은 축에 짝지어 그려준다. 필드는 남겨두지만 GUI 에서는 고를 수
    # 없다 -- 이름을 바꿔서 얻을 것보다 짝이 깨져서 잃을 것이 크다.
    action_column_name_overrides: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    # GUI 에서 고를 수 없는 필드들. 저장된 JSON 에 옛 값이 남아 있어도 무시한다
    # -- 다이얼로그에서 뺐다는 이유만으로 고정이 되지는 않는다. 이 목록이 실제
    # 강제 지점이고, 다이얼로그는 그 결과를 보여줄 뿐이다.
    #
    # depth 수집은 lerobot 0.5.0 RealSenseCamera 가 read_latest_depth 를
    # 지원하지 않아 당분간 비활성화한다. 코드(버퍼/저장 경로)는 남겨두고
    # 플래그만 강제 Off 로 막는다 (fix/depth-gate).
    _FIXED = ("action_space", "action_include_gripper", "gripper_action_match_obs",
              "action_column_name_overrides",
              "save_agentview_depth", "save_eye_in_hand_depth")

    @classmethod
    def from_json(cls, s: str) -> "DatasetSchemaConfig":
        data = json.loads(s)
        valid = {f.name for f in fields(cls)} - set(cls._FIXED)
        filtered = {k: v for k, v in data.items() if k in valid}
        cfg = cls(**filtered)
        # depth 플래그가 True 로 저장돼 있어도 드라이버 미지원으로 인해 Off.
        # warnings 는 stderr 로만 가서 데스크톱 아이콘 실행에서는 아무 데도 안
        # 남는다 (collect_workspace.py 의 stderr 주석 참조) -- 무시된 플래그를
        # 인스턴스 속성으로도 들고 있어 GUI 가 로그 뷰가 생긴 뒤 보이는 로그로
        # 재보고할 수 있게 한다. dataclass 필드가 아니므로 to_json 에는 안 실린다.
        ignored = [flag for flag in ("save_agentview_depth", "save_eye_in_hand_depth")
                   if data.get(flag)]
        if ignored:
            warnings.warn(
                f"{', '.join(ignored)}=True 인 설정을 무시합니다: "
                "카메라 드라이버(lerobot RealSenseCamera)가 depth 읽기를 "
                "지원하지 않아 수집이 비활성화되어 있습니다.",
                stacklevel=2,
            )
            cfg.ignored_depth_flags = ignored
        return cfg


def load_schema_config(path: Path = DEFAULT_CONFIG_PATH) -> DatasetSchemaConfig:
    """Never raises -- a missing/corrupt config file just means "no custom
    config saved yet", not a startup failure."""
    try:
        return DatasetSchemaConfig.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return DatasetSchemaConfig()


def save_schema_config(cfg: DatasetSchemaConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.to_json(), encoding="utf-8")


def selftest() -> None:
    """버저닝 규약 자체 검증 (issue #41). 하드웨어·파일 불필요."""
    # 별칭: 옛 표기는 현재 버전으로 풀린다 (파일 소급 수정 없이)
    assert normalize_schema_version("scene-v1") == "knu-1.0.0"
    assert normalize_schema_version("") == "knu-1.0.0"
    assert normalize_schema_version("knu-1.1.0") == "knu-1.1.0"
    assert parse_schema_version("scene-v1") == (1, 0, 0)
    assert parse_schema_version("모르는버전") is None

    # 같은 MAJOR 안이면 위·아래 MINOR 모두 읽는다 (MINOR = 추가만)
    assert schema_is_readable("knu-1.0.0", reader="knu-1.0.0")
    assert schema_is_readable("knu-1.0.0", reader="knu-1.1.0")   # 후방 호환
    assert schema_is_readable("knu-1.1.0", reader="knu-1.0.0")   # 전방 호환
    assert not schema_is_readable("knu-2.0.0", reader="knu-1.0.0")  # 옛 리더엔 shim 이 없다
    # 2.x 리더는 1.x 를 읽는다 -- frame_table 이 두 레이아웃을 모두 표로 준다.
    # 이게 깨지면 이미 모은 2,434 에피소드가 "못 읽는 버전" 이 된다.
    assert schema_is_readable("knu-1.3.0", reader="knu-2.0.0")
    assert schema_is_readable("knu-1.0.0", reader="knu-2.0.0")
    assert schema_is_readable("knu-2.0.0", reader="knu-2.0.0")
    assert not schema_is_readable("이상한거")

    # 현재 버전은 필드 목록을 갖고 있고, 그 목록이 문서와 같은 정본이다
    cur = schema_required_fields(SCHEMA_VERSION)
    assert cur is not None and SCHEMA_VERSION in SCHEMA_FIELDS
    assert set(cur) == {"episode_datasets", "obs_datasets",
                        "episode_attrs", "metadata_attrs"}
    # depth 는 아직 없다 -- 드라이버 미지원으로 수집 자체가 꺼져 있다.
    # 되살아나면 '추가'라 또 한 번 MINOR 를 올린다 (docs/dataset-schema.md)
    assert not any("depth" in f for f in cur["obs_datasets"])
    assert "save_agentview_depth" in DatasetSchemaConfig._FIXED
    # 포스·토크는 1.1.0 에서 들어왔다. 1.0.0 은 그대로 없어야 한다 --
    # MINOR 는 추가만 하므로 옛 규칙을 고치면 옛 파일 검사가 틀어진다.
    assert all(f in cur["obs_datasets"] for f in FT_OBS_KEYS)
    old = schema_required_fields("knu-1.0.0")
    assert not any("torque" in f or "wrench" in f or "contact" in f
                   for f in old["obs_datasets"])
    assert len(cur["obs_datasets"]) == len(old["obs_datasets"]) + len(FT_OBS_KEYS)
    # 키가 겹치면 뒤엣것이 앞엣것을 덮어써 조용히 한 필드가 사라진다
    assert len(set(FT_OBS_KEYS)) == len(FT_OBS_KEYS)
    # 탈락한 후보는 현재 버전에 없어야 한다 -- 값이 안 오는 것으로 실측된
    # 필드들이라, 다시 들어오면 knu-1.1.0 과 같은 사고가 반복된다.
    assert not (set(FT_OBS_KEYS) & set(FT_OBS_REJECTED))
    for f in FT_OBS_REJECTED:
        assert f not in cur["obs_datasets"], f
    # 폐기된 1.1.0 은 정의가 남아 있어야 한다 -- 그 버전으로 찍힌 파일이
    # 검증을 통과해야 하기 때문이다 (새로 찍는 것은 런처가 막는다).
    dead = schema_required_fields("knu-1.1.0")
    assert dead is not None and "desired_joint_torques" in dead["obs_datasets"]
    # 1.2.x 가 더한 것은 metadata attrs 뿐 -- obs 는 1.1.1 과 같아야 한다.
    prev = schema_required_fields("knu-1.1.1")
    assert cur["obs_datasets"] == prev["obs_datasets"]
    # 1.2.x 가 더한 metadata attrs. **SCHEMA_VERSION 이 아니라 1.2.1 을 본다** --
    # 예전에는 cur(=그때의 최신)을 봤는데, 1.2.2 가 provenance_source 를 더하는
    # 순간부터 이 단언이 깨진 채로 남아 있었다 (이 셀프테스트는 스위트에서
    # 돌지 않아 아무도 몰랐다). 여기서 묻고 싶은 것은 "지금 최신이 무엇을
    # 더했나" 가 아니라 "1.2.x 가 무엇을 더했나" 다.
    v121 = schema_required_fields("knu-1.2.1")
    added = set(v121["metadata_attrs"]) - set(prev["metadata_attrs"])
    assert added == {META_PAYLOAD_MASS, META_PAYLOAD_COM,
                     META_RESET_POSE, META_RESET_QPOS}, added
    # 1.2.0 은 부하 모델만, 1.2.1 이 리셋 자세를 더한다 (2026-09-07).
    v120 = schema_required_fields("knu-1.2.0")
    assert set(v120["metadata_attrs"]) - set(prev["metadata_attrs"]) == {
        META_PAYLOAD_MASS, META_PAYLOAD_COM}
    assert set(v121["metadata_attrs"]) - set(v120["metadata_attrs"]) == {
        META_RESET_POSE, META_RESET_QPOS}

    # 모르는 버전은 필드 목록이 없다 -> 검증기가 "모르는 스키마 버전" 으로 잡는다
    assert schema_required_fields("knu-9.9.9") is None
    print("dataset_schema selftest 통과")


if __name__ == "__main__":
    selftest()
