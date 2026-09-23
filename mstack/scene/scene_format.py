"""Scene 기반 HDF5 저장 (scene-v1).

파일 하나 = scene 하나(책상 위의 한 가지 물리적 배치). 한 파일 안에 서로 다른
instruction 의 에피소드가 공존한다. 파일명에는 scene ID 만 들어가고,
instruction 은 **episode attrs 안에만** 존재한다 -- legacy v0 에서 파일명이
사실상 source of truth 라서 문장을 고칠 때마다 파일이 갈라지던 사고(Notion
프로토콜 §A)를 구조적으로 없애는 것이 이 포맷의 목적이다.

    scene_012.hdf5
    ├── metadata                    <- scene 단위. 그룹 attrs + 기준 사진
    │   attrs: scene_id, description, objects(JSON), layout(JSON),
    │          station, dataset_version, created, next_episode_idx
    │   └── reference_image         <- (H, W, 3) uint8 정면 사진 (배치 재현용)
    ├── episode_000
    │   ├── obs/..., actions, rewards, dones   <- legacy demo_N 과 동일 페이로드
    │   attrs: scene_id, instruction_id, episode_id, episode_uid,
    │          instruction(따옴표 없는 순수 문자열), success, quality_status,
    │          collector, timestamp,
    │          num_samples, action_space, gripper_action_convention,
    │          action_column_names, crop_params, station
    └── episode_001

legacy(``<task>_demo.hdf5``, mstack/data/libero_format.py)와의 의도적 차이:

- ``problem_info``/``env_args`` 스텁을 쓰지 않는다. 이 저장소 안에 env_args 를
  읽는 코드는 없고(2026-08 조사), problem_info 소비자는 전부 새 포맷 지원으로
  고친다. 외부 LIBERO/robomimic 리더 호환은 포기한다 -- 결정 사항.
- instruction 은 따옴표로 감싸지 않는다. legacy 는 ``f'"{...}"'`` 로 감싸
  저장했고 읽는 쪽이 두 가지 방식으로 벗기고 있었다.
- 프레임·instruction 은 편집하지 않는다. 1차 큐레이션은 ``quality_status``
  재판정(변환이 success 만 내보냄), 실패·튀는 궤적은 푸시 전에 **삭제**
  (2026-08-14 결정) -- legacy 와 같이 삭제 후 ``renumber_scene_episodes``
  로 그룹 번호·episode_id·slot E번호(uid)를 다시 매긴다. uid 는 "그 slot 의
  몇 번째" 로 완전히 파생되는 값이라 보존할 이력이 없다. 끝만 자르는
  트림(episode_trim)도 허용.
- 에피소드 안쪽 페이로드(obs/actions/rewards/dones 와 provenance attrs)는
  legacy 와 완전히 같다 (:func:`mstack.data.libero_format.write_episode_payload`
  공유). 변환기가 두 포맷의 에피소드 내부를 같은 코드로 읽게 하기 위해서다.

layout 은 격자 존 기반 구조화 JSON 이다 (결정 사항 -- scene 다양성 추천이
배치 거리를 계산할 수 있어야 한다)::

    {
      "grid": [3, 3],                       # [rows, cols] 작업공간 분할
      "placements": {                       # instance ID -> 존 [row, col]
        "OBJ-CUP-BLU-01": {"zone": [0, 2]},
        "OBJ-BOWLS-YEL-01": {"zone": [1, 1]}
      },
      "relations": [                        # 선택: 존만으로 표현 안 되는 관계
        ["OBJ-CUP-BLU-01", "next_to", "OBJ-BOWLS-YEL-01"]
      ]
    }

존은 로봇 기준 작업공간을 카메라(agentview) 프레임에서 rows x cols 로 나눈
것이고, [0, 0] 이 왼쪽 위다. cm 좌표는 요구하지 않는다 -- 기준 위치에서 수 cm
흔드는 controlled variation(§4)은 같은 존 안의 이동이고, 존 경계를 넘으면
새 scene ID 다.

distractor 에 대하여 -- 개념은 유지하되 필드로 구현하지 않는다 (2026-08-13
결정). distractor 는 "책상에 놓여 있고 카메라에 찍히지만 그 scene 의 어떤
instruction 에도 등장하지 않는 물체"로, 언어조건 조작의 표준 robustness
평가축이다 (BC-Z, RT-1 의 distractor 평가; 이론적 배경은 causal confusion in
imitation learning). 다만 지금 스케일에서는 별도 필드가 주는 분석력보다
수집자가 역할(objects vs distractors)을 오지정하는 비용이 크다. 그런 물체도
``objects`` 에 넣고 ``description`` 에 사람 말로 적어 둔다 -- "어떤 물체가
지칭되지 않았는가"는 나중에 계획 파일의 instruction 집합과 대조해 파생할 수
있고, 정말 필드가 필요해지면 scene-v2 에서 되살린다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from mstack.data.dataset_schema import (
    META_PAYLOAD_COM,
    META_COLLECTOR_COMMIT,
    META_FR3_SYSTEM_VERSION,
    META_PAYLOAD_MASS,
    META_PROVENANCE_SOURCE,
    META_PYLIBFRANKA_VERSION,
    META_RESET_POSE,
    META_RESET_QPOS,
    SCHEMA_VERSION,
    DatasetSchemaConfig,
    normalize_schema_version,
    schema_version_key,
)
from mstack.data.crop import default_crop_params
from mstack.data.libero_format import (
    LiberoEpisodeBuffer,
    _mark_close_on_exec,
    write_episode_payload,
)

#: 파일에 적히는 형식 버전. 2026-08-31 부터 SemVer 표기 ``knu-X.Y.Z``
#: (issue #41) 이고, 정본과 규약은 mstack.data.dataset_schema 에 있다.
#: 그 전 파일은 전부 ``scene-v1`` 인데 필드 구성이 knu-1.0.0 과 같아
#: 소급 기록 없이 별칭으로 해석한다 (normalize_schema_version).
SCENE_DATASET_VERSION = SCHEMA_VERSION

# 표준 격자 (2026-08-13 결정). 포맷 자체는 파일마다 grid 를 기록하므로 나중에
# 바꿔도 기존 파일은 읽히지만, 새 scene 생성은 이 값만 허용한다 -- 수집자마다
# 격자 해석이 갈리는 것을 막고, "같은 존 안 이동 = 같은 scene, 존 경계를
# 넘으면 새 scene ID" 라는 §4 controlled variation 규칙의 단위가 된다.
# FR3 작업공간을 3x3 으로 나누면 칸 하나가 약 20x13cm 라 기준 위치에서
# 수 cm 흔드는 것은 거의 항상 같은 존에 머문다.
STANDARD_GRID = [3, 3]

# ------------------------------------------------------------------- 품질
# 낱말 자체는 mstack/config/quality.py 에 있다 -- 데이터 동기화와 변환
# 스크립트도 같은 낱말을 쓰는데, 여기 두면 화살표가 양쪽으로 생긴다.
from mstack.data.edit_marker import mark_scene_edited  # noqa: F401  (재수출)
from mstack.config.quality import (  # noqa: F401  (여기서 재수출한다)
    QUALITY_BAD_DATA,
    QUALITY_DEPRECATED,
    QUALITY_FAILED,
    QUALITY_RETAKE,
    QUALITY_STATUSES,
    QUALITY_SUCCESS,
)

# ------------------------------------------------------------------- ID 규칙
#
# scene ID 는 **불투명**하다 -- ``S7QK3M2A`` 처럼 순서도 뜻도 없는 값이다.
# 예전에는 ``S000`` 부터 세는 번호였고, 번호에는 두 가지가 딸려 온다:
#
# * 중간을 지우면 "다시 채워야 할 것 같은" 구멍이 남는다. 2026-09-17 에
#   실제로 그렇게 했다 -- scene 6개를 빼고 **22개를 재넘버링**했고, 그
#   순간 이미 밖으로 나간 모든 ``episode_uid`` 가 다른 것을 가리키게 됐다
#   (Hub 데이터셋, 논문 표, 사람의 메모). 되돌릴 방법은
#   ``scene_renumber_20260917.json`` 뿐이다.
# * 번호는 "몇 번째"라는 뜻을 풍긴다. 그래서 "0번은 reset 으로 예약하자"
#   같은 생각이 자연스러워지는데, 그것은 위치에 의미를 싣는 일이고 같은
#   사고를 한 번 더 부른다 (PlanSlot.kind 주석).
#
# 사람이 읽을 이름은 색인이 준다 (``dataset_index`` 의 ``scenes.tsv``) --
# R2R 처럼 "짧은 불투명 ID + 별도의 넘버링 문서" 다. 그래서 ID 자체는 짧고
# 헷갈리지 않기만 하면 된다.
#
# 글자는 Crockford base32 에서 ``I L O U`` 를 뺀 것이다: 손으로 옮겨 적거나
# 읽어 줄 때 1/I, 0/O 가 섞이지 않고, U 는 빼면 우연히 만들어지는 낱말이
# 준다. 8글자면 32^8 = 1.1e12 라 실수로 겹칠 일이 없다.
SCENE_ID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
SCENE_ID_LEN = 8
#: 새로 만드는 ID 의 모양.
SCENE_ID_OPAQUE_RE = re.compile(rf"^S[{SCENE_ID_ALPHABET}]{{{SCENE_ID_LEN}}}$")
#: 번호 시절의 모양. **읽기는 계속 받는다.** 새 ID 를 쓰기로 한 것이지 옛
#: 파일을 못 읽게 하기로 한 것이 아니다 -- fr3-tabletop 의 24개는 논문 분석에
#: 계속 쓰이고, 거부하면 그 폴더 전체가 안 열린다.
SCENE_ID_LEGACY_RE = re.compile(r"^S(\d{3,})$")
SCENE_ID_RE = re.compile(
    rf"^S(?:[{SCENE_ID_ALPHABET}]{{{SCENE_ID_LEN}}}|\d{{3,}})$")
INSTRUCTION_ID_RE = re.compile(r"^I(\d{3,})$")
# 파일명은 scene ID 에서 기계적으로 파생된다. 이 정규식은 파일 목록에만 쓴다
# -- 파일명에서 task/instruction 을 판별하는 용도가 아니다(그건 metadata 가
# 정본이고, 이 포맷에는 애초에 파일명에 instruction 이 없다).
SCENE_FILE_RE = re.compile(
    rf"^scene_(?:[{SCENE_ID_ALPHABET}]{{{SCENE_ID_LEN}}}|\d{{3,}})\.hdf5$")
EPISODE_GROUP_RE = re.compile(r"^episode_(\d{3,})$")


def scene_filename(scene_id: str) -> str:
    m = SCENE_ID_RE.match(scene_id)
    if not m:
        raise ValueError(
            f"잘못된 scene ID: {scene_id!r} "
            f"(S + {SCENE_ID_ALPHABET} 에서 {SCENE_ID_LEN}글자, 예: 'S7QK3M2A')")
    if SCENE_ID_LEGACY_RE.match(scene_id):
        # 번호 시절 파일명은 0 을 채운 3자리였다 -- S24 와 S024 가 같은
        # 파일을 가리켰으므로 그 규칙을 유지한다.
        return f"scene_{int(scene_id[1:]):03d}.hdf5"
    return f"scene_{scene_id[1:]}.hdf5"


def iter_scene_files(root: Path) -> list[Path]:
    """``root`` 아래 scene 파일들. legacy ``*_demo.hdf5`` 와는 글롭이 겹치지
    않아 두 포맷이 같은 디렉터리에 있어도 서로 안 보인다.

    정렬은 **파일명 사전순**이다. ID 에 순서가 없으므로 이것은 "만든 순서"가
    아니고, 그렇게 읽혀서도 안 된다 -- 목록을 시간순으로 보고 싶으면
    metadata 의 ``created`` 를 쓴다. 여기서 정렬하는 이유는 같은 폴더가
    언제 읽어도 같은 순서를 주게 하기 위해서다 (diff·로그가 흔들리지 않게).
    """
    root = Path(root)
    out = [p for p in root.glob("scene_*.hdf5") if SCENE_FILE_RE.match(p.name)]
    out.sort(key=lambda p: p.name)
    return out


def new_scene_id(root: Path | None = None) -> str:
    """새 scene ID. **번호가 아니라 난수다** -- 순서도 뜻도 없다.

    ``root`` 를 주면 그 폴더에 같은 ID 가 있는지 확인하고 다시 뽑는다.
    32^8 에서 부딪힐 일은 없지만, 확인이 한 번의 디렉터리 읽기라서 둔다.
    """
    import secrets

    taken = {p.name for p in iter_scene_files(root)} if root is not None else set()
    while True:
        sid = "S" + "".join(secrets.choice(SCENE_ID_ALPHABET)
                            for _ in range(SCENE_ID_LEN))
        if scene_filename(sid) not in taken:
            return sid


#: (파일 지문) -> {scene_id: 순번}. list_scene_episodes 와 같은 방식으로
#: 캐시한다 -- 정본은 파일이고, 폴더가 바뀌면 지문이 달라져 무효가 된다.
_ORDINAL_CACHE: dict = {}


def scene_ordinals(root: Path) -> dict:
    """``{scene_id: 1,2,3...}`` -- **만든 순서**. 화면에서만 쓴다.

    scene ID 가 불투명해지면서(``S7QK3M2A``) 사람이 "몇 번째 scene" 을 말할
    방법이 없어졌다. 그 자리를 이 번호가 메운다.

    **어디에도 저장하지 않는다.** 저장하는 순간 두 번째 식별자가 되고, 앞의
    scene 을 지우면 낡는다 -- 2026-09-17 의 재넘버링이 그래서 문제였다
    (CLAUDE.md). 파일·episode_uid·Hub 가 쓰는 이름은 언제나 scene_id 다.
    지우면 뒤 번호가 당겨지는 것이 정상이고, 그래서 번호로 파일을 찾으면 안 된다.

    metadata attrs 만 읽는다 (이미지 청크는 안 건드린다). created 를 모르는
    파일은 뒤로, 같은 값이면 scene_id 로 가른다 -- 두 번 읽어도 같은 표가
    나와야 한다.
    """
    root = Path(root)
    files = iter_scene_files(root)
    try:
        fp = tuple((p.name, p.stat().st_mtime_ns) for p in files)
    except OSError:
        fp = None
    if fp is not None and fp in _ORDINAL_CACHE:
        return dict(_ORDINAL_CACHE[fp])
    rows = []
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                a = f["metadata"].attrs
                rows.append((str(a.get("created", "")), str(a["scene_id"])))
        except Exception:  # noqa: BLE001 -- 잠겼거나 깨진 파일은 번호에서 뺀다
            continue
    rows.sort(key=lambda r: (not r[0], r[0], r[1]))
    out = {sid: n for n, (_c, sid) in enumerate(rows, start=1)}
    if fp is not None:
        _ORDINAL_CACHE.clear()          # 폴더 하나 분량만 들고 있으면 된다
        _ORDINAL_CACHE[fp] = dict(out)
    return out


def scene_label(scene_id: str, ordinals: "dict | None" = None) -> str:
    """화면에 쓰는 짧은 이름 -- ``#7``. 번호를 모르면 ID 그대로.

    ID 를 함께 보여주지 않는 이유는 길이다. 여덟 글자 난수는 목록에서 읽히지
    않고 자리만 먹는다 (조작자, 2026-09-23). 전체 ID 가 필요한 자리 -- 툴팁,
    상세 카드, 로그 -- 에는 그대로 남는다. 파일을 찾을 때 쓰는 것은 ID 다.
    """
    n = (ordinals or {}).get(scene_id)
    return f"#{n}" if n else str(scene_id)


def next_scene_id(root: Path) -> str:
    """옛 이름. ``new_scene_id`` 를 부른다 -- "다음" 이라는 말이 순서를
    풍기지만 더 이상 순서가 없다. 부르는 곳을 옮긴 뒤 지운다."""
    return new_scene_id(root)


def episode_uid(scene_id: str, instruction_id: str, episode_idx: int) -> str:
    """``EP-S012-I003-E007`` -- HDF5, 수집 로그, QA 기록, Hub manifest,
    evaluation 결과에서 전부 이 하나의 이름을 쓴다 (§2).

    ``episode_idx`` 는 **slot(=scene×instruction) 로컬** 번호다 -- 각 slot 의
    첫 에피소드가 E000 (2026-08-13 결정). 파일 안 그룹 이름(episode_NNN)은
    별개로 파일 전체 append 순서를 유지한다."""
    return f"EP-{scene_id}-{instruction_id}-E{episode_idx:03d}"


# ------------------------------------------------------------ scene metadata
@dataclass
class SceneMetadata:
    """파일당 1회 기록되는 scene 정의. ``objects`` 는 색 이름이 아니라
    configs/scenes/props.yaml 의 instance ID 다.

    ``description`` 은 사람이 쓰는 자유 문장이다 -- 파싱 대상도 정본도 아니고
    (검증·추천·재현은 전부 구조화 필드가 담당), 배치 의도나 "지칭하지 않는
    물체(distractor)" 같은 관례를 사람 말로 남기는 자리다.
    """

    scene_id: str
    objects: list[str]
    layout: dict
    description: str = ""
    station: str = ""
    dataset_version: str = SCENE_DATASET_VERSION
    created: str = ""
    #: 기록 시점의 로봇 부하 모델 (knu-1.2.0). 질량 kg, 무게중심 m.
    #: 이 값이 없으면 파일의 절대 힘값을 해석할 수 없다 -- 미신고 질량이
    #: 그대로 외력 추정에 섞이기 때문이다 (dataset_schema 의 META_PAYLOAD_* 참조).
    #: 부하를 못 알려주는 로봇(시뮬레이터 등)에서는 None 이고, 그때는 attrs 를
    #: 쓰지 않아 파일이 1.1.1 규칙으로 검사된다.
    payload_mass: Optional[float] = None
    payload_com: Optional[list] = None
    #: 기록 시점의 리셋 자세 (knu-1.2.1). 별칭과 7관절 절대값을 함께 적는다 --
    #: 이름만으로는 FR3_RESET_POSES 가 바뀌면 옛 파일을 잘못 읽고, 값만으로는
    #: 사람이 그것이 무엇인지 알아보지 못한다.
    reset_pose: Optional[str] = None
    reset_qpos: Optional[list] = None
    #: 이 파일을 만든 소프트웨어 (knu-1.2.2). 물리 셋업은 위에서 적고 있었는데
    #: 그것을 돌린 코드·펌웨어는 어디에도 없었다 (조작자, 2026-09-13).
    #: 자동으로 읽은 것만 담고, 못 읽은 항목은 None 이라 attrs 에도 안 쓴다.
    collector_commit: Optional[str] = None
    pylibfranka_version: Optional[str] = None
    fr3_system_version: Optional[str] = None
    #: "live" (수집하며 적음) / "backfilled <날짜>" (나중에 채움).
    provenance_source: Optional[str] = None

    def validate(self, known_prop_ids: Optional[set[str]] = None) -> None:
        """구조가 틀린 metadata 로 파일을 만드는 것을 생성 시점에 막는다.
        ``known_prop_ids`` 를 주면(보통 mstack.scene.props.active_prop_ids()) 인벤토리에
        없는 ID 도 잡는다 -- 안 주면 형식 검사만 한다."""
        if not SCENE_ID_RE.match(self.scene_id):
            raise ValueError(f"잘못된 scene ID: {self.scene_id!r} (예: 'S000')")
        if not self.objects:
            raise ValueError("objects 가 비어 있다 -- scene 에는 물체가 최소 1개 필요하다")
        placed = list(self.objects)
        dup = {o for o in placed if placed.count(o) > 1}
        if dup:
            raise ValueError(f"objects 에 중복 instance ID: {sorted(dup)}")
        for oid in placed:
            if not oid.startswith("OBJ-"):
                raise ValueError(
                    f"instance ID 가 아니다: {oid!r} -- 색 이름이 아니라 "
                    "configs/scenes/props.yaml 의 OBJ-* ID 를 적는다 (§3)"
                )
            if known_prop_ids is not None and oid not in known_prop_ids:
                raise ValueError(f"인벤토리에 없는 instance ID: {oid!r} (configs/scenes/props.yaml)")

        grid = self.layout.get("grid")
        if (
            not isinstance(grid, (list, tuple))
            or len(grid) != 2
            or not all(isinstance(g, int) and g > 0 for g in grid)
        ):
            raise ValueError(f"layout.grid 는 [rows, cols] 양의 정수 2개여야 한다: {grid!r}")
        if list(grid) != STANDARD_GRID:
            raise ValueError(
                f"layout.grid 는 표준 {STANDARD_GRID} 만 허용한다 (현재: {list(grid)!r}) -- "
                "격자가 scene 경계 판정의 단위라 수집자마다 다르면 안 된다. "
                "바꾸려면 STANDARD_GRID 와 프로토콜 §4 를 함께 바꾼다"
            )
        placements = self.layout.get("placements")
        if not isinstance(placements, dict) or not placements:
            raise ValueError("layout.placements 가 비어 있다 -- 모든 물체의 존을 기록한다")
        for oid, spec in placements.items():
            if oid not in placed:
                raise ValueError(f"layout.placements 의 {oid!r} 가 objects 에 없다")
            zone = (spec or {}).get("zone")
            if (
                not isinstance(zone, (list, tuple))
                or len(zone) != 2
                or not all(isinstance(z, int) for z in zone)
                or not (0 <= zone[0] < grid[0] and 0 <= zone[1] < grid[1])
            ):
                raise ValueError(f"{oid} 의 zone 이 격자를 벗어났다: {zone!r} (grid={grid})")
        missing = [o for o in placed if o not in placements]
        if missing:
            raise ValueError(f"layout.placements 에 존이 없는 물체: {missing}")
        for rel in self.layout.get("relations", []):
            if not (isinstance(rel, (list, tuple)) and len(rel) == 3):
                raise ValueError(f"relations 항목은 [주어, 관계, 목적어] 3개여야 한다: {rel!r}")


def _opt_str(meta: h5py.Group, key: str):
    """있으면 문자열, 없으면 None. **없는 것과 빈 것을 구분한다** -- 판번호
    필드는 "안 적혔다" 와 "빈 값으로 적혔다" 가 다른 뜻이다."""
    return str(meta.attrs[key]) if key in meta.attrs else None


def _read_metadata(meta: h5py.Group) -> SceneMetadata:
    return SceneMetadata(
        scene_id=str(meta.attrs["scene_id"]),
        objects=json.loads(meta.attrs["objects"]),
        layout=json.loads(meta.attrs["layout"]),
        description=str(meta.attrs.get("description", "")),
        station=str(meta.attrs.get("station", "")),
        # 옛 표기(scene-v1)는 여기서 knu-1.0.0 으로 풀어 준다 -- 읽는 쪽은
        # 어느 시절 파일인지 신경 쓰지 않고 SemVer 하나만 보면 된다.
        dataset_version=normalize_schema_version(
            meta.attrs.get("dataset_version", "")),
        created=str(meta.attrs.get("created", "")),
        payload_mass=(float(meta.attrs[META_PAYLOAD_MASS])
                      if META_PAYLOAD_MASS in meta.attrs else None),
        payload_com=(json.loads(meta.attrs[META_PAYLOAD_COM])
                     if META_PAYLOAD_COM in meta.attrs else None),
        reset_pose=(str(meta.attrs[META_RESET_POSE])
                    if META_RESET_POSE in meta.attrs else None),
        reset_qpos=(json.loads(meta.attrs[META_RESET_QPOS])
                    if META_RESET_QPOS in meta.attrs else None),
        collector_commit=_opt_str(meta, META_COLLECTOR_COMMIT),
        pylibfranka_version=_opt_str(meta, META_PYLIBFRANKA_VERSION),
        fr3_system_version=_opt_str(meta, META_FR3_SYSTEM_VERSION),
        provenance_source=_opt_str(meta, META_PROVENANCE_SOURCE),
    )


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _video_frames(grp: h5py.Group) -> "int | None":
    """이 에피소드에 **사진이 몇 장** 있는가. 단일 축 파일이면 None.

    ``num_samples`` 와 다른 값이다. knu-2.0.0 에서 그것은 control 축의 행
    수이고, 120 Hz 로 찍으면 857 이 되는데 사진은 30 fps 로 214 장이다.
    화면에서 사람이 넘기는 단위는 사진이므로 (트림 슬라이더 한 칸 = 한 장),
    목록에 857 을 띄우면 슬라이더 끝 번호와 네 배 어긋나 보인다.

    None 은 "이 파일은 축이 하나라 둘이 같은 값" 이라는 뜻이고, 그때는
    부르는 쪽이 ``num_samples`` 를 그대로 쓰면 된다.
    """
    t = grp.get("t")
    if t is None or not hasattr(t, "keys"):
        return None
    for axis in ("agent", "wrist"):
        if axis in t:
            return int(t[axis].shape[0])
    return None


def _episode_summary(name: str, grp: h5py.Group) -> dict:
    success = grp.attrs.get("success")
    return {
        "name": name,
        #: 사진 장수 (knu-2.0.0). 단일 축 파일은 None -- num_samples 와 같다.
        "video_frames": _video_frames(grp),
        "episode_id": int(grp.attrs["episode_id"]),
        "slot_episode_idx": int(grp.attrs.get("slot_episode_idx", -1)),
        "episode_uid": str(grp.attrs["episode_uid"]),
        "instruction_id": str(grp.attrs["instruction_id"]),
        "instruction": str(grp.attrs["instruction"]),
        "quality_status": str(grp.attrs["quality_status"]),
        "success": None if success is None else bool(success),
        "collector": str(grp.attrs.get("collector", "")),
        "timestamp": str(grp.attrs.get("timestamp", "")),
        "num_samples": int(grp.attrs["num_samples"]),
    }


# ------------------------------------------------------------------- writer
def _meta_gaps(metadata) -> list:
    """도장을 막는 세션 메타 중 아직 없는 것들, 사람이 읽는 이름으로.

    세 갈래를 두 자리(새 파일·pending)에서 같은 말로 부르기 위한 것이다 --
    한쪽만 고치면 같은 결함이 다른 이름으로 보고된다.
    """
    gaps = []
    if metadata.payload_mass is None:
        gaps.append("부하 모델")
    if not (metadata.reset_pose and metadata.reset_qpos):
        gaps.append("리셋 자세")
    if not metadata.provenance_source:
        gaps.append("판번호 출처")
    return gaps


def stampable_version(want: str, has_payload: bool,
                      has_reset: bool = False,
                      has_provenance: bool = False) -> str:
    """찍어도 되는 가장 높은 버전. 못 채우는 요구가 있으면 내린다.

    생성 시점에 모를 수 있는 것은 세션이 밖에서 받아 오는 값들이다 -- 부하
    모델(knu-1.2.0), 리셋 자세(knu-1.2.1), 그리고 판번호(knu-1.2.2: git 커밋과
    로봇 쪽 버전). 나머지 요구(관측·에피소드 attrs)는 이 세션이 직접 쓰는
    값이라 늘 채워진다. 새 요구가 생기면 여기에 조건을 더한다.

    **"모르면 있는 것으로 친다"(have.get(a, True))가 기본이라 새 요구를 여기
    안 적으면 조용히 통과한다.** 2026-09-13 에 판번호를 더하면서 실제로
    그랬고, doctor 테스트가 잡았다 (판번호 없는 파일이 knu-1.1.2 로 찍혔다).
    """
    from mstack.data.dataset_schema import SCHEMA_FIELDS

    want = normalize_schema_version(want)
    if want not in SCHEMA_FIELDS:
        return want
    from mstack.data.dataset_schema import META_PROVENANCE_SOURCE

    have = {META_PAYLOAD_MASS: has_payload, META_RESET_POSE: has_reset,
            META_PROVENANCE_SOURCE: has_provenance}

    def _ok(version: str) -> bool:
        need = SCHEMA_FIELDS[version].get("metadata_attrs", ())
        return all(have.get(a, True) for a in need)

    if _ok(want):
        return want
    lower = [v for v in SCHEMA_FIELDS
             if schema_version_key(v) <= schema_version_key(want) and _ok(v)]
    return max(lower, key=schema_version_key) if lower else want


class SceneWriter:
    """Owns one ``scene_XXX.hdf5``: one file per scene, one ``episode_NNN``
    per demonstration, instruction 은 에피소드마다 다를 수 있다.

    legacy :class:`~mstack.data.libero_format.LiberoTaskWriter` 와 같은 스레드 규칙:
    파일을 만지는 호출(save_buffer / set_quality_status / list_episodes /
    close)은 호출자가 한 스레드로 직렬화한다 (GUI 에서는 EpisodeSaver).

    새 scene::

        meta = SceneMetadata(scene_id=next_scene_id(root), objects=[...], layout={...})
        w = SceneWriter(root, metadata=meta)

    기존 scene 이어찍기::

        w = SceneWriter(root, scene_id="S012", resume=True)   # metadata 는 파일에서

    instruction 은 저장 시점에 명시적으로 받는다 -- writer 상태로 들고 있지
    않는다. 저장이 백그라운드 스레드라, 조작자가 다음 slot 으로 넘어간 뒤에
    직전 에피소드가 저장되는 경합에서 "그 에피소드가 실제로 수행한 문장"이
    찍혀야 하기 때문이다 (crop_params 를 buffer 에 싣는 것과 같은 이유).
    """

    def __init__(
        self,
        root: Path,
        scene_id: Optional[str] = None,
        metadata: Optional[SceneMetadata] = None,
        resume: bool = False,
        schema: Optional[DatasetSchemaConfig] = None,
        crop_params: Optional[dict] = None,
        collector: str = "",
        known_prop_ids: Optional[set[str]] = None,
        session_version: Optional[str] = None,
        session_payload: Optional[dict] = None,
        session_reset: Optional[dict] = None,
        session_provenance: Optional[dict] = None,
        metadata_pending: bool = False,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema = schema or DatasetSchemaConfig()
        self.crop_params = crop_params or default_crop_params()
        self.collector = collector
        self._buffer = LiberoEpisodeBuffer(self.schema, self.crop_params)
        #: 이어찍기에서 버전 도장을 어떻게 했는지 -- 상류가 로그로 보여준다.
        self.version_note = ""

        if resume:
            if metadata is not None:
                raise ValueError("resume=True 면 metadata 는 파일에서 읽는다 -- 둘 다 주지 않는다")
            if scene_id is None:
                raise ValueError("resume=True 면 scene_id 가 필요하다")
            self.path = self.root / scene_filename(scene_id)
            if not self.path.exists():
                raise FileNotFoundError(f"{self.path} 가 없다 -- 새 scene 이면 metadata 를 주고 resume=False 로")
            self._file = h5py.File(self.path, "a")
            _mark_close_on_exec(self._file)
            self._meta = self._file["metadata"]
            self.metadata = _read_metadata(self._meta)
            if self.metadata.scene_id != scene_id:
                # 파일명이 아니라 내부 metadata 가 정본이다. 어긋났다는 것은
                # 파일이 손으로 rename 됐다는 뜻이므로 조용히 진행하지 않는다.
                raise ValueError(
                    f"{self.path.name} 내부 scene_id 는 {self.metadata.scene_id!r} 다 "
                    f"(요청: {scene_id!r}) -- 파일명이 아니라 metadata 를 믿는다"
                )
            self._resume_version(session_version, session_payload,
                                 session_reset, session_provenance)
        else:
            if metadata is None:
                raise ValueError("새 scene 에는 metadata 가 필요하다")
            if scene_id is not None and scene_id != metadata.scene_id:
                raise ValueError(f"scene_id 인자({scene_id!r})와 metadata.scene_id({metadata.scene_id!r})가 다르다")
            metadata.validate(known_prop_ids=known_prop_ids)
            self.path = self.root / scene_filename(metadata.scene_id)
            if self.path.exists():
                raise FileExistsError(
                    f"{self.path} already exists; pass resume=True to append episodes."
                )
            self._file = h5py.File(self.path, "a")
            _mark_close_on_exec(self._file)
            self._meta = self._file.create_group("metadata")
            self.metadata = metadata
            if not metadata.created:
                metadata.created = _now_iso()
            # **찍는 버전은 채울 수 있는 것까지다.** 세션 버전이 요구하는
            # metadata attr 을 못 채우면서 그 도장을 찍으면, 파일이 갖지 않은
            # 필드를 가졌다고 주장하는 상태가 된다 -- 검증이 통째로 실패하고,
            # 그것을 만든 조작이 knu-1.1.0 사고였다 (#47).
            #
            # 2026-09-07 에 실제로 그렇게 됐다: 로봇이 부하를 못 알려준 세션에서
            # S017~S019 가 payload 없이 knu-1.2.0 으로 찍혀 57 에피소드가
            # 검증 불가가 됐다. _resume_version 은 이 검사를 하고 있었는데,
            # **새 파일 경로에는 없었다.**
            #
            # ``metadata_pending`` 는 그 판단을 **미룬다** -- 구성기가 로봇에
            # 붙기 전에 만드는 빈 scene 이다. 거기서 빠진 값은 "물어봤는데
            # 없다"가 아니라 "아직 안 물어봤다"라서, 내려 찍으면 2.0.0 에서
            # MAJOR 를 넘어가 그 파일에 영영 녹화할 수 없게 된다 (S024).
            # #47 방어는 없어지지 않고 **Connect 로 옮겨간다**: 에피소드가
            # 하나라도 쓰이기 전에 _settle_pending 이 채우거나 거절한다.
            asked = normalize_schema_version(metadata.dataset_version)
            if metadata_pending:
                metadata.dataset_version = asked
                gaps = _meta_gaps(metadata)
                if gaps:
                    self.version_note = (
                        f"{asked} 로 만들었습니다. {' · '.join(gaps)} 는 아직 "
                        f"모르므로 Connect 할 때 채웁니다 -- 그때도 모르면 "
                        f"녹화를 시작하지 않습니다.")
            else:
                metadata.dataset_version = stampable_version(
                    asked, metadata.payload_mass is not None,
                    bool(metadata.reset_pose and metadata.reset_qpos),
                    bool(metadata.provenance_source))
            if metadata.dataset_version != asked:
                # **말없이 내리지 않는다.** _resume_version 이 못 올릴 때
                # 이유를 남기는 것과 같은 이유다 -- 마법사에서 고른 버전과
                # 파일에 찍힌 버전이 다른데 아무도 말해 주지 않으면, 그것을
                # 아는 방법이 나중에 검증기를 돌리는 것뿐이 된다.
                self.version_note = (
                    f"{asked} 로 만들려 했는데 {' · '.join(_meta_gaps(metadata))} 를 "
                    f"몰라 {metadata.dataset_version} 로 찍었습니다. 나중에 "
                    f"Doctor 에서 채워 올릴 수 있습니다.")
            self._meta.attrs["scene_id"] = metadata.scene_id
            self._meta.attrs["objects"] = json.dumps(metadata.objects, ensure_ascii=False)
            self._meta.attrs["layout"] = json.dumps(metadata.layout, ensure_ascii=False)
            self._meta.attrs["description"] = metadata.description
            self._meta.attrs["station"] = metadata.station
            self._meta.attrs["dataset_version"] = metadata.dataset_version
            self._meta.attrs["created"] = metadata.created
            # 부하 모델은 있을 때만 쓴다 -- 못 주는 로봇에서 0 을 적으면
            # "부하가 0 이었다"로 읽혀 없느니만 못하다.
            if metadata.payload_mass is not None:
                self._meta.attrs[META_PAYLOAD_MASS] = float(metadata.payload_mass)
                self._meta.attrs[META_PAYLOAD_COM] = json.dumps(
                    list(metadata.payload_com or []))
            # 리셋 자세도 있을 때만 (같은 이유 -- 모르는 값을 0 으로 적으면
            # "0 이었다" 로 읽힌다).
            if metadata.reset_pose and metadata.reset_qpos:
                self._meta.attrs[META_RESET_POSE] = str(metadata.reset_pose)
                self._meta.attrs[META_RESET_QPOS] = json.dumps(
                    [float(x) for x in metadata.reset_qpos])
            # 판번호도 **있을 때만**. 못 읽은 것을 "?" 로 적으면 읽은 값처럼
            # 보인다 (knu-1.1.0 의 0 으로 찬 힘 필드가 그 교훈이다).
            for attr, value in (
                    (META_COLLECTOR_COMMIT, metadata.collector_commit),
                    (META_PYLIBFRANKA_VERSION, metadata.pylibfranka_version),
                    (META_FR3_SYSTEM_VERSION, metadata.fr3_system_version),
                    (META_PROVENANCE_SOURCE, metadata.provenance_source)):
                if value:
                    self._meta.attrs[attr] = str(value)
            self._meta.attrs["next_episode_idx"] = 0

        if "next_episode_idx" not in self._meta.attrs:
            # 정상 파일에는 항상 있다. 없다면 손상이므로 기존 이름에서 복원하되
            # max+1 -- 어떤 경우에도 번호를 재사용하지 않는다.
            existing = [
                int(EPISODE_GROUP_RE.match(k).group(1))
                for k in self._file.keys()
                if EPISODE_GROUP_RE.match(k)
            ]
            self._meta.attrs["next_episode_idx"] = max(existing, default=-1) + 1
        self._file.flush()

    def _resume_version(self, session_version: "str | None",
                        session_payload: "dict | None" = None,
                        session_reset: "dict | None" = None,
                        session_provenance: "dict | None" = None) -> None:
        """이어찍기: 파일의 버전 도장을 이번 세션 버전에 맞춘다.

        이어 찍으면 **이번 세션이 쓰는 필드**가 그 파일에 들어간다. 그런데
        도장은 파일이 처음 만들어질 때 찍힌 그대로였다 -- 2026-09-06 에
        scene_015 가 정확히 그렇게 됐다: 에피소드를 다 지우고 새 필드 구성으로
        40개를 다시 찍었는데 도장은 옛 knu-1.1.0 이라, 그 버전이 요구하는 (이제는
        기록하지 않는) 필드가 없다며 검증에 걸렸다.

        **올리기만 한다** (2026-09-06 사용자 결정). 내려 찍으면 이웃 에피소드가
        가진 열을 잃은 파일이 되고, 그건 버저닝이 막으려는 바로 그 상황이다.
        낮은 버전으로 찍고 싶으면 새 데이터셋으로 시작하는 것이 올바른 신호다.

        이미 있는 에피소드가 새 버전의 필수 관측을 갖췄을 때만 올린다 -- 아니면
        도장이 파일 내용을 넘어서 약속하는 셈이라, 옛 도장을 두고 사유만 남긴다.
        """
        from mstack.data.dataset_schema import (parse_schema_version,
                                                schema_required_fields)

        cur = normalize_schema_version(self.metadata.dataset_version)
        want = normalize_schema_version(session_version or "")
        if not want:
            return
        if want == cur:
            # 도장은 같아도 **요구하는 값이 아직 없을 수 있다** -- 구성기가
            # metadata_pending 으로 만든 빈 scene 이 그렇다. 여기서 채우지
            # 않으면 그 파일은 자기가 갖지 않은 필드를 가졌다고 주장한 채로
            # 에피소드를 받는다 (#47 과 똑같은 모양).
            self._settle_pending(want, session_payload, session_reset,
                                 session_provenance)
            return
        req = schema_required_fields(want)
        if req is None:
            self.version_note = f"모르는 버전이라 도장을 두었습니다: {want}"
            return
        # MAJOR 가 다르면 **이어찍지 않는다.** MINOR 올림과 성격이 다르다:
        # 2.0.0 은 계열마다 시간축이 따로인 구조라, 1.x 에피소드와 2.x
        # 에피소드가 한 파일에 섞이면 그 파일은 어느 도장을 찍어도 거짓이
        # 된다 (1.3.0 이라 하면 2.x 에피소드가 옛 리더를 깨고, 2.0.0 이라
        # 하면 1.x 에피소드에 t/control 이 없다).
        a, b = parse_schema_version(want), parse_schema_version(cur)
        if a and b and a[0] != b[0]:
            raise ValueError(
                f"이 scene 은 {cur} 로 기록됐고 이번 세션은 {want} 입니다 -- "
                "MAJOR 가 다르면 한 파일에 섞을 수 없습니다. 새 scene 으로 "
                "시작하세요 (옛 파일은 그대로 읽힙니다).")
        if schema_version_key(want) < schema_version_key(cur):
            self.version_note = (
                f"이 파일은 {cur} 인데 이번 세션은 {want} 입니다 -- 버전은 "
                f"내리지 않습니다. 낮은 버전으로 찍으려면 새 데이터셋으로 "
                f"시작하세요.")
            return
        missing = self._episodes_missing(req["obs_datasets"],
                                         req.get("episode_datasets", ()))
        if missing:
            self.version_note = (
                f"{cur} -> {want} 로 올리지 못했습니다: 기존 에피소드 "
                f"{len(missing)}개에 {want} 필수 관측이 없습니다. 도장을 그대로 둡니다.")
            return
        # 관측만 보면 부족하다. 새 버전이 metadata attrs 를 요구할 수도 있고
        # (knu-1.2.0 의 부하 모델), 그건 에피소드가 아니라 파일에 있다. 이걸
        # 빼먹으면 도장은 올라갔는데 그 버전이 요구하는 값이 없는 파일이 된다
        # -- 고치려던 것과 똑같은 모양의 결함이다.
        need_meta = [k for k in req["metadata_attrs"] if k not in self._meta.attrs]
        if need_meta and not self._fill_meta(need_meta, session_payload,
                                             session_reset, session_provenance):
            self.version_note = (
                f"{cur} -> {want} 로 올리지 못했습니다: {want} 가 요구하는 "
                f"{', '.join(need_meta)} 를 이번 세션이 알지 못합니다. "
                f"도장을 그대로 둡니다.")
            return
        self.metadata.dataset_version = want
        self._meta.attrs["dataset_version"] = want
        self.version_note = f"버전 도장을 {cur} -> {want} 로 올렸습니다."

    def _settle_pending(self, want: str, payload: "dict | None",
                        reset: "dict | None",
                        provenance: "dict | None") -> None:
        """도장과 세션 버전이 같을 때, 아직 빈 필수 metadata 를 이번 세션 값으로
        채운다. 못 채우고 **빈 scene** 이면 거절한다.

        구성기의 [✚ 새 Scene 만들기] 는 로봇에 붙기 전에 파일을 만든다
        (2026-09-06 결정: 미리 여러 개 짜 둘 수 있어야 한다). 그때 부하 모델은
        알 수 없는데, 그것을 이유로 도장을 내리면 2.0.0 에서는 MAJOR 를 넘어가
        **그 파일에 영영 녹화할 수 없게 된다** (2026-09-23 S024). 그래서 도장은
        요청한 버전으로 두고, 판단을 이 자리로 미룬다.

        여기서 거절하는 편이 구성기에서 거절하는 것보다 낫다: 로봇이 이미
        붙어 있으므로 남는 사유가 "이 로봇이 부하 모델을 주지 않는다" 하나뿐이고,
        그건 조작자가 알아야 하는 진짜 문제다.

        이미 에피소드가 있는 파일은 건드리지 않는다 -- 옛 파일은 자기 도장대로
        검사받으면 되고, 여기서 값을 넣으면 그 에피소드들이 찍힌 때의 값이
        아닌 것이 섞인다.
        """
        from mstack.data.dataset_schema import schema_required_fields

        req = schema_required_fields(want)
        if req is None:
            return
        need = [k for k in req["metadata_attrs"] if k not in self._meta.attrs]
        if not need:
            return
        if self._fill_meta(need, payload, reset, provenance):
            self.version_note = (
                f"{want} 가 요구하는 {', '.join(need)} 를 이번 세션 값으로 "
                f"채웠습니다.")
            return
        if self._episode_names():
            self.version_note = (
                f"{want} 가 요구하는 {', '.join(need)} 가 이 파일에 없는데 "
                f"이번 세션도 모릅니다. 이미 에피소드가 있어 도장을 그대로 둡니다.")
            return
        raise ValueError(
            f"이 scene 은 {want} 로 만들어졌는데 그 버전이 요구하는 "
            f"{', '.join(need)} 가 아직 비어 있고, 이번 세션도 그 값을 "
            "알지 못합니다 (로봇이 부하 모델을 주지 않으면 이렇게 됩니다). "
            "그대로 녹화하면 파일이 갖지 않은 필드를 가졌다고 주장하게 되어 "
            "시작하지 않습니다.")

    def _episode_names(self) -> list:
        """이 파일의 에피소드 그룹 이름들."""
        return [n for n in self._file if EPISODE_GROUP_RE.match(n)]

    def _fill_meta(self, need: list, payload: "dict | None",
                   reset: "dict | None" = None,
                   provenance: "dict | None" = None) -> bool:
        """올리는 데 필요한 metadata attrs 를 이번 세션 값으로 채운다.

        채울 수 있는 것만 채우고, 하나라도 모르면 **아무것도 쓰지 않고** False.
        절반만 채워 두면 그 다음 검사에서 통과해 버려, 모르는 값이 빈 채로
        도장만 올라간 파일이 남는다.

        **세 갈래를 모두 안다.** 판번호(knu-1.2.2)가 요구사항에 들어온
        2026-09-13 에 여기에는 그 갈래를 더하지 않았고, 그래서 값을 아는
        세션도 도장을 올리지 못했다 -- S023 이 그렇게 되었다: 로그에
        ``[부하] 850 g`` 와 ``[판번호] collector_commit=...`` 를 찍은 바로 그
        초에 "이번 세션이 알지 못합니다" 가 나왔고, 모두-아니면-전무 규칙
        때문에 알고 있던 부하까지 버려졌다. 1.3.0 내용(timing/*)을 담은 파일이
        knu-1.1.1 도장으로 남았다 (2026-09-18, Doctor 로 사후 복구).
        """
        known = {}
        if payload and payload.get("mass") is not None:
            known[META_PAYLOAD_MASS] = float(payload["mass"])
            known[META_PAYLOAD_COM] = json.dumps(list(payload.get("com") or []))
        if reset and reset.get("name") and reset.get("qpos") is not None:
            known[META_RESET_POSE] = str(reset["name"])
            known[META_RESET_QPOS] = json.dumps(
                [float(x) for x in reset["qpos"]])
        if provenance:
            # 이 세션이 그 자리에서 읽은 값이므로 ``live`` 다 -- Doctor 가
            # 나중에 채울 때 쓰는 ``backfilled <날짜>`` 와 구분된다.
            known[META_PROVENANCE_SOURCE] = "live"
        if any(k not in known for k in need):
            return False
        for k in need:
            self._meta.attrs[k] = known[k]
        if META_PAYLOAD_MASS in need:
            self.metadata.payload_mass = known[META_PAYLOAD_MASS]
            self.metadata.payload_com = json.loads(known[META_PAYLOAD_COM])
        if META_RESET_POSE in need:
            self.metadata.reset_pose = known[META_RESET_POSE]
            self.metadata.reset_qpos = json.loads(known[META_RESET_QPOS])
        if META_PROVENANCE_SOURCE in need:
            self.metadata.provenance_source = known[META_PROVENANCE_SOURCE]
            # 선택 항목은 요구사항이 아니라서 need 에 없다. 아는 값이면
            # 함께 적는다 -- 나중에 "어느 코드에서 나왔나" 를 묻는 쪽은
            # 필수/선택을 구분하지 않는다.
            for attr, key in ((META_COLLECTOR_COMMIT, "collector_commit"),
                              (META_PYLIBFRANKA_VERSION, "pylibfranka"),
                              (META_FR3_SYSTEM_VERSION, "fr3_system")):
                val = (provenance or {}).get(key)
                if val and attr not in self._meta.attrs:
                    self._meta.attrs[attr] = str(val)
        return True

    def _episodes_missing(self, need, need_episode=()) -> list:
        """필수 관측이 빠진 기존 에피소드 이름들 (데이터는 읽지 않는다).

        ``need_episode`` are paths under the episode group itself
        (``timing/frame`` for knu-1.3.0). Checking obs alone would raise an old
        file's stamp to a version whose timing its episodes never had.
        """
        out = []
        for name in self._file:
            if not EPISODE_GROUP_RE.match(name):
                continue
            grp = self._file[name]
            obs = grp.get("obs")
            if (obs is None or any(k not in obs for k in need)
                    or any(k not in grp for k in need_episode)):
                out.append(name)
        return out


    # ------------------------------------------------------- 버퍼 (legacy 미러)
    def start_episode(self) -> None:
        self._buffer.clear()

    def add_frame(self, **kwargs: Any) -> None:
        self._buffer.add_frame(**kwargs)

    def discard_episode(self) -> None:
        self._buffer.clear()

    def expect_capture(self, roles) -> None:
        """이 역할들의 이미지는 capture 에서 올 것이므로, 20 Hz 루프가 집어간
        프레임을 버퍼에 쌓지 않는다 (버퍼만 만진다)."""
        self._buffer.capture_roles = set(roles)

    def add_command(self, t: float, joints, gripper: float) -> None:
        """명령 틱 하나 (버퍼만 만진다)."""
        self._buffer.add_command(t, joints, gripper)

    def set_control_hz(self, hz: float) -> None:
        """이 에피소드 control 축의 목표 주기. 버퍼만 만진다."""
        self._buffer.control_hz = float(hz)

    def set_capture(self, axis: str, frames: list) -> None:
        """카메라 한 대가 이 에피소드 동안 준 프레임 전부를 버퍼에 싣는다.

        ``add_frame`` 과 같이 **버퍼만 만지는** 호출이라 저장 스레드를 거치지
        않는다. 에피소드가 끝난 직후 detach_buffer 전에 불러야 한다.
        """
        self._buffer.set_capture(axis, frames)

    def detach_buffer(self) -> LiberoEpisodeBuffer:
        buf = self._buffer
        self._buffer = LiberoEpisodeBuffer(self.schema, self.crop_params)
        #: 이어찍기에서 버전 도장을 어떻게 했는지 -- 상류가 로그로 보여준다.
        self.version_note = ""
        return buf

    # ------------------------------------------------------------- 기준 사진
    def set_reference_image(self, img: np.ndarray) -> None:
        """scene 정면 사진 (H, W, 3) uint8. Scene Sheet 의 "사진 1장 필수"(§6)를
        파일 안에 넣는다 -- 배치 재현과 갤러리 대표 이미지가 이걸 쓴다.
        수집 시작 전 다시 찍을 수 있게 덮어쓰기는 허용한다."""
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
            raise ValueError(f"(H, W, 3) uint8 이어야 한다: shape={arr.shape}, dtype={arr.dtype}")
        if "reference_image" in self._meta:
            del self._meta["reference_image"]
        # 단일 이미지라 프레임 축이 없다 -- 통째로 한 청크로 둔다. gzip 도
        # 무손실이라 기준 사진의 목적(배치 재현·대표 이미지)에 그대로 맞는다.
        self._meta.create_dataset("reference_image", data=arr,
                                  compression="gzip", compression_opts=4,
                                  chunks=arr.shape)
        self._file.flush()

    @property
    def has_reference_image(self) -> bool:
        """기준 사진 유무. 자동 캡처(첫 에피소드의 agentview)가 수동 촬영본을
        덮어쓰지 않도록 쓰기 전에 확인하는 용도다."""
        return "reference_image" in self._meta

    # ----------------------------------------------------------------- 저장
    def save_buffer(
        self,
        buf: LiberoEpisodeBuffer,
        *,
        instruction: str,
        instruction_id: str,
        success: Optional[bool] = None,
        quality_status: Optional[str] = None,
        collector: Optional[str] = None,
        pool: "Any" = None,
        timestamp: Optional[str] = None,
    ) -> Optional[str]:
        """Commits one detached episode buffer as the next ``episode_NNN``.

        instruction 관련 규칙:
        - ``instruction`` 은 따옴표 없는 순수 문장. legacy 습관으로 감싸진
          문자열이 들어오면 조용히 저장하지 않고 거부한다.
        - ``quality_status`` 를 안 주면 ``success`` 에서 파생한다. 라벨이 아예
          없는 에피소드는 거부한다 -- 배포 필터링이 quality_status 만 보므로,
          라벨 없는 에피소드는 나중에 아무도 판정할 수 없다.

        Returns the group name, or None if the buffer was empty.
        """
        n = len(buf)
        if n < 2:
            buf.clear()
            return None

        if not INSTRUCTION_ID_RE.match(instruction_id):
            raise ValueError(f"잘못된 instruction ID: {instruction_id!r} (예: 'I000')")
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("instruction 이 비어 있다")
        if len(instruction) >= 2 and instruction[0] == '"' and instruction[-1] == '"':
            raise ValueError(
                f"instruction 이 따옴표로 감싸져 있다: {instruction!r} -- "
                "scene 포맷은 순수 문자열만 저장한다 (legacy 의 f'\"{...}\"' 관례를 가져오지 않는다)"
            )
        if quality_status is None:
            if success is None:
                raise ValueError(
                    "success 나 quality_status 중 하나는 있어야 한다 -- "
                    "scene 포맷은 라벨 없는 에피소드를 허용하지 않는다"
                )
            quality_status = QUALITY_SUCCESS if success else QUALITY_FAILED
        if quality_status not in QUALITY_STATUSES:
            raise ValueError(f"잘못된 quality_status: {quality_status!r} (허용: {QUALITY_STATUSES})")

        idx = int(self._meta.attrs["next_episode_idx"])
        # slot(=instruction) 로컬 E번호: 각 slot 은 E000 부터 센다 (2026-08-13
        # 결정). 기존 같은 slot 에피소드의 uid E-부분 최대+1 로 계산 --
        # 과거 파일(전역 번호 시절)에 이어붙여도 uid 가 충돌하지 않는다.
        # 삭제 뒤에는 renumber 가 E 를 0..k-1 로 다시 채우므로 새 번호는 k.
        slot_idx = _next_slot_idx(self._file, self._meta, instruction_id)
        self._meta.attrs["next_episode_idx"] = idx + 1
        name = f"episode_{idx:03d}"
        grp = self._file.create_group(name)
        write_episode_payload(grp, buf, self.schema, success=success, pool=pool)

        sid = self.metadata.scene_id
        grp.attrs["scene_id"] = sid
        grp.attrs["instruction_id"] = instruction_id
        grp.attrs["episode_id"] = idx
        grp.attrs["slot_episode_idx"] = slot_idx
        grp.attrs["episode_uid"] = episode_uid(sid, instruction_id, slot_idx)
        grp.attrs["instruction"] = instruction
        grp.attrs["quality_status"] = quality_status
        grp.attrs["collector"] = self.collector if collector is None else collector
        grp.attrs["timestamp"] = timestamp or _now_iso()

        self._file.flush()
        buf.clear()
        return name

    def delete_episode(self, name: str) -> None:
        """에피소드 삭제 (큐레이션: 실패·튀는 궤적을 푸시 전에 지운다).

        legacy 와 같은 규칙 -- 지운 뒤 **번호를 다시 매긴다**: 그룹 이름
        ``episode_000..N-1`` 연속, ``episode_id`` attr 도 맞추고, 각 slot 의
        E번호(``slot_episode_idx``/``episode_uid``)도 순서대로 다시 부여한다.
        uid 는 "그 slot 의 몇 번째" 로 완전히 파생되는 값이라 저장된 이력을
        보존할 것이 없다 (2026-08-14 결정: 큐레이션은 보존 대상이 아니다).
        지운 것이 이미 Hub 에 있으면 다음 전체 처리가 '삭제됨' 으로 잡아
        재빌드를 요구한다 -- 그때 사이드카도 새 uid 로 다시 만들어진다.
        파일 크기는 공간 회수 전까지 줄지 않는다 (HDF5 특성).
        """
        if name not in self._file or not EPISODE_GROUP_RE.match(name):
            raise KeyError(f"{name!r} not found in {self.path}")
        del self._file[name]
        renumber_scene_episodes(self._file, self._meta)
        self._file.flush()

    def set_quality_status(self, name: str, status: str) -> None:
        """QA 재판정 (1차 큐레이션). 프레임·instruction 은 편집하지 않는다;
        지우려면 delete_episode (삭제 후 renumber), 끝만 자르려면 트림."""
        if status not in QUALITY_STATUSES:
            raise ValueError(f"잘못된 quality_status: {status!r} (허용: {QUALITY_STATUSES})")
        if name not in self._file or not EPISODE_GROUP_RE.match(name):
            raise KeyError(f"{name!r} not found in {self.path}")
        self._file[name].attrs["quality_status"] = status
        self._file[name].attrs["success"] = status == QUALITY_SUCCESS
        self._file.flush()

    # ----------------------------------------------------------------- 조회
    @property
    def num_episodes(self) -> int:
        return sum(1 for k in self._file.keys() if EPISODE_GROUP_RE.match(k))

    def list_episodes(self) -> list[dict]:
        """에피소드 요약을 번호순으로 -- 갤러리·slot 카운트가 쓰는 형태."""
        items = [
            _episode_summary(k, self._file[k])
            for k in self._file.keys()
            if EPISODE_GROUP_RE.match(k)
        ]
        items.sort(key=lambda d: d["episode_id"])
        return items

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "SceneWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ------------------------------------------------------------------- 읽기
def read_scene_metadata(path: Path) -> SceneMetadata:
    with h5py.File(path, "r") as f:
        return _read_metadata(f["metadata"])


def read_reference_image(path: Path) -> Optional[np.ndarray]:
    with h5py.File(path, "r") as f:
        ds = f["metadata"].get("reference_image")
        return None if ds is None else ds[...]


def _next_slot_idx(f: h5py.File, meta: h5py.Group, instruction_id: str) -> int:
    """slot(=instruction) 로컬 다음 E번호 = 같은 slot 의 살아 있는 에피소드
    uid E-부분 최대 + 1. 삭제 후에는 renumber_scene_episodes 가 E 를 0..k-1 로
    다시 채우므로 결과적으로 '그 slot 의 개수' 와 같다."""
    slot_idx = 0
    for k in f.keys():
        if not EPISODE_GROUP_RE.match(k):
            continue
        g = f[k]
        if str(g.attrs.get("instruction_id", "")) != instruction_id:
            continue
        m = re.search(r"-E(\d+)$", str(g.attrs.get("episode_uid", "")))
        if m:
            slot_idx = max(slot_idx, int(m.group(1)) + 1)
    return slot_idx




def renumber_scene_episodes(f: h5py.File, meta: h5py.Group) -> dict:
    """삭제로 생긴 빈자리를 메운다 -- legacy ``renumber_episodes`` 의 scene 판.

    - 그룹 이름 ``episode_NNN`` 을 현재 번호 오름차순으로 0..N-1 재부여
      (오름차순 처리라 임시 이름 없이 충돌 없음: k번째 에피소드의 현재 번호는
      항상 k 이상이다)
    - ``episode_id`` attr = 새 그룹 번호
    - slot 별로 현재 순서대로 ``slot_episode_idx``/``episode_uid`` 재부여
    - ``next_episode_idx`` = N
    - 편집 마커 ``edit_count`` 증가 (:func:`mark_scene_edited` -- uid 가
      재배정되므로 이 파일은 이후 resume 대상이 될 수 없다)
    빈자리가 없어도 마커는 올라간다 -- 이 함수가 불렸다는 것 자체가 삭제가
    있었다는 뜻이다 (호출자는 delete 경로뿐).

    돌려주는 것: **uid 가 바뀐 에피소드의** ``{옛 uid: 새 uid}``. 프록시 클립이
    uid 로 이름 붙어 있어서, 이 표가 있으면 씬 통째를 버리지 않고 파일 이름만
    옮길 수 있다 (2026-09-14 -- 에피소드 하나를 지워도 씬 전체를 다시 구워야
    했다).
    """
    names = sorted((k for k in f.keys() if EPISODE_GROUP_RE.match(k)),
                   key=lambda k: int(EPISODE_GROUP_RE.match(k).group(1)))
    for new_idx, name in enumerate(names):
        new_name = f"episode_{new_idx:03d}"
        if name != new_name:
            f.move(name, new_name)
        f[new_name].attrs["episode_id"] = new_idx
    meta.attrs["next_episode_idx"] = len(names)
    sid = str(meta.attrs.get("scene_id", ""))
    per_slot: dict = {}
    moved: dict = {}
    for i in range(len(names)):
        g = f[f"episode_{i:03d}"]
        iid = str(g.attrs.get("instruction_id", ""))
        e = per_slot.get(iid, 0)
        per_slot[iid] = e + 1
        g.attrs["slot_episode_idx"] = e
        if sid and iid:
            old = str(g.attrs.get("episode_uid", ""))
            new = episode_uid(sid, iid, e)
            g.attrs["episode_uid"] = new
            if old and old != new:
                moved[old] = new
    mark_scene_edited(meta)
    return moved


def delete_scene_episodes(path: Path, names: list) -> tuple:
    """세션이 파일을 쥐고 있지 않을 때 GUI 가 직접 쓰는 삭제 경로 (규칙은
    SceneWriter.delete_episode 와 동일: 삭제 후 renumber). 이름 하나라도 없으면
    아무것도 지우지 않고 KeyError.

    돌려주는 것: ``(지운 uid 목록, {옛 uid: 새 uid})`` -- 프록시 캐시를 씬
    통째로 버리지 않고 맞추는 데 쓴다 (proxy_clip.remap_scene_caches).
    """
    with h5py.File(path, "a") as f:
        meta = f["metadata"]
        missing = [n for n in names if n not in f or not EPISODE_GROUP_RE.match(n)]
        if missing:
            raise KeyError(", ".join(missing))
        deleted = [str(f[n].attrs.get("episode_uid", "")) for n in names]
        for n in names:
            del f[n]
        moved = renumber_scene_episodes(f, meta)
    return [u for u in deleted if u], moved


#: (해석된 경로) -> (stat 지문, 결과). list_scene_episodes 전용.
_EPISODES_CACHE: dict[str, tuple[tuple, list[dict]]] = {}


def list_scene_episodes(path: Path) -> list[dict]:
    """파일을 열지 않고 있는(writer 없는) 호출자용 에피소드 요약. attrs 만
    읽으므로 이미지 청크는 건드리지 않는다.

    (size, mtime_ns) 로 캐시한다. "두 개의 진실 금지"를 어기지 않는다 --
    파일이 바뀌면 지문이 달라져 무효가 되므로 정본은 여전히 파일이고,
    hub_upload_state 장부가 쓰는 것과 같은 판정이다. 파일 하나가 8~18GB 라
    비싼 건 읽는 바이트가 아니라 여는 횟수다: 데이터셋 16개를 훑는 데
    543ms 가 들고, 그것이 저장·연결마다 메인 스레드에서 일어났다
    (2026-09-04 실측).
    """
    path = Path(path)
    try:
        st = path.stat()
        key = str(path.resolve())
        fp = (st.st_size, st.st_mtime_ns)
        hit = _EPISODES_CACHE.get(key)
        if hit is not None and hit[0] == fp:
            return [dict(d) for d in hit[1]]   # 호출자가 고쳐도 캐시는 그대로
    except OSError:
        key = fp = None
    with h5py.File(path, "r") as f:
        items = [
            _episode_summary(k, f[k]) for k in f.keys() if EPISODE_GROUP_RE.match(k)
        ]
    items.sort(key=lambda d: d["episode_id"])
    if key is not None:
        # 읽는 사이에 또 바뀌었을 수 있다 -- 읽고 난 뒤의 stat 으로 기록해야
        # 낡은 결과를 새 지문에 붙이지 않는다.
        try:
            st2 = path.stat()
            _EPISODES_CACHE[key] = ((st2.st_size, st2.st_mtime_ns),
                                    [dict(d) for d in items])
        except OSError:
            pass
    return items


def empty_zones(layout: dict) -> list[tuple[int, int]]:
    """placements 에서 파생한 빈 존 목록 (행 우선 정렬).

    파생 함수이지 저장 필드가 아니다 -- 빈 존을 metadata 에 따로 적으면
    placements 와 어긋날 수 있는 두 번째 진실이 생긴다. legacy 사고(파일명 vs
    attrs)의 교훈 그대로: 같은 사실은 한 곳에만 적는다."""
    rows, cols = layout["grid"]
    occupied = {tuple(spec["zone"]) for spec in layout["placements"].values()}
    return [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in occupied]


def describe_scene(md: SceneMetadata) -> str:
    """metadata 에서 파생한 사람용 요약 -- 채운 존과 빈 존이 한눈에 보이는
    ASCII 격자 지도. QA 검사기(scripts/check/check_scene_file.py), GUI 의 scene
    표시, 다양성 추천의 제안 카드가 전부 이 하나의 렌더러를 쓴다 -- 어디서든
    같은 그림이 보이는데 저장은 placements 한 곳인 구조.

    소품 종류(category/color)는 저장돼 있지 않고 인벤토리에서 조회한다 --
    조회가 안 되는 ID(은퇴/테스트용)는 ID 만 보여준다."""
    try:
        from mstack.scene.props import props_by_id
        inv = props_by_id()
    except Exception:  # noqa: BLE001 -- 인벤토리가 없어도 요약은 나와야 한다
        inv = {}

    def _kind(oid: str) -> str:
        p = inv.get(oid)
        return f"({p.category}/{p.color})" if p else ""

    lines = [f"{md.scene_id} · station {md.station or '(미기록)'} · "
             f"{md.dataset_version} · created {md.created or '(미기록)'}"]
    if md.description:
        lines.append(f'"{md.description}"')
    lines.append("objects: " + ", ".join(f"{o}{_kind(o)}" for o in md.objects))

    rows, cols = md.layout["grid"]
    cell: dict[tuple[int, int], list[str]] = {}
    for oid, spec in md.layout["placements"].items():
        label = oid[4:] if oid.startswith("OBJ-") else oid
        cell.setdefault(tuple(spec["zone"]), []).append(label)
    width = max([13] + [len("+".join(v)) + 2 for v in cell.values()])

    def _row(cells: list[str]) -> str:
        return "│" + "│".join(s.center(width) for s in cells) + "│"

    def _rule(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * width for _ in range(cols)) + right

    lines.append(f"grid {rows}x{cols} (agentview, [0,0]=왼쪽 위)")
    lines.append(_rule("┌", "┬", "┐"))
    for r in range(rows):
        lines.append(_row(["+".join(cell.get((r, c), [])) or "·" for c in range(cols)]))
        lines.append(_rule("├", "┼", "┤") if r < rows - 1 else _rule("└", "┴", "┘"))
    empties = empty_zones(md.layout)
    lines.append("빈 존: " + (" ".join(f"({r},{c})" for r, c in empties) or "(없음)"))
    for a, rel, b in md.layout.get("relations", []):
        lines.append(f"관계: {a} {rel} {b}")
    return "\n".join(lines)


def count_by_slot(path: Path) -> dict[str, dict[str, int]]:
    """slot(=instruction_id)별 수집 현황: ``{instruction_id: {"total", "usable"}}``.
    usable 은 quality_status == "success" 만 센다 -- 계획 파일의 target 과
    비교하는 것은 이 값이다 ("collected 는 파일을 읽어 계산한다", §11)."""
    out: dict[str, dict[str, int]] = {}
    for ep in list_scene_episodes(path):
        slot = out.setdefault(ep["instruction_id"], {"total": 0, "usable": 0})
        slot["total"] += 1
        if ep["quality_status"] == QUALITY_SUCCESS:
            slot["usable"] += 1
    return out
