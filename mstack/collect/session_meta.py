"""파일에 한 번 적히는 **세션 메타** -- 부하 모델·리셋 자세·판번호.

세 값 모두 ``dataset_schema`` 가 metadata attrs 로 요구한다 (knu-1.2.0 /
1.2.1 / 1.2.2). 그래서 이 값을 모르면 ``SceneWriter`` 가 도장을 내려 찍고,
그 결과가 2.0.0 에서는 **못 쓰는 파일**이 된다: MAJOR 가 다른 scene 에는
이어찍을 수 없으므로 (scene_format._resume_version), 1.x 로 내려 찍힌 빈
scene 에는 영원히 녹화할 수 없다. 2026-09-23 에 S024 가 그렇게 됐다.

그래서 이것을 읽는 자리가 **둘**이고, 둘 다 같은 값을 얻어야 한다:

* 수집 워커 -- Connect 할 때 (``collect/worker.py``)
* Scene 구성기 -- [✚ 새 Scene 만들기] 를 누를 때, 로봇에 붙기 전

셋 중 **부하 모델만** 로봇 노드를 거친다. 리셋 자세는 station 설정에 있고
수집기 커밋은 git 에 있으므로, 구성기도 노드 없이 그 둘은 채울 수 있다.

``mstack/collect/`` 에 있는 이유는 레이어 규칙이다 -- 리셋 자세를 풀려면
``mstack/robots/`` 의 표가 필요한데 ``scene/`` 과 ``data/`` 는 그것을
import 할 수 없다. ``collect/`` 는 구체적인 하드웨어를 알아도 되는 층이다.
"""
from __future__ import annotations

from typing import Optional

from mstack.data.provenance import collector_commit


def reset_pose_from_station() -> Optional[dict]:
    """이번 세션의 리셋 자세 -- 별칭과 7관절 절대값. 로봇이 필요 없다.

    station 설정에서 이름을 읽고 ``FR3_RESET_POSES`` 에서 값을 푼다. 둘을
    함께 적는 이유는 ``dataset_schema.META_RESET_POSE`` 주석에 있다 -- 이름만
    적으면 그 표가 바뀔 때 옛 파일을 잘못 읽는다.
    """
    try:
        from mstack.config.station import load_station
        from mstack.robots.franka_fr3 import FR3_RESET_POSES

        name = str(load_station().robot.reset_pose or "")
        q = FR3_RESET_POSES.get(name)
        if not name or q is None:
            return None
        return {"name": name, "qpos": [float(x) for x in q]}
    except Exception:  # noqa: BLE001 -- 못 읽으면 그 버전을 안 찍을 뿐이다
        return None


def payload_from_node(timeout_ms: int = 3000) -> dict:
    """로봇 노드의 부하 모델. 노드가 없거나 안 주면 빈 dict.

    **0 을 채워 넣지 않는다.** 0 은 "부하가 없었다" 로 읽혀 없느니만 못하다
    (``SceneMetadata.payload_mass`` 주석). 비어 있으면 그 버전으로 못 올라가
    눈에 띈다 -- 그것이 이 값의 유일한 방어선이다 (#47).
    """
    try:
        from mstack.comm.zmq_core.robot_node import probe_payload
        from mstack.config.station import load_station

        node = load_station().node
        info = probe_payload(node.host, int(node.port), timeout_ms=timeout_ms)
    except Exception:  # noqa: BLE001 -- 노드가 없는 것은 정상 경로다
        return {}
    if not info or info.get("mass") is None:
        return {}
    return info


def offline_provenance() -> dict:
    """로봇 없이 아는 판번호 -- 수집기 커밋뿐.

    ``pylibfranka``·``fr3_system`` 은 노드 쪽 인터프리터에만 있어서 여기서는
    못 읽는다. 스키마가 **필수로 요구하는 것은 ``provenance_source`` 하나**
    이고 나머지는 있으면 적는 값이라, 커밋만으로도 요구는 채워진다
    (``dataset_schema._PROVENANCE_REQUIRED``).
    """
    commit = collector_commit()
    return {"collector_commit": commit} if commit else {}


def gripper_from_station() -> dict:
    """달려 있는 그리퍼. 로봇이 필요 없다 -- station 설정에 적혀 있다.

    그리퍼 열이 0~1 정규화값이라 (실측 0.0026~0.9624), 최대 벌림을 모르면
    미터로 되돌릴 수 없다. 그래서 리셋 자세와 같이 파일에 적는다.
    """
    try:
        from mstack.config.station import load_station

        r = load_station().robot
        return {"name": r.gripper, "max_width": r.gripper_max_width}
    except Exception:  # noqa: BLE001 -- 못 읽으면 그 버전을 안 찍을 뿐이다
        return {}


def apply_to_metadata(meta, payload: Optional[dict], reset: Optional[dict],
                      prov: Optional[dict], gripper: Optional[dict] = None) -> None:
    """읽어 온 값을 ``SceneMetadata`` 에 싣는다 (새 scene 을 만들기 직전).

    워커와 구성기가 같은 방식으로 실어야 한다 -- 한쪽만 ``provenance_source``
    를 빼먹으면 그쪽에서 만든 파일만 조용히 낮은 버전으로 찍힌다.
    """
    if payload:
        meta.payload_mass = float(payload["mass"])
        meta.payload_com = list(payload.get("com") or [])
    if reset:
        meta.reset_pose = reset["name"]
        meta.reset_qpos = list(reset["qpos"])
    gripper = gripper if gripper is not None else gripper_from_station()
    if gripper:
        meta.gripper = gripper["name"]
        meta.gripper_max_width = float(gripper["max_width"])
    prov = prov or {}
    meta.collector_commit = prov.get("collector_commit") or None
    meta.pylibfranka_version = prov.get("pylibfranka") or None
    meta.fr3_system_version = prov.get("fr3_system") or None
    meta.provenance_source = "live" if prov else None
