"""이 파일을 **무엇이 만들었는가** -- 자동으로 읽히는 것만.

데이터는 물리 셋업(station·payload·reset 자세)을 적어 왔지만 그것을 돌린
소프트웨어는 어디에도 없었다. 정책이 이상하게 움직일 때 첫 질문이 "이 데이터는
어느 컨트롤러, 어느 수집기 코드에서 나왔나" 인데 그 답이 사람 기억뿐이었다
(조작자, 2026-09-13).

여기 있는 것은 **읽을 수 있는 것만**이다:

* 수집기 커밋 -- 이 저장소의 git HEAD. 제어 상수가 전부 저장소 안에 있으므로
  이 한 줄이면 그때의 v_max·저크 클램프·가드 임계값을 정확히 되짚을 수 있다.
* pylibfranka / FR3 시스템 이미지 -- 로봇 쪽(`FrankaFR3.versions`)에서 온다.
  GUI 인터프리터에는 pylibfranka 가 없고 로봇 IP 도 노드가 쥐고 있어서,
  payload 와 같은 길(ZMQ `versions`)로 받아 온다.

**못 읽으면 적지 않는다.** "?" 나 0 을 적으면 측정한 것처럼 읽힌다 --
knu-1.1.0 이 모든 프레임 0 인 힘 필드를 적었다가 그 값을 아무도 믿을 수 없게
된 것이 이 규칙의 출처다.
"""

from __future__ import annotations

import json
import ssl
import subprocess
import urllib.request
from pathlib import Path

#: 저장소 뿌리 (이 파일 기준 mstack/data/ -> 두 칸 위).
REPO_ROOT = Path(__file__).resolve().parents[2]


def collector_commit(repo_root=None, timeout: float = 3.0) -> str:
    """수집기 저장소의 git 커밋. 못 읽으면 빈 문자열.

    작업 트리가 더러우면 ``-dirty`` 를 붙인다 -- 커밋만 적으면 "그 커밋을
    돌렸다" 가 되는데 실제로는 고치다 만 코드였을 수 있다. 재현하려는 사람이
    그 차이를 알아야 한다.
    """
    root = Path(repo_root or REPO_ROOT)
    def _git(*args) -> str:
        out = subprocess.run(("git", "-C", str(root), *args),
                             capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else ""
    try:
        sha = _git("rev-parse", "--short=12", "HEAD")
        if not sha:
            return ""
        dirty = _git("status", "--porcelain")
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return ""


def robot_versions(timeout: float = 2.0) -> dict:
    """**지금 이 리그의** 판번호. 로봇 세션도 노드도 필요 없다.

    수집 세션을 거쳐야만 알 수 있는 값이 아니다 -- 둘 다 지금 바로 읽힌다:

    * FR3 시스템 이미지 -- 로봇 Desk 의 ``GET /admin/api/system-version``.
      HTTPS 한 번이라 GUI 인터프리터가 직접 친다 (FCI 도 pylibfranka 도
      필요 없다). 로봇이 꺼져 있으면 그 항목이 빠진다.
    * pylibfranka -- 노드 venv 의 인터프리터에게 물어본다. GUI 쪽
      (lerobot-venv)에는 그 패키지가 없지만, 어느 파이썬이 노드를 띄우는지는
      station 설정이 알고 있다 (``node.python``).

    닥터가 옛 파일을 채울 때 쓰는 값이다. **그때 읽은 값이 아니므로**
    부르는 쪽이 ``backfilled`` 로 표시해야 한다 -- 그 구분은 여기서 하지 않고
    fill_and_raise 가 한다.
    """
    out: dict = {}
    try:
        from mstack.config.station import load_station

        cfg = load_station()
    except Exception:  # noqa: BLE001
        return out
    out.update(_desk_version(getattr(cfg.robot, "ip", "") or "", timeout))
    ver = _node_pylibfranka(cfg.node.python_path, timeout)
    if ver:
        out["pylibfranka_version"] = ver
    return out


def _desk_version(ip: str, timeout: float) -> dict:
    """Desk 가 말하는 시스템 이미지. 못 읽으면 빈 dict.

    자체 서명 인증서라 검증을 끈다 (사설망의 로봇 한 대다). 실측 응답
    (2026-09-13, 172.16.0.2)::

        "5.10.0\nec764230...\n340b9610...\n"
    """
    if not ip:
        return {}
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(f"https://{ip}/admin/api/system-version",
                                    timeout=timeout, context=ctx) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 -- 로봇이 꺼져 있어도 나머지는 읽는다
        return {}
    # 응답은 **JSON 문자열 하나**다 -- 줄바꿈이 진짜 개행이 아니라 ``\n``
    # 두 글자로 들어 있다 (실측 2026-09-14: od 로 확인). json 으로 풀어야
    # 줄이 나뉜다; 못 풀면 원문 그대로 다룬다.
    try:
        text = json.loads(body)
        if not isinstance(text, str):
            text = body
    except ValueError:
        text = body
    parts = [x.strip() for x in text.strip().strip('"').splitlines() if x.strip()]
    if not parts:
        return {}
    out = {"fr3_system_version": parts[0]}
    if len(parts) > 1:
        out["fr3_system_build"] = " ".join(parts[1:])
    return out


def _node_pylibfranka(python_path: str, timeout: float) -> str:
    """노드 venv 의 pylibfranka 버전. 못 읽으면 빈 문자열."""
    if not python_path or not Path(python_path).exists():
        return ""
    try:
        out = subprocess.run(
            (python_path, "-c",
             "import pylibfranka,json;print(json.dumps(getattr(pylibfranka,'__version__','')))"),
            capture_output=True, text=True, timeout=max(timeout, 5.0))
    except (OSError, subprocess.SubprocessError):
        return ""
    if out.returncode != 0:
        return ""
    try:
        return str(json.loads(out.stdout.strip()) or "")
    except ValueError:
        return ""
