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

import subprocess
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
