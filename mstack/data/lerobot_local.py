"""로컬 LeRobot 변환 결과가 **무엇을 담고 있는가** -- 네트워크 없이.

업로드 화면의 첫 질문은 "무엇이 변환됐나" 인데, 지금까지 그 답은 Hub 을
찌르는 버튼(이어붙이기·전체 처리) 안에만 있었다. 누르기 전에는 알 수 없고,
누르면 네트워크를 기다린다 (조작자, 2026-09-13: "upload 탭에서 변환된 게
무엇인지 먼저 알려주고").

여기서 읽는 것은 변환기가 써 둔 `meta/info.json` 하나다. LeRobot 포맷의
정본이고, 변환이 끝난 순간의 사실이라 세는 데 몇 ms 도 안 걸린다. Hub 과의
대조는 여전히 그 버튼들의 일이다 -- 이 함수는 **로컬만** 말한다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def local_lerobot_status(root) -> dict:
    """``{"exists", "episodes", "frames", "tasks", "at", "error"}``.

    ``at`` 은 info.json 의 수정 시각 문자열(없으면 ""). 폴더가 없거나 아직
    변환한 적이 없으면 ``exists=False`` 이고 나머지는 0 이다 -- 그것도 답이다
    ("아직 아무것도 변환 안 됨").
    """
    out = {"exists": False, "episodes": 0, "frames": 0, "tasks": 0,
           "at": "", "error": ""}
    if not root:
        return out
    info = Path(root).expanduser() / "meta" / "info.json"
    if not info.is_file():
        return out
    try:
        data = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["exists"] = True
    out["episodes"] = int(data.get("total_episodes") or 0)
    out["frames"] = int(data.get("total_frames") or 0)
    out["tasks"] = int(data.get("total_tasks") or 0)
    try:
        out["at"] = time.strftime("%m/%d %H:%M",
                                  time.localtime(info.stat().st_mtime))
    except OSError:
        pass
    return out
