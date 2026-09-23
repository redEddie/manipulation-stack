"""Pure text/stream helpers used by the collector GUIs.

Split out of the former mstack/gui/gui_widgets.py so subprocess log filtering and repo-id
validation can be imported without dragging in the full widget collection.
"""

from __future__ import annotations

import re
import time

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def repo_id_error(repo_id: str) -> "str | None":
    """Why `repo_id` is not a usable Hub id, or None if it is.

    Checked before anything is stored or run because the failure it prevents is
    slow and confusing: an id with a bad namespace passes every local step and
    dies at the very end with `403 ... rights to create a dataset under the
    namespace "r"`, after (in one real run) 15.6 minutes of repacking. And a
    typo that reaches Recents becomes the default for the automatic buttons,
    so the same failure repeats without anyone retyping it.
    """
    if not repo_id:
        return "Repo ID를 입력하세요."
    if "/" not in repo_id:
        return f"'{repo_id}' 에 네임스페이스가 없습니다. <조직 또는 사용자>/<이름> 형식이어야 합니다."
    if repo_id.count("/") > 1:
        return f"'{repo_id}' 에 '/' 가 너무 많습니다. <네임스페이스>/<이름> 하나뿐이어야 합니다."
    if not _REPO_ID_RE.match(repo_id):
        return f"'{repo_id}' 는 사용할 수 없는 형식입니다 (영문/숫자로 시작, 나머지는 영문·숫자·. _ - )."
    ns = repo_id.split("/")[0]
    if len(ns) < 2:
        return f"네임스페이스 '{ns}' 가 너무 짧습니다 — 오타로 보입니다."
    return None



_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\r")
_PROGRESS_RE = re.compile(r"\d+%\|")


#: 진행률을 말하는 세 가지 꼴. 자식 프로세스마다 다르게 찍는다 --
#: tqdm 은 "45%|", 공간 회수는 "진행  12.3%", 업로드는 "[3/27]",
#: 그리고 공간 회수의 MB 카운터는 "(120/480 MB)".
_FRAC_RES = (
    re.compile(r"(\d+(?:\.\d+)?)\s*%"),
    re.compile(r"[\[(](\d+)\s*/\s*(\d+)"),
)


def parse_progress_fraction(line: str, state: "dict | None" = None) -> "float | None":
    """진행률 줄에서 0~1 을 뽑는다. 못 읽으면 None.

    남은 시간을 말하려면 "얼마나 왔나" 가 필요한데, 그것을 아는 것은 자식
    프로세스뿐이고 그 앎은 stdout 한 줄로만 나온다 (2026-09-13). 꼴이 여럿이라
    한 곳에서 다 받는다 -- 못 읽으면 None 이고, 그러면 화면은 경과 시간만
    말한다. **거짓 추정은 하지 않는다.**

    ``state`` (프로세스별 dict) 를 주면 **작업 자신의 카운터를 우선**한다.
    LeRobot 변환은 에피소드마다 lerobot 이 짧은 tqdm 막대를 띄우는데, 그
    막대는 끝날 때마다 100% 라 상태바가 계속 100% 에 붙어 있었다 (2026-09-18).
    한 번이라도 ``[n/total]`` 꼴을 본 작업에서는 그 뒤로 tqdm 막대를 무시한다.
    """
    if state is not None and _FRAC_RES[1].search(line):
        state["saw_counter"] = True
    if (state is not None and state.get("saw_counter")
            and _PROGRESS_RE.search(line) and not _FRAC_RES[1].search(line)):
        return None
    m = _FRAC_RES[0].search(line)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1)) / 100.0))
        except ValueError:
            return None
    m = _FRAC_RES[1].search(line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            return max(0.0, min(1.0, done / total))
    return None


def is_progress_line(line: str) -> bool:
    """tqdm 진행률 줄인가. 로그에 쌓지 않고 한 줄을 갱신하는 데 쓴다."""
    return bool(_PROGRESS_RE.search(line))


def clean_stream_lines(data: str, state: dict, every_s: float = 3.0) -> list[str]:
    """Split subprocess output into log-worthy lines, de-spamming progress."""
    out = []
    for raw in _ANSI_RE.sub("\n", data).splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if _PROGRESS_RE.search(line):
            now = time.monotonic()
            # Always keep a finished bar; throttle the rest.
            done = line.lstrip().startswith("100%") or "100%|" in line
            if not done and now - state.get("t", 0.0) < every_s:
                continue
            if done and state.get("last_done") == line:
                continue
            state["t"] = now
            if done:
                state["last_done"] = line
        out.append(line)
    return out


