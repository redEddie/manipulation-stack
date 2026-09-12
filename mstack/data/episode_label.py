"""에피소드 한 줄 표기 -- 격자 타일과 목록이 **같은 글자**를 쓰게 하는 한 곳.

같은 에피소드가 갤러리 타일에는 ``I000-E003 ✓ 152f`` 로, 왼쪽 목록에는
``I000-E003`` + ``✓`` + ``152`` 로 나온다. 표기가 같아야 두 화면을 오갈 때
같은 것을 보고 있다는 확신이 서는데, 2026-09-12 감사 전까지는 포맷이 두
파일에 복사돼 있었다 (mstack/gui/clip_grid.py 와 dataset/ops.py). 실제로
갈라져 있기도 했다: 목록은 legacy 의 ``success`` 불리언을 판정으로 받아
주는데 타일은 안 받아서, 같은 에피소드가 한쪽은 ✓ 다른 쪽은 · 였다.

Qt 를 모르는 순수 함수다 -- 위젯 없이 시험된다.
"""

from __future__ import annotations

from mstack.config.quality import QUALITY_FAILED, QUALITY_SUCCESS

#: 판정 → 한 글자. 모르는 값(bad_data·retake 등)과 미판정은 같은 가운뎃점이다
#: -- 큐레이션 화면이 묻는 것은 "쓸 것인가" 하나라, 그 밖은 전부 "아직 아님".
_MARKS = {QUALITY_SUCCESS: "✓", QUALITY_FAILED: "✗"}


def quality_of(ep) -> str:
    """에피소드의 판정. scene-v1 은 ``quality_status``, legacy 는 ``success``
    불리언뿐이라 그것도 받아 준다. 둘 다 없으면 빈 문자열."""
    q = ep.get("quality_status")
    if q:
        return str(q)
    ok = ep.get("success")
    if ok is None:
        return ""
    return QUALITY_SUCCESS if ok else QUALITY_FAILED


def quality_mark(ep) -> str:
    return _MARKS.get(quality_of(ep), "·")


def episode_label(ep) -> str:
    """``I000-E003`` -- 지시문 안에서의 자리. E 번호는 uid 의 마지막 조각이라
    slot 로컬이다 (파일 전체 번호가 아니다)."""
    uid = ep.get("episode_uid", "")
    return f"{ep.get('instruction_id', '')}-{uid.rsplit('-', 1)[-1]}"


def episode_caption(ep, marked: bool = False) -> str:
    """``🗑 I000-E003 ✓ 152f`` -- 타일 밑에 붙는 한 줄.

    프레임 수를 늘 보여주는 이유는 큐레이션 대상 셋 중 하나가 "2~3틱만 찍힌
    것" 이라서다 -- 영상으로 찾을 것이 아니라 숫자로 바로 보여야 한다.
    """
    head = "🗑 " if marked else ""
    return (f"{head}{episode_label(ep)} {quality_mark(ep)} "
            f"{ep.get('num_samples', 0)}f")
