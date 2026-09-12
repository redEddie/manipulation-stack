"""오래 걸리는 일이 도는 동안 머리줄에 한 줄로 말하는 표시.

`features/collection/header.py` 에 있던 것을 옮겼다 (2026-09-12). 라벨
자체는 그 머리줄이 만들지만, **부르는 쪽은 기능 셋**이다 (수집·갤러리·
분석). 한 기능의 파일을 다른 기능이 import 하기 시작하면 폴더 이름이 더는
의존 방향을 말해 주지 못한다 -- 여럿이 함께 쓰는 헬퍼는 shared 에 둔다
(CLAUDE.md 의 apps/workspace 배치 규칙).
"""

from __future__ import annotations


def set_busy(win, text: str = "") -> None:
    """머리줄에 "⏳ <하는 일>" 을 띄우거나(텍스트 있음) 지운다(빈 문자열).

    같은 UI 스레드에서 도는 일이 많아(분석 스캔 등) 화면이 갱신될 틈이
    없으므로, 켤 때만 한 번 ``processEvents`` 로 실제로 그린다. 끌 때는
    부르지 않는다 -- 끄는 쪽은 대개 일이 끝난 직후라 곧 다시 그려진다.
    """
    lab = getattr(win, "hud_busy", None)
    if lab is None:
        return
    try:
        lab.setText(f"⏳ {text}" if text else "")
        lab.setVisible(bool(text))
        if text:
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
    except RuntimeError:      # 창이 닫히는 중이면 C++ 쪽이 이미 없다
        pass
