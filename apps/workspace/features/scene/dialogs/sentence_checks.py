"""Sentence checklist grouped by skill -- the grammar's candidates for one scene.

Shared by the scene recommendation dialog (step 2) and the instruction plan
editor, so a scene with no instructions yet can be filled by ticking sentences
instead of typing them (2026-09-17: S025 had none and only a free-text row).
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from apps.workspace.shared.badges import ClickableBadge
from mstack.gui.i18n import tr
from mstack.scene.skill_stats import rank_instructions


def build_sentence_checks(md, props: dict, counts, into, exclude=()) -> list:
    """문장 체크리스트를 **동작별로 나눠** into 에 만들고, 체크박스 전량을
    준다 (고르는 쪽 계약은 그대로 -- 평평한 목록이다).

    전에는 문장 서른 몇 개가 한 줄로 늘어서 있었다. 다 켜진 채로 시작하니
    "무엇을 뺄까" 를 서른 몇 개 사이에서 한 번에 판단해야 했고, 그러면
    빼야 할 것을 놓친다 (2026-09-07 사용자).

    동작 뱃지를 누르면 그 동작의 문장만 보인다. 한 번에 보는 수가 서넛으로
    줄고, 뱃지에 "고른 수/전체" 가 적혀 있어 안 보는 동안에도 그쪽이 몇
    개인지 안다 -- 좁혀서 보되 전체를 잃지 않는 것이 요점이다.
    """
    skip = set(exclude)
    ranked = [r for r in rank_instructions(md, props, counts or {})
              if r[0] not in skip]
    checks: list[QCheckBox] = []
    if not ranked:
        note = QLabel(tr("(문법상 생성 가능한 문장이 없음)"))
        note.setStyleSheet("color:#666;")
        into.addWidget(note)
        return checks

    by_skill: dict[str, list] = {}
    for s, sk, n in ranked:
        by_skill.setdefault(sk, []).append((s, n))

    badges: dict[str, ClickableBadge] = {}
    pages: dict[str, QWidget] = {}
    summary = QLabel("")
    summary.setStyleSheet("color:#333;")

    def _refresh() -> None:
        for sk, page in pages.items():
            on = [cb for cb in page.findChildren(QCheckBox)]
            badges[sk].set_note(
                f"{sum(1 for cb in on if cb.isChecked())}/{len(on)}")
        summary.setText(tr("고른 문장 {n} / {m}").format(
            n=sum(1 for cb in checks if cb.isChecked()), m=len(checks)))

    def _show(sk: str) -> None:
        for key, page in pages.items():
            page.setVisible(key == sk)
            badges[key].set_selected(key == sk)

    strip = QHBoxLayout()
    strip.setSpacing(6)
    into.addLayout(strip)

    for sk in sorted(by_skill):
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 4, 0, 0)
        col.setSpacing(2)
        for sent, n in by_skill[sk]:
            cb = QCheckBox(sent)
            cb.setChecked(True)
            # **끄지 않는다.** 전에는 지시문 파일 경로가 없으면 회색으로
            # 잠갔는데, 그러면 "빼야 할 문장" 을 뺄 수가 없다 (2026-09-07
            # 사용자: "체크가 선택해제가 안 되서 불가능한 걸 빼기가
            # 불가능한데요?"). 고르는 것은 언제나 되어야 하고, 경로가
            # 없어서 못 하는 것은 **등록**이다 -- 그건 아래 등록
            # 체크박스가 이유와 함께 말한다.
            cb.setToolTip(tr("스킬 {sk} · 지금까지 {n} 에피소드 수집")
                          .format(sk=sk, n=n))
            cb.toggled.connect(lambda _v: _refresh())
            checks.append(cb)
            col.addWidget(cb)
            # 누적 수집량은 문장 아래 한 줄로 -- 체크박스 문장에 붙이면
            # 문장이 길어져 읽기 어렵다. 스킬은 이제 뱃지가 말한다.
            cnt = QLabel(tr("{n}개 수집").format(n=n))
            cnt.setContentsMargins(28, 0, 0, 4)
            # #888 은 밝은 바탕에서 대비 미달(2.9:1)이라 #666 으로.
            cnt.setStyleSheet("color:#666;")
            col.addWidget(cnt)
        pages[sk] = page
        into.addWidget(page)

        b = ClickableBadge(sk, "")
        b.clicked.connect(_show)
        badges[sk] = b
        strip.addWidget(b)
    strip.addStretch(1)

    row = QHBoxLayout()
    row.addWidget(summary, 1)
    for label, want in ((tr("이 동작 전부"), True), (tr("전부 해제"), False)):
        btn = QPushButton(label)
        btn.clicked.connect(
            lambda _c=False, w=want: _set_visible_checks(pages, w))
        row.addWidget(btn)
    into.addLayout(row)

    _show(sorted(by_skill)[0])
    _refresh()
    return checks


def _set_visible_checks(pages: dict, on: bool) -> None:
    """지금 보이는 동작의 체크박스만 켜거나 끈다 -- 안 보는 것을 건드리면
    무엇이 바뀌었는지 알 수 없다."""
    for page in pages.values():
        if page.isVisible():
            for cb in page.findChildren(QCheckBox):
                cb.setChecked(on)
