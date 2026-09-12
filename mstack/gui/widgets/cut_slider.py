"""재생 위치 슬라이더 + 잘릴 지점을 그리는 빨간 선."""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QSlider, QStyle, QStyleOptionSlider


#: 잘려나갈 구간의 색. 플롯의 빨간 음영(plot_widgets.SeriesPlot.set_cut)과
#: 같은 색이라야 "같은 것을 가리킨다"가 읽힌다.
CUT_COLOR = "#c0392b"


class CutSlider(QSlider):
    """끝 다듬기용 재생바.

    Trim 탭에서 "어디까지 자를지"를 말해 주던 것은 위치 라벨의 글자뿐이었다
    (" ← 잘린 뒤 마지막"). 글자는 **그 프레임에 정확히 서 있을 때만** 나오고,
    그래서 자를 양을 −5 로 바꿔도 재생바에는 아무 변화가 없었다 (조작자,
    2026-09-12: "-1, -5처럼 어디까지 자를지 표시하는게 없네, 아예 빨간
    구분선이 있도록 하던지?").

    여기서는 그것을 **그림**으로 그린다: 잘릴 지점에 빨간 세로선, 그 뒤로
    사라질 구간에 옅은 빨간 띠. 눈금이 아니라 홈(groove) 위에 그리므로
    슬라이더의 길이나 동작에는 영향이 없다.
    """

    def __init__(self, orientation=Qt.Orientation.Horizontal) -> None:
        super().__init__(orientation)
        self._cut: int | None = None

    def set_cut(self, cut: int | None) -> None:
        """``cut`` = 마지막으로 **남는** 프레임의 인덱스. None 이면 안 그린다."""
        if cut != self._cut:
            self._cut = cut
            self.update()

    def cut(self) -> int | None:
        """지금 그려진 잘림 지점. 화면 코드는 set_cut 만 쓰고, 이 getter 의
        유일한 소비자는 계약 테스트다 (test_trim_controls)."""
        return self._cut

    def _x_for(self, value: int) -> float:
        """값이 놓이는 화면 x -- 손잡이 중심이 가는 자리."""
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()
        groove = style.subControlRect(QStyle.ComplexControl.CC_Slider, opt,
                                      QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(QStyle.ComplexControl.CC_Slider, opt,
                                      QStyle.SubControl.SC_SliderHandle, self)
        span = groove.width() - handle.width()
        lo, hi = self.minimum(), self.maximum()
        frac = 0.0 if hi <= lo else (value - lo) / (hi - lo)
        return groove.x() + handle.width() / 2.0 + span * frac

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if self._cut is None or self.maximum() <= self.minimum():
            return
        x = self._x_for(max(self.minimum(), min(self.maximum(), self._cut)))
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = self.height()
        band = QColor(CUT_COLOR)
        band.setAlpha(60)
        p.fillRect(QRectF(x, 2, max(0.0, self.width() - 2 - x), h - 4), band)
        p.fillRect(QRectF(x - 1, 1, 2, h - 2), QColor(CUT_COLOR))
