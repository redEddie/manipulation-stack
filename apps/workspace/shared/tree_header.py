"""Tree header whose first section expands or collapses every row at once.

The arrow is drawn where the tree draws its own branch arrows, so the header
reads as one more row of the same tree: pointing down when every group is open,
right otherwise. Clicking the first section toggles all groups (2026-09-17: the
Configure plan table had no way to fold 29 scenes but one by one).

The state is read from the tree at paint time rather than stored, so a refill
(clear + expandAll) cannot leave the arrow pointing the wrong way.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QPainter, QPalette, QPolygonF
from PyQt6.QtWidgets import QHeaderView, QTreeWidget


class ExpandAllHeader(QHeaderView):
    def __init__(self, tree: QTreeWidget) -> None:
        super().__init__(Qt.Orientation.Horizontal, tree)
        self._tree = tree
        self.setSectionsClickable(True)
        # Left-aligned like the rows, so "Scene" sits where scene ids start.
        self.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setToolTip("모두 펼치기 / 접기")
        tree.itemExpanded.connect(lambda _i: self.viewport().update())
        tree.itemCollapsed.connect(lambda _i: self.viewport().update())

    def _groups(self) -> list:
        t = self._tree
        return [t.topLevelItem(i) for i in range(t.topLevelItemCount())
                if t.topLevelItem(i).childCount()]

    def all_expanded(self) -> bool:
        groups = self._groups()
        return bool(groups) and all(g.isExpanded() for g in groups)

    def toggle(self) -> None:
        if self.all_expanded():
            self._tree.collapseAll()
        else:
            self._tree.expandAll()
        self.viewport().update()

    def label_indent(self) -> str:
        """Leading spaces that put the section label past the arrow, where the
        tree's own item text starts."""
        space = max(1, self.fontMetrics().horizontalAdvance(" "))
        return " " * (-(-self._tree.indentation() // space))

    def paintSection(self, painter, rect, index) -> None:  # noqa: N802 - Qt override
        painter.save()
        super().paintSection(painter, rect, index)
        painter.restore()
        if index != 0 or not self._groups():
            return
        # Painted by hand. The app sets a style sheet (GROUP_BOX_QSS), and under
        # QStyleSheetStyle PE_IndicatorBranch draws nothing outside the tree's
        # own row painting -- offscreen tests without the sheet showed the
        # arrow while the running GUI did not (2026-09-17).
        size = max(6, min(self._tree.indentation(), rect.height()) // 2 - 1)
        cx = rect.x() + self._tree.indentation() / 2
        cy = rect.y() + rect.height() / 2
        half = size / 2
        if self.all_expanded():                  # pointing down
            pts = [QPointF(cx - half, cy - half / 2), QPointF(cx + half, cy - half / 2),
                   QPointF(cx, cy + half / 2 + 1)]
        else:                                    # pointing right
            pts = [QPointF(cx - half / 2, cy - half), QPointF(cx - half / 2, cy + half),
                   QPointF(cx + half / 2 + 1, cy)]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._tree.palette().color(QPalette.ColorRole.WindowText))
        painter.drawPolygon(QPolygonF(pts))
        painter.restore()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.logicalIndexAt(event.position().toPoint()) == 0:
            self.toggle()
            return
        super().mousePressEvent(event)
