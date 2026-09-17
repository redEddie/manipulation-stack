"""Tree header whose first section expands or collapses every row at once.

The arrow is drawn where the tree draws its own branch arrows, so the header
reads as one more row of the same tree: pointing down when every group is open,
right otherwise. Clicking the first section toggles all groups (2026-09-17: the
Configure plan table had no way to fold 29 scenes but one by one).

The state is read from the tree at paint time rather than stored, so a refill
(clear + expandAll) cannot leave the arrow pointing the wrong way.
"""

from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtWidgets import QHeaderView, QStyle, QStyleOption, QTreeWidget


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
        # Drawn as the tree draws its rows' arrows -- the tree's palette, style
        # and widget. With the header's own, Fusion painted nothing (2026-09-17).
        opt = QStyleOption()
        opt.initFrom(self._tree)
        opt.rect = QRect(rect.x(), rect.y(), self._tree.indentation(), rect.height())
        opt.state = (QStyle.StateFlag.State_Children | QStyle.StateFlag.State_Item
                     | QStyle.StateFlag.State_Enabled)
        if self.all_expanded():
            opt.state |= QStyle.StateFlag.State_Open
        self._tree.style().drawPrimitive(
            QStyle.PrimitiveElement.PE_IndicatorBranch, opt, painter, self._tree)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.logicalIndexAt(event.position().toPoint()) == 0:
            self.toggle()
            return
        super().mousePressEvent(event)
