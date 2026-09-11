"""프록시 클립 여러 개를 한 화면에서 동시에 재생하는 격자 -- 큐레이션의 본체.

**무엇을 잡으려는 화면인가** (조작자, 2026-09-10): 지시문이 바뀌었는데 이전
명령으로 찍힌 것, 녹화를 너무 늦게 끝낸 것. (2~3틱짜리는 영상이 아니라
``num_samples`` 로 거른다 -- 여기서 다루지 않는다.)

**시작만 맞추고 시간은 정규화하지 않는다.** 이것이 이 화면의 핵심 규칙이다.
진행률을 0~1 로 정규화해 나란히 돌리면 모든 타일이 같은 순간에 끝나서
**"너무 늦게 끝낸 것" 이 안 보인다** -- 길이 자체가 신호인데 그것을 지우는
설계다. 그래서 다 같이 0프레임에서 출발하고, 각자 자기 실시간으로 가고,
먼저 끝난 타일은 **마지막 프레임에서 멈춘다.** 멈춘 것들 사이에서 혼자
계속 움직이는 타일이 곧 답이다.

끝난 타일이 되감아 다시 돌면 그 신호가 사라지므로 되감지 않는다. **전부**
끝나고 ``REST_TICKS`` 만큼 쉰 뒤에야 다 같이 되감는다.

**왜 원본이 아니라 프록시인가.** 원본은 에피소드 하나가 480x640 카메라 2대로
280 MB 이고 gzip 청크가 19프레임 깊이라, 12개를 나란히 돌리는 일이 불가능하다
(``mstack.data.proxy_clip`` 참고). 프록시는 12타일 동시 스트리밍이 틱당
2.4 ms -- 20 Hz 예산 50 ms 의 4.8% 다.

프록시가 없는 에피소드는 빈 타일로 두고 그렇다고 말한다. 큐레이션 도중에
클립을 굽기 시작하면 화면이 멎는다 -- 굽는 것은 툴바의 별도 작업이다.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mstack.data.proxy_clip import PROXY_DIR, proxy_path

#: 기본 격자 (열 x 행). 조작자 요구: 최소 10개가 한 눈에 (2026-09-10).
DEFAULT_COLS, DEFAULT_ROWS = 4, 3

#: 전부 끝난 뒤 되감기까지 쉬는 틱 수. 20 Hz 기준 1초 --
#: "다 끝났다" 를 눈이 알아차릴 틈이다.
REST_TICKS = 20

#: 선택된 타일 테두리.
_SEL = "#3498db"
#: 삭제 표시된 타일 테두리. 선택(파랑)과 다른 뜻이라 색을 나눈다 --
#: 선택은 "지금 보고 있다", 표시는 "지울 것이다" 이고 둘은 겹칠 수 있다.
_MARK = "#c0392b"
_DONE_DIM = 0.45


class _Tile(QFrame):
    """클립 하나. 자기 캡처와 진행 상태만 갖는다."""

    clicked = pyqtSignal(object, object)   # (tile, modifiers)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(2)
        self.ep = None
        self.cap = None
        self.n_frames = 0
        self.i = 0
        self.done = True
        self.selected = False
        self.marked = False
        self._last = None
        col = QVBoxLayout(self)
        col.setContentsMargins(2, 2, 2, 2)
        col.setSpacing(2)
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(QSize(120, 90))
        self.view.setSizePolicy(QSizePolicy.Policy.Ignored,
                                QSizePolicy.Policy.Ignored)
        self.view.setStyleSheet("background:#111;")
        col.addWidget(self.view, 1)
        self.caption = QLabel("")
        self.caption.setWordWrap(False)
        self.caption.setStyleSheet("color:#bbb; font-size:11px;")
        col.addWidget(self.caption)
        self._restyle()

    # ------------------------------------------------------------------ 상태
    def _restyle(self) -> None:
        if self.ep is None:
            # 빈 칸에는 테두리도 없다. 테두리가 있으면 "여기 뭔가 있는데
            # 안 움직인다" 로 읽힌다 -- 없는 것은 없어 보여야 한다.
            self.setStyleSheet("QFrame { border:none; }")
            self.caption.setText("")
            return
        # 표시가 선택을 이긴다 -- 둘 다일 때 알아야 하는 것은 "지울 것" 쪽이다.
        border = _MARK if self.marked else (_SEL if self.selected else "#333")
        width = 3 if self.marked else 2
        self.setStyleSheet(f"QFrame {{ border:{width}px solid {border}; }}")
        self.caption.setText(self._caption_text())

    def set_selected(self, on: bool) -> None:
        self.selected = bool(on)
        self._restyle()

    def set_marked(self, on: bool) -> None:
        self.marked = bool(on)
        self._restyle()

    def mousePressEvent(self, ev) -> None:  # noqa: N802 (Qt)
        self.clicked.emit(self, ev.modifiers())

    # ------------------------------------------------------------------ 클립
    def load(self, ep, camera: str, proxy_dir=None) -> None:
        """``ep`` 는 ``list_scene_episodes`` 가 주는 dict 그대로. None 이면 빈 타일.

        키를 번역하지 않는다 -- 갤러리·트리·통계가 모두 그 모양을 쓰는데
        여기만 다른 이름을 요구하면 사이에 변환 계층이 하나 생기고, 그 계층이
        새 필드가 생길 때마다 낡는다.
        """
        self.release()
        self.ep = ep
        self.i = 0
        self._last = None
        if ep is None:
            # **clear() 여야 한다.** setText("") 는 픽스맵을 지우지 않아서
            # 빈 칸이 직전 화면의 프레임을 그대로 들고 있었다 (2026-09-11
            # 조작자 보고). 그러면 "에피소드가 없는 칸" 과 "짧아서 금방 멈춘
            # 에피소드" 가 화면에서 똑같아 보인다 -- 둘 다 안 움직이는
            # 그림이라서다. 3프레임짜리를 찾는 것이 이 화면의 목적 중 하나인데
            # 그것이 빈 칸에 묻힌다.
            self.view.clear()
            self.done = True
            self._restyle()
            return
        uid = ep.get("episode_uid", "")
        p = proxy_path(uid, camera, proxy_dir or PROXY_DIR)
        if not uid or not Path(p).exists():
            self.view.setText("no proxy")
            self.view.setStyleSheet("background:#111; color:#666;")
            self.caption.setText(self._caption_text())
            self.done = True
            return
        import cv2

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            cap.release()
            self.view.setText("unreadable")
            self.done = True
            return
        self.cap = cap
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.view.setStyleSheet("background:#111;")
        self.done = False
        self.caption.setText(self._caption_text())
        self.step()          # 첫 프레임을 바로 보여준다 -- 정지 상태에서도 보인다

    def _caption_text(self) -> str:
        """``I000-E003 ✓ 152f`` -- 갤러리 목록과 같은 표기.

        E번호는 slot 로컬(uid 의 마지막 조각)이다. 프레임 수를 늘 보여주는
        이유는 큐레이션 대상 셋 중 하나가 "2~3틱만 찍힌 것" 이라서다 --
        영상으로 찾을 것이 아니라 여기 숫자로 바로 보여야 한다.
        """
        ep = self.ep
        if ep is None:
            return ""
        uid = ep.get("episode_uid", "")
        mark = {"success": "✓", "failed": "✗"}.get(
            ep.get("quality_status", ""), "·")
        head = "🗑 " if self.marked else ""
        return (f"{head}{ep.get('instruction_id', '')}-{uid.rsplit('-', 1)[-1]} "
                f"{mark} {ep.get('num_samples', 0)}f")

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.n_frames = 0
        self.done = True
        self._last = None

    def rewind(self) -> None:
        if self.cap is None:
            return
        import cv2

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.i = 0
        self.done = False

    def step(self) -> bool:
        """한 프레임 나아간다. 더 없으면 False 를 돌려주고 마지막 화면에 머문다."""
        if self.cap is None or self.done:
            return False
        ok, frame = self.cap.read()
        if not ok:
            self.done = True
            self._dim()
            self._paint(smooth=True)   # 멈춘 타일은 곱게 -- 여기서부터 들여다본다
            return False
        self.i += 1
        h, w = frame.shape[:2]
        # cv2 는 BGR 이다. QImage 에 BGR888 로 넘겨 변환을 아낀다.
        img = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        self._last = QPixmap.fromImage(img.copy())
        self._paint(smooth=False)
        return True

    def _paint(self, smooth: bool = True) -> None:
        """마지막 프레임을 타일 크기에 맞춰 그린다.

        **재생 중에는 Fast, 멈추면 Smooth.** 실측(12타일 200틱)으로 Smooth 가
        틱당 11.0 ms, Fast 가 4.9 ms, 스케일을 아예 안 하면 2.3 ms 다 -- 20 Hz
        예산 50 ms 에서 Smooth 는 22% 를 먹는다. 움직이는 동안에는 계단이
        눈에 안 띄고, 멈춰서 들여다볼 때가 화질이 필요한 때다.
        """
        if self._last is None:
            return
        mode = (Qt.TransformationMode.SmoothTransformation if smooth
                else Qt.TransformationMode.FastTransformation)
        self.view.setPixmap(self._last.scaled(
            self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, mode))

    def repaint_smooth(self) -> None:
        """멈춘 상태에서 다시 곱게 그린다."""
        self._paint(smooth=True)

    def _dim(self) -> None:
        """끝난 타일을 흐리게 -- 아직 도는 것이 눈에 띄어야 한다."""
        self.caption.setStyleSheet("color:#666; font-size:11px;")

    def undim(self) -> None:
        self.caption.setStyleSheet("color:#bbb; font-size:11px;")

    def resizeEvent(self, ev):  # noqa: N802 (Qt)
        super().resizeEvent(ev)
        self._paint(smooth=True)


class ClipGrid(QWidget):
    """타일 격자 + 한 개의 타이머. 페이지 단위로 클립을 갈아 끼운다."""

    #: 선택이 바뀔 때 -- 선택된 에피소드 dict 목록
    selection_changed = pyqtSignal(list)
    #: 타일을 더블클릭했을 때 (크게 보기)
    activated = pyqtSignal(object)

    def __init__(self, cols: int = DEFAULT_COLS, rows: int = DEFAULT_ROWS,
                 parent=None) -> None:
        super().__init__(parent)
        self.cols, self.rows = cols, rows
        self.camera = "agentview_rgb"
        #: 이 격자가 볼 프록시 디렉터리. 데이터셋마다 다르다 -- uid 는 데이터셋
        #: 안에서만 유일해서, 한 곳에 섞으면 다른 데이터셋의 영상이 나온다
        #: (mstack.data.proxy_clip.dataset_tag 참고). 창이 꽂는다.
        self.proxy_dir = None
        self.episodes: list = []
        self.page = 0
        self._rest = 0
        #: ``ep -> bool`` -- 이 에피소드가 삭제 표시되었나. 창이 꽂는다.
        #: 격자는 장바구니를 모른다 (그래야 화면 없이 시험된다).
        self.is_marked = None
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        self.tiles: list = []
        for r in range(rows):
            for c in range(cols):
                t = _Tile(self)
                t.clicked.connect(self._on_tile_clicked)
                grid.addWidget(t, r, c)
                self.tiles.append(t)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ 내용
    @property
    def per_page(self) -> int:
        return self.cols * self.rows

    @property
    def n_pages(self) -> int:
        if not self.episodes:
            return 1
        return (len(self.episodes) + self.per_page - 1) // self.per_page

    def set_episodes(self, episodes, camera: str = None) -> None:
        if camera:
            self.camera = camera
        self.episodes = list(episodes or [])
        self.page = 0
        self._load_page()

    def set_camera(self, camera: str) -> None:
        self.camera = camera
        self._load_page()

    def set_page(self, page: int) -> None:
        self.page = max(0, min(page, self.n_pages - 1))
        self._load_page()

    def _load_page(self) -> None:
        self._load_page_quiet()
        self._emit_selection()

    def refresh_marks(self) -> None:
        """표시 상태를 다시 물어 화면에 반영한다. 장바구니가 바뀌면 부른다."""
        if self.is_marked is None:
            return
        for t in self.tiles:
            t.set_marked(t.ep is not None and self.is_marked(t.ep))

    # ------------------------------------------------------------------ 재생
    def start(self, fps: float = 20.0) -> None:
        self.timer.start(max(1, int(1000 / max(fps, 1))))

    def stop(self) -> None:
        self.timer.stop()
        for t in self.tiles:
            t.repaint_smooth()

    @property
    def playing(self) -> bool:
        return self.timer.isActive()

    def rewind_all(self) -> None:
        for t in self.tiles:
            t.rewind()
            t.undim()
            t.step()
        self._rest = 0

    def _tick(self) -> None:
        live = 0
        for t in self.tiles:
            if t.step():
                live += 1
        if live:
            self._rest = 0
            return
        # 전부 끝났다. 잠깐 세워 두었다가 다 같이 되감는다 -- 먼저 끝난 것이
        # 혼자 되감으면 "누가 길었나" 가 사라진다.
        self._rest += 1
        if self._rest >= REST_TICKS:
            self.rewind_all()

    # ------------------------------------------------------------------ 선택
    def _on_tile_clicked(self, tile, modifiers) -> None:
        if tile.ep is None:
            return
        multi = bool(modifiers & (Qt.KeyboardModifier.ControlModifier
                                  | Qt.KeyboardModifier.ShiftModifier))
        if not multi:
            for t in self.tiles:
                t.set_selected(t is tile)
        else:
            tile.set_selected(not tile.selected)
        self._emit_selection()

    def selected_episodes(self) -> list:
        return [t.ep for t in self.tiles if t.selected and t.ep is not None]

    def page_of(self, ep) -> int:
        """이 에피소드가 있는 쪽. 없으면 -1."""
        name = (ep or {}).get("name")
        for i, e in enumerate(self.episodes):
            if e.get("name") == name:
                return i // self.per_page
        return -1

    def show_selection(self, episodes) -> None:
        """바깥이 정한 선택을 그린다. 신호를 내지 않는다.

        선택을 격자가 아니라 **창이** 들고 있기 때문에 필요하다. 격자는 한
        번에 12개만 보여주는데 목록 뷰는 걸러진 전부를 보여주므로, 선택이
        격자 안에 있으면 쪽을 넘길 때마다 사라진다. 두 뷰가 같은 선택을
        그리려면 선택이 둘 중 어느 쪽에도 살면 안 된다.

        선택된 것이 지금 쪽에 없으면 **그 쪽으로 넘어간다** -- 목록에서 고른
        것이 화면 밖에 있으면 "같이 선택된다" 는 말이 무의미하다. 여러 개가
        여러 쪽에 걸쳐 있으면 첫 번째가 있는 쪽으로 간다.
        """
        names = {e.get("name") for e in (episodes or []) if e}
        if names and not any(t.ep is not None and t.ep.get("name") in names
                             for t in self.tiles):
            pg = self.page_of(next(iter(episodes)))
            if pg >= 0 and pg != self.page:
                self.page = pg
                self._load_page_quiet()
        for t in self.tiles:
            t.set_selected(t.ep is not None and t.ep.get("name") in names)

    def _load_page_quiet(self) -> None:
        """쪽만 갈아 끼운다 -- selection_changed 를 내지 않는다.
        (show_selection 이 부르므로, 신호를 내면 선택이 되돌아와 맴돈다.)"""
        start = self.page * self.per_page
        chunk = self.episodes[start:start + self.per_page]
        for i, t in enumerate(self.tiles):
            t.load(chunk[i] if i < len(chunk) else None, self.camera,
                   self.proxy_dir)
            t.undim()
            t.selected = False
            t.marked = bool(self.is_marked and t.ep is not None
                            and self.is_marked(t.ep))
            t._restyle()
        self._rest = 0

    def _emit_selection(self) -> None:
        self.selection_changed.emit(self.selected_episodes())

    def closeEvent(self, ev):  # noqa: N802 (Qt)
        self.stop()
        for t in self.tiles:
            t.release()
        super().closeEvent(ev)
