"""큐레이션 격자 -- 동시 재생의 규칙.

여기서 못박는 것 중 조용히 깨지기 쉬운 것들:

  * **시간을 정규화하지 않는다.** 진행률을 0~1 로 맞춰 돌리면 모든 타일이
    같은 순간에 끝나서 "녹화를 너무 늦게 끝낸 것" 이 안 보인다 -- 길이 자체가
    신호인데 그것을 지우는 설계다. 짧은 클립은 짧게 끝나야 한다.
  * **먼저 끝난 타일은 되감지 않는다.** 혼자 되감으면 "누가 길었나" 가
    사라진다. 전부 끝나고 REST_TICKS 뒤에 다 같이 되감는다.
  * **프록시가 없어도 죽지 않는다.** 큐레이션 도중 클립을 굽기 시작하면
    화면이 멎으므로, 없는 것은 없다고 말하고 넘어간다.

합성 클립으로 돈다. 조작자의 .hdf5 는 열지 않는다.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
sys.argv = ["t"]

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)

from mstack.data.proxy_clip import encode_clip, proxy_path  # noqa: E402
import mstack.gui.clip_grid as cg  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="clipgrid_"))
# 몽키패치하지 않는다 -- 격자가 자기 proxy_dir 을 갖는 것이 실제 경로다.
# 데이터셋마다 캐시 자리가 다르기 때문이다: episode_uid 는 데이터셋 안에서만
# 유일해서, 한 곳에 섞으면 다른 데이터셋의 영상이 나온다 (2026-09-11 에 실제로
# libero_datasets/sangtae 를 보던 화면이 fr3-tabletop 의 클립을 틀었다).


def make(uid, n_frames):
    """움직이는 사각형 -- 전부 같은 그림이면 코덱이 프레임을 합쳐 버린다."""
    fr = np.zeros((n_frames, 48, 64, 3), np.uint8)
    for i in range(n_frames):
        fr[i, 10:20, (i % 50):(i % 50) + 8] = 255
    encode_clip(fr, proxy_path(uid, "agentview_rgb", TMP), fps=20, scale=1.0)
    return {"episode_uid": uid, "name": f"episode_{uid[-3:]}",
            "instruction_id": uid.split("-")[2], "instruction": "x",
            "quality_status": "success", "num_samples": n_frames}


SHORT, LONG = 20, 60
eps = [make("EP-S000-I000-E000", SHORT), make("EP-S000-I000-E001", LONG)]
eps.append({"episode_uid": "EP-S000-I000-E999", "name": "episode_999",
            "instruction_id": "I000", "instruction": "x",
            "quality_status": "failed", "num_samples": 5})   # 프록시 없음
print(f"1. 합성 클립 {SHORT}프레임 / {LONG}프레임 + 프록시 없는 것 1개 OK")


# ------------------------------------------------------- 로드 · 없는 프록시
g = cg.ClipGrid(cols=2, rows=2)
g.proxy_dir = TMP
g.resize(400, 300)
g.set_episodes(eps)
opened = [t for t in g.tiles if t.cap is not None]
assert len(opened) == 2, f"클립이 {len(opened)}개 열렸다 (2 여야 함)"
missing = g.tiles[2]
assert missing.ep is not None and missing.cap is None
assert "no proxy" in missing.view.text(), missing.view.text()
assert g.tiles[3].ep is None, "빈 칸이어야 한다"
print("2. 프록시 없는 타일이 죽지 않고 그렇다고 말함 OK")


# ------------------------------------------------- 빈 칸은 정말 비어 있어야 한다
# setText("") 는 QLabel 의 픽스맵을 지우지 않는다. 그래서 목록을 좁히면 남는
# 칸이 **직전 화면의 프레임을 그대로 들고** 있었다 (2026-09-11 조작자 보고).
# 그러면 "에피소드가 없는 칸" 과 "짧아서 금방 멈춘 에피소드" 가 화면에서
# 똑같아 보인다 -- 둘 다 안 움직이는 그림이라서다. 3프레임짜리를 찾는 것이
# 이 화면의 목적 중 하나인데 그것이 빈 칸에 묻힌다.
g2 = cg.ClipGrid(cols=2, rows=2)
g2.proxy_dir = TMP
g2.resize(400, 300)
g2.set_episodes([eps[0], eps[1], make("EP-S000-I000-E002", 30),
                 make("EP-S000-I000-E003", 30)])
for _ in range(40):
    g2._tick()
assert all(t.view.pixmap() is not None and not t.view.pixmap().isNull()
           for t in g2.tiles), "네 칸 다 그림이 있어야 한다"
g2.set_episodes([eps[0]])            # 목록을 하나로 좁힌다
for t in g2.tiles[1:]:
    pm = t.view.pixmap()
    assert t.ep is None
    assert pm is None or pm.isNull(), "빈 칸이 직전 프레임을 들고 있다"
    assert "none" in t.styleSheet(), "빈 칸에 테두리가 남아 있다"
    assert t.caption.text() == "", f"빈 칸에 캡션이 남아 있다: {t.caption.text()!r}"
# 짧은 에피소드는 반대로 그림·테두리·캡션이 남아야 구분된다
short_ep = g2.tiles[0]
assert short_ep.view.pixmap() is not None and not short_ep.view.pixmap().isNull()
assert "none" not in short_ep.styleSheet()
assert f"{SHORT}f" in short_ep.caption.text()
print("2b. 빈 칸이 정말 빔 (짧은 에피소드와 구분됨) OK")

# 캡션에 프레임 수가 늘 보인다 -- 2~3틱짜리를 숫자로 잡는 자리다
assert f"{SHORT}f" in g.tiles[0].caption.text(), g.tiles[0].caption.text()
assert "5f" in missing.caption.text(), missing.caption.text()
print(f"3. 캡션 {g.tiles[0].caption.text()!r} / {missing.caption.text()!r} OK")


# ------------------------------------------- 정규화 없음: 짧은 것이 먼저 멈춘다
g.rewind_all()
short_t, long_t = g.tiles[0], g.tiles[1]
for _ in range(SHORT + 2):
    g._tick()
assert short_t.done, "짧은 클립이 자기 길이에서 안 멈췄다 -- 정규화가 끼었나?"
assert not long_t.done, "긴 클립이 짧은 것과 같이 끝났다 -- 시간이 정규화됐다"
frozen = short_t.i
print(f"4. 짧은 것 {frozen}프레임에서 정지, 긴 것은 계속 ({long_t.i}프레임) OK")

# 먼저 끝난 타일은 되감지 않는다 -- 길이 신호가 사라지면 안 된다
for _ in range(5):
    g._tick()
assert short_t.i == frozen, f"끝난 타일이 되감겼다 ({short_t.i})"
assert short_t.done
print("5. 먼저 끝난 타일이 되감지 않음 OK")

# 전부 끝나면 REST_TICKS 뒤에 다 같이 되감는다.
# 고정 횟수로 돌리면 안 된다 -- 넘기면 이미 되감긴 뒤라 done 이 도로 False 다.
for _ in range(LONG * 2):
    if long_t.done:
        break
    g._tick()
assert long_t.done, f"긴 클립이 {long_t.i}프레임에서 안 끝났다"
# 마지막 타일이 끝난 그 틱이 쉬는 시간의 1틱째다 -- 남은 것은 REST_TICKS-1 틱.
assert g._rest == 1, g._rest
for _ in range(cg.REST_TICKS - 2):
    g._tick()
    assert short_t.done and long_t.done, f"쉬는 중({g._rest})에 되감았다"
g._tick()
assert not short_t.done and not long_t.done, "전부 끝난 뒤 다 같이 되감아야 한다"
assert short_t.i <= 1 and long_t.i <= 1
print(f"6. 전부 끝난 뒤 {cg.REST_TICKS}틱 쉬고 다 같이 되감기 OK")


# ------------------------------------------------------------------ 선택 · 쪽
from PyQt6.QtCore import Qt  # noqa: E402

got = []
g.selection_changed.connect(lambda eps: got.append(list(eps)))
g._on_tile_clicked(g.tiles[0], Qt.KeyboardModifier.NoModifier)
assert got[-1] and got[-1][0]["episode_uid"] == "EP-S000-I000-E000"
g._on_tile_clicked(g.tiles[1], Qt.KeyboardModifier.ControlModifier)
assert len(got[-1]) == 2, f"Ctrl+클릭 다중선택이 안 된다: {got[-1]}"
g._on_tile_clicked(g.tiles[1], Qt.KeyboardModifier.NoModifier)
assert len(got[-1]) == 1, "맨클릭은 단일선택이어야 한다"
g._on_tile_clicked(g.tiles[3], Qt.KeyboardModifier.NoModifier)   # 빈 칸
assert len(got[-1]) == 1, "빈 칸 클릭이 선택을 바꾸면 안 된다"
print("7. 선택 (단일 · Ctrl 다중 · 빈 칸 무시) OK")

many = [make(f"EP-S000-I001-E{i:03d}", 10) for i in range(9)]
g.set_episodes(many)
assert g.n_pages == 3, f"2x2 격자에 9개면 3쪽 ({g.n_pages})"
g.set_page(2)
assert sum(1 for t in g.tiles if t.ep is not None) == 1, "마지막 쪽은 1개"
g.set_page(99)
assert g.page == 2, "쪽 번호가 범위를 넘으면 안 된다"
print(f"8. 쪽 나눔 ({g.n_pages}쪽, 범위 고정) OK")

print("\n큐레이션 격자 통과")
