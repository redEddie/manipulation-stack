"""QPainter 플롯 위젯 계약 -- 2026-09-12 구조 감사에서 테스트가 0 이던 곳.

`mstack/gui/plot_widgets.py` 는 Analysis·Trim 두 화면의 그림을 전부 그리는데
tests/ 어디에서도 참조되지 않았다. 여기서 보는 것은 **그리다 죽지 않는가**와
**분포 통계가 맞는가** 둘이다 -- 픽셀은 보지 않는다 (그건 사람이 본다).

DistStrip 은 2026-09-12 에 BarStrip(차원마다 막대 하나)을 대체했다:
같은 차원이라도 에피소드마다 얼마나 흔들리는지가 큐레이션의 질문이라,
평균 하나로는 답이 안 됐다.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import helpers  # noqa: E402
helpers.isolate_state()

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from mstack.gui.plot_widgets import (  # noqa: E402
    DistStrip,
    Histogram,
    LegendStrip,
    SeriesPlot,
)

# ------------------------------------------------------------ 1. DistStrip
d = DistStrip()
d.resize(420, 200)
rng = np.random.default_rng(0)
rows = [(f"joint{i + 1}", rng.normal(0.004 + i * 0.001, 0.0008, 40)) for i in range(7)]
d.set_rows(rows)
d.grab()                                    # paintEvent 가 돈다
assert d.minimumHeight() >= d.ROW_H * 7, d.minimumHeight()

# 빈 목록 -- 씬을 막 바꾼 직후에 실제로 온다. 그려도 죽지 않아야 한다.
d.set_rows([])
d.grab()

# 에피소드 하나짜리(σ = 0): 상자가 선으로 눌려도 그린다.
d.set_rows([("joint1", [0.004])])
d.grab()

# 값이 전부 0 인 축 -- vmax 가 0 이 되어 나눗셈이 터지지 않아야 한다.
d.set_rows([("joint1", [0.0, 0.0, 0.0])])
d.grab()
print("1. DistStrip OK (7줄 · 빈 목록 · 한 개 · 전부 0)")

# ---------------------------------------------- 2. 막대(BarStrip) 는 갔다
import mstack.gui.plot_widgets as pw  # noqa: E402

assert not hasattr(pw, "BarStrip"), (
    "차원별 σ 를 막대 하나로 그리던 BarStrip 이 돌아왔다 -- 분포(DistStrip)로 "
    "바꾼 판단을 다시 하려면 이 줄을 지워라 (2026-09-12).")
print("2. BarStrip stay-gone OK")

# ------------------------------------------------------------ 3. LegendStrip
lg = LegendStrip()
lg.resize(500, 18)
lg.grab()
assert lg.height() == 18, lg.height()       # 한 줄 고정 -- 플롯을 밀지 않는다
LegendStrip("빨간 음영 = 잘려나갈 구간").grab()
print("3. LegendStrip OK")

# ------------------------------------------------------------ 4. SeriesPlot
sp = SeriesPlot("joint1.pos, joint2.pos")
sp.resize(400, 150)
sp.grab()                                   # 데이터 없이도 그린다 ("데이터 없음")
t = np.linspace(0, 1, 120, dtype=np.float32)[:, None]
series = {"state": t.repeat(8, axis=1), "action": t.repeat(8, axis=1) * 1.01,
          "commanded": None, "n": 120}
sp.set_data(series, [(0, "joint1.pos"), (1, "joint2.pos")])
sp.set_cursor(40)
sp.set_cut(90)
sp.grab()
# "n" 은 계열이 아니다 -- 정수를 배열처럼 잘라 쓰면 그때 죽는다.
assert "n" not in sp._series and "commanded" not in sp._series, sp._series.keys()
sp.clear()
sp.grab()
print("4. SeriesPlot OK (빈 상태 · 커서 · 잘림 음영 · 'n' 은 계열 아님)")

# ------------------------------------------------------------- 5. Histogram
h = Histogram("에피소드 평균 |Δa| 분포")
h.resize(400, 110)
h.grab()
h.set_values([0.004, 0.005, 0.006, 0.02], [(0.005, "중앙값"), (0.02, "이 에피소드")])
h.grab()
h.set_values([])
h.grab()
print("5. Histogram OK")

print("test_plot_widgets OK")
