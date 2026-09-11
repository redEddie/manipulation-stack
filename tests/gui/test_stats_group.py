"""Analysis 큐레이션 후보의 범위 계약 -- 그룹 콤보는 **없고**, 왼쪽 패널
(Scene 콤보 + Instruction 목록)이 (씬 → 지시문) 범위를 정한다 (조작자,
2026-09-11).

2026-09-12 에 바꾼 이유: 같은 (씬 → 지시문) 축이 Analysis 의 그룹 콤보와
왼쪽 패널 두 군데에 있었다. 큐레이션은 오직 (씬 → 지시문) 안에서만 하므로
Analysis 는 왼쪽 패널을 따른다. 정렬 콤볏도 뺐다 -- 왼쪽 목록·격자·Analysis 가
같은 집합을 다른 순서로 보여주면 어느 것이 어느 것인지 헷갈리므로, 순위표는
에피소드 번호 순으로 고정이다. 튀는 것은 '평균과 차이' 열 색과 '튀는 것만
선택' 버튼이 잡는다.

아래 scan_dataset 부분은 예전 그대로다: (scene, 문장) 그룹 분리 자체는
변하지 않았다 -- 바뀐 것은 GUI 가 그 그룹을 **자기 콤보로** 고르지 않고
왼쪽 패널 선택을 따른다는 것뿐이다."""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

WT = str(Path(__file__).resolve().parents[2])   # 리포 루트
sys.path.insert(0, WT)

from mstack.data.episode_stats import scan_dataset, task_table  # noqa: E402
from mstack.scene.scene_format import SceneMetadata, SceneWriter, list_scene_episodes  # noqa: E402

d = Path(tempfile.mkdtemp(prefix="statsgrp_"))
subprocess.run([sys.executable, WT + "/scripts/check/check_scene_file.py",
                "--selftest", "--keep", str(d)], check=True, capture_output=True)
scene0 = d / "scene_000.hdf5"
eps0 = list_scene_episodes(scene0)
sentence = eps0[0]["instruction"]

# 같은 문장으로 S001 을 하나 더 만든다 -- 훨씬 느린(작은 |Δa|) 궤적으로
md = SceneMetadata(scene_id="S001", objects=["OBJ-CUP-BLU-01", "OBJ-BOWLS-WHT-01"],
                   layout={"grid": [3, 3], "placements": {
                       "OBJ-CUP-BLU-01": {"zone": [0, 0]},
                       "OBJ-BOWLS-WHT-01": {"zone": [2, 2]}}})
w = SceneWriter(d, metadata=md)
r = np.random.default_rng(0)
for _ in range(3):
    w.start_episode()
    q = np.zeros(7, np.float32)
    for _ in range(30):
        q = q + 0.001                          # 아주 느린 움직임
        w.add_frame(
            agentview_rgb=r.integers(0, 255, (48, 64, 3), dtype=np.uint8),
            eye_in_hand_rgb=r.integers(0, 255, (48, 64, 3), dtype=np.uint8),
            joint_positions=q, gripper_position=0.5, ee_pos_quat=np.zeros(7),
            gripper_closed=False, commanded_joint_positions=q, commanded_gripper=0.0)
    w.save_buffer(w.detach_buffer(), instruction=sentence, instruction_id="I000",
                  success=True, collector="t")
w.close()

stats = scan_dataset([str(scene0), str(d / "scene_001.hdf5"),
                      str(d / "selftest_task_demo.hdf5")])
by_group = {}
for s in stats:
    by_group.setdefault(s.group, []).append(s)
groups_same_sentence = [g for g in by_group if g[1] == sentence]
scenes = sorted(g[0] for g in groups_same_sentence)
assert scenes == ["S000", "S001"], scenes                # 같은 문장이 scene 별로 분리
# S001 그룹은 자기들끼리만 비교 -> 전부 느려도 그룹 내 편차는 ~0 (튀는 것 없음)
s1 = by_group[("S001", sentence)]
assert all(abs(s.task_dev) < 1e-6 for s in s1), [s.task_dev for s in s1]
assert not any(s.flagged for s in s1)
# 그룹 라벨은 scene 이 앞에 붙는다, legacy 는 문장만
assert s1[0].group_label.startswith("S001 · ")
leg = [s for s in stats if s.scene == ""]
assert leg and leg[0].group_label == leg[0].task
# task_table 도 그룹 단위 행
rows = task_table(stats)
assert any(r_["scene"] == "S001" and r_["task"] == sentence for r_ in rows)
assert any(r_["scene"] == "S000" and r_["task"] == sentence for r_ in rows)
print(f"통과: (scene,문장) 그룹 분리 -- 그룹 {len(by_group)}개, S001 느린 궤적이 S000 기준으로 튀지 않음")

# ---- GUI: 후보 목록은 왼쪽 패널(Scene 콤보 + Instruction 목록)을 따른다 ----
sys.path.insert(0, WT + "/apps")
sys.argv = ["t"]
from PyQt6.QtWidgets import QApplication, QTreeWidgetItem  # noqa: E402

app = QApplication(sys.argv)
import collect_workspace as cw  # noqa: E402

cw.CameraOps.refresh_cameras = lambda self: None
cw.CameraOps.restart_previews = lambda self: None
cw.SystemOps.startup_tuning = lambda self: None   # pkexec 비밀번호 창 차단
cw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
win = cw.WorkspaceWindow(None)
win.session.stats = stats

# **계약 1: 범위 축은 왼쪽 패널 하나뿐이다.** 그룹 콤보가 다시 생기면 걸린다.
assert not hasattr(win, "group_combo"), \
    "그룹 축이 두 군데가 됐다 -- 왼쪽 패널(씬 → 지시문)이 정본"
# 정렬은 **남아 있어야 한다.** 상황에 따라 이상치를 찾는 수단이다
# (조작자, 2026-09-12): "늘어짐" 은 녹화를 늦게 끝낸 것을, "짧음" 은 2~3틱짜리를
# 위로 끌어올린다. 다만 **기본은 에피소드 순**이다 -- 왼쪽 목록·격자와 같은
# 순서라야 세 화면을 오갈 때 헷갈리지 않고, 기준을 바꾼 것이 눈에 띈다.
assert hasattr(win, "rank_combo"), "정렬 수단이 사라졌다"
assert win.rank_combo.currentData() is None, (
    "기본 정렬이 에피소드 순이 아니다: %r" % win.rank_combo.currentData())
assert win.rank_combo.count() >= 5, win.rank_combo.count()
# 그룹 열은 없앴다 -- 한 지시문 안에서만 보므로 같은 값만 반복한다.
assert win.rank_tree.columnCount() == 4, win.rank_tree.columnCount()

scene1 = d / "scene_001.hdf5"

# 왼쪽 패널을 (씬=S001, 지시문=I000) 으로 놓는다 -- 갤러리 로더 대신 직접.
# Gallery 가 채우는 것과 같은 모양: instruction 목록 첫 줄은 (전체)=None,
# 그 다음 줄에 instruction_id.
win.gallery_scene_combo.addItem("scene_001", str(scene1))
win.gallery_scene_combo.setCurrentIndex(win.gallery_scene_combo.count() - 1)
eps1 = list_scene_episodes(scene1)
win._gallery_shown = eps1
lst = win.instruction_list
lst.blockSignals(True)
lst.clear()
all_it = QTreeWidgetItem(["(all instructions)", str(len(eps1))])
all_it.setData(0, cw.Qt.ItemDataRole.UserRole, None)
lst.addTopLevelItem(all_it)
iid_it = QTreeWidgetItem([f"I000  {sentence[:44]}", str(len(eps1))])
iid_it.setData(0, cw.Qt.ItemDataRole.UserRole, "I000")
lst.addTopLevelItem(iid_it)
lst.setCurrentItem(all_it)
lst.blockSignals(False)

# (전체) 지시문 + scene_001 선택: 파일 좁히기만 한다.
out = win.stats_ops.filtered_stats()
assert len(out) == len(s1), (len(out), len(s1))
assert all(e.path == str(scene1) for e in out), [e.path for e in out]

# 지시문 I000 으로 좁히면: _gallery_shown 의 문장과 같은 task 만 남는다.
# S001 의 지시문이 I000 이고 문장이 sentence 하나뿐이므로 집합은 그대로다.
lst.setCurrentItem(iid_it)
out_i = win.stats_ops.filtered_stats()
assert len(out_i) == len(s1), (len(out_i), len(s1))

# _gallery_shown 이 비어 있으면 지시문 필터는 걸지 않는다 (깨진 상태로
# 전부 날리지 않는다).
win._gallery_shown = []
out_empty = win.stats_ops.filtered_stats()
assert len(out_empty) == len(s1), (len(out_empty), len(s1))
win._gallery_shown = eps1

# 순위표: 4열, 에피소드 번호 순, 선택된 (씬 → 지시문) 집합만.
win.stats_ops.refresh_rank_list()
n_rows = win.rank_tree.topLevelItemCount()
assert n_rows == len(s1), (n_rows, len(s1))
keys = [win.rank_tree.topLevelItem(k).data(0, cw.Qt.ItemDataRole.UserRole)
        for k in range(n_rows)]
assert all(p == str(scene1) for p, _ in keys), keys
import re  # noqa: E402

nums = [int(re.search(r"(\d+)", demo).group(1)) for _, demo in keys]
assert nums == sorted(nums), nums                # 에피소드 번호 순 고정
for k in range(n_rows):
    assert win.rank_tree.topLevelItem(k).columnCount() == 4   # 'task' 열은 없다

# scene 콤보 선택을 빼면 전 파일이 범위가 된다 (지시문은 (전체) 인 채).
win.gallery_scene_combo.setCurrentIndex(-1)
out_all = win.stats_ops.filtered_stats()
assert len(out_all) == len(stats), (len(out_all), len(stats))
win.stats_ops.refresh_rank_list()
assert win.rank_tree.topLevelItemCount() == len(stats)

print("통과: 후보 목록은 왼쪽 패널이 범위를 정한다 -- 콤보 없음, 에피소드 번호 순, 4열")
import os  # noqa: E402

# os._exit 는 버퍼를 비우지 않는다 -- 먼저 비운다. 없으면 이 파일의
# 출력이 통째로 사라져서, 검사가 실제로 돌았는지 사람이 볼 수 없다
# (스위트는 종료 코드만 본다).
sys.stdout.flush()
os._exit(0)
