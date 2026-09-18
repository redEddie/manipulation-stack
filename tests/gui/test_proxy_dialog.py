"""Proxy build dialog: bake one instruction at a time, select all from the header.

Pinned here (2026-09-17):
  * each scene file expands into one row per instruction with its own clip count;
  * the header checkbox selects/clears everything and shows [-] for a partial
    selection, and a file row goes partial when only some instructions are checked;
  * only the checked instructions are baked -- the rest of the scene stays missing.

Runs on a synthetic scene; the operator's .hdf5 files are never opened.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "tests" / "gui"))
from helpers import isolate_state  # noqa: E402

isolate_state()

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from apps.workspace.features.dataset.proxy_dialog import ProxyBuildDialog  # noqa: E402
from mstack.data.proxy_clip import build_file, plan_file, scan_file  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)

d = Path(tempfile.mkdtemp(prefix="proxydlg_"))
subprocess.run([sys.executable, str(WT / "scripts/check/check_scene_file.py"),
                "--selftest", "--keep", str(d)], check=True, capture_output=True)
scene = d / "scene_000.hdf5"
proxies = Path(tempfile.mkdtemp(prefix="proxydlg_clips_"))

sc = scan_file(scene, proxies)
iids = sorted({sc["slots"][name][0] for name, _u, _c in sc["missing"]})
assert len(iids) >= 2, f"need a scene with several instructions, got {iids}"
print(f"1. scan: {len(sc['missing'])} clips over instructions {iids} OK")


class _Win(QWidget):
    def log(self, _msg):
        pass


dlg = ProxyBuildDialog(_Win(), [scene], "", proxies)
file_row = dlg.tree.topLevelItem(0)
assert file_row.childCount() == len(iids), file_row.childCount()
assert sum(int(file_row.child(j).text(1)) for j in range(file_row.childCount())) \
    == len(sc["missing"])
assert dlg.header.check_state() == Qt.CheckState.Checked
print("2. one row per instruction, everything checked by default OK")

dlg.set_all_checked(False)
assert dlg.header.check_state() == Qt.CheckState.Unchecked
assert file_row.checkState(0) == Qt.CheckState.Unchecked
assert dlg._checked_selection() == {}

first = file_row.child(0)
first.setCheckState(0, Qt.CheckState.Checked)        # a user click, signals on
assert file_row.checkState(0) == Qt.CheckState.PartiallyChecked
assert dlg.header.check_state() == Qt.CheckState.PartiallyChecked, "header must show [-]"
sel = dlg._checked_selection()
assert set(sel) == {str(scene)} and sel[str(scene)] == set(first.data(0, Qt.ItemDataRole.UserRole + 1))
assert str(first.data(0, Qt.ItemDataRole.UserRole + 2)) in dlg.status.text()

# 클립은 카메라마다 하나다. 개수만 보이면 에피소드의 배수로 읽히므로 상태줄이
# 에피소드 수도 말해야 한다 (2026-09-18, 조작자가 14 에피소드를 28 로 읽었다).
dlg.set_all_checked(True)
n_eps = len({name for name, _u, _c in sc["missing"]})
n_cams = len({cam for _n, _u, cam in sc["missing"]})
assert n_cams >= 2, f"이 검사에는 카메라가 둘 이상인 장면이 필요하다: {n_cams}"
assert len(sc["missing"]) == n_eps * n_cams, (n_eps, n_cams, len(sc["missing"]))
text = dlg.status.text()
assert f"에피소드 {n_eps}개" in text, text
assert f"클립 {n_eps * n_cams}개" in text, text
print(f"2b. status names both units: {n_eps} episodes x {n_cams} cameras OK")

dlg.set_all_checked(True)
assert dlg.header.check_state() == Qt.CheckState.Checked
assert file_row.checkState(0) == Qt.CheckState.Checked
dlg.set_all_checked(False)
dlg.set_all_checked(True)
print("3. header select-all / clear / partial [-] OK")

# ---- only the chosen instruction is baked
names = sel[str(scene)]
want = plan_file(scene, proxies, names)
assert want and all(t[0] in names for t in want)
r = build_file(scene, proxy_dir=proxies, episodes=names)
assert r["made"] == len(want) and r["failed"] == 0, r
left = scan_file(scene, proxies)["missing"]
assert len(left) == len(sc["missing"]) - len(want), (len(left), len(sc["missing"]))
assert not any(t[0] in names for t in left)
print(f"4. baked {r['made']} clips of one instruction, {len(left)} others still missing OK")

dlg.reject()
print("test_proxy_dialog 통과")
