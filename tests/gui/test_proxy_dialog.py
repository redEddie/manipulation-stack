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
scene = d / "scene_AAAAAAA1.hdf5"   # selftest 의 고정 픽스처 ID
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
