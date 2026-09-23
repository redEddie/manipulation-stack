"""에피소드 판정(set_verdict) 경로 검증.


2026-09-11 계약 변경: 성공<->실패 **뒤집기**에서 주어진 값으로 **정하기**로
바꿨다 (뒤집기는 다중 선택의 결과를 예측할 수 없어서). 검증 내용은 같다:
- owned 파일(세션이 쥔 파일): h5py.File 재오픈 없이 saver 큐로만 전달.
- 캐시에 없는 이름: 건너뜀 집계 + 로그, 예외 없음.
- 비소유 파일: scene 직접 수정 (호출 경로가 scene 만 넘긴다 -- legacy 분기 없음).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py

WT = str(Path(__file__).resolve().parents[2])  # 리포 루트
sys.path.insert(0, WT)
sys.path.insert(0, WT + "/apps")
sys.argv = ["t"]

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)
import collect_workspace as cw  # noqa: E402
from mstack.scene.scene_format import list_scene_episodes  # noqa: E402

d = Path(tempfile.mkdtemp(prefix="relabel_"))
subprocess.run([sys.executable, WT + "/scripts/check/check_scene_file.py",
                "--selftest", "--keep", str(d)], check=True, capture_output=True)
scene = d / "scene_AAAAAAA1.hdf5"   # selftest 의 고정 픽스처 ID
legacy = d / "selftest_task_demo.hdf5"

cw.CameraOps.refresh_cameras = lambda self: None
cw.CameraOps.restart_previews = lambda self: None
cw.SystemOps.startup_tuning = lambda self: None   # pkexec 비밀번호 창 차단

win = cw.WorkspaceWindow(None)


class StubSaver:
    def __init__(self):
        self.calls = []

    def enqueue_set_success(self, name, success):
        self.calls.append((name, success))


class StubWorker:
    def __init__(self, saver):
        self.saver = saver

    def cmd_set_episode_success(self, name, success):
        self.saver.enqueue_set_success(name, success)


stub_saver = StubSaver()
win.worker = StubWorker(stub_saver)

# ---- 1. owned scene 파일 재판정: h5py.File 호출 금지, saver 큐로 전달 ----
eps = list_scene_episodes(scene)
success_ep = next(e for e in eps if e["quality_status"] == "success")
bad_data_ep = next(e for e in eps if e["quality_status"] == "bad_data")

real_h5py_File = h5py.File
h5py_calls = []


def fail_h5py_File(*args, **kwargs):
    h5py_calls.append((args, kwargs))
    raise RuntimeError("owned 파일 재판정에서 h5py.File 호출 금지")


h5py.File = fail_h5py_File
try:
    win.session.active_file_path = scene
    win.session.active_episode_cache = eps
    ok = win.dataset_ops.set_verdict(
        {scene: [success_ep["name"], bad_data_ep["name"]]}, False)
    assert ok
    # bad_data 는 success/failed 가 아니므로 skipped 되고 saver 호출은 1개뿐
    assert len(stub_saver.calls) == 1
    assert stub_saver.calls[0] == (success_ep["name"], False)
    assert len(h5py_calls) == 0, h5py_calls
finally:
    h5py.File = real_h5py_File
print("1 통과: owned scene 재판정이 h5py.File 없이 saver 큐로 전달")

# ---- 2. 캐시에 없는 이름은 건너뜀 집계 + 로그, 예외 없음 ----
stub_saver.calls.clear()
logs: list = []
win.log = lambda msg: logs.append(str(msg))
win.session.active_episode_cache = [success_ep]
# 2026-09-11 계약 변경: 뒤집기(relabel_episodes) 에서 set_verdict 로 바꿨다.
# 뒤집기는 여러 개를 고르면 결과를 예측할 수 없어(성공 3 + 실패 2 를 고르면
# 성공 2 + 실패 3) 명시 값으로 **정하는** 방식이 됐다 -- 아래 단언도 옛
# "바뀌었는지"에서 "정한 값으로 바뀌었는지"를 보는 것으로 갱신했다.
ok = win.dataset_ops.set_verdict({scene: [success_ep["name"], bad_data_ep["name"]]}, False)
assert ok
assert len(stub_saver.calls) == 1
assert stub_saver.calls[0] == (success_ep["name"], False)
# saver 호출 수 간접 확인만으로는 "조용한 무시" 회귀를 못 잡는다 --
# 건너뜀이 사유와 함께 집계·로그되는 것까지 직접 단언한다.
assert any("캐시에 없어 건너뜀" in m for m in logs), logs
assert any("1개 건너뜀 (세션 캐시에 없음)" in m for m in logs), logs
print("2 통과: 캐시에 없는 이름은 건너뜀 집계 + 로그")

# ---- 3. 비소유 scene 파일 경로 회귀 ----
# 비소유 직접 쓰기 경로. (legacy 는 호출 경로가 scene 만 넘기므로 이 함수에
# 도달하지 않는다 -- 도달 불가 분기를 검증하는 테스트는 두지 않는다.)
win.session.active_file_path = None
win.session.active_episode_cache = None

eps = list_scene_episodes(scene)
target_scene = eps[0]
old_q = target_scene["quality_status"]
want = old_q != "success"
ok = win.dataset_ops.set_verdict({scene: [target_scene["name"]]}, want)
assert ok
eps_after = list_scene_episodes(scene)
new_q = next(e["quality_status"] for e in eps_after
             if e["name"] == target_scene["name"])
assert new_q == ("success" if want else "failed")
print("3 통과: 비소유 scene 파일 직접 수정 회귀")

# ---- 4. 화면의 판정 버튼이 그 경로에 실제로 닿는가 ----
# 버튼 글자는 두 번 바뀌었고(정정 → Pass → Success) 그때마다 배선은 아무도
# 확인하지 않았다. 파일을 실제로 바꾸는 버튼이라 한 번 눌러 본다.
calls = []
win.dataset_ops.set_verdict = lambda by_file, ok: (calls.append((by_file, ok)), True)[1]
win.gallery_ops.selected_keys = lambda: [(str(scene), target_scene["name"])]
# 판정이 성공하면 목록과 격자를 다시 그린다 -- 여기서 보려는 것은 **배선**
# 이므로 그 뒷단은 막는다 (이 스텁 창에는 세션이 없어 다시 그리다 죽는다).
win.dataset_ops.refresh_dataset_tree = lambda *a, **k: None
win.gallery_ops.refresh_gallery = lambda *a, **k: None
win.verdict_ok_btn.click()
win.verdict_fail_btn.click()
assert [ok for _b, ok in calls] == [True, False], calls
assert all(list(b.values()) == [[target_scene["name"]]] for b, _ok in calls), calls
assert "Success" in win.verdict_ok_btn.text() and "Failed" in win.verdict_fail_btn.text(), (
    win.verdict_ok_btn.text(), win.verdict_fail_btn.text())
print("4 통과: 판정 버튼 두 개가 set_verdict 로 닿는다 (뒤집기가 아니라 정한다)")

print("\n재판정 경로 검증 통과")
import os  # noqa: E402

# os._exit 는 버퍼를 비우지 않는다 -- 먼저 비운다. 없으면 이 파일의
# 출력이 통째로 사라져서, 검사가 실제로 돌았는지 사람이 볼 수 없다
# (스위트는 종료 코드만 보므로 통과로 지나간다).
sys.stdout.flush()
os._exit(0)
