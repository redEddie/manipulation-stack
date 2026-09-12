"""Trim 탭의 조작들 -- 2026-09-12 조작자 요청 다섯 가지를 못박는다.

1. 재생바에 **잘릴 지점**이 빨간 선으로 그려진다 (−5/−1 을 누르면 따라 움직인다)
2. " ← 잘린 뒤 마지막" 문구는 **돌아오지 않는다** (뜻이 읽히지 않아 뺐다)
3. [행동취소] 는 마지막 한 걸음만, [원래대로] 는 통째로 0 으로
4. Trim 에도 배속이 있다 (Playback 탭에 있던 것과 같은 값)
5. 플롯을 한 번에 켜고 끄는 [전체]/[해제]

그리고 Analysis 의 [Trim 에서 재생] 이 **Trim 탭**을 연다 (예전엔 Playback).

**Playback 탭은 2026-09-12 에 없앴다.** 같은 일(한 에피소드를 크게 보며
재생)을 Trim 이 전부 하게 되어, 같은 것을 하는 화면이 둘이 되었다. 9번이
그것이 되살아나지 않는지 지킨다 -- 되살리려면 그 판단을 다시 하고 이 줄을
지우면 된다.

로봇도 카메라도 필요 없다 (offscreen). 합성 scene 파일 하나로 돈다.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import helpers  # noqa: E402
helpers.isolate_state()

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

N_FRAMES = 60


def _episode(f, i: int, step: float, iid: str, sentence: str, sid: str) -> None:
    g = f.create_group(f"episode_{i:03d}")
    g.attrs.update({
        "scene_id": sid, "instruction_id": iid, "instruction": sentence,
        "episode_uid": f"EP-{sid}-{iid}-E{i:03d}", "episode_id": i,
        "quality_status": "success", "collector": "tester",
        "num_samples": N_FRAMES,
    })
    # arm[t, j] = t * step  ->  |Δa| 가 모든 프레임에서 step 이라
    # mean_da == step 이 된다 (task_dev 를 원하는 값으로 맞추기 쉽다).
    t = np.arange(N_FRAMES, dtype=np.float32)[:, None]
    arm = (t * step).repeat(7, axis=1)
    g.create_dataset("actions", data=arm)
    obs = g.create_group("obs")
    obs.create_dataset("joint_states", data=arm)
    obs.create_dataset("gripper_states", data=np.zeros((N_FRAMES, 1), np.float32))


def _make_outlier_scene(root: Path) -> str:
    """늘어짐 하나가 든 두 번째 씬 -- [튀는 것만 선택] 이 데려갈 곳."""
    path = root / "scene_001.hdf5"
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["scene_id"] = "S001"
        meta.attrs["objects"] = json.dumps(["OBJ-CUP-WHT-02"])
        meta.attrs["layout"] = json.dumps(
            {"grid": [3, 3], "placements": {"OBJ-CUP-WHT-02": {"zone": [0, 0]}}})
        sentence = "drag the small gray bowl"
        for i in range(9):                       # 평범한 것들
            _episode(f, i, 0.0100 + i * 0.00005, "I000", sentence, "S001")
        _episode(f, 9, 0.0160, "I000", sentence, "S001")   # 튄 것 (+0.005)
    return str(path)


def _make(root: Path) -> None:
    with h5py.File(root / "scene_000.hdf5", "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["scene_id"] = "S000"
        meta.attrs["objects"] = json.dumps(["OBJ-CUP-WHT-02"])
        meta.attrs["layout"] = json.dumps(
            {"grid": [3, 3], "placements": {"OBJ-CUP-WHT-02": {"zone": [0, 0]}}})
        for i in range(2):
            g = f.create_group(f"episode_{i:03d}")
            g.attrs.update({
                "scene_id": "S000", "instruction_id": "I000",
                "instruction": "pick up the white cup",
                "episode_uid": f"EP-S000-I000-E{i:03d}", "episode_id": i,
                "quality_status": "success", "collector": "tester",
                "num_samples": N_FRAMES,
            })
            t = np.linspace(0, 1, N_FRAMES, dtype=np.float32)
            arm = np.stack([t * (j + 1) * 0.1 for j in range(7)], axis=1)
            g.create_dataset("actions", data=arm)
            obs = g.create_group("obs")
            obs.create_dataset("joint_states", data=arm)
            obs.create_dataset("gripper_states",
                               data=np.zeros((N_FRAMES, 1), np.float32))


def main() -> None:
    import apps.collect_workspace as cw
    from mstack.gui.widgets import CutSlider

    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make(root)
        path = str(root / "scene_000.hdf5")
        win = cw.WorkspaceWindow(None)
        win.root_edit.setText(str(root))
        win.dataset_ops.on_root_changed()
        win._set_activity("dataset")
        ops = win.trim_ops

        # ---------------------------------------------- 1. 잘림 선
        assert isinstance(win.trim_slider, CutSlider), type(win.trim_slider)
        ops.show_trim_for(path, "episode_000")
        assert win.trim.n == N_FRAMES, win.trim.n
        assert win.trim_slider.cut() is None, "자를 것이 없는데 선이 그려졌다"
        ops.trim_add(5)
        assert ops.trim_keep() == N_FRAMES - 5
        assert win.trim_slider.cut() == N_FRAMES - 6, win.trim_slider.cut()
        ops.trim_add(5)
        assert win.trim_slider.cut() == N_FRAMES - 11, "선이 따라 움직이지 않는다"
        # 재생은 **잘릴 자리에서 한 번 선다** -- 통째로 흘려보내면 새 끝이
        # 어느 프레임인지 재생만으로는 알 수 없다 (2026-09-12).
        keep = ops.trim_keep()
        ops.trim_seek(keep - 2)
        ops.on_trim_play()
        ops.trim_tick()                       # keep - 1 (아직 남는 마지막)
        assert win.trim_slider.value() == keep - 1
        assert win.trim.timer.isActive()
        ops.trim_tick()                       # 잘릴 첫 프레임 -- 여기서 선다
        assert not win.trim.timer.isActive(), "잘림 지점에서 안 섰다"
        assert win.trim_slider.value() == keep - 1, "선 자리가 잘림 지점이 아니다"
        assert "이어보기" in win.trim_play_btn.text(), win.trim_play_btn.text()
        # 한 번 더 누르면 잘려나갈 구간까지 이어서 본다 (전체 재생은 그대로).
        ops.on_trim_play()
        assert win.trim.timer.isActive()
        ops.trim_tick()
        assert win.trim_slider.value() == keep, "이어보기가 안 된다"
        assert "#c0392b" in win.trim_views["agent"].styleSheet(), "잘릴 구간인데 테두리가 안 빨갛다"
        ops.trim_seek(0)
        assert "transparent" in win.trim_views["agent"].styleSheet()
        # 재생의 끝은 **원본 길이**다 (자를 양을 바꿔도 같은 길이로 보인다).
        ops.trim_seek(N_FRAMES - 3)
        for _ in range(5):
            ops.trim_tick()
        assert win.trim_slider.value() == N_FRAMES - 1, win.trim_slider.value()
        assert not win.trim.timer.isActive()
        # 재생바 폭은 불변이다 -- 위치 라벨이 글자 길이로 넓어지면 그 폭을
        # 슬라이더에서 뺏는다 (2026-09-12 조작자: "길이가 무조건 불변").
        assert win.trim_pos.minimumWidth() == win.trim_pos.maximumWidth(), (
            win.trim_pos.minimumWidth(), win.trim_pos.maximumWidth())
        print("1. 빨간 잘림선 · 잘림 지점에서 정지 · 빨간 테두리 · 고정 폭 재생바 OK")

        # ---------------------------------------------- 2. 사라진 문구
        ops.trim_seek(ops.trim_keep() - 1)
        assert "잘린 뒤 마지막" not in win.trim_pos.text(), win.trim_pos.text()
        src = (Path(__file__).resolve().parents[2] /
               "apps/workspace/features/trim/ops.py").read_text(encoding="utf-8")
        assert 'tr(" ← 잘린 뒤 마지막")' not in src, "빼기로 한 문구가 돌아왔다"
        print("2. '잘린 뒤 마지막' stay-gone OK")

        # ---------------------------------------------- 3. 행동취소 / 원래대로
        assert win.trim_undo_btn.isEnabled(), "되돌릴 걸음이 있는데 꺼져 있다"
        ops.trim_undo()
        assert ops.trim_pending() == 5, ops.trim_pending()      # 한 걸음만
        ops.trim_add(1)
        ops.trim_reset()
        assert ops.trim_pending() == 0
        ops.trim_undo()
        assert ops.trim_pending() == 6, "원래대로도 되돌릴 수 있어야 한다"
        while win.trim.undo:
            ops.trim_undo()
        assert ops.trim_pending() == 0 and not win.trim_undo_btn.isEnabled()
        # 다른 에피소드를 물면 이전 걸음은 못 되돌린다
        ops.trim_add(3)
        ops.show_trim_for(path, "episode_001")
        assert win.trim.undo == [] and ops.trim_pending() == 0
        print("3. 행동취소 / 원래대로 OK")

        # ---------------------------------------------- 4. 배속
        labels = [win.trim_speed_combo.itemText(i)
                  for i in range(win.trim_speed_combo.count())]
        assert labels == ["0.5x", "1x", "2x", "3x"], labels
        win.trim_speed_combo.setCurrentIndex(labels.index("1x"))
        ops.on_trim_play()                      # 타이머를 만든다
        base = win.trim.timer.interval()
        assert base == 50, base                 # 20Hz
        win.trim_speed_combo.setCurrentIndex(labels.index("2x"))
        assert win.trim.timer.interval() == 25, \
            win.trim.timer.interval()
        ops.on_trim_play()                      # 멈춘다
        assert not win.trim.timer.isActive()
        print("4. Trim 배속 OK")

        # ---------------------------------------------- 5. 플롯 전체 / 해제
        checks = win.trim_plot_checks
        assert [t for t, c in checks.items() if c.isChecked()] == ["gripper"]
        btns = {b.text(): b for b in win.center_tab_widgets["trim"].findChildren(
            type(win.trim_play_btn))}
        assert "전체" in btns and "해제" in btns, sorted(btns)
        btns["전체"].click()
        assert all(c.isChecked() for c in checks.values())
        btns["해제"].click()
        assert not any(c.isChecked() for c in checks.values())
        btns["전체"].click()
        print("5. 플롯 전체/해제 OK")

        # ------------------------------------- 6. [Trim 에서 재생] 은 Trim 으로
        win.stats_ops.refresh_analysis(force=True)
        assert win.rank_tree.topLevelItemCount() >= 1, "순위표가 비었다"
        first = win.rank_tree.topLevelItem(0).text(0)
        assert first.startswith("S000 · episode_"), first   # 짧은 scene 이름
        assert "scene_000" not in first, first
        win.rank_tree.setCurrentItem(win.rank_tree.topLevelItem(0))
        from apps.workspace.shared.tabs import show_center_tab
        show_center_tab(win, "analysis")
        ops.on_rank_trim()
        cur = win.center_tabs.currentWidget()
        assert cur is win.center_tab_widgets["trim"], \
            "Playback 이 아니라 Trim 으로 가야 한다"
        print("6. Trim 에서 재생 OK · 후보 목록의 짧은 이름 OK")

        # ------------------------------- 7. 목록이 무엇의 목록인지 말한다
        scope = win.stats_ops.scope_label()
        assert scope.startswith("S000 · pick up the white cup"), scope
        assert "2개" in scope, scope
        assert scope in win.dim_box.title(), win.dim_box.title()
        assert scope in win.filt_box.title(), win.filt_box.title()
        # 회색 줄은 **하나**뿐이고, 남은 한 마디는 판정선이다.
        assert "±0.004" in win.stats_hint.text(), win.stats_hint.text()
        assert not hasattr(win, "analysis_summary"), "요약 텍스트 줄이 돌아왔다"
        # 요약 줄은 무엇을 하라고 시키지 않는다 -- 그 일을 하는 버튼이 아래 있다.
        from mstack.data.episode_stats import summarize
        v = summarize(win.session.stats)["verdict"]
        assert "재생해서 확인" not in v, v
        print("7. 범위 표시 · 요약 문구 OK")

        # ------------------------- 8. [튀는 것만 선택] 이 **데려간다**
        # 지금 목록(S000)에는 튄 것이 없고, 옆 씬(S001)에 하나 있다.
        other = _make_outlier_scene(root)
        win.gallery_ops.refresh_gallery_scenes()
        for _ in range(80):
            app.processEvents()
            if win._gallery_episodes:
                break
            time.sleep(0.02)
        win.stats_ops.refresh_analysis(force=True)
        flagged = [e for e in win.session.stats if e.flagged]
        assert len(flagged) == 1 and flagged[0].scene == "S001", \
            [(e.scene, e.demo, round(e.task_dev, 5)) for e in flagged]
        # S000 을 보고 있는 상태에서 누른다
        cb = win.gallery_scene_combo
        cb.setCurrentIndex(cb.findData(path))
        for _ in range(80):
            app.processEvents()
            if cb.currentData() == path and win._gallery_episodes:
                break
            time.sleep(0.02)
        win.stats_ops.on_select_flagged()
        for _ in range(120):
            app.processEvents()
            if cb.currentData() == other and win._focus_episode is None:
                break
            time.sleep(0.02)
        assert cb.currentData() == other, "튄 것이 있는 씬으로 안 옮겼다"
        picked = [i.text(0) for i in win.rank_tree.selectedItems()]
        assert picked and flagged[0].demo in picked[0], (picked, flagged[0].demo)
        assert "S001" in win.filt_box.title(), win.filt_box.title()
        print("8. 튀는 것만 선택 -> 다른 씬으로 데려가기 OK")

        # ---------------------------------------------- 9. Playback stay-gone
        from apps.workspace.constants import CENTER_TABS, CENTER_TABS_BY_ACTIVITY
        assert "playback" not in dict(CENTER_TABS), "Playback 탭이 돌아왔다"
        for act, keys in CENTER_TABS_BY_ACTIVITY.items():
            assert "playback" not in keys, act
        assert "playback" not in win.center_tab_widgets
        for gone in ("play_views", "play_btn", "play_slider", "play_caption",
                     "speed_combo"):
            assert not hasattr(win, gone), gone
        for gone in ("play_episode", "on_play_toggle", "on_play_tick",
                     "show_frame", "stop_playback"):
            assert not hasattr(ops, gone), gone
        # 갤러리 타일 더블클릭도 Trim 으로 온다 (예전엔 Playback 이었다).
        show_center_tab(win, "gallery")
        win.gallery_ops.on_gallery_activated({"name": "episode_001"})
        assert win.center_tabs.currentWidget() is win.center_tab_widgets["trim"]
        print("9. Playback stay-gone · 갤러리 더블클릭도 Trim OK")

        # -------------------- 10. 순위표 선택 = **공유 선택** (통로 하나)
        tree = win.rank_tree
        assert tree.topLevelItemCount() >= 2, tree.topLevelItemCount()
        key = tree.topLevelItem(1).data(0, Qt.ItemDataRole.UserRole)
        tree.setCurrentItem(tree.topLevelItem(1))   # 목록을 다시 그리므로
        app.processEvents()                          # 행 포인터는 여기서 죽는다
        assert win.gallery_ops.selected_keys() == [key], \
            (win.gallery_ops.selected_keys(), key)
        assert win.trim.key == key, win.trim.key
        # 왼쪽 Episode 목록도 같은 것을 가리킨다 (예전엔 순위표만 움직였다)
        in_tree = [i.data(0, Qt.ItemDataRole.UserRole)["name"]
                   for i in win.dataset_tree.selectedItems()]
        assert in_tree == [key[1]], (in_tree, key)
        # 다시 그려도 순위표가 그 선택을 그대로 비춘다 (clear() 가 지우던 것)
        win.stats_ops.refresh_rank_list()
        back = [i.data(0, Qt.ItemDataRole.UserRole) for i in tree.selectedItems()]
        assert back == [key], back
        # 삭제로 가는 문은 하나 -- Analysis 상자의 중복 버튼은 없앴다
        assert not hasattr(win.stats_ops, "on_rank_delete")
        from PyQt6.QtWidgets import QPushButton
        labels = [b.text() for b
                  in win.center_tab_widgets["analysis"].findChildren(QPushButton)]
        assert not any("Mark for delete" in t for t in labels), labels
        print("10. 순위표 선택 = 공유 선택 · 삭제 문 하나 OK")

        for loader in (win.trim.loader,):
            if loader is not None:
                loader.wait(3000)
    print("test_trim_controls OK")


main()
