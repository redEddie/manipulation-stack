"""업로드 계열 대화상자 셋이 **같은 모양**인가 (2026-09-13 조작자 요청).

재압축·HDF5 업로드·LeRobot 변환은 하는 일이 같다: 어느 .hdf5 를 고를지 정하고
그 동작의 옵션을 정한다. 그런데 화면은 셋이 따로 자라서, 재압축만 체크 목록이고
나머지 둘은 경로를 공백으로 이어 붙인 한 줄 입력칸이었다 -- *"각각 다른 팝업이
떠서 동일한 작업을 한다는 느낌을 전혀 주지 못해요."*

여기서 지키는 것:

1. 셋 다 같은 표(`Hdf5FileTable`)로 파일을 고른다. 한 줄 입력칸은 **안 돌아온다.**
2. 기본 체크가 동작마다 뜻이 있다 (재압축=안 된 것, 업로드=장부가 올리라는 것,
   변환=전부).
3. Repo ID 칸은 줄바꿈되고, 접힌 값을 복사해 붙여도 유효한 id 가 된다.
4. **argv 는 그대로다.** 화면을 바꿨지 스크립트의 CLI 를 바꾼 것이 아니다.

네트워크는 쓰지 않는다 -- hf_account 는 whoami() 를 부르므로 스텁으로 막는다.
"""
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import helpers  # noqa: E402
helpers.isolate_state()

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

import mstack.gui.dialogs.hf_account as hf_mod  # noqa: E402

hf_mod.hf_account = lambda *a, **k: ("HF 로그인: tester", "#27ae60")
import mstack.gui.dialogs.convert as convert_mod  # noqa: E402
import mstack.gui.dialogs.upload as upload_mod  # noqa: E402

convert_mod.hf_account = hf_mod.hf_account
upload_mod.hf_account = hf_mod.hf_account

from mstack.gui.dialogs.convert import LerobotConvertDialog  # noqa: E402
from mstack.gui.dialogs.parts import Hdf5FileTable, RepoIdEdit  # noqa: E402
from mstack.gui.dialogs.repack import RepackDialog  # noqa: E402
from mstack.gui.dialogs.upload import HdfUploadDialog  # noqa: E402
from mstack.data.hub_upload_state import record_uploaded  # noqa: E402

REPO = "knu-physical-ai/fr3-tabletop"


def _make(root: Path, n: int = 3) -> list:
    out = []
    for i in range(n):
        p = root / f"scene_{i:03d}.hdf5"
        with h5py.File(p, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["scene_id"] = f"S{i:03d}"
            g = f.create_group("episode_000")
            g.attrs["episode_uid"] = f"EP-S{i:03d}-I000-E000"
            g.create_dataset("actions", data=np.zeros((5, 7), np.float32))
            # 이미지 압축이 재압축 판정의 기준이다 -- 하나만 gzip 으로 둬서
            # 그 파일이 "이미 됨" 으로 빠지는지 본다.
            obs = g.create_group("obs")
            # 판정 기준은 **이미지 압축**이다 (libero_format.hdf5_repack_status).
            # 데이터가 무작위여야 gzip 이 줄이지 못해 "죽은 공간" 비율이 낮게
            # 나온다 -- 0 으로 채우면 gzip 이 거의 0바이트로 줄여서, 파일 크기
            # 대비 저장 바이트가 작아지고 재압축 대상으로 잡힌다.
            rng = np.random.default_rng(i)
            obs.create_dataset(
                "agentview_rgb",
                data=rng.integers(0, 255, (200, 32, 32, 3), dtype=np.uint8),
                compression="gzip" if i == 0 else None)
        out.append(str(p))
    return out


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        paths = _make(root)

        # ---------------------------------------- 1. 셋이 같은 표를 쓴다
        rep = RepackDialog(None, paths)
        up = HdfUploadDialog(None, str(root))
        cv = LerobotConvertDialog(None, str(root))
        for dlg, name in ((rep, "재압축"), (up, "업로드"), (cv, "변환")):
            assert isinstance(dlg.table, Hdf5FileTable), name
            heads = [dlg.table.tree.headerItem().text(c) for c in range(2)]
            assert heads == ["파일", "크기"], (name, heads)
            assert dlg.table.tree.topLevelItemCount() == 3, name
        # 공백으로 이어 붙이던 한 줄 입력칸은 돌아오지 않는다
        for dlg, attr in ((up, "file_edit"), (cv, "files_edit")):
            assert not hasattr(dlg, attr), attr
        print("1. 셋 다 같은 표 · 한 줄 입력칸 stay-gone OK")

        # ------------------------------- 2. 기본 체크가 동작마다 뜻이 있다
        # 재압축: gzip 인 첫 파일은 빠지고 나머지 둘만
        assert len(rep.selected()) == 2, rep.selected()
        # 변환: 전부 (변환은 파일 상태와 무관하다)
        assert len(cv.table.checked_paths()) == 3
        # 업로드: repo 를 모르면 아무것도 안 고른다
        assert up.table.checked_paths() == []
        up.repo_id_edit.set_text(REPO)
        assert len(up.table.checked_paths()) == 3, "신규 파일이 안 골라졌다"
        # 장부에 하나를 '올린 것' 으로 적으면 그것만 빠진다
        record_uploaded(REPO, Path(paths[0]))
        up.repo_id_edit.set_text(REPO + "-x")      # 값이 바뀌어야 다시 그린다
        up.repo_id_edit.set_text(REPO)
        picked = up.table.checked_paths()
        assert paths[0] not in picked and len(picked) == 2, picked
        print("2. 기본 체크 OK (재압축 2/3 · 변환 3/3 · 업로드는 장부대로 2/3)")

        # ------------------------------------------ 3. Repo ID 칸은 접힌다
        box = RepoIdEdit(["a/b"], "")
        box.edit.setPlainText("knu-physical-ai/\n  fr3-tabletop\n")
        # 접힌 값을 그대로 복사해 붙여도 공백이 섞이지 않는다 -- 빈칸으로
        # 이으면 'org/ name' 이 되어 검증에서 튕긴다.
        assert box.text() == "knu-physical-ai/fr3-tabletop", box.text()
        from PyQt6.QtWidgets import QPlainTextEdit
        assert isinstance(box.edit, QPlainTextEdit), "줄바꿈 안 되는 칸으로 돌아갔다"
        assert box.edit.lineWrapMode() == QPlainTextEdit.LineWrapMode.WidgetWidth
        # 접을 이유가 실제로 있는가: 실제 id 가 좁은 패널보다 넓다.
        ident = "knu-physical-ai/fr3-tabletop-lerobot-2026-09"
        fm = box.edit.fontMetrics()
        assert fm.horizontalAdvance(ident) > 240, fm.horizontalAdvance(ident)
        assert box.edit.height() >= 2 * fm.lineSpacing(), box.edit.height()
        # 업로드 페이지(왼쪽 패널)의 칸도 같은 조각이라야 한다 -- 거기가
        # 제일 좁아서 잘림이 제일 잘 보인다.
        page_src = (Path(__file__).resolve().parents[2]
                    / "apps/workspace/features/upload/page.py").read_text(encoding="utf-8")
        assert "RepoIdEdit(" in page_src and "QLineEdit(win._recents_valid_repo" not in page_src
        print("3. Repo ID 줄바꿈 · 공백 제거 OK "
              f"(id {fm.horizontalAdvance(ident)}px > 패널 240px)")

        # ------------------------------------------------ 4. argv 는 그대로
        up.path_in_repo_edit.setText("")
        args = up.build_args()
        assert args[:2] == picked[:2] or set(picked).issubset(set(args)), args
        assert args[-3:] == ["--repo-id", REPO, "--no-private"], args[-3:]
        up.private_check.setChecked(True)
        assert up.build_args()[-1] == "--private"

        cv.repo_id_edit.set_text(REPO + "-lerobot")
        cv.out_root_edit.setCurrentText(str(root / "out"))
        a = cv.build_args()
        assert a[:3] == paths[:3], a[:3]
        assert a[3:] == ["--repo-id", REPO + "-lerobot", "--root",
                         str(root / "out"), "--fps", "20"], a[3:]
        cv.only_success_check.setChecked(True)
        assert cv.build_args()[-1] == "--only-success"
        cv.mode_push_only.setChecked(True)
        assert cv.build_args() == ["--repo-id", REPO + "-lerobot", "--root",
                                   str(root / "out"), "--push-only", "--no-private"]
        assert not cv.table.isEnabled(), "업로드만 모드인데 파일 표가 살아 있다"
        print("4. argv 그대로 OK (업로드 · 변환 · 업로드만)")

    print("test_upload_dialogs OK")


main()
