"""fps 배선 인수 테스트 (2026-09-22).

배경: --fps 기본값 20 이 변환기에 하드코딩돼 있었고, 자동 파이프라인 몇
곳(LeRobot 자동 / 이어붙이기 / 전체 처리 다이얼로그)은 --fps 를 아예 안
넘겨 그 기본값에 맡겼다. 그리고 --resume 은 기존 Hub 데이터셋의
meta/info.json 에서 fps 를 물려받으면서 넘어온 --fps 를 조용히 무시했다.
fps 는 메타데이터가 아니라 모든 프레임의 timestamp 와 비디오 스트림 속도를
만드는 값이다.

여기서 못박는 것:

1. 변환 argv 를 만드는 세 자리(UploadOps 의 자동/이어붙이기 단계,
   PipelineDialog.steps, 변환 대화상자 build_args) 가 스테이션 설정
   (load_station().fps) 의 fps 를 **명시로** 넘긴다.
2. 변환기의 --resume 검사(_check_resume_fps) 가 기존 fps 와 다른 --fps 를
   SystemExit 로 거부하고, 같으면 통과한다.
3. 변환 대화상자의 FPS 칸이 스테이션 값으로 채워지고, 칸을 비워도 스테이션 값이 나간다.

로봇도 카메라도 필요 없다 (offscreen). 네트워크도 쓰지 않는다 -- hf_account
는 whoami() 를 부르므로 스텁으로 막는다.
"""
import os
import sys
import tempfile
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import helpers  # noqa: E402
helpers.isolate_state()

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from mstack.config.station import load_station  # noqa: E402

STATION_FPS = str(load_station().fps)

from apps.workspace.constants import (  # noqa: E402
    CONVERT_SCRIPT,
    REPACK_SCRIPT,
    UPLOAD_SCRIPT,
)
from apps.workspace.features.upload.ops import lerobot_convert_step  # noqa: E402
import apps.workspace.features.upload.pipeline_dialog as pipe_mod  # noqa: E402
from apps.workspace.features.upload.pipeline_dialog import PipelineDialog  # noqa: E402
import mstack.gui.dialogs.convert as convert_mod  # noqa: E402


def _fps_of(args: list) -> str:
    assert "--fps" in args, f"--fps 가 argv 에 없다: {args}"
    return args[args.index("--fps") + 1]


# ---------------- 1. UploadOps 의 자동/이어붙이기 변환 단계는 fps 를 명시한다
paths = ["/data/a_demo.hdf5", "/data/b_demo.hdf5"]
for resume in (False, True):
    step = lerobot_convert_step(paths, "org/ds", "/out", resume)
    a = step["args"]
    assert a[0] == CONVERT_SCRIPT, a
    assert a[1:3] == paths, a
    assert _fps_of(a) == STATION_FPS, (a, STATION_FPS)
    assert ("--resume" in a) is resume, a
    assert step["clear_root"] == "/out", step
auto = lerobot_convert_step(paths, "org/ds", "/out", resume=False)
res = lerobot_convert_step(paths, "org/ds", "/out", resume=True)
assert "전체 재빌드" in auto["name"] and "이어붙이기" in res["name"], \
    (auto["name"], res["name"])
print(f"1. UploadOps 자동/이어붙이기 변환 단계가 --fps {STATION_FPS} 명시 OK")


# ------------- 2. PipelineDialog.steps() 의 변환 단계도 fps 를 명시한다
# hf_account 는 whoami() 를 부른다 -- 네트워크 없이 돌아야 하므로 스텁으로 막는다.
pipe_mod.hf_account = lambda *a, **k: ("HF 로그인: tester", "#27ae60")

tmp = Path(tempfile.mkdtemp(prefix="fps-pipe-"))
fa = tmp / "scene_000.hdf5"
fa.write_bytes(b"x")     # 단계를 만드는 데 내용은 안 읽는다 -- 있기만 하면 된다
plan = {"action": "up_to_date", "rows": [], "ambiguous": [],
        "local_total": 0, "hub_total": 0, "paths": [str(fa)]}
# scripts 는 GUI 가 넘겨주는 실제 상수를 그대로 쓴다 -- 사본을 적어 두면 GUI 쪽
# 경로가 바뀌어도 테스트는 통과해 버린다 (test_hub_upload_state 와 같은 규칙).
dlg = PipelineDialog(None, str(tmp), plan, "org/lr", "org/hr", str(tmp / "lr"),
                     scripts={"repack": REPACK_SCRIPT,
                              "convert": CONVERT_SCRIPT,
                              "upload": UPLOAD_SCRIPT})
dlg.repack_check.setChecked(False)
dlg.hdf5_check.setChecked(False)
for resume in (False, True):
    dlg.mode_rebuild.setChecked(not resume)
    dlg.mode_resume.setChecked(resume)
    steps = dlg.steps()
    conv = [s for s in steps if s.get("name", "").startswith("LeRobot 변환")]
    assert len(conv) == 1, [s.get("name") for s in steps]
    a = conv[0]["args"]
    assert a[0] == CONVERT_SCRIPT, a
    assert _fps_of(a) == STATION_FPS, (a, STATION_FPS)
    assert ("--resume" in a) is resume, a
print(f"2. PipelineDialog.steps() 변환 단계가 --fps {STATION_FPS} 명시 OK "
      "(재빌드/이어붙이기 둘 다)")


# ---- 3. 변환 대화상자: FPS 칸은 스테이션 값, argv 도 그 값 (빈칸 폴백에도)
convert_mod.hf_account = lambda *a, **k: ("HF 로그인: tester", "#27ae60")


def _make_hdf5(root: Path, n: int = 1) -> list:
    """test_upload_dialogs._make 와 같은 최소 scene 파일 -- 표 채우기용."""
    out = []
    for i in range(n):
        p = root / f"scene_{i:03d}.hdf5"
        with h5py.File(p, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["scene_id"] = f"S{i:03d}"
            g = f.create_group("episode_000")
            g.attrs["episode_uid"] = f"EP-S{i:03d}-I000-E000"
            g.create_dataset("actions", data=np.zeros((5, 7), np.float32))
            rng = np.random.default_rng(i)
            g.create_dataset(
                "obs/agentview_rgb",
                data=rng.integers(0, 255, (200, 32, 32, 3), dtype=np.uint8))
        out.append(str(p))
    return out


with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    made = _make_hdf5(root)
    cv = convert_mod.LerobotConvertDialog(None, str(root))
    assert cv.fps_edit.text() == STATION_FPS, \
        f"FPS 칸이 스테이션 값({STATION_FPS})이 아니다: {cv.fps_edit.text()!r}"
    cv.repo_id_edit.set_text("org/x")
    cv.out_root_edit.setCurrentText(str(root / "out"))
    assert _fps_of(cv.build_args()) == STATION_FPS
    # 조작자가 칸을 지워도(공백) 스테이션 값이 나간다 -- 상수 20 폴백은 없다.
    cv.fps_edit.setText("")
    assert _fps_of(cv.build_args()) == STATION_FPS
print(f"3. 변환 대화상자 FPS 칸({STATION_FPS}) + 빈칸 폴백 OK")


# ----------- 4. 변환기 --resume 검사: fps 불일치는 거부, 일치는 통과
import scripts.convert.convert_libero_to_lerobot as conv  # noqa: E402

conv._check_resume_fps(int(STATION_FPS), int(STATION_FPS))   # 같으면 조용히 통과
print(f"4a. --resume fps 일치 통과 OK ({STATION_FPS})")
try:
    conv._check_resume_fps(20, 30)
except SystemExit as e:
    msg = str(e)
    assert "20" in msg and "30" in msg and "fps" in msg, msg
    print("4b. --resume fps 불일치 SystemExit OK")
    print("    " + msg.replace("\n", "\n    "))
else:
    raise AssertionError("fps 가 다른데 --resume 검사가 거부하지 않았다")

print("\nfps 배선 인수 통과")
