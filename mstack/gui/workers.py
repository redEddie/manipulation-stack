"""Background QThread workers used by the collector GUIs.

Split out of the former mstack/gui/gui_widgets.py so camera/depth/episode work can be
imported without dragging in the full widget collection.
"""

from __future__ import annotations

import os

# Must run before numpy/cv2/h5py/torch(via lerobot) are imported below --
# these each read the env var once at their own C-level init and spin up a
# BLAS/parallel-executor thread pool sized to the CPU core count (measured:
# 39 extra OS threads on this 20-core machine, for a GUI that does no heavy
# matrix math or bulk image processing at all). Setting these first keeps
# that pool at 1 with zero measured functional difference for this script's
# actual workload (light resize/color-convert calls). Don't copy this into
# scripts/convert/convert_libero_to_lerobot.py -- that one genuinely benefits from
# parallel video encoding.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

import h5py
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from mstack.data.dataset_schema import OBS_AGENTVIEW_RGB, OBS_EYE_IN_HAND_RGB
from mstack.gui.i18n import tr

class EpisodeLoadWorker(QThread):
    """Reads one episode's two image streams into RAM, off the UI thread.

    Playback could read frame-by-frame instead -- gzip images cost ~1.7 ms per
    frame, 3% of a 20 Hz tick -- but that puts an h5py read inside the timer
    callback, where a stall from a cold page cache or a competing repack shows
    up directly as a dropped frame. A whole episode is ~47 MB for both cameras
    and loads in ~0.4 s, so it is bought once here and played from memory,
    which also keeps the file handle open for the shortest possible time (the
    collection session may want it back).
    """

    loaded = pyqtSignal(str, str, object, object)  # path, demo, agent, wrist
    failed = pyqtSignal(str)

    def __init__(self, path: str, demo: str) -> None:
        super().__init__()
        self.path = path
        self.demo = demo

    def run(self) -> None:
        try:
            with h5py.File(self.path, "r") as f:
                # legacy 는 data/demo_N, scene(scene-v1)은 루트의 episode_NNN.
                # 에피소드 안쪽 페이로드는 동일해서 그룹만 찾으면 같은 코드다.
                grp = f[self.demo] if self.demo in f else f["data"][self.demo]
                obs = grp["obs"]
                agent = obs[OBS_AGENTVIEW_RGB][:] if OBS_AGENTVIEW_RGB in obs else None
                wrist = obs[OBS_EYE_IN_HAND_RGB][:] if OBS_EYE_IN_HAND_RGB in obs else None
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        if agent is None and wrist is None:
            self.failed.emit(tr("이 에피소드에는 이미지가 없습니다."))
            return
        self.loaded.emit(self.path, self.demo, agent, wrist)



class GalleryLoadWorker(QThread):
    """scene 갤러리(#31)용: 썸네일 캐시 생성 + 에피소드 요약을 UI 스레드
    밖에서 읽는다. 첫 로드에서 에피소드 수만큼 프레임을 읽으므로(예: 50개
    ≈ 수 초) 타이머 콜백에 넣지 않는다. 캐시가 차면 이후에는 목록 조회뿐."""

    loaded = pyqtSignal(str, list, object)  # scene_path, episodes(+thumb), ref_thumb|None
    failed = pyqtSignal(str)

    def __init__(self, scene_path: str) -> None:
        super().__init__()
        self.scene_path = scene_path

    def run(self) -> None:
        try:
            from mstack.scene.scene_format import read_scene_metadata
            from mstack.gui.scene_gallery import build_gallery, reference_thumb

            episodes = build_gallery(self.scene_path)
            sid = read_scene_metadata(Path(self.scene_path)).scene_id
            ref = reference_thumb(self.scene_path, sid)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.loaded.emit(self.scene_path, episodes, ref)



class CameraPreviewWorker(QThread):
    """카메라 노드 구독 미리보기 (2026-08-25 3-프로세스 분리).

    장치를 직접 열지 않는다 -- mstack/comm/camera_node.py 가 카메라를 독점 소유하고
    이 스레드는 ZMQ 로 최신 프레임만 받아온다. 그래서 수집 worker 와 장치를
    두고 경합하지 않고(예전 'device busy'), 멈추는 것도 소켓 닫기라 즉시다.
    노드가 자가복구 중이면 죽지 않고 기다린다 (5초에 한 번 상태만 알림).
    """

    frame_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, role: str, serial: str) -> None:
        super().__init__()
        self.role = role
        self.serial = serial
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        import time as _time

        from mstack.comm.camera_client import NodeCamera

        cam = NodeCamera(self.serial)
        try:
            cam.connect(warmup_s=6.0)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"{type(e).__name__}: {e}")
            return
        last_err = 0.0
        try:
            while self._running:
                try:
                    frame = cam.read_latest(max_age_ms=2000)
                except Exception as e:  # noqa: BLE001
                    if not self._running:
                        break
                    now = _time.monotonic()
                    if now - last_err > 5.0:
                        last_err = now
                        self.error.emit(f"{type(e).__name__}: {e}")
                    self.msleep(300)   # 노드 자가복구를 기다린다
                    continue
                if self._running:
                    self.frame_ready.emit(frame)
                self.msleep(33)  # preview only needs ~30 fps
        finally:
            cam.disconnect()



class DepthCloudWorker(QThread):
    """카메라 노드에서 depth 를 받아 Depth/Point Cloud 탭 그림을 만든다.

    (2026-08-25 3-프로세스 분리) 장치를 직접 열지 않는다:
    - "depth" 모드: 노드의 raw depth 토픽 구독 -> (H,W) float32 m
    - "cloud" 모드: 노드 제어 채널에 정렬(depth->color) 프레임 1쌍을 요청해
      내부 파라미터로 역투영 -- 표시 전용이라 2.5Hz 요청이면 충분하고,
      기록 경로(비정렬 raw)와 완전히 분리된다.
    예전처럼 미리보기와 카메라를 뺏고 빼앗길 일이 없다.
    """

    cloud_ready = pyqtSignal(object, object)   # points (N,3) f32, colors (N,3) u8
    depth_ready = pyqtSignal(object)           # (H,W) float32 m -- Depth 탭용 원해상도
    error = pyqtSignal(str)

    def __init__(self, role: str, serial: str = "", stride: int = 3,
                 interval_ms: int = 400, mode: str = "cloud") -> None:
        super().__init__()
        self.role = role
        self.serial = serial
        self.stride = stride
        self.interval_ms = interval_ms
        # "cloud" | "depth" -- 보이는 탭에 필요한 계산만 한다. GUI 가 탭 전환
        # 때 바꾼다 (단순 속성 읽기라 락 불필요).
        self.mode = mode
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:  # noqa: C901
        import time as _time

        from mstack.comm.camera_client import NodeCamera, fetch_aligned

        cam = NodeCamera(self.serial)
        try:
            cam.connect(warmup_s=6.0)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"카메라 열기 실패: {e}")
            return
        last_err = 0.0
        try:
            while self._running:
                try:
                    if self.mode == "depth":
                        z16 = cam.read_latest_depth(max_age_ms=2000)
                        scale = cam.depth_scale or 0.001
                        z_full = z16[:, :, 0].astype(np.float32) * scale
                        if self._running:
                            self.depth_ready.emit(z_full)
                    else:
                        al = fetch_aligned(self.serial, ctl_port=cam.ctl_port,
                                           host=cam.host)
                        if al is None:
                            raise TimeoutError("정렬 프레임 응답 없음 "
                                               "(노드/카메라 복구 중?)")
                        z = al["z"][::self.stride, ::self.stride]
                        rgb = al["rgb"][::self.stride, ::self.stride]
                        intr = al["intrinsics"]
                        h, w = al["z"].shape
                        vs, us = np.mgrid[0:h:self.stride,
                                          0:w:self.stride].astype(np.float32)
                        valid = (z > 0.05) & (z < 2.0)
                        zf = z[valid]
                        pts = np.stack(
                            [(us[valid] - intr["ppx"]) * zf / intr["fx"],
                             (vs[valid] - intr["ppy"]) * zf / intr["fy"],
                             zf], axis=1).astype(np.float32)
                        if self._running:
                            self.cloud_ready.emit(pts, rgb[valid])
                except Exception as e:  # noqa: BLE001
                    if not self._running:
                        break
                    now = _time.monotonic()
                    if now - last_err > 5.0:
                        last_err = now
                        self.error.emit(f"{type(e).__name__}: {e}")
                    self.msleep(400)
                    continue
                self.msleep(self.interval_ms)
        finally:
            cam.disconnect()




class ProxyBuildWorker(QThread):
    """큐레이션용 mp4 프록시를 UI 스레드 밖에서 굽는다.

    왜 QProcess 가 아니라 QThread 인가. 인코딩은 PyAV 안에서 일어나고
    (``mstack.data.proxy_clip``) PyAV 는 GIL 을 놓는다 -- 굽는 동안 화면이
    멎지 않는다. 별도 프로세스로 띄우면 진행 상황을 다시 파이프로 실어
    날라야 하는데 얻는 것이 없다. 원시 상태 로거와는 사정이 다르다: 저쪽은
    1 kHz 제어 루프가 GIL 전환에 걸려서 프로세스를 나눴다.

    원본 .hdf5 는 읽기 전용으로만 연다. 파일당 한 번 열고 그 안의 모든
    에피소드를 처리한다 -- 비싼 것은 읽기(카메라 2대 2.24초)이고 인코딩은
    0.4초뿐이라, 파일을 다시 여는 것이 가장 나쁘다.
    """

    #: (지금까지, 전체, 무엇을)
    progress = pyqtSignal(int, int, str)
    #: 파일 하나가 끝날 때마다 -- {"path", "made", "failed", "bytes", "error"}
    file_done = pyqtSignal(dict)
    #: 전부 끝났을 때 -- {"made", "failed", "bytes", "stopped"}
    done = pyqtSignal(dict)

    def __init__(self, paths, scale: float, crf: int, proxy_dir=None,
                 parent=None) -> None:
        super().__init__(parent)
        self.paths = [str(p) for p in paths]
        self.proxy_dir = proxy_dir
        self.scale = float(scale)
        self.crf = int(crf)
        self._stop = False

    def stop(self) -> None:
        """다음 클립부터 멈춘다. 이미 구운 것은 남는다 -- 다시 돌리면 이어서
        만든다 (있는 것은 건너뛰므로)."""
        self._stop = True

    def run(self) -> None:
        from mstack.data.proxy_clip import PROXY_DIR, build_file, plan_file

        # 전체 개수를 먼저 센다. 진행바가 "남은 일" 을 알아야 하고, plan_file
        # 은 이미지 없이 attrs 만 읽으므로 싸다.
        per_file = {}
        total = 0
        for p in self.paths:
            try:
                n = len(plan_file(p, self.proxy_dir) if self.proxy_dir else plan_file(p))
            except Exception:  # noqa: BLE001 -- 못 여는 파일은 0개로 두고 넘어간다
                n = 0
            per_file[p] = n
            total += n

        made = failed = 0
        nbytes = 0
        base = 0
        for p in self.paths:
            if self._stop:
                break
            if per_file[p] == 0:
                continue

            def _prog(done_in_file, _total_in_file, label, _base=base):
                self.progress.emit(_base + done_in_file, total, label)

            try:
                r = build_file(p, scale=self.scale, crf=self.crf,
                               proxy_dir=self.proxy_dir or PROXY_DIR,
                               progress=_prog, should_stop=lambda: self._stop)
            except Exception as e:  # noqa: BLE001 -- 한 파일이 전체를 멈추지 않는다
                r = {"made": 0, "failed": per_file[p], "bytes": 0,
                     "total": per_file[p],
                     "error": f"{type(e).__name__}: {e}"}
            base += per_file[p]
            made += r["made"]
            failed += r["failed"]
            nbytes += r["bytes"]
            self.file_done.emit({"path": p, **r})
        self.done.emit({"made": made, "failed": failed, "bytes": nbytes,
                        "stopped": self._stop})
