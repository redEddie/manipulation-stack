"""큐레이션용 mp4 프록시 클립 캐시.

**왜 필요한가.** 저장 포맷이 동영상 재생에 부적합하다. 에피소드 하나가
480x640 카메라 2대로 280 MB 이고, gzip 청크가 19프레임 깊이라 한 프레임을
보려 해도 19프레임어치를 푼다. 실측으로 통째 읽기가 1.16초인데 ``d[::2]`` 로
솎아 읽으면 15.9초 -- **솎아 읽기가 14배 느리다.** 12개를 나란히 재생하는
일은 원본으로는 불가능하다.

프록시는 그 반대다: 에피소드당 0.086 MB(카메라 2대), 클립 전체 디코드가
26 ms, 프레임당 0.14 ms 다. 12타일이 프레임마다 1.7 ms 로 20 Hz 예산 50 ms 의
3.4% -- 미리 램에 올릴 필요 없이 열어 두고 흘려보내면 된다.

**왜 전체 화각인가.** 큐레이션이 잡으려는 것은 세 가지다 (조작자, 2026-09-10):
지시문이 바뀌었는데 이전 명령으로 찍힌 것, 녹화를 너무 늦게 끝낸 것, 2~3틱만
찍힌 것. 앞의 둘은 **동작 전체와 끝나는 순간**을 봐야 하므로 LeRobot 이 쓰는
224 정사각 크롭(zoom 1.2)으로는 잘려 나간다. 세 번째는 애초에 영상이 아니라
``num_samples`` 로 걸러야 한다 -- 여기서 다루지 않는다.

**왜 LeRobot 변환용과 겸하지 않는가 -- 닫힌 질문이다 (2026-09-11 조작자 결정).**
겸하는 길은 실제로 있고 아주 빠르다. 클립을 LeRobot 과 똑같이 (224 정사각 +
에피소드별 ``crop_params``, ``libsvtav1 crf30 g=2``) 구워 두면, 변환이 다시
인코딩하지 않고 **패킷만 옮겨 담으면(remux)** 된다. 실측 2.0 ms/에피소드로
재인코딩(3100 ms)의 1550배이고 프레임 수도 정확했다 -- 전체 재빌드의 영상
쪽이 131분에서 5초가 된다.

그런데도 하지 않는다: **한 번 제대로 수집하면 잘 안 바꾸므로, 변환은 늘 새로
인코딩한다.** 겸하면 클립이 학습 데이터의 원본이 되어, ``crop_params`` 가
에피소드별 attrs 인 만큼 나중에 크롭을 바꿨을 때 캐시가 조용히 틀린 영상이
된다. 느린 변환보다 틀린 학습 데이터가 훨씬 나쁘고, 그것을 막으려면 사이드카와
무효화 규칙이 한 벌 더 필요하다 -- 자주 일어나지 않는 일에 그만한 장치를 두지
않는다.

그래서 **이 캐시는 큐레이션 전용이고 학습 데이터에 닿지 않는다.** 틀려도
사람이 영상을 잘못 보는 것으로 끝난다.

덧붙여, 겸했다면 재생도 바뀌어야 했다. LeRobot 기본 코덱 AV1 은 이 기계의
OpenCV 가 디코드하지 못한다 (0프레임, "platform doesn't support hardware
accelerated AV1"). PyAV 로는 되고 12타일에 2.2 ms/틱으로 h264(2.4 ms)와
같지만, 그리드가 PyAV 에 묶이게 된다. h264 는 둘 다로 읽힌다.

참고로 변환이 느린 이유는 인코딩이 아니다. 실측(157프레임, 카메라 2대,
LeRobot 기본 ``libsvtav1 crf30 g=2``): HDF5 읽기 2.24초(74%), 224 크롭·리사이즈
0.49초(16%), 인코딩 0.40초(13%). **읽기가 병목이다.**

**캐시이지 데이터셋이 아니다.** ``episode_uid`` 는 renumber 때 재배정되므로
(지운 에피소드의 uid 를 뒤 에피소드가 물려받는다) 프록시는 본질적으로
파생물이고 수시로 버려져야 한다. 썸네일 캐시와 같은 자리·같은 무효화 규칙을
쓴다 -- 스키마 버전도 닥터 검사도 두지 않는다. 나중에 "원본 없는 사람에게
큐레이션을 넘긴다" 가 실제로 생기면 이 디렉터리를 묶는 것이 승격의 전부다.

**코덱.** PyAV 의 libx264 로 굽는다. OpenCV ``VideoWriter`` 는 이 기계에서
h264 장치를 못 찾고(``h264_v4l2m2m``), mp4v 로 떨어지면 ``VIDEOWRITER_PROP_QUALITY``
가 무시된다 -- crf 10/50/95 가 전부 같은 용량이었다. 그리고 mp4v 는 같은
해상도에서 x264 보다 8.6배 크다 (0.37 vs 0.043 MB). 읽기는 무엇으로도 된다.
"""

from __future__ import annotations

import os
import hashlib
import os
import re
from pathlib import Path

import numpy as np

from mstack.config.paths import state_dir

#: 프록시 디렉터리를 바꾸는 환경 변수. 두 체크아웃(main/dev)이
#: ``MSTACK_PROXY_DIR`` 을 같은 곳으로 가리키면 캐시를 공유한다.
PROXY_DIR_ENV = "MSTACK_PROXY_DIR"

#: 프록시가 쌓이는 곳. 썸네일 캐시(``thumbs/``) 옆이다.
#:
#: **왜 환경 변수로 덮을 수 있는가.** 프록시는 데이터셋의 파생물이지 세션의
#: 파생물이 아니다. 두 체크아웃이 같은 ``~/libero_datasets`` 를 본다면 캐시도
#: 같이 봐야 한다. 그런데 기본값이 ``state_dir()`` 아래라서, dev 아이콘이
#: ``GELLO_STATE_DIR`` 을 다른 경로로 주면 **같은 데이터셋인데도** 이미 구워
#: 둔 클립을 못 보고 다시 굽는다 (2026-09-11 에 dev 로 띄웠더니 480개를 못
#: 봤다). 그래서 공유 캐시 자리를 이 환경 변수로 밀어 넣을 수 있게 했다.
#: 기본값은 그대로 ``state_dir()`` 아래다 -- 인수 테스트가 조작자의 진짜
#: 캐시를 건드리지 않게 테스트마다 격리된 자리를 쓰게 하려는 것이다.
PROXY_DIR = Path(os.environ.get(PROXY_DIR_ENV) or (state_dir() / "proxy"))

#: 긴 변을 이 비율로 줄인다. 0.5 면 640x480 -> 320x240.
#: 3x4 그리드에서 중앙 패널이 1000px 안팎이면 타일이 250x188 이라 이보다 큰
#: 프록시는 버려진다.
DEFAULT_SCALE = 0.5

#: x264 CRF. 낮을수록 좋고 크다. 실측(157프레임 320x240, 카메라 1대):
#: crf 23 -> 0.075 MB, 28 -> 0.043 MB, 32 -> 0.030 MB.
#: 28 은 "무엇을 하는 중인지"가 또렷하되 용량이 무시할 만한 지점이다.
DEFAULT_CRF = 28

#: 이미지 카메라로 볼 obs 데이터셋의 차원. (T, H, W, C)
_IMAGE_NDIM = 4

#: uid 안에 scene_id 가 들어가는 형식 -- ``EP-<scene>-<instr>-E<idx>``.
_UID_RE = re.compile(r"^EP-([^-]+)-")


def dataset_tag(dataset_root) -> str:
    """데이터셋 뿌리를 구분하는 짧은 꼬리표. 캐시의 하위 디렉터리 이름이 된다.

    **왜 필요한가.** ``episode_uid`` 는 한 데이터셋 안에서만 유일하다 --
    ``EP-S000-I004-E000`` 은 어느 데이터셋에나 있을 수 있다. 캐시를 uid 로만
    키하면 데이터 경로를 바꿨을 때 **다른 데이터셋의 영상을 보여준다.**
    2026-09-11 에 실제로 그랬다: 경로가 ``libero_datasets/sangtae`` 로 바뀐
    화면이 ``fr3-tabletop`` 의 클립을 틀고 있었다.

    같은 이유로 무효화도 위험했다. scene 무효화는 ``EP-<scene>-*`` 글롭인데,
    뿌리가 섞이면 **다른 데이터셋의 캐시를 지운다** -- 같은 날 fr3-tabletop 의
    클립 120개가 두 번 사라진 것이 이것이었다.

    이름을 그대로 쓰지 않고 경로 해시를 붙이는 이유는 폴더 이름이 겹칠 수
    있어서다 (``a/scene`` 과 ``b/scene``). 이름은 사람이 읽으라고 남긴다.
    """
    p = Path(dataset_root).expanduser().resolve()
    return f"{p.name}-{hashlib.sha1(str(p).encode()).hexdigest()[:8]}"


def proxy_dir_for(dataset_root, base: Path = None) -> Path:
    """이 데이터셋의 프록시가 쌓이는 곳. 호출부는 이것만 넘기면 된다."""
    return Path(base or PROXY_DIR) / dataset_tag(dataset_root)


def proxy_path(episode_uid: str, obs_key: str,
               proxy_dir: Path = PROXY_DIR) -> Path:
    """``<uid>__<obs_key>.mp4``. 카메라가 늘면 파일이 늘 뿐 규칙은 그대로다."""
    return Path(proxy_dir) / f"{episode_uid}__{obs_key}.mp4"


def camera_keys(obs) -> list:
    """``obs`` 에서 이미지 카메라 키를 이름 순으로.

    이름을 하드코딩하지 않는다 -- 카메라가 늘어나면 늘어난 만큼 프록시가
    생겨야 한다는 것이 조작자 요구다 (2026-09-10). (T, H, W, C) 모양이면
    카메라로 본다. 깊이(H, W)나 1차원 신호는 걸리지 않는다.
    """
    return sorted(k for k in obs
                  if getattr(obs[k], "ndim", 0) == _IMAGE_NDIM
                  and obs[k].shape[-1] in (1, 3, 4))


#: 지운 기록을 남기는 파일. 캐시 디렉터리 안에 둔다.
AUDIT_NAME = ".deleted.log"


def _audit(proxy_dir, what: str, removed: int) -> None:
    """클립을 지웠다는 사실을 캐시 옆에 남긴다.

    **왜 필요한가.** 2026-09-11 에 scene_000 의 클립 120개가 세 번 사라졌는데
    원인을 못 찾았다. 무효화 경로에 스택 추적을 걸고, 인수 스위트를 돌리고,
    GUI 로그를 뒤지고, 데이터셋별로 캐시를 갈라 봐도 잡히지 않았다. 파일을
    지우는 곳은 이 모듈의 두 함수뿐인데도 그랬다.

    지우는 동작이 **아무 흔적도 남기지 않는 것**이 문제였다. 호출자가 GUI 면
    거기 로그가 남지만, 스크립트·테스트·다른 프로세스에서 부르면 어디에도
    안 남는다. 그래서 지우는 쪽에서 직접 남긴다 -- 캐시는 다시 구우면 되지만
    "왜 없어졌나" 를 모르는 채로 두면 계속 다시 굽게 된다.

    실패해도 조용히 넘긴다. 감사 기록을 못 남기는 것이 무효화를 막을 이유는
    없다.
    """
    if not removed:
        return
    import time
    import traceback

    try:
        where = " <- ".join(
            f"{Path(f.filename).name}:{f.lineno}"
            for f in traceback.extract_stack()[-5:-2])
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {what}  "
                f"{removed}개  [{where}]\n")
        with open(Path(proxy_dir) / AUDIT_NAME, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:  # noqa: BLE001
        pass


def invalidate_scene_proxies(scene_id: str, proxy_dir: Path = PROXY_DIR) -> int:
    """해당 scene 의 프록시를 전부 지운다. 반환: 지워진 파일 수.

    썸네일과 같은 이유다 -- renumber 가 slot E번호(=uid 꼬리)를 재배정하므로
    살아남은 에피소드도 예전 uid 의 클립을 물려받을 수 있다. 지워진 것만
    지우는 것으로는 부족하다.
    """
    removed = 0
    for p in Path(proxy_dir).glob(f"EP-{scene_id}-*.mp4"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    _audit(proxy_dir, f"scene {scene_id}", removed)
    return removed


def scene_id_of(episode_uid: str) -> str:
    """uid 에서 scene_id 를 뽑는다. 형식이 아니면 빈 문자열."""
    m = _UID_RE.match(episode_uid or "")
    return m.group(1) if m else ""


def encode_clip(frames, out: Path, fps: float = 20.0,
                scale: float = DEFAULT_SCALE, crf: int = DEFAULT_CRF) -> Path:
    """``frames`` (T, H, W, 3) uint8 을 mp4 로 굽는다. 경로를 돌려준다.

    부분 파일을 남기지 않는다 -- ``.part`` 로 쓰고 다 되면 이름을 바꾼다.
    중간에 죽으면 다음 실행이 "없음"으로 보고 다시 만든다. 반쯤 쓰인 mp4 를
    "있음"으로 착각하면 그 에피소드는 영영 깨진 채로 남는다.
    """
    import av
    import cv2

    frames = np.asarray(frames)
    if frames.ndim != _IMAGE_NDIM or len(frames) == 0:
        raise ValueError(f"프레임 모양이 (T, H, W, C) 가 아니다: {frames.shape}")
    h, w = frames.shape[1:3]
    # x264 는 짝수 치수를 요구한다 (yuv420p 크로마 서브샘플링).
    tw = max(2, int(round(w * scale)) // 2 * 2)
    th = max(2, int(round(h * scale)) // 2 * 2)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    try:
        # format 을 명시한다 -- 확장자가 .part 라 PyAV 가 컨테이너를 추론하지
        # 못하고 "Could not determine output format" 으로 죽는다.
        container = av.open(str(tmp), "w", format="mp4")
        stream = container.add_stream("libx264", rate=int(round(fps)) or 1)
        stream.width, stream.height = tw, th
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        for f in frames:
            if f.shape[-1] == 1:
                f = np.repeat(f, 3, axis=-1)
            elif f.shape[-1] == 4:
                f = f[..., :3]
            if (f.shape[1], f.shape[0]) != (tw, th):
                f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(f), format="rgb24")):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        tmp.replace(out)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return out


def record_fps(default: float = 20.0) -> float:
    """기록 주기 (Hz). 파일에는 없고 station 설정이 정본이다.

    에피소드는 프레임별 시각을 남기지 않는다 -- ``timestamp`` 는 에피소드
    attrs 한 개(녹화 시각)다. 그래서 재생 속도는 설정에서 온다. 이 값이 틀리면
    클립이 빠르거나 느리게 보일 뿐 무엇이 찍혔는지는 그대로다.
    """
    try:
        from mstack.config.station import load_station

        return float(load_station().fps) or default
    except Exception:  # noqa: BLE001 -- 설정을 못 읽어도 프록시는 만들어야 한다
        return default


def episode_uid_of(group) -> str:
    """에피소드 그룹의 uid. 없으면 빈 문자열 (legacy 파일)."""
    v = group.attrs.get("episode_uid", "")
    return v.decode() if isinstance(v, bytes) else str(v or "")


def scan_file(path, proxy_dir: Path = PROXY_DIR) -> dict:
    """``{"episodes", "missing"}`` -- 이 파일의 에피소드 수와 만들어야 할 목록.

    ``missing`` 은 (에피소드, uid, 카메라) 이고 이미 있는 것은 빠진다 --
    진행바가 "남은 일" 을 세야 하고, 중단한 뒤 다시 돌리면 이어서 되어야 한다.

    ``episodes`` 를 따로 세는 이유는 **0 의 뜻이 둘**이기 때문이다: 다 만들어
    놓았거나, 파일이 비었거나. 화면에서 그 둘을 "이미 있음" 한 마디로 뭉치면
    빈 파일이 완료된 것처럼 보인다 (2026-09-10, scene_006 이 실제로 그랬다).

    파일을 읽기 전용으로만 연다 -- 조작자의 데이터다.
    """
    import h5py

    n_eps = 0
    todo = []
    with h5py.File(str(path), "r") as f:
        names = [k for k in f if k.startswith("episode_")]
        if not names and "data" in f:
            names = [f"data/{k}" for k in f["data"]]
        for name in sorted(names):
            n_eps += 1
            grp = f[name]
            uid = episode_uid_of(grp)
            if not uid:
                continue        # legacy 는 uid 가 없어 캐시 키를 만들 수 없다
            obs = grp.get("obs")
            if obs is None:
                continue
            for cam in camera_keys(obs):
                if not proxy_path(uid, cam, proxy_dir).exists():
                    todo.append((name, uid, cam))
    return {"episodes": n_eps, "missing": todo}


def plan_file(path, proxy_dir: Path = PROXY_DIR) -> list:
    """만들어야 할 (에피소드, uid, 카메라) 목록만. :func:`scan_file` 의 축약."""
    return scan_file(path, proxy_dir)["missing"]


def build_file(path, scale: float = DEFAULT_SCALE, crf: int = DEFAULT_CRF,
               fps: "float | None" = None, proxy_dir: Path = PROXY_DIR,
               progress=None, should_stop=None) -> dict:
    """한 파일의 빠진 프록시를 만든다. ``{"made", "skipped", "failed", "bytes"}``.

    ``progress(done, total, label)`` 는 매 클립마다, ``should_stop()`` 이 참이면
    다음 클립을 시작하지 않고 돌아온다 -- 이미 만든 것은 남는다 (다음 실행이
    이어서 만든다).

    **한 에피소드의 카메라를 한 번에 굽는다.** 비싼 것은 읽기(카메라 2대
    2.24초)이고 인코딩은 0.4초뿐이라, 카메라별로 파일을 다시 여는 것이 가장
    나쁘다. 원본은 읽기 전용으로만 연다 -- 조작자의 데이터다.
    """
    import h5py

    todo = plan_file(path, proxy_dir)
    by_ep: dict = {}
    for name, uid, cam in todo:
        by_ep.setdefault((name, uid), []).append(cam)

    fps = float(fps if fps is not None else record_fps())
    made = failed = 0
    error = None
    nbytes = 0
    done = 0
    total = len(todo)
    with h5py.File(str(path), "r") as f:
        for (name, uid), cams in sorted(by_ep.items()):
            if should_stop is not None and should_stop():
                break
            obs = f[name]["obs"]
            for cam in cams:
                if should_stop is not None and should_stop():
                    break
                out = proxy_path(uid, cam, proxy_dir)
                try:
                    encode_clip(obs[cam][:], out, fps=fps, scale=scale, crf=crf)
                    made += 1
                    nbytes += out.stat().st_size
                except Exception as e:  # noqa: BLE001 -- 한 클립이 전체를 멈추지 않는다
                    failed += 1
                    # 첫 이유는 반드시 남긴다. 개수만 세면 "40개 실패" 만 보이고
                    # 무엇이 잘못됐는지는 영영 안 나온다 (2026-09-10 에 실제로
                    # 그래서 한 바퀴를 헛돌았다).
                    if error is None:
                        error = f"{uid} {cam}: {type(e).__name__}: {e}"
                done += 1
                if progress is not None:
                    progress(done, total, f"{Path(path).name} {uid} {cam}")
    return {"made": made, "failed": failed, "bytes": nbytes,
            "total": total, "error": error}


def invalidate_episode_proxies(episode_uid: str,
                               proxy_dir: Path = PROXY_DIR) -> int:
    """에피소드 하나의 클립을 카메라 수와 무관하게 전부 지운다.

    **트림이 이것을 불러야 한다.** 트림은 꼬리 프레임을 지우지만 uid 도
    번호도 바꾸지 않으므로, 무효화하지 않으면 클립이 **이미 없는 프레임을
    계속 보여준다.** 썸네일은 첫 프레임이라 트림에 안 변하지만 클립은 변한다
    -- 둘의 수명이 다르다.

    반환: 지워진 파일 개수.
    """
    if not episode_uid:
        return 0
    removed = 0
    for p in Path(proxy_dir).glob(f"{episode_uid}__*.mp4"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    _audit(proxy_dir, f"episode {episode_uid}", removed)
    return removed


def episode_uid_at(path, group_name: str) -> str:
    """파일 안의 한 에피소드 uid 를 읽는다. 못 읽으면 빈 문자열.

    트림처럼 uid 를 손에 들고 있지 않은 자리에서 쓴다. 읽기 전용으로 연다.
    """
    import h5py

    try:
        with h5py.File(str(path), "r") as f:
            grp = f[group_name] if group_name in f else f["data"][group_name]
            return episode_uid_of(grp)
    except Exception:  # noqa: BLE001 -- 못 읽으면 무효화를 건너뛸 뿐이다
        return ""
