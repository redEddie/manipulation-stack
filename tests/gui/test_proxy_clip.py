"""mp4 프록시 클립 캐시 -- 큐레이션 그리드의 전제.

여기서 못박는 것 중 조용히 깨지기 쉬운 것들:

  * **카메라 이름을 하드코딩하지 않는다.** (T,H,W,C) 인 obs 데이터셋을 전부
    카메라로 본다 -- 카메라가 늘면 늘어난 만큼 클립이 생겨야 한다는 것이
    조작자 요구다 (2026-09-10). 깊이(H,W)나 1차원 신호는 걸리면 안 된다.
  * **부분 파일을 "있음" 으로 착각하지 않는다.** 중간에 죽으면 다음 실행이
    다시 만들어야 한다. 반쯤 쓰인 mp4 가 남으면 그 에피소드는 영영 깨진다.
  * **크기가 짝수다.** yuv420p 크로마 서브샘플링이 홀수 치수를 못 받는다.
  * **삭제 뒤에는 맞춘다, 버리지 않는다.** uid 는 renumber 때 재배정되므로
    (지운 에피소드의 uid 를 뒤가 물려받는다) 지워진 것만 지우면 살아남은
    에피소드가 남의 클립을 물려받는다. 그래서 예전에는 씬 통째를 버렸는데,
    그러면 하나만 지워도 다시 구워야 했다 (2026-09-14). 이제 renumber 의
    uid 대응표로 당겨진 클립의 이름을 옮긴다 (7b). 대응표가 없는 경로(수집
    세션 중 삭제)만 여전히 씬 통째를 버린다 (7).

합성 데이터로 돈다. 조작자의 .hdf5 는 열지 않는다.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
from mstack.data.proxy_clip import (  # noqa: E402
    DEFAULT_CRF,
    camera_keys,
    encode_clip,
    invalidate_scene_proxies,
    proxy_path,
    scene_id_of,
)

TMP = Path(tempfile.mkdtemp(prefix="proxy_"))


# ------------------------------------------------------------------ 경로 규칙
p = proxy_path("EP-S003-I001-E007", "agentview_rgb", TMP)
assert p.name == "EP-S003-I001-E007__agentview_rgb.mp4", p.name
assert scene_id_of("EP-S003-I001-E007") == "S003"
assert scene_id_of("") == "" and scene_id_of("garbage") == ""
print("1. 경로·uid 규칙 OK")


# -------------------------------------------------------------- 카메라 열거
class FakeDS:
    def __init__(self, shape): self.ndim = len(shape); self.shape = shape


obs = {
    "agentview_rgb": FakeDS((100, 480, 640, 3)),
    "eye_in_hand_rgb": FakeDS((100, 480, 640, 3)),
    "wrist2_rgb": FakeDS((100, 240, 320, 3)),      # 카메라가 늘어난 경우
    "agentview_depth": FakeDS((100, 480, 640)),     # 3차원 -- 카메라 아님
    "joint_states": FakeDS((100, 7)),               # 2차원 -- 카메라 아님
    "seg": FakeDS((100, 480, 640, 7)),              # 채널이 1/3/4 가 아님
}
cams = camera_keys(obs)
assert cams == ["agentview_rgb", "eye_in_hand_rgb", "wrist2_rgb"], cams
assert "agentview_depth" not in cams and "joint_states" not in cams
assert "seg" not in cams, "채널 7개짜리를 카메라로 보면 안 된다"
print(f"2. 카메라 열거 (이름 무관, 늘어난 만큼) {cams} OK")


# ------------------------------------------------------------------ 굽기
rng = np.random.default_rng(0)
frames = np.zeros((30, 48, 64, 3), np.uint8)
for i in range(30):                    # 움직이는 사각형 -- 전부 같으면 압축이 0
    frames[i, 10:20, i:i + 8] = 255
out = encode_clip(frames, TMP / "a.mp4", fps=20, scale=0.5, crf=DEFAULT_CRF)
assert out.exists() and out.stat().st_size > 0
assert not (TMP / "a.part").exists(), "부분 파일이 남았다"

import cv2  # noqa: E402

cap = cv2.VideoCapture(str(out))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n = 0
while cap.read()[0]:
    n += 1
cap.release()
assert (w, h) == (32, 24), f"scale 0.5 가 안 먹었다: {w}x{h}"
assert n == 30, f"프레임이 {n}개 (30 이어야 함)"
print(f"3. 굽기·되읽기 {w}x{h} {n}프레임 OK")

# 홀수 치수 -> 짝수로 내림 (yuv420p)
odd = encode_clip(frames, TMP / "odd.mp4", scale=0.7)   # 64*0.7=44.8 -> 44
cap = cv2.VideoCapture(str(odd))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
assert w % 2 == 0 and h % 2 == 0, f"홀수 치수: {w}x{h}"
print(f"4. 짝수 치수 강제 {w}x{h} OK")

# crf 가 실제로 듣는다 (OpenCV VideoWriter 의 PROP_QUALITY 는 무시된다)
lo = encode_clip(frames, TMP / "crf20.mp4", crf=20).stat().st_size
hi = encode_clip(frames, TMP / "crf38.mp4", crf=38).stat().st_size
assert lo > hi, f"crf 가 안 듣는다: 20->{lo}B, 38->{hi}B"
print(f"5. CRF 조절 (20 -> {lo}B, 38 -> {hi}B) OK")

# 빈 입력은 파일을 만들지 않고 거절한다
for bad in (np.zeros((0, 8, 8, 3), np.uint8), np.zeros((5, 8, 8), np.uint8)):
    try:
        encode_clip(bad, TMP / "bad.mp4")
        raise AssertionError("빈/잘못된 입력을 받아들였다")
    except ValueError:
        pass
assert not (TMP / "bad.mp4").exists()
print("6. 잘못된 입력 거절 OK")


# ------------------------------------------------------------------ 무효화
for uid in ("EP-S003-I000-E000", "EP-S003-I001-E000", "EP-S004-I000-E000"):
    for cam in ("agentview_rgb", "eye_in_hand_rgb"):
        proxy_path(uid, cam, TMP).write_bytes(b"x")
removed = invalidate_scene_proxies("S003", TMP)
assert removed == 4, f"지워진 것 {removed}개 (4 여야 함)"
assert proxy_path("EP-S004-I000-E000", "agentview_rgb", TMP).exists(), \
    "다른 scene 을 지웠다"
assert not proxy_path("EP-S003-I001-E000", "eye_in_hand_rgb", TMP).exists()
print(f"7. scene 통째 무효화 ({removed}개, 다른 scene 은 보존) OK")

# ---- 7b. 삭제 뒤에는 맞춘다: 지운 것만 지우고, 당겨진 것은 이름만 옮긴다 ----
# 같은 slot 에서 E1 을 지우면 E2→E1, E3→E2 로 당겨진다. 새 이름이 다른
# 에피소드의 옛 이름이라 한 번에 옮기면 순서에 따라 덮어쓴다 -- 두 단계로 옮기는
# 것이 지켜지는지 **내용**으로 본다.
from mstack.data.proxy_clip import remap_scene_proxies  # noqa: E402

R = Path(tempfile.mkdtemp(prefix="remap_"))
for e in range(4):
    proxy_path(f"EP-S005-I000-E{e:03d}", "agentview_rgb", R).write_bytes(f"clip{e}".encode())
proxy_path("EP-S005-I001-E000", "agentview_rgb", R).write_bytes(b"other-slot")
r = remap_scene_proxies(["EP-S005-I000-E001"],
                        {"EP-S005-I000-E002": "EP-S005-I000-E001",
                         "EP-S005-I000-E003": "EP-S005-I000-E002"}, R)
assert r == {"removed": 1, "renamed": 2}, r
got = {p.name.split("__")[0]: p.read_bytes() for p in R.glob("*.mp4")}
assert got == {"EP-S005-I000-E000": b"clip0", "EP-S005-I000-E001": b"clip2",
               "EP-S005-I000-E002": b"clip3", "EP-S005-I001-E000": b"other-slot"}, got
assert not list(R.glob("*.remap")), "임시 이름이 남았다"
print("7b. 삭제 뒤 맞추기 (지운 1개 삭제 · 당겨진 2개 이름만 옮김 · 다른 slot 보존) OK")

print("\n프록시 클립 캐시 통과")


# --------------------------------------- 데이터셋마다 캐시 자리가 달라야 한다
# episode_uid 는 **한 데이터셋 안에서만** 유일하다. 두 데이터 경로에 같은
# scene_id 가 있으면 uid 가 그대로 겹친다. 2026-09-11 에 실제로 그랬다:
# 데이터 경로를 libero_datasets/sangtae 로 바꾼 화면이 fr3-tabletop 의 클립을
# 틀고 있었고(둘 다 EP-S000-I004-E000 을 갖는다), scene 무효화가
# EP-S000-* 글롭으로 **다른 데이터셋의 클립 120개를 두 번 지웠다.**
from mstack.data.proxy_clip import dataset_tag, proxy_dir_for  # noqa: E402

a = "/data/sets/fr3-tabletop"
b = "/data/sets/sangtae"
assert dataset_tag(a) != dataset_tag(b)
assert proxy_dir_for(a, TMP) != proxy_dir_for(b, TMP)
# 이름이 같아도 경로가 다르면 갈린다
assert dataset_tag("/x/scene") != dataset_tag("/y/scene"), "이름만 쓰면 겹친다"
# 같은 경로는 표기가 달라도 같은 자리 (끝 슬래시·상대경로·~)
assert dataset_tag(a) == dataset_tag(a + "/")
# 사람이 읽을 수 있게 폴더 이름이 앞에 남는다
assert dataset_tag(a).startswith("fr3-tabletop-"), dataset_tag(a)

# 같은 uid 라도 데이터셋이 다르면 다른 파일이다
uid = "EP-S000-I004-E000"
pa = proxy_path(uid, "agentview_rgb", proxy_dir_for(a, TMP))
pb = proxy_path(uid, "agentview_rgb", proxy_dir_for(b, TMP))
assert pa != pb, "데이터셋이 달라도 같은 파일을 가리킨다 -- 남의 영상이 나온다"
pa.parent.mkdir(parents=True, exist_ok=True)
pb.parent.mkdir(parents=True, exist_ok=True)
pa.write_bytes(b"x")
pb.write_bytes(b"y")
# 한쪽 scene 무효화가 다른 데이터셋을 건드리면 안 된다
assert invalidate_scene_proxies("S000", proxy_dir_for(a, TMP)) == 1
assert not pa.exists() and pb.exists(), "다른 데이터셋의 캐시를 지웠다"
print(f"8. 데이터셋별 캐시 분리 ({dataset_tag(a)} / {dataset_tag(b)}) OK")

# ---- 9. 지웠다는 사실이 캐시 옆에 남는다 ----
# 2026-09-11 에 클립 120개가 세 번 사라졌는데 스택 추적·스위트·GUI 로그
# 어느 것으로도 못 잡았고, 이 감사 로그를 넣고서야 범인(테스트를 단독으로
# 돌려 진짜 상태 폴더를 쓴 것)이 드러났다. 그 증거 통로가 살아 있는지 본다.
from mstack.data.proxy_clip import AUDIT_NAME  # noqa: E402

audit = proxy_dir_for(a, TMP) / AUDIT_NAME
assert audit.exists(), f"{AUDIT_NAME} 이 안 쓰였다 -- 다음에 사라지면 또 못 찾는다"
last = audit.read_text(encoding="utf-8").strip().splitlines()[-1]
assert "S000" in last and "1" in last, last
print(f"9. 삭제 감사 기록 OK ({last.strip()[:60]})")

print("\n프록시 클립 캐시 통과 (데이터셋 분리 · 감사 기록 포함)")
