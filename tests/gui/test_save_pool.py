"""압축 워커 풀과 저장 게이트 인수 시험.

풀은 성능 최적화지만, **출력이 달라지면 최적화가 아니라 새 포맷이다.** 그래서
여기서 가장 중요한 단언은 속도가 아니라 "풀 경로와 직렬 경로가 같은 파일을
만든다" 이다. HDF5 의 gzip 필터가 기대하는 것이 정확히 zlib 스트림이라
성립하는 성질이고, 그 전제가 깨지면 여기서 먼저 걸린다.

게이트 쪽은 한 가지만 본다: 앞 에피소드가 디스크에 안 닿았으면 다음 녹화가
시작되지 않는다. 버퍼가 비압축이라 에피소드당 최대 0.74 GB 이고, 겹쳐 쌓이면
디스크보다 메모리가 먼저 무너진다.
"""
import os
import sys
import tempfile
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)
sys.argv = ["t"]

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402

from mstack.data.libero_format import (  # noqa: E402
    _POOL_MIN_FRAMES,
    gzip4,
    write_image_dataset,
)

TMP = tempfile.mkdtemp(prefix="savepool-")
# 압축률이 현실적이어야 의미가 있다 -- 무작위 노이즈는 안 줄고, 상수는 과하게
# 준다. 부드러운 기울기 + 약한 잡음이 카메라 프레임에 가깝다.
rng = np.random.default_rng(0)
base = np.linspace(0, 255, 64 * 64 * 3, dtype=np.float64).reshape(64, 64, 3)
FRAMES = np.stack([
    np.clip(base + rng.normal(0, 3, base.shape) + i, 0, 255).astype(np.uint8)
    for i in range(12)
])


def write(pool):
    p = os.path.join(TMP, f"w{id(pool)}-{rng.integers(1 << 30)}.h5")
    with h5py.File(p, "w") as f:
        write_image_dataset(f, "x", FRAMES, pool)
    with h5py.File(p, "r") as f:
        d = f["x"]
        out = (d[:].copy(), d.compression, d.compression_opts, d.chunks)
    return p, out


def main() -> None:
    # ------------------------------------------------ 1. 직렬 경로의 모양
    p0, (data0, comp, opts, chunks) = write(None)
    assert np.array_equal(data0, FRAMES), "직렬 경로가 내용을 바꿨다"
    assert comp == "gzip" and opts == 4, (comp, opts)
    assert chunks == (1,) + FRAMES.shape[1:], f"프레임 청크가 아니다: {chunks}"
    n0 = os.path.getsize(p0)
    print(f"1. 직렬: gzip-4, 청크 {chunks}, {n0/1e3:.1f} KB OK")

    # ------------------------------------------------ 2. 풀 경로가 같은 파일
    pool = ProcessPoolExecutor(2, mp_context=get_context("spawn"))
    try:
        list(pool.map(gzip4, [b""] * 2))          # 예열
        p1, (data1, comp1, opts1, chunks1) = write(pool)
        n1 = os.path.getsize(p1)
        assert np.array_equal(data1, FRAMES), "풀 경로가 내용을 바꿨다"
        assert (comp1, opts1, chunks1) == (comp, opts, chunks), "코덱/청크가 갈렸다"
        assert n1 == n0, (
            f"풀 경로 파일이 직렬과 다르다: {n1} vs {n0} 바이트 -- "
            "최적화가 아니라 다른 포맷을 쓰고 있다는 뜻이다")
        print(f"2. 풀 경로가 **바이트까지 동일한 파일** OK ({n1/1e3:.1f} KB)")
    finally:
        pool.shutdown()

    # ------------------------------------------------ 3. 죽은 풀이면 직렬 폴백
    # 압축 하나 때문에 조작자의 에피소드를 잃는 쪽이 훨씬 나쁘다. 그리고
    # 데이터셋을 만든 뒤 지우는 폴백이면 HDF5 가 그 자리를 회수하지 않아
    # 파일이 부푼다 -- 그래서 압축을 먼저 하고 데이터셋을 나중에 만든다.
    dead = ProcessPoolExecutor(2, mp_context=get_context("spawn"))
    dead.shutdown()
    p2, (data2, *_) = write(dead)
    n2 = os.path.getsize(p2)
    assert np.array_equal(data2, FRAMES), "폴백이 내용을 바꿨다"
    assert n2 == n0, (
        f"폴백 파일이 직렬과 다르다: {n2} vs {n0} 바이트 -- 만들었다 지운 "
        "자리를 HDF5 가 회수하지 않은 것이다")
    print("3. 죽은 풀 -> 직렬 폴백, 파일은 직렬과 동일 OK")

    # ------------------------------------------------ 4. 짧은 에피소드는 직렬
    short = FRAMES[:_POOL_MIN_FRAMES - 1]
    p3 = os.path.join(TMP, "short.h5")
    # 절대 부르면 안 되는 풀을 준다 -- 부르면 여기서 터진다.
    class Boom:
        def map(self, *a, **k):
            raise AssertionError(
                f"{_POOL_MIN_FRAMES}프레임 미만인데 풀을 깨웠다 -- "
                "프레임당 IPC 왕복이 있어 짧은 에피소드는 직렬이 빠르다")
    with h5py.File(p3, "w") as f:
        write_image_dataset(f, "x", short, Boom())
    with h5py.File(p3, "r") as f:
        assert np.array_equal(f["x"][:], short)
    print(f"4. {_POOL_MIN_FRAMES}프레임 미만은 풀을 안 쓴다 OK")

    # ------------------------------------------------ 5. 게이트: pending()
    from mstack.collect.worker import EpisodeSaver  # noqa: E402
    s = EpisodeSaver()
    assert s.pending() == 0, "아무것도 안 넣었는데 대기가 있다"
    s.enqueue_save(object(), True)
    assert s.pending() == 1, s.pending()
    s.enqueue_save(object(), True)
    assert s.pending() == 2, s.pending()
    # 쓰는 중인 한 개도 세어야 한다 -- 큐에서 꺼낸 순간 qsize 는 줄지만
    # 그 에피소드는 아직 디스크에 없다.
    s._busy = True
    s._q.get(); s._q.get()
    assert s.pending() == 1, (
        f"쓰는 중인 에피소드를 안 세고 있다 ({s.pending()}) -- 게이트가 "
        "저장이 끝나기 전에 다음 녹화를 열어 준다")
    s._busy = False
    assert s.pending() == 0
    print("5. saver.pending() 이 큐 + 쓰는 중 을 함께 센다 OK")

    print("\n압축 풀 + 저장 게이트 인수 통과")


if __name__ == "__main__":
    main()
