"""삭제 대기 목록 -- 되돌릴 수 없는 조작의 유일한 문.

여기서 못박는 것:

  * 표시는 **토글**이다. 같은 것을 두 번 눌러 쌓이면 확인창의 개수가 거짓말을
    한다.
  * ``by_file`` 이 파일별로 묶는다. 한 파일에서 하나씩 지우고 매번 번호를
    다시 매기면 두 번째부터 이미 밀린 이름을 지운다 -- 실행부가 묶음을 받는
    이유다.
  * 순서를 지킨다. 확인창이 표시한 순서로 보여야 조작자가 자기가 무엇을
    골랐는지 따라갈 수 있다.
  * ``drop_file`` 이 실행 뒤 뒷정리를 한다. 남으면 다음 실행이 이미 없는
    에피소드를 지우려 든다.

로봇도 화면도 파일도 없이 돈다.
"""
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
from mstack.gui.curation_basket import CurationBasket  # noqa: E402

A, B = "/data/scene_000.hdf5", "/data/scene_001.hdf5"
b = CurationBasket()
assert len(b) == 0 and not b.items()

# 토글: 두 번 누르면 풀린다
assert b.toggle((A, "episode_000")) is True
assert len(b) == 1 and (A, "episode_000") in b
assert b.toggle((A, "episode_000")) is False
assert len(b) == 0, "두 번 눌렀는데 남았다"
print("1. 표시 토글 OK")

# 중복 add 가 쌓이지 않는다
b.add((A, "episode_000"))
assert b.add((A, "episode_000")) is False
assert len(b) == 1, "같은 것이 두 번 들어갔다 -- 확인창 개수가 거짓말을 한다"
print("2. 중복 방지 OK")

# 순서 유지
b.clear()
for n in ("episode_005", "episode_001", "episode_003"):
    b.add((A, n))
assert [n for _, n in b.items()] == ["episode_005", "episode_001", "episode_003"], b.items()
print("3. 표시한 순서 유지 OK")

# 파일별 묶기
b.add((B, "episode_002"))
grouped = b.by_file()
assert set(grouped) == {Path(A), Path(B)}, grouped
assert grouped[Path(A)] == ["episode_005", "episode_001", "episode_003"]
assert grouped[Path(B)] == ["episode_002"]
assert b.count_for(A) == 3 and b.count_for(B) == 1
print(f"4. 파일별 묶기 {[(p.name, len(v)) for p, v in grouped.items()]} OK")

# 경로 표기가 달라도 같은 파일이다
assert (Path(A), "episode_005") in b
assert b.count_for(Path(A)) == 3
print("5. 경로 표기 정규화 OK")

# 실행 뒤 뒷정리
assert b.drop_file(A) == 3
assert len(b) == 1 and b.count_for(A) == 0
assert b.count_for(B) == 1, "다른 파일의 표시를 지웠다"
assert b.discard((B, "없는것")) is False
print("6. drop_file 뒷정리 (다른 파일 보존) OK")

print("\n삭제 대기 목록 통과")
