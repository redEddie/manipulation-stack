"""데이터셋 텍스트 색인 생성 — scene HDF5 를 읽어 3종 TSV 를 만든다.

    scenes.tsv     scene 재고 (metadata 그룹 attrs)
    cells.tsv      (scene_id, instruction_id) 칸 단위 요약 -- train/eval
                   subset 을 짜는 파일. 에피소드 단위로 쪼개면 같은 칸의
                   에피소드가 양쪽에 들어가 누출이 되니 칸 단위가 끝단위다.
    episodes.tsv   에피소드 상세 (에피소드 그룹 attrs + actions.shape[0])

사용:
    python scripts/analyze/build_dataset_index.py <dataset_root> [--out <dir>]

``--out`` 이 없으면 dataset_root 에 쓴다. HDF5 는 읽기만 한다 -- attrs 만
읽으므로 관측 청크를 건드리지 않고 대용량 파일도 빠르게 훑는다. 깨진 파일은
걸러내고 stderr 에 한 줄 남긴다 (나머지는 만든다).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mstack.scene.dataset_index import build_dataset_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_root", type=Path,
                    help="scene_*.hdf5 가 있는 데이터셋 루트")
    ap.add_argument("--out", type=Path, default=None,
                    help="TSV 출력 디렉터리 (기본: dataset_root)")
    args = ap.parse_args()
    result = build_dataset_index(args.dataset_root, out_dir=args.out)
    for kind, path in sorted(result.paths.items()):
        print(f"{kind}.tsv: {len(getattr(result, kind))}행 -> {path}")
    if result.errors:
        print(f"주의: {len(result.errors)}개 파일을 건너뛰었다 (위 stderr 참조)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
