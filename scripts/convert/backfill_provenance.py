#!/usr/bin/env python3
"""이미 찍힌 scene 파일에 **판번호 칸**을 만들어 준다 (knu-1.x.x → 1.x.(x+1)).

무엇을 채우고 무엇을 비우는지가 이 스크립트의 전부다.

* ``provenance_source`` = ``backfilled <날짜>`` -- **이 값들은 그때 읽은 것이
  아니다.** 수집 시점에 읽은 파일은 ``live`` 라고 적힌다. 이 구분이 없으면
  나중에 이 필드 전체를 믿을 수 없게 된다.
* ``pylibfranka_version`` -- ``--pylibfranka`` 로 준 값만 적는다. 지금 기계에
  깔린 버전을 그냥 적지 않는 이유: 그 버전이 그때도 깔려 있었는지는 설치
  시각으로 **따로 확인해야 하는 사실**이다 (2026-09-13 확인: 0.21.2 가
  2026-07-17 에 설치됐고 첫 수집은 2026-09-06 이라, 그 사이에 다른 버전이
  없었다면 이 데이터셋 전체에 0.21.2 를 적어도 된다).
* ``fr3_system_version`` -- ``--fr3-system`` 으로 준 값만.
* ``collector_commit`` -- **적지 않는다.** 어느 커밋으로 찍혔는지는 파일에도
  이력에도 없다. 타임스탬프로 추측할 수는 있지만 그것은 추론이지 사실이
  아니고, 이 필드의 쓸모는 "정확히 그 코드" 를 되짚는 것뿐이다. 모르면
  비워 두는 편이 훨씬 낫다 -- 그래서 도장도 올리지 않는다(아래).

**도장(dataset_version)은 커밋을 아는 파일에서만 올린다.** 판번호 버전이
요구하는 것은 ``collector_commit`` 과 ``provenance_source`` 둘인데, 여기서는
커밋을 모르므로 요구를 채우지 못한다. 그래서 이 스크립트는 기본적으로
**칸만 채우고 버전은 그대로 둔다** -- 파일이 갖지 않은 것을 가졌다고 주장하지
않는다 (knu-1.1.0 사고와 같은 모양을 피한다).

쓰는 법::

    python scripts/convert/backfill_provenance.py ~/libero_datasets/fr3-tabletop \\
        --pylibfranka 0.21.2 --fr3-system 5.10.0 [--apply]

``--apply`` 없이는 무엇을 쓸지만 보여준다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py  # noqa: E402

from mstack.data.dataset_schema import (  # noqa: E402
    META_FR3_SYSTEM_BUILD,
    META_FR3_SYSTEM_VERSION,
    META_PROVENANCE_SOURCE,
    META_PYLIBFRANKA_VERSION,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="scene_*.hdf5 가 있는 폴더")
    ap.add_argument("--pylibfranka", default="",
                    help="그때 쓰던 바인딩 버전 (예: 0.21.2). 확인한 값만 주세요")
    ap.add_argument("--fr3-system", default="",
                    help="그때의 FR3 시스템 이미지 (예: 5.10.0)")
    ap.add_argument("--fr3-build", default="", help="그 이미지의 빌드 해시들")
    ap.add_argument("--apply", action="store_true", help="실제로 쓴다")
    args = ap.parse_args()

    stamp = f"backfilled {time.strftime('%Y-%m-%d')}"
    write = {META_PROVENANCE_SOURCE: stamp}
    if args.pylibfranka:
        write[META_PYLIBFRANKA_VERSION] = args.pylibfranka
    if args.fr3_system:
        write[META_FR3_SYSTEM_VERSION] = args.fr3_system
    if args.fr3_build:
        write[META_FR3_SYSTEM_BUILD] = args.fr3_build

    files = sorted(Path(args.root).glob("scene_*.hdf5"))
    if not files:
        print(f"{args.root} 에 scene_*.hdf5 가 없습니다.")
        return 1
    print(f"쓸 값: " + " · ".join(f"{k}={v}" for k, v in write.items()))
    print(f"대상 {len(files)}개" + ("" if args.apply else "  (미리보기 -- --apply 로 실제 기록)"))
    changed = skipped = 0
    for path in files:
        mode = "a" if args.apply else "r"
        try:
            with h5py.File(path, mode) as f:
                meta = f.get("metadata")
                if meta is None:
                    print(f"  [건너뜀] {path.name}: metadata 가 없습니다 (legacy?)")
                    skipped += 1
                    continue
                have = str(meta.attrs.get(META_PROVENANCE_SOURCE, ""))
                if have.startswith("live"):
                    # 수집하며 적힌 값을 나중 추정으로 덮지 않는다.
                    print(f"  [건너뜀] {path.name}: 이미 live 로 적혀 있습니다")
                    skipped += 1
                    continue
                ver = str(meta.attrs.get("dataset_version", "?"))
                if args.apply:
                    for k, v in write.items():
                        meta.attrs[k] = v
                print(f"  {path.name:<22} {ver:<10} ← {stamp}")
                changed += 1
        except OSError as e:
            print(f"  [실패] {path.name}: {e}")
            skipped += 1
    print(f"\n{'기록함' if args.apply else '기록할 것'} {changed}개 · 건너뜀 {skipped}개")
    if args.apply:
        print("주의: 파일의 mtime 이 바뀌었습니다 -- 업로드 장부가 전부 "
              "'변경' 으로 봅니다. 다음 업로드에서 다시 읽어 올립니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
