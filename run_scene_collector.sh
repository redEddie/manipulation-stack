#!/bin/bash
# Scene 기반 수집 GUI 런처. 바탕화면 아이콘이 이것을 부른다.
#
# 이 스크립트는 자기가 놓인 체크아웃에서 돈다 (아래 cd 참고). 그래서 한 벌로
# 수집용/개발용 아이콘을 모두 감당한다 -- 바탕화면에 아이콘이 둘이면 하나는
# 실제 수집용 체크아웃(main), 다른 하나는 병합 전 확인용(dev)을 가리키는 것이다.
#
# 실행 전에 자동으로 git pull 을 시도한다 -- "아이콘은 눌렀는데 옛 코드가
# 돌고 있었다" 사고를 원천 차단하기 위해서다. 오프라인이거나 pull 이
# 실패하면 경고만 찍고 현재 코드로 그냥 실행한다 (수집을 막는 쪽이 더
# 나쁘다). 이미 떠 있는 GUI 는 pull 의 영향을 받지 않는다 -- 재시작해야
# 새 코드다.
#
# GUI 를 돌릴 venv 는 MSTACK_VENV 로 바꿀 수 있다 (기본값은 이 기계의 것).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
git pull --ff-only 2>/dev/null || echo "[scene-collector] git pull 실패 (오프라인?) -- 현재 코드로 실행합니다"
source "${MSTACK_VENV:-$HOME/lerobot-venv}/bin/activate"
# 마법사를 건너뛰려면: python apps/collect_workspace.py
exec python apps/collect_launcher.py
