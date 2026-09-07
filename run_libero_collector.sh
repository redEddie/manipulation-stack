#!/bin/bash
# 마법사를 건너뛰고 수집 워크스페이스를 바로 연다. GUI 안에 로봇 노드
# (launch_nodes.py, pylibfranka venv) 를 띄우고 내리는 버튼이 있으므로
# 바탕화면 바로가기로는 이것 하나면 된다.
#
# venv 는 MSTACK_VENV 로 바꿀 수 있다.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source "${MSTACK_VENV:-$HOME/lerobot-venv}/bin/activate"
exec python apps/collect_workspace.py
