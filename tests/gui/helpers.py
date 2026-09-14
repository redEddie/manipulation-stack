"""Shared helpers for GUI tests."""

import os
import tempfile
import time

from PyQt6.QtWidgets import QApplication


def isolate_state() -> str:
    """상태 디렉터리를 임시 폴더로 돌린다. **mstack 을 임포트하기 전에 부른다.**

    ``run_all.sh`` 가 GELLO_STATE_DIR 을 임시 폴더로 주지만, 테스트 하나를
    손으로 돌릴 때는 그것이 없다. 그러면 조작자의 진짜 상태 폴더를 쓰게 되고,
    파일을 지우는 테스트는 **실제 캐시를 지운다.**

    2026-09-11 에 실제로 그랬다: test_scene_edit 를 디버깅하느라 단독으로
    돌렸는데, 그 테스트가 만드는 합성 씬의 scene_id 가 하필 "S000" 이라
    fr3-tabletop 의 S000 프록시 클립 120개가 지워졌다. 네 번 반복됐고 원인을
    한참 못 찾았다 (감사 로그를 넣고서야 잡혔다).

    경로 상수들이 임포트 시점에 state_dir() 로 정해지므로 **임포트보다 먼저**
    불러야 한다. 이미 설정돼 있으면 그대로 둔다 (스위트가 준 것을 덮지 않게).
    """
    # 실험실 리그에 묻지 않는다. 로봇이 켜졌는지에 따라 결과가 달라지는
    # 테스트는 테스트가 아니다 (2026-09-14: 닥터 테스트가 실제로 그랬다).
    os.environ.setdefault("MSTACK_NO_RIG_QUERY", "1")
    cur = os.environ.get("GELLO_STATE_DIR")
    if not cur:
        cur = tempfile.mkdtemp(prefix="gui-test-state-")
        os.environ["GELLO_STATE_DIR"] = cur
    return cur


def _wait_recs(dlg, timeout_iters: int = 600, sleep_sec: float = 0.01):
    """RecommendDialog 의 백그라운드 추천 계산이 끝날 때까지 기다린다."""
    app = QApplication.instance()
    for _ in range(timeout_iters):
        if dlg._worker is None or not dlg._worker.isRunning():
            break
        app.processEvents()
        time.sleep(sleep_sec)
    else:
        raise AssertionError("RecommendDialog worker timeout")
    app.processEvents()
