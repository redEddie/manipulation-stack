"""파생 캐시를 버리는 **화면 쪽 문** -- 로그와 예외 처리를 한 벌만 둔다.

무효화 자체는 `mstack.data.proxy_clip` 이 한다 (거기가 "캐시가 늘면 한 곳만
늘린다" 는 단일 문이다). 여기 있는 것은 그것을 부르는 **화면의 규약**이다:

* 캐시 정리 실패는 부른 쪽의 실패가 아니다. 삭제도 트림도 이미 성공했고,
  남은 캐시는 다음 굽기에서 덮인다 -- 그래서 삼키고 로그로만 남긴다.
* 로그 문구를 한 벌로 둔다. 세 경로(목록 삭제·세션 중 삭제·트림)가 각자
  같은 try/except/log 를 베껴 쓰고 있었다 (kimi 구조 감사, 2026-09-12).
  캐시 종류가 늘면 여기 한 곳만 고친다.
"""

from __future__ import annotations

from mstack.data.proxy_clip import invalidate_episode_proxies, invalidate_scene_caches


def drop_scene_caches(win, scene_id, label: str = "") -> None:
    """씬 하나의 파생 캐시를 전부 버린다 (에피소드 번호가 재배정된 뒤).

    ``scene_id`` 는 값이거나 **값을 구해 오는 무인자 함수**다 -- 파일에서
    읽어야 하는 경로가 있고, 그 읽기도 같은 try 안에 들어야 한다.
    """
    try:
        sid = scene_id() if callable(scene_id) else scene_id
        if not sid:
            return
        n = invalidate_scene_caches(sid, win.dataset_ops.dataset_root())["proxies"]
        if n:
            win.log(f"[캐시] {label or sid}: 프록시 {n}개 무효화")
    except Exception as e:  # noqa: BLE001 -- 캐시 정리 실패는 조작의 실패가 아니다
        win.log(f"[캐시 정리 실패] {label or scene_id}: {e}")


def drop_episode_caches(win, uid, label: str = "") -> None:
    """에피소드 **하나**의 프록시만. 트림은 uid 도 번호도 바꾸지 않으므로
    씬 통째를 버릴 이유가 없다."""
    try:
        if not uid:
            return
        n = invalidate_episode_proxies(uid, win.gallery_ops.proxy_dir())
        if n:
            win.log(f"[캐시] {label or uid}: 프록시 {n}개 무효화")
    except Exception as e:  # noqa: BLE001
        win.log(f"[캐시 정리 실패] {label or uid}: {e}")
