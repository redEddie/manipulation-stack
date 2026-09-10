"""리더를 놓쳤는지 판정한다 -- 텔레옵 안전층의 판단 부분.

**무엇을 대신하는가.** 2026-09-10 이전에는 리더를 놓치거나 난폭하게 흔들면
libfranka 가 명령을 거부하며 acceleration_discontinuity 반사로 제어 루프를
죽였다. 설계된 안전장치가 아니라 부작용이었지만 실제로 팔을 세우는 유일한
것이었고, v/a 를 올리고 v_max 테이퍼를 고치면서 그 경로가 사라졌다. 추종이
좋아진 만큼 필터가 **떨어진 리더를 끝까지 충실히 쫓아가게** 된 것이다.

**왜 여기(워커)인가.** 판단에는 "리더" 라는 개념과 100 ms 의 이력이 필요한데
노드는 둘 다 모른다. 반대로 팔을 세우는 것은 노드만 할 수 있다 (필터의 현재
출력 ``_q_cmd`` 를 아는 것이 노드뿐이다). 그래서 **언제** 세울지는 여기가,
**어떻게** 세울지는 ``FrankaFR3Robot.hold`` 가 맡는다.

**설정점 간격으로는 안 된다.** 처음에는 "설정점이 필터 출력에서 얼마나
떨어졌나" 로 잡으려 했는데, 실제 낙하에서 그 값이 0.322 rad 까지밖에 안 갔다
-- 정상 텔레옵이 0.532 를 찍은 세션이 있으므로 둘이 겹친다. 이유는 물리적으로
분명하다: 놓인 리더는 팔로워의 v_max(1.5 rad/s) 안에서 떨어지므로 간격이
벌어지지 않는다. 위험한 것은 속도가 아니라 **아무도 의도하지 않은 자세까지
꾸준히 가는 것**이고, 그 "꾸준히" 가 이 모듈이 재는 값이다.

지표와 임계의 근거는 :data:`mstack.config.constants.LEADER_DROP_SPEED_RAD_S`
에 있다. 원시 로그에서 같은 값을 다시 내려면
``scripts/analyze/leader_speed.py``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from mstack.config.constants import LEADER_DROP_SPEED_RAD_S, LEADER_DROP_WINDOW_S


class LeaderDropGuard:
    """리더 관절각을 받아 "놓쳤는가" 를 판정한다. 상태만 있고 부작용은 없다.

    한 에피소드에 하나 만들고 매 명령 틱마다 :meth:`update` 를 부른다.
    판정의 결과로 무엇을 할지는 부르는 쪽이 정한다 -- 이 클래스는 로봇도
    시그널도 모른다 (그래야 하드웨어 없이 시험할 수 있다).
    """

    def __init__(
        self,
        limit: float = LEADER_DROP_SPEED_RAD_S,
        window_s: float = LEADER_DROP_WINDOW_S,
    ) -> None:
        self.limit = float(limit)
        self.window_s = float(window_s)
        self.peak = 0.0          # 이번 에피소드의 최고값 -- 임계를 다시 볼 근거
        self._t: deque = deque()
        self._q: deque = deque()

    def reset(self) -> None:
        """에피소드 경계. 이력을 버린다 -- 에피소드 사이의 홈 복귀·정렬 이동이
        다음 에피소드 첫 판정에 섞이면 안 된다."""
        self._t.clear()
        self._q.clear()
        self.peak = 0.0

    def update(self, q_arm, t: float) -> "float | None":
        """샘플 하나를 넣고 지금의 지표를 돌려준다.

        창을 채우기 전에는 ``None`` -- "아직 모른다" 와 "0 이다" 는 다르다.
        에피소드 시작 직후 100 ms 가 여기 해당하는데, 그때는 조작자가 리더를
        잡고 게이트를 막 통과한 참이라 판정할 것도 없다.

        시각은 부르는 쪽이 준다. ``time.monotonic()`` 을 쓸 것 --
        ``time.time()`` 은 NTP 보정으로 뒤로 갈 수 있고, 그러면 dt 가 음수가
        되어 속도가 폭발한다.
        """
        q = np.asarray(q_arm, dtype=float)
        self._t.append(float(t))
        self._q.append(q.copy())

        # 창보다 오래된 것을 버리되, 창을 **덮는** 가장 오래된 샘플 하나는
        # 남긴다. 그것까지 버리면 실제로 재는 구간이 창보다 짧아져 평균이
        # 짧은 구간의 값으로 부풀어 오른다.
        while len(self._t) > 2 and self._t[-1] - self._t[1] >= self.window_s:
            self._t.popleft()
            self._q.popleft()

        span = self._t[-1] - self._t[0]
        if len(self._t) < 2 or span < self.window_s:
            return None

        # 관절별 이동거리 / 시간 = 관절별 평균 속력. 순 변위가 아니라
        # **이동거리**다 (절대값을 먼저 취한다) -- 실측 기준과 같은 정의이고,
        # 떨어지는 팔은 한 방향으로 가므로 둘이 같지만 손떨림은 거리만 키운다.
        arr = np.asarray(self._q)
        speed = np.abs(np.diff(arr, axis=0)).sum(axis=0) / span
        value = float(np.linalg.norm(speed))
        if value > self.peak:
            self.peak = value
        return value

    def tripped(self, value: "float | None") -> bool:
        """:meth:`update` 가 준 값이 임계를 넘었나. ``None`` 은 넘지 않은 것."""
        return value is not None and value > self.limit
