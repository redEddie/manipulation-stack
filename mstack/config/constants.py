"""Shared runtime constants for the Gello system.

Values here are the "source of truth" for both robot control and the GUI so
that what the operator sees on screen matches what the motors do.
"""

#: Maximum joint error at which pose-match assist may engage (rad).
#: The GUI "pose matching" gauge uses the same threshold so that the gauge
#: being green coincides with the wall starting to pull (GitHub issue #37A).
MATCH_GATE_RAD = 0.5

#: Floor for the match-current cap (mA).
#: This is a *lower bound on the cap*, not a holding current. Some joints
#: need ~430 mA just to hold position, so 200 mA would make convergence
#: impossible if it were an upper limit.
IDLE_MIN_CURRENT = 200.0

#: Pose-match aborts when a *roll* joint (J1/J3/J5/J7 on the FR3) is further
#: than this from its target (rad). Roll joints turn without end, so pulling
#: one from far away can wind the cable; bending joints being far apart is
#: just a different pose and is no reason to stop.
#:
#: Deliberately looser than MATCH_GATE_RAD: the gate decides when *teleop* may
#: start, this decides when aligning becomes unsafe for the cable, and the
#: second is the more forgiving question. 0.5 rad stopped alignments that were
#: perfectly fine to run (2026-09-01, measured on the robot).
ROLL_ABORT_RAD = 1.0

#: 리더를 놓쳤다고 판정하는 속도 (rad/s).
#:
#: 재는 것: 리더 관절속도를 ``LEADER_DROP_WINDOW_S`` 창으로 **관절별 평균**
#: 낸 뒤 그 7개 값의 **L2 노름**. 순서가 중요하다 -- 절대값 → 관절별 평균 →
#: L2 다.
#:
#: **왜 순간값이 아니라 평균인가.** 2026-09-10 에 조작자가 리더를 일부러
#: 놓았다 잡기를 반복한 에피소드를 재보니, 순간 최대는 정상 4.25 / 낙하 6.64
#: 로 겹쳤다 -- 양쪽 다 1샘플짜리 글리치라 정보가 없다. 놓인 리더는 **빠른
#: 것이 아니라 오래간다.** 100 ms 평균에서야 정상 1.13~1.53 / 낙하 2.47~3.21
#: 로 갈렸다.
#:
#: **왜 가속도가 아닌가.** 같은 데이터에서 낙하의 가속도가 정상보다 *낮았다*
#: (641 vs 1008 rad/s^2). 이차미분이 잡는 것은 리더의 움직임이 아니라 엔코더
#: 양자화라, 중력방향으로 분해해도 결과는 같다.
#:
#: **왜 관절 최대가 아니라 L2 인가.** 떨어지는 팔은 여러 관절이 함께 움직이고
#: 의도적인 동작은 한두 관절에 몰린다. 같은 에피소드 안에서 분리비가 관절
#: 최대 1.59 / L2 1.92 였다. J6(손목 피치)만 보면 2.66 로 더 크지만, 손목을
#: 많이 쓰는 작업에서 깨질 지표라 쓰지 않는다.
#:
#: **2.4 의 근거.** 실측 스트림을 이 코드에 그대로 흘려 본 결과다
#: (``scripts/analyze/leader_speed.py --replay``):
#:
#: * 정상 에피소드 5개(28.9초) 최고값 1.30 / 1.36 / 1.38 / 1.48 / 1.51
#: * 낙하 에피소드(9.5초) 낙하 2회에서 2.84 와 2.70
#:
#: 1.8~2.6 어디에 두어도 정상 오검 0건이면서 낙하 2회를 다 잡는다 -- 2.4 는
#: 그 고원의 가운데다. 절벽 위에 놓인 값이 아니므로 재유도 없이 옮겨도 된다.
#:
#: 여유는 대칭이 아니다: 정상 최대의 1.59배지만 낙하 최고의 0.85배다. 낙하가
#: 15% 만 얌전해도 놓친다는 뜻이라, 놓치는 쪽이 더 아프면 2.2 로 내리는 것이
#: 같은 데이터 안에서 정당하다.
#:
#: 표본은 낙하 1건, 조작자 1명, 작업 1종, 하루 저녁이다. 정상 5개가 1.30~1.51
#: 로 몰려 있는 것이 유일한 위안이다.
LEADER_DROP_SPEED_RAD_S = 2.4

#: 위 평균을 내는 창 (초). 100 ms 는 분리비가 가장 좋았던 길이다
#: (30 ms 1.76 / 50 ms 1.73 / 80 ms 1.85 / 100 ms 2.19).
#:
#: 검출이 이만큼 늦는다는 뜻이기도 하다. v_max 1.5 rad/s 에서 0.15 rad 를 더
#: 가고, 급정거 거리 0.19 rad 를 더해 약 0.34 rad 다. 낙하 자체가 만든 설정점
#: 간격이 0.32 였으니 비례하는 크기다.
LEADER_DROP_WINDOW_S = 0.10
