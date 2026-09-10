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

#: 리더 놓침 감시가 **보는 관절** (0-based). J2 / J4 / J6 -- FR3 의 세 피치 축.
#:
#: 나머지 넷(J1/J3/J5/J7)은 롤이다. 중력은 회전축과 나란한 방향으로 모멘트를
#: 만들지 못하므로 **롤 관절은 떨어지지 않는다.** 실측이 그대로 말한다
#: (2026-09-10, 100 ms 평균 최고 속력):
#:
#:     ::
#:
#:              J1    J2    J3    J4    J5    J6    J7
#:       정상  0.43  0.96  0.51  1.10  1.12  0.87  1.01
#:       낙하  0.11  2.15  0.20  0.95  0.26  2.09  0.44
#:
#: 롤 넷은 낙하 때가 **정상보다 느리다** -- 지표에 넣으면 낙하 신호는 없이
#: 정상 쪽 바닥만 올린다. 롤 L2 만 보면 분리비가 0.37 로 뒤집혀 있다.
#: 그래서 롤에는 더 큰 임계를 주는 대신 아예 보지 않는다. 어떤 값을 줘도
#: 낙하에서는 안 걸리므로, 근거 없는 숫자를 하나 더 두는 것이 될 뿐이다
#: (같은 이유로 2026-09-10 에 설정점 간격 상한 0.9 를 걷어냈다).
#:
#: 세 피치를 다 넣는 이유. 이번 낙하에서 실제로 떨어진 것은 J2(1.87)와
#: J6(2.02)이고 J4 는 0.67 로 조용했다. 수치만 보면 J2+J6 가 분리비 2.85 로
#: 제일 좋지만, **어느 피치가 떨어지느냐는 자세에 따라 달라진다** -- 한
#: 자세에서 안 움직였다고 빼는 것은 낙하 1건에 맞추는 과적합이다. 빼는 기준은
#: "이번에 안 움직인 관절" 이 아니라 "중력이 모멘트를 못 만드는 축" 이어야
#: 자세가 바뀌어도 성립한다.
LEADER_DROP_JOINTS = (1, 3, 5)

#: 리더를 놓쳤다고 판정하는 속도 (rad/s).
#:
#: 재는 것: :data:`LEADER_DROP_JOINTS` 의 관절속도를 ``LEADER_DROP_WINDOW_S``
#: 창으로 **관절별 평균** 낸 뒤 그 값들의 **L2 노름**. 순서가 중요하다 --
#: 절대값 → 관절별 평균 → L2 다.
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
#: **왜 관절 최대가 아니라 L2 인가.** 떨어지는 팔은 여러 피치가 함께
#: 움직이고 의도적인 동작은 한 관절에 몰린다. 같은 에피소드 안에서 분리비가
#: 관절 최대 1.59 / L2 1.92 였다.
#:
#: **2.1 의 근거.** 실측 스트림을 이 코드에 그대로 흘려 본 결과다
#: (``scripts/analyze/leader_speed.py --sweep``):
#:
#: * 정상 에피소드 5개(28.9초) 최고값 최대 1.41
#: * 낙하 에피소드(9.5초) 낙하 2회에서 2.83 과 2.69
#:
#: 1.6~2.8 어디에 두어도 정상 오검 0건이면서 낙하 2회를 다 잡는다 -- 2.1 은
#: 그 고원에서 두 실측의 가운데다(정상의 1.49배, 낙하의 0.74배). 절벽 위에
#: 놓인 값이 아니므로 재유도 없이 옮겨도 된다.
#:
#: 표본은 낙하 1건, 조작자 1명, 작업 1종, 하루 저녁이다. 정상 5개가 좁게
#: 몰려 있는 것이 유일한 위안이다.
LEADER_DROP_SPEED_RAD_S = 2.1

#: 위 평균을 내는 창 (초). 100 ms 는 분리비가 가장 좋았던 길이다
#: (30 ms 1.76 / 50 ms 1.73 / 80 ms 1.85 / 100 ms 2.19).
#:
#: 검출이 이만큼 늦는다는 뜻이기도 하다. v_max 1.5 rad/s 에서 0.15 rad 를 더
#: 가고, 급정거 거리 0.19 rad 를 더해 약 0.34 rad 다. 낙하 자체가 만든 설정점
#: 간격이 0.32 였으니 비례하는 크기다.
LEADER_DROP_WINDOW_S = 0.10
