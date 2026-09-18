#!/bin/bash
# GUI 인수 테스트 일괄 실행 (offscreen, 로봇/카메라 불필요)
# 사용: bash tests/gui/run_all.sh [python]
PY="${1:-python}"
# 사람이 없는 자리(밤샘 러너, CI)에서 관리자 비밀번호 창이 뜨면 답할 사람이
# 없어 그대로 멈춘다. 리더암이 꽂힌 채 재연결/재부팅된 뒤에만 뜨므로 기계
# 상태에 따라 떴다 안 떴다 한다 -- 2026-09-01 에 실제로 막혔다.
export GELLO_NO_PRIVILEGED=1
# 이 스위트는 로봇도 카메라도 없이 돈다. 마법사 하드웨어 페이지가 미리보기를
# 위해 카메라 노드를 띄우므로, 여기서 막지 않으면 테스트가 실제 카메라를
# 붙잡아 조작자의 GUI 를 방해한다 (2026-09-05).
export GELLO_NO_CAMERA_NODE=1
# 같은 이유로 로봇 노드도 막는다 -- 데이터세트 버전 [확인] 이 노드를 직접
# 띄우므로, 여기서 막지 않으면 테스트가 FCI 를 잡는다 (2026-09-05).
export GELLO_NO_ROBOT_NODE=1
# 단계 표지 PUB 소켓도 열지 않는다 -- 인수 테스트가 포트를 잡으면 같은
# 기계에서 도는 실제 수집과 부딪힌다 (카메라/로봇 노드와 같은 계약).
export GELLO_NO_PHASE_BUS=1
# 원시 상태 로거 프로세스도 띄우지 않는다 -- 같은 이유.
export GELLO_NO_RAW_LOGGER=1
# 이 스위트는 네트워크 없이도 돌아야 한다. 앱 글꼴(D2Coding)은 캐시가 비어
# 있으면 21MB 를 받는데, 그 대기와 실패가 테스트 결과에 섞이면 안 된다.
# 캐시가 이미 있으면 그건 그대로 쓴다 -- 막는 것은 네트워크지 글꼴이 아니다.
export GELLO_NO_FONT_DOWNLOAD=1
# 테스트는 조작자의 상태 파일(격자·크롭·recents·업로드 장부·스키마 설정)을
# 절대 건드리면 안 된다. 2026-09-06 에 실제로 덮어썼다 -- 격자 테스트가
# collect_workspace.save_grid_store 를 패치했지만 features/camera/ops.py 가
# 같은 함수를 직접 임포트해 들고 있어서, 그 이름을 통한 호출이 진짜 파일로
# 갔고 조작자가 맞춰 둔 3×3 격자가 스위트를 돌릴 때마다 초기화됐다.
# 호출 하나를 패치하는 방식은 임포트 방식에 따라 새므로, 뿌리를 옮긴다.
GELLO_STATE_DIR="$(mktemp -d -t gello-test-state-XXXXXX)"
export GELLO_STATE_DIR
STATE_DIRS="$GELLO_STATE_DIR"
trap 'rm -rf $STATE_DIRS' EXIT
# The shared stores follow their own env vars, not GELLO_STATE_DIR. Run from a
# shell that exported them (the dev icon does), the suite would write test
# sessions and phase lines into the operators' real history.
unset MSTACK_HISTORY MSTACK_PROXY_DIR MSTACK_HUB_STATE
cd "$(dirname "$0")"
fail=0

# 세 개가 스위트 시간의 절반을 쓴다 (2026-09-18 실측: 전체 156 s 중
# diversity_cloud 47.5, recommend_register 11.8, quick_resume 11.1). 모두
# 추천기가 CP-SAT 로 실제 배치를 푸는 값이라 줄일 수 있는 성질이 아니다.
# 그래서 이 셋만 먼저 띄우고, 나머지 56 개가 순차로 도는 동안 같이 끝나게
# 한다. 전체를 병렬로 돌리지 않는 이유는 아래 상태 디렉터리 때문이다 --
# 격리해야 할 대상이 셋이면 눈으로 확인되지만 쉰아홉이면 그렇지 않다.
SLOW="test_diversity_cloud test_recommend_register test_quick_resume"
slow_pids=""
for t in $SLOW; do
  # 상태 디렉터리를 공유하면 동시에 도는 테스트끼리 서로의 격자·캐시·장부를
  # 밟는다. 순차 실행일 때는 하나로 충분했다.
  d="$(mktemp -d -t gello-test-state-XXXXXX)"
  STATE_DIRS="$STATE_DIRS $d"
  GELLO_STATE_DIR="$d" QT_QPA_PLATFORM=offscreen MSTACK_NO_RIG_QUERY=1 \
    timeout 240 "$PY" -u "$t.py" >"/tmp/$t.out" 2>&1 &
  slow_pids="$slow_pids $!"
done

for t in test_phase4a test_grid_replay test_plan_form test_right_scene \
         test_gate_reset test_plan_edit_replay test_h5view \
         test_depth17 \
         test_scene_edit test_stats_group test_relabel test_dataset_sync \
         test_hub_upload_state test_camera_node test_match_gate \
         test_app_structure test_ui_surface test_domain_attrs \
         test_episode_io test_layer_rules test_signal_slots test_instruction_counter \
         test_dataset_meta test_launcher test_key_autorepeat \
         test_station_save test_wheel_guard test_waypoint_kinematics \
         test_state_isolation test_resume_version test_workflow_gui test_node_diag \
         test_collect_layout test_script_bootstrap test_relation_shape \
         test_scene_repair test_doctor_tab test_info_card \
         test_doctor_contrast test_doctor_progress test_doctor_schema test_dataset_right test_phase_log test_frame_timing test_scene_sampler \
         test_leader_guard test_proxy_clip test_clip_grid test_curation_basket \
         test_trim_controls test_plot_widgets test_jobs test_upload_dialogs test_provenance test_proxy_dialog test_repack; do
  if QT_QPA_PLATFORM=offscreen MSTACK_NO_RIG_QUERY=1 timeout 240 "$PY" -u "$t.py" >"/tmp/$t.out" 2>&1; then
    echo "$t OK"
  else
    echo "$t FAIL"; tail -5 "/tmp/$t.out"; fail=1
  fi
done

# 먼저 띄운 것들을 거둔다. 순차 목록이 길어 대개 이미 끝나 있다.
set -- $SLOW
for pid in $slow_pids; do
  t="$1"; shift
  if wait "$pid"; then
    echo "$t OK"
  else
    echo "$t FAIL"; tail -5 "/tmp/$t.out"; fail=1
  fi
done
exit $fail
