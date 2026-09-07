# GUI 인수 테스트 (offscreen)

로봇·카메라 없이 도는 수집 GUI 회귀 테스트. QMessageBox/카메라 열거는
스텁하고, scene 파일은 합성하거나 잠금(수집 세션 중)을 허용하도록 짜여
있다 -- 수집이 돌아가는 중에도 실행할 수 있다.

```bash
bash tests/gui/run_all.sh ~/lerobot-venv/bin/python
```

현재 45개가 `run_all.sh` 순서대로 돈다.

| 파일 | 검증 |
|---|---|
| test_phase4a | Phase 4a 인수 테스트: 계획 로더 + slot 드롭다운/카운트/다음 slot/불일치 경고 |
| test_grid_replay | 3×3 격자 오버레이 + 실로봇 재생 버튼 검증 |
| test_plan_form | 계획 폼 편집기 + Configure 계획 문장 드롭다운 검증 |
| test_right_scene | 오른쪽 패널 scene 배치도 검증 |
| test_gate_reset | 게이트 자동정렬 범위 조건 + 리셋 중 프레임 방출 + 시작 버튼 잠금 검증 |
| test_plan_edit_replay | 계획 편집 다이얼로그 + replay 로더 검증 |
| test_h5view | Hdf5TreeDialog 스모크 — selftest 로 만든 scene 파일을 트리로 탐색 |
| test_diversity_cloud | scene 다양성 추천(#33) + Point Cloud 탭 검증 |
| test_recommend_register | RecommendDialog 문장 체크리스트 + 계획 등록, SceneComposer lint |
| test_depth17 | depth 기록(#17) 검증 + depth 수집 게이트 검증 |
| test_scene_edit | scene 에피소드 편집(삭제·트림) 검증 -- 삭제 후 renumber(그룹·episode_id· |
| test_stats_group | Analysis 그룹 기준 = (scene, 문장) 검증 -- 같은 문장이라도 scene 이 다르면 |
| test_relabel | 에피소드 재판정 경로 검증 |
| test_dataset_sync | dataset_sync Hub 조회 revision 핀 검증 |
| test_hub_upload_state | 업로드 장부(mstack/hub_upload_state.py) 검증 -- 2026-08-25 도입 |
| test_camera_node | 카메라 노드/클라이언트 검증 |
| test_match_gate | 정렬 힘 우물 검증 (issue #37A) -- 로봇 없이 가짜 드라이버로 |
| test_app_structure | apps/ 구조 계약 -- 이름 하나에 정의 하나, 화살표는 한쪽 |
| test_ui_surface | GUI 표면 계약 -- 리팩토링이 눈에 보이는 것을 바꾸지 않았는지 |
| test_domain_attrs | 도메인이 창에서 읽는 이름이 실제로 있는지 |
| test_episode_io | 에피소드 쓰기 경로 왕복 검증 -- 데이터 파이프라인 분리 전 안전망 |
| test_layer_rules | mstack/ 계층 규칙 -- 화살표는 아래로만, 상위는 구체 하드웨어를 모른다 |
| test_signal_slots | 시그널이 실제로 슬롯에 붙는지 |
| test_instruction_counter | slot 카운터 (issue #38) 검증 -- 수집 화면의 현재 (scene, instruction) |
| test_dataset_meta | dataset-identity.json 메타 + discover_datasets + plan_progress 검증 |
| test_launcher | 런처 마법사 검증 (offscreen) — 모드 분기, 새 데이터셋 생성, 이어서 하기 |
| test_key_autorepeat | 단축키가 '누르고 있는 것'을 연타로 읽지 않는지 검증 |
| test_station_save | 새 스테이션 저장 -- 카메라 대수와 저장된 YAML 모양 검증 |
| test_wheel_guard | 휠이 콤보·스핀 값을 바꾸지 않고 페이지를 스크롤하는지 검증 |
| test_waypoint_kinematics | waypoint 액션 변환 검증 — 서버와 byte-identical 인 쪽과 클라이언트가 쓰는 쪽 |
| test_state_isolation | 테스트가 조작자의 상태 파일을 건드리지 않는지 검증 |
| test_resume_version | 이어찍기에서 버전 도장이 이번 세션 것으로 올라가는지 검증 |
| test_workflow_gui | 2026-09-06 GUI 개선분 인수 테스트 |
| test_quick_resume | 빠른 재개(⚡ Quick resume)의 고르기 규칙 + homing 속도 상한 |
| test_node_diag | 노드가 죽었을 때 **왜** 죽었는지가 화면까지 오는가 |
| test_collect_layout | 수집 워크플로 화면 개편 인수 |
| test_script_bootstrap | scripts/ 의 mstack 임포트가 sys.path 부트스트랩 뒤에 오는지 |
| test_relation_shape | 관계-모양 규칙 -- 목적지가 그릇이면 언제나 inside, on 은 평평한 것에만 |
| test_scene_repair | 찍은 뒤 바로잡기(mstack/scene/scene_repair.py) 검증 |
| test_doctor_tab | 기록 닥터(Doctor 활동탭) 인수 테스트 |
| test_info_card | 데이터 정보 표시 모듈의 계약 |
| test_doctor_contrast | Doctor 화면의 글자 대비 검사 |
| test_doctor_progress | 진행 닥터 (#48) 인수 테스트 |
| test_doctor_schema | 스키마 닥터 (#47) 인수 테스트 |
| test_dataset_right | Dataset 의 우측 패널 -- 고른 에피소드의 값과 그 scene 의 배치 |
