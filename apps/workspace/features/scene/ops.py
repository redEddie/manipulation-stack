"""Scene configuration and core scene identity operations."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

from mstack.config.station import load_station
from mstack.scene.collection_plan import ensure_scene
from mstack.scene.dataset_meta import plan_path as dataset_plan_path
from apps.workspace.shared.tabs import show_center_tab
from mstack.gui.i18n import tr
from mstack.scene.props import active_prop_ids
from mstack.collect.session_meta import (
    apply_to_metadata,
    offline_provenance,
    payload_from_node,
    reset_pose_from_station,
)
from mstack.scene.scene_format import (
    INSTRUCTION_ID_RE,
    SceneWriter,
    count_by_slot,
    iter_scene_files,
    next_scene_id,
    read_scene_metadata,
    scene_filename,
)

STATION = load_station()


class SceneOps:
    """Scene configuration, file identity, and right-panel metadata."""

    def __init__(self, win) -> None:
        self.win = win

    def refresh_scene_combo(self) -> None:
        """저장 경로의 scene_*.hdf5 목록. 파일명이 아니라 내부 metadata 로
        표시한다 (경로 역산 금지)."""
        self.win.scene_combo.blockSignals(True)
        self.win.scene_combo.clear()
        root = Path(self.win.root_edit.text().strip() or ".")
        try:
            sid_next = next_scene_id(root)
        except Exception:  # noqa: BLE001
            sid_next = "S???"
        self.win.scene_combo.addItem(tr("— 새 Scene ({sid}) —").format(sid=sid_next), None)
        try:
            for p in iter_scene_files(root):
                try:
                    md = read_scene_metadata(p)
                except Exception as e:  # noqa: BLE001
                    self.win.scene_combo.addItem(f"{p.name} (읽기 실패: {type(e).__name__})", None)
                    continue
                label = f"{md.scene_id} · 물체 {len(md.objects)}개"
                if md.description:
                    label += f" · {md.description[:28]}"
                self.win.scene_combo.addItem(label, md.scene_id)
        except Exception:  # noqa: BLE001
            pass
        self.win.scene_combo.blockSignals(False)
        # 저장 경로가 바뀌었을 수 있으니 데이터셋 귀속 계획(instructions.json)
        # 표시·slot 패널·scene 정보를 함께 갱신한다.
        self.win.scene_planning.on_plan_changed()

    def on_scene_selected(self, *_args) -> None:
        self.win.scene_planning.refresh_start_instruction()
        self.win.collection.refresh_instruction()
        sid = self.win.scene_combo.currentData()
        if sid is None:
            self.win.scene_info.setText(tr(
                "Scene 탭에서 배치를 짜고 오른쪽의 [✚ 새 Scene 만들기] 를 누르면 "
                "여기 목록에 생깁니다. 여러 개를 미리 만들어 둘 수 있습니다."))
            return
        root = Path(self.win.root_edit.text().strip() or ".")
        try:
            path = root / scene_filename(sid)
            if self.win.session.active_file_path is not None and path == self.win.session.active_file_path:
                # 세션이 파일을 쥐고 있다 -- 캐시 요약으로 대신한다
                counts = self.win.scene_planning.session_instruction_counts()
                lines = [tr("{s} — 수집 세션 진행 중 (배치도는 오른쪽 패널에)")
                         .format(s=sid)]
                if counts:
                    lines.append("slot: " + "  ".join(
                        f"{iid} {c.get('usable', 0)}/{c.get('total', 0)}"
                        for iid, c in sorted(counts.items())))
                self.win.scene_info.setText("\n".join(lines))
                return
            md = read_scene_metadata(path)
            counts = count_by_slot(path)
            extra = []
            plan = self.win.scene_planning.current_plan()
            if plan is not None and plan.slots_for(sid):
                extra.append((tr("지시문 파일({n})").format(n=plan.path.name),
                              "  ".join(
                    f"{s.instruction_id} {counts.get(s.instruction_id, {}).get('usable', 0)}"
                    f"/{s.target}" for s in plan.slots_for(sid))))
            self.win.scene_info.set_scene(md, counts=counts or None, extra=extra)
        except BlockingIOError:
            self.win.scene_info.setText(tr(
                "(다른 프로세스가 파일을 사용 중입니다 — 공간 회수/변환이 끝난 "
                "뒤 새로고침하세요)"))
        except Exception as e:  # noqa: BLE001
            self.win.scene_info.setText(f"(scene 정보 읽기 실패: {type(e).__name__}: {e})")

    def on_scene_activated(self, *_args) -> None:
        """드롭다운을 **사람이** 건드렸을 때만 온다 (``activated``).

        "— 새 Scene (Sxxx) —" 을 고르는 것이 곧 "새로 짜겠다"는 뜻이라 Scene
        탭을 연다 -- 옛 [새 Scene 구성...] 버튼을 대신한다.

        ``currentIndexChanged`` 가 아니라 ``activated`` 인 이유: 목록을 다시
        채울 때마다(경로 변경·새로고침·세션 종료) 인덱스는 0 으로 돌아가는데,
        그때마다 탭을 열면 보고 있던 화면을 빼앗는다 (실제로 그랬다).
        """
        if self.win.scene_combo.currentData() is None and self.win.worker is None:
            self.on_new_scene()

    def on_new_scene(self) -> None:
        """Scene 탭을 열고 다음 scene 번호로 맞춘다 (2026-09-06: 대화상자 ->
        탭). 파일은 우측 패널의 [✚ 새 Scene 만들기] 를 누를 때 생긴다 --
        Connect 를 기다리지 않는다 (계약은 on_compose_done).

        안내문은 비운다 -- "아직 저장 안 됨" 은 그 버튼 밑의 상태 줄이 늘
        말하고 있고(composer.save_state_text), 지난 결과가 새 구성 옆에
        남아 있으면 그게 이번 것인 줄 읽힌다."""
        root = Path(self.win.root_edit.text().strip() or ".")
        try:
            sid = next_scene_id(root)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self.win, tr("경로 오류"),
                                tr("저장 경로를 확인하세요: {e}").format(e=e))
            return
        # 지시문은 데이터셋 폴더 안 instructions.json 하나뿐이다 (고정 파일명).
        # **파일이 아직 없어도 경로는 넘긴다** -- 없으면 만들면 되는 것이지
        # 등록을 막을 사유가 아니다 (2026-09-07 조작자 지적). 전에는
        # ``pp if pp.is_file() else None`` 이라, 새 데이터셋에서 추천을 받으면
        # "Configure 에서 지시문 파일을 먼저 고르세요" 라는 회색 체크박스를
        # 만났다 -- 그런 자리는 9/6 에 없어졌는데 문구만 남아 있었다.
        self.win.scene_composer.set_context(
            sid, root, dataset_plan_path(root),
            STATION.name, self.win.schema_version)
        self.win.scene_compose_hint.setText("")
        show_center_tab(self.win, "scene")

    def refresh_composer_context(self) -> None:
        """구성기에 지금 데이터셋의 경로·다음 scene 번호를 물린다.

        on_new_scene 이 하던 일 중 **맥락을 채우는 부분만** 떼어 낸 것이다.
        Scene 탭은 그 명령을 거치지 않고 탭 클릭으로도 열려서, 그때는 구성기가
        경로 없이 남아 있었다.
        """
        root = Path(self.win.root_edit.text().strip() or ".")
        try:
            sid = next_scene_id(root)
        except Exception:  # noqa: BLE001 -- 경로가 아직 없을 수 있다
            return
        self.win.scene_composer.set_context(
            sid, root, dataset_plan_path(root),
            STATION.name, self.win.schema_version)

    def on_compose_done(self) -> None:
        """우측 패널의 [✚ 새 Scene 만들기] -- **그 자리에서 파일을 만든다**.

        2026-09-06 사용자 결정. 전에는 구성을 메모리에 하나 얹어 두고
        Connect 때 파일을 만들었는데, 화면이 그 사실을 말하지 않아 "지금
        파일이 생긴 건가? 여러 개 미리 짤 수 있나?" 를 알 수 없었다 (둘 다
        아니었다 -- 대기는 하나였고 다시 구성하면 조용히 덮였다).

        이제는 누르는 순간 ``scene_002.hdf5`` 가 생긴다 (에피소드 0개). 그래서
        S002·S003·S004 를 미리 짜 두고 나중에 골라 찍을 수 있다. 대가는 알고
        택한 것이다: 구성만 하고 안 찍은 빈 scene 이 데이터셋에 남아 진행률에
        0/N 으로 섬긴다. 지우는 수단은 Dataset 의 [파일 삭제] 다.
        """
        md = self.win.scene_composer.build_valid()
        if md is None:
            return
        root = Path(self.win.root_edit.text().strip() or ".")
        # **번호는 만드는 순간 다시 센다.** 구성기가 들고 있던 번호는 맥락을
        # 물린 시점의 것이라, 그 사이에 데이터셋이 바뀌었거나 맥락을 못 받은
        # 채였으면 낡았다 -- 실제로 S000 인 채로 남아 이미 있는 파일과 부딪혔다
        # (2026-09-07 실기). 화면 표시가 틀리는 것은 불편이지만, 그 번호로
        # 파일을 만드는 것은 사고다.
        try:
            md.scene_id = next_scene_id(root)
        except Exception:  # noqa: BLE001 -- 경로가 이상하면 아래에서 잡힌다
            pass
        # **세션 메타를 만들기 전에 채운다.** 부하 모델·리셋 자세·판번호는
        # 스키마가 metadata attrs 로 요구하고(knu-1.2.0/1.2.1/1.2.2), 없으면
        # SceneWriter 가 도장을 내려 찍는다. 예전에는 여기서 아무것도 안
        # 채워 knu-1.1.1 로 찍혔고, 1.x 시절엔 Connect 가 도장을 올려 주어
        # 무해했다. 2.0.0 부터는 MAJOR 가 달라 이어찍기가 거부되므로, 그
        # 파일은 **영원히 녹화할 수 없는 빈 scene** 이 된다 (2026-09-23 S024).
        #
        # 셋 중 리셋 자세(station 설정)와 판번호(git)는 로봇 없이 채워진다.
        # 부하 모델만 노드를 거치는데, 노드가 안 떠 있어도 여기서 막지는
        # 않는다 -- "로봇 없이 scene 을 미리 짜 둔다"가 이 버튼의 기능이다
        # (2026-09-06). 대신 도장은 요청한 버전으로 찍고(metadata_pending),
        # 부하 모델은 Connect 가 채운다. 못 채우면 **그때** 녹화를 거절한다:
        # 로봇이 붙은 뒤의 거절이라야 "이 로봇이 부하를 안 준다"는 진짜
        # 문제만 남는다.
        want = self.win.schema_version
        payload = payload_from_node(timeout_ms=1500)
        apply_to_metadata(md, payload, reset_pose_from_station(),
                          offline_provenance())
        md.dataset_version = want
        try:
            # metadata 만 있는 파일. SceneWriter 는 생성 시점에 파일을 쓰므로
            # 여기서 닫으면 그대로 빈 scene 이 된다 -- Connect 는 resume 으로
            # 이어 쓴다 (기존 scene 을 고르는 것과 완전히 같은 경로).
            writer = SceneWriter(root, metadata=md,
                                 known_prop_ids=active_prop_ids(),
                                 session_version=want,
                                 metadata_pending=True)
            if writer.version_note:
                self.win.log(f"[스키마] {writer.version_note}")
            writer.close()
        except FileExistsError:
            QMessageBox.warning(self.win, tr("이미 있음"), tr(
                "{s} 파일이 이미 있습니다. 목록을 새로고침한 뒤 다시 "
                "만드세요.").format(s=md.scene_id))
            self.refresh_scene_combo()
            return
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self.win, tr("Scene 만들기 실패"),
                                f"{type(e).__name__}: {e}")
            return
        # 계획에도 같이 적는다 (2026-09-06 사용자 결정: **배치가 주**).
        # 전에는 scene 하나를 쓰려면 파일과 계획 항목을 각각 손으로 만들어야
        # 했고, 그것이 "scene 추가하는 곳이 두 군데"의 실체였다. 지시문은
        # 비워 둔다 -- 무엇을 시킬지는 사람이 정한다.
        added = False
        try:
            added = ensure_scene(dataset_plan_path(root), md.scene_id)
        except OSError as e:
            self.win.log(f"[지시문] {md.scene_id} 항목을 추가하지 못했습니다: {e}")
        self.win.log(f"[Scene] {md.scene_id} 생성 (물체 {len(md.objects)}개, "
                     f"에피소드 0개) — {scene_filename(md.scene_id)}"
                     + (f" · 지시문에 {md.scene_id} 추가" if added else ""))
        self.refresh_scene_combo()
        for i in range(self.win.scene_combo.count()):
            if self.win.scene_combo.itemData(i) == md.scene_id:
                self.win.scene_combo.setCurrentIndex(i)
                break
        self.win.scene_compose_hint.setText(tr(
            "{f} 를 만들고 지시문에 {s} 를 넣었습니다. 이제 이 scene 에서 무엇을 "
            "시킬지 적으세요.").format(f=scene_filename(md.scene_id), s=md.scene_id))
        # 다음 번호로 갈아 끼워 둔다 -- 연달아 여러 개를 짜는 것이 이 화면의
        # 새 용도다.
        self.win.scene_composer.set_context(
            next_scene_id(root), root, dataset_plan_path(root),
            STATION.name, self.win.schema_version)
        # 지시문을 적으러 보낸다. scene 을 만든 사람의 다음 질문이 늘
        # "여기서 무엇을 시키지?" 라서, 그 화면으로 데려다 주는 편이 낫다.
        self.win.scene_planning.refresh_plan_progress()
        show_center_tab(self.win, "instruction")

    def scene_config_from_ui(self):
        """Connect 시점의 scene 설정 검증. (meta, scene_id, resume, error) --
        error 가 None 이 아니면 연결을 중단하고 그 메시지를 보여준다."""
        lang = self.win.lang_edit.text().strip()
        iid = self.win.scene_iid_edit.text().strip()
        collector = self.win.collector_edit.text().strip()
        # **계획을 맨 먼저 본다** -- 계획이 없으면 시작 지시문도 있을 수
        # 없으므로, 빈 칸을 나무라기 전에 진짜 원인을 말해야 한다.
        plan = self.win.scene_planning.current_plan()
        if plan is None:
            return None, None, False, tr(
                "이 데이터셋에는 지시문이 없습니다.\n\n"
                "Instruction 탭의 [지시문 편집] 으로 scene 과 지시문을 적으세요. "
                "지시문은 지시문 파일에서만 옵니다.")
        if not lang:
            return None, None, False, tr(
                "시작 지시문이 없습니다 — Instruction 탭에서 줄을 눌러 고르세요.")
        if lang.startswith('"') and lang.endswith('"'):
            return None, None, False, tr("instruction 은 따옴표 없는 순수 문장이어야 합니다.")
        if not INSTRUCTION_ID_RE.match(iid):
            return None, None, False, tr("시작 지시문 ID 형식이 틀렸습니다 (예: I000).")
        if not collector:
            return None, None, False, tr("수집자 식별자를 입력하세요 (에피소드 필수 attr).")
        # 지시문은 데이터셋의 instructions.json 에서만 온다 (2026-09-06) --
        # 자유 입력이 계획 밖 지시문을 실데이터에 만든 적이 있고(ID-문장
        # 갈라짐), 그 길을 아예 없앴다. 계획 없이 찍은 파일은 나중에 무엇을
        # 얼마나 모았는지 셀 수가 없다.
        psid = self.configure_scene_id()
        # "계획에 없다"와 "지시문을 아직 안 적었다"는 다른 사건이다 -- 앞은
        # 이제 거의 안 나고(만들 때 함께 적힌다), 뒤는 새 scene 을 만든 직후
        # 늘 나는 정상 상태다. 같은 문구로 말하면 무엇을 하라는 것인지 모른다.
        if not plan.has_scene(psid or ""):
            return None, None, False, tr(
                "지시문에 scene {s} 가 없습니다.\n\n"
                "Instruction 탭의 [지시문 편집] 에서 추가하세요.").format(s=psid)
        slots = plan.slots_for(psid)
        if not slots:
            return None, None, False, tr(
                "{s} 에 지시문이 없습니다.\n\n"
                "Instruction 탭의 [지시문 편집] 에서 이 scene 에서 무엇을 시킬지 "
                "적으세요.").format(s=psid)
        if not any(s.instruction_id == iid and s.instruction == lang
                   for s in slots):
            return None, None, False, tr(
                "시작 지시문이 지시문 목록에 없습니다 ({i}: {t!r}).\n\n"
                "Instruction 탭에서 줄을 눌러 고르세요.").format(i=iid, t=lang[:40])
        sid = self.win.scene_combo.currentData()
        if sid is None:
            # 새 scene 은 Scene 탭에서 **만들어진 뒤** 목록에 뜬다 -- 연결
            # 시점에 만드는 경로는 없어졌다 (2026-09-06).
            return None, None, False, tr(
                "찍을 scene 을 드롭다운에서 고르세요.\n\n"
                "없으면 Scene 탭에서 배치를 짜고 오른쪽의 [✚ 새 Scene 만들기] 를 "
                "누르면 목록에 생깁니다.")
        return None, sid, True, None

    def configure_scene_id(self):
        """Configure 가 가리키는 scene ID -- 고른 것이 없으면 다음 발번 예정 ID.

        "대기 구성" 은 없어졌다 (2026-09-06): 새 scene 은 Scene 탭에서 만드는
        순간 파일이 되고 목록에 뜬다.
        """
        sid = self.win.scene_combo.currentData()
        if sid is not None:
            return sid
        try:
            return next_scene_id(Path(self.win.root_edit.text().strip() or "."))
        except Exception:  # noqa: BLE001
            return None

    def selected_scene_path(self):
        """Configure 의 Scene 콤보가 가리키는 기존 scene 파일 (새 scene 이면 None)."""
        sid = self.win.scene_combo.currentData()
        if sid is None:
            return None
        return Path(self.win.root_edit.text().strip() or ".") / scene_filename(sid)

    def session_scene_id(self):
        if self.win.worker is None:
            return None
        cfg = self.win.worker.cfg
        if getattr(cfg, "scene_metadata", None) is not None:
            return cfg.scene_metadata.scene_id
        return getattr(cfg, "scene_id", None)

    def scene_session_file(self):
        if not self.win.session.scene_session or self.win.session.active_file_path is None:
            return None
        return self.win.session.active_file_path

    def set_right_scene(self, md, sid=None) -> None:
        """오른쪽 패널의 '수집 중 scene 배치도'를 갱신한다."""
        if not hasattr(self.win, "right_scene_view"):
            return
        if md is not None:
            self.win.right_scene_view.set_scene(md)
        elif sid:
            self.win.right_scene_view.setText(
                tr("{s} — 배치 정보를 읽지 못했습니다").format(s=sid))
        else:
            self.win.right_scene_view.setText(tr("(scene 세션 없음)"))

    def apply_session_config(self, cfg: dict) -> list:
        """Puts a file's recorded session_config back into the widgets that
        produced it (see mstack/collect/worker.py's record_session_config).

        Returns the labels of what was actually restored, so the hint can say
        what changed rather than claim more than it did -- older files were
        written before some of these keys existed.
        """
        done = []
        for key, combo, label in (("reset_pose", self.win.reset_pose_combo, tr("Reset pose")),
                                  ("grip", self.win.grip_combo, tr("Grip"))):
            val = cfg.get(key)
            if val is None:
                continue
            i = combo.findText(str(val))
            if i >= 0:
                combo.setCurrentIndex(i)
                done.append(label)
        for key, edit, label in (("max_episode_seconds", self.win.eplen_edit, tr("에피소드 길이")),
                                 ("reset_wait_seconds", self.win.resetwait_edit, tr("리셋 대기"))):
            val = cfg.get(key)
            if val is not None:
                edit.setText(str(int(val)))
                done.append(label)
        if cfg.get("enable_wall") is not None:
            self.win.wall_check.setChecked(bool(cfg["enable_wall"]))
            done.append(tr("관절 한계 벽"))
        return done

    # on_start_sentence_edited 를 지웠다 (2026-09-06). 시작 문장을 손으로
    # 치는 길이 없어졌기 때문이다 -- 계획이 필수가 되면서 지시문은
    # instructions.json 에서만 온다 (scene_config_from_ui 가 막는다).
