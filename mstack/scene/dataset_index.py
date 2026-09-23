"""데이터셋 텍스트 색인 — scene HDF5 에서만 파생하는 3종 TSV.

scene_010.hdf5 는 13.9 GB 인데 그 안의 메타데이터를 읽는 데 필요한 바이트는
0.47 MB (0.0034%) 다. 그런데 읽기 요청이 865회라 원격(HTTP range)에서는
파일 하나에 70초가 걸린다. HDF5 를 페이지 정렬하면 요청이 6회로 줄지만
용량이 9~19% 늘어 못 쓴다. 그래서 **텍스트 색인**을 만든다 — 받기 전에
내용을 알고, 필요한 파일만 고르기 위한 것이다. 여기 적히는 값은 거의 전부
HDF5 (metadata 그룹 attrs · 에피소드 그룹 attrs · actions shape) 에서 온다.
유일한 예외는 ``kind`` 열이다 — 그것은 파일이 아니라 **계획 파일**
(instructions.json) 에서 온다. HDF5 에피소드 attrs 에는 kind 가 없고
앞으로도 없을 수 있다: kind 는 "이 슬롯이 무엇을 모으는 자리인가"를
말하는 계획 개념이라 파일 단위로 얽매이지 않는다. 계획이 없거나 읽기
실패하면 전부 ``task`` (기존 계획 파일의 기본값) 로 두고 색인은 끝까지
만든다 — 아래 견고성 규칙과 같은 취지다.

3종 TSV:

- ``scenes.tsv`` — scene 재고 한 줄 요약 (metadata 그룹 attrs).
- ``cells.tsv`` — (scene_id, instruction_id) **칸 단위** 요약. **train/eval
  subset 을 짜는 파일로 가장 중요하다.** 같은 칸의 에피소드를 train/eval 로
  쪼개면 거의 같은 장면·같은 문장이 양쪽에 들어가 누출(leakage)이 된다.
  칸 단위로 나누면 한 칸이 통째로 어느 한쪽에만 가므로 구조적으로 막힌다.
- ``episodes.tsv`` — 에피소드 상세 (에피소드 그룹 attrs + actions.shape[0]).

견고성 규칙:

- 옛 스키마 파일에 없는 attr 은 ``-`` 로 적고 넘어간다 (예외로 죽지 않는다).
- 파일 하나가 열리지 않아도 나머지는 만들고, 실패한 파일은 stderr 에 한 줄.
- 값에 탭·개행이 들어가면 공백으로 바꾼다 (TSV 라서).

cells.tsv 의 skill/object/target 은 지시문에서 파생한다. 스킬은
:func:`mstack.scene.instruction_grammar.skill_of` 가 정본이고, 역할(조작
물체·목적지) 해석은 ``scripts/analyze/audit_position_bias.py`` 의
``instruction_roles`` 와 같은 규칙을 라이브러리 안에 둔 것이다 (그 함수는
스크립트에 있어 라이브러리가 가져다 쓸 수 없다). 거기에 없는 것 하나:
stack-all 문장은 지칭 복수구를 인벤토리로 풀어 해당하는 oid 전부를 ``+`` 로
잇는다 — 같은 (색, 종류) 가 여럿이라 집합 지칭이 생긴 문장이므로.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py

from mstack.scene.collection_plan import KIND_TASK, load_plan
from mstack.scene.dataset_meta import plan_path
from mstack.scene.instruction_grammar import (
    _DRAG_RE,
    _PICK_RE,
    _STACK_RE,
    _TIDY_RE,
    _parse_group_phrase,
    resolve_reference,
    skill_of,
)
from mstack.scene.props import Prop, props_by_id
from mstack.scene.scene_format import (
    EPISODE_GROUP_RE,
    SceneMetadata,
    _read_metadata,
    iter_scene_files,
)

SCENES_HEADER = ("ordinal", "scene_id", "created", "n_episodes", "bytes",
                 "schema_version", "objects")
CELLS_HEADER = ("scene_id", "instruction_id", "kind", "skill", "object",
                "target", "n_episodes", "n_ok")
EPISODES_HEADER = ("episode_uid", "scene_id", "instruction_id", "kind",
                   "n_frames", "ok", "collector", "timestamp")


@dataclass
class DatasetIndex:
    """색인 결과. 행은 헤더 이름 -> 값 dict 이고, 값이 없는 자리는 None
    (TSV 로 쓸 때 ``-`` 가 된다). episodes 행에는 집계용 내부 키
    ``_instruction``/``_ok`` 가 붙는다 — TSV 헤더에 없어서 파일에는 안 나온다."""

    scenes: list
    cells: list
    episodes: list
    errors: list = field(default_factory=list)       # [(파일명, 오류)]
    paths: dict = field(default_factory=dict)        # {종류: 쓴 경로}


def _clean(value) -> str:
    """TSV 칸 안전 문자열. None -> ``-``, 탭·개행 -> 공백."""
    if value is None:
        return "-"
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _objects_list(raw) -> Optional[str]:
    """metadata 의 objects JSON attr -> 쉼표로 이어 붙인 문자열."""
    if raw is None:
        return None
    try:
        items = json.loads(str(raw))
    except (ValueError, TypeError):
        return None
    if not isinstance(items, list):
        return None
    return ",".join(str(o) for o in items)


def _instruction_roles(sentence: str, md: Optional[SceneMetadata],
                       props: "dict[str, Prop]") -> tuple:
    """지시문 -> (조작 물체 oid, 목적지 oid). 못 풀면 None.

    스크립트 ``audit_position_bias.py`` 의 ``instruction_roles`` 와 같은
    해석이다 (docstring 참고). QUALIFIER 가 붙은 지칭은 그대로 넘겨
    resolve_reference 가 물체 하나로 푼다.
    """
    if md is None:
        return None, None
    s = (sentence or "").strip().strip('"')
    if s in ("open the top drawer", "close the top drawer"):
        d = [o for o in md.objects
             if o in props and props[o].category == "drawer"]
        return (d[0] if d else None), None
    st = _STACK_RE.match(s)
    if st is not None:
        parsed = _parse_group_phrase(st.group("grp"))
        if parsed is None:
            return None, None
        color, cat = parsed
        oids = [o for o in md.objects
                if o in props and props[o].category == cat
                and props[o].color == color]
        return ("+".join(oids) if oids else None), None
    for rx in (_PICK_RE, _DRAG_RE, _TIDY_RE):
        m = rx.match(s)
        if m is None:
            continue
        qual = m.groupdict().get("qual")
        obj = "the " + m.group("obj") + (" " + qual if qual else "")
        return (resolve_reference(obj, md, props),
                resolve_reference("the " + m.group("tgt"), md, props))
    return None, None


def _as_text(value) -> str:
    """HDF5 attr 값 -> 비교용 문자열. 고정 길이 문자열 attr 은 bytes 로
    돌아오는데 계획의 scene_id/instruction_id 는 str 이라, 그대로 비교하면
    같은 값도 영원히 매칭이 안 된다."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _plan_kind_map(root: Path) -> dict:
    """계획 파일 -> {(scene_id, instruction_id): kind}.

    kind 는 파일에서 오는 게 아니라 계획에서 온다 — HDF5 에피소드 attrs 에
    kind 필드가 없고 앞으로도 없을 수 있다(수집 시점에는 전부 task 였고,
    종류는 나중에 계획에 붙은 개념이라 파일에 얽매이지 않는다). 계획 파일이
    없거나 읽기/검증에 실패하면 빈 사상을 돌려 전부 ``task`` 가 되게 한다 —
    색인은 계획 없이도 만들어져야 하고(옛 데이터셋), 깨진 계획이 색인까지
    죽이면 안 된다.
    """
    path = plan_path(root)
    if not path.is_file():
        return {}
    try:
        plan = load_plan(path)
    except Exception as e:  # noqa: BLE001 -- 깨진 계획은 kind 만 task 로 폰백
        print(f"[색인] 계획 읽기 실패 — kind 는 전부 task 로 둔다: "
              f"{path.name}: {e}", file=sys.stderr)
        return {}
    out = {}
    for sp in plan.scenes:
        for slot in sp.slots:
            out[(sp.scene_id, slot.instruction_id)] = slot.kind
    return out


def _kind_of(kind_map: dict, scene_id, instruction_id) -> str:
    """(scene_id, instruction_id) 의 kind. 사상에 없으면 ``task`` — 계획에
    없는 slot 의 옛 에피소드라도 색인 행 자체는 남아야 하므로, 기본값으로
    돌아간다."""
    if scene_id is None or instruction_id is None:
        return KIND_TASK
    return kind_map.get((_as_text(scene_id), _as_text(instruction_id)),
                        KIND_TASK)


def _scan_file(path: Path) -> tuple:
    """한 scene 파일 -> (scenes 행, episodes 행, metadata|None).

    attrs 만 읽는다 — 관측 청크는 건드리지 않아서 대용량 파일도 빠르다.
    """
    size = path.stat().st_size    # 심볼릭 링크면 대상 파일 크기 (stat 가 따라간다)
    with h5py.File(path, "r") as f:
        meta = f.get("metadata")
        mattrs = meta.attrs if isinstance(meta, h5py.Group) else {}
        meta_scene_id = (str(mattrs["scene_id"]) if "scene_id" in mattrs
                         else None)
        scene_row = {
            # **여기서는 비워 둔다.** 순번은 한 파일만 보고는 못 정한다 --
            # 폴더 전체를 created 순으로 세워야 나온다 (아래 _number_scenes).
            "ordinal": None,
            "scene_id": meta_scene_id,
            "created": mattrs.get("created"),
            "n_episodes": 0,
            "bytes": size,
            "schema_version": mattrs.get("dataset_version"),
            "objects": _objects_list(mattrs.get("objects")),
        }
        md = None
        if meta is not None:
            try:
                md = _read_metadata(meta)
            except Exception:  # noqa: BLE001 -- 옛/깨진 metadata 라도 행은 만든다
                md = None
        ep_rows = []
        names = sorted((k for k in f if EPISODE_GROUP_RE.match(k)),
                       key=lambda k: int(EPISODE_GROUP_RE.match(k).group(1)))
        for name in names:
            g = f[name]
            a = g.attrs
            actions = g.get("actions")
            ep_rows.append({
                "episode_uid": a.get("episode_uid"),
                # scene_id 는 에피소드 attr 이 정본. 없으면 metadata 로 폰백.
                "scene_id": a["scene_id"] if "scene_id" in a else meta_scene_id,
                "instruction_id": a.get("instruction_id"),
                "n_frames": actions.shape[0] if actions is not None else None,
                "ok": (1 if bool(a["success"]) else 0)
                      if "success" in a else None,
                "collector": a.get("collector"),
                "timestamp": a.get("timestamp"),
                "_instruction": a.get("instruction"),
                "_ok": bool(a["success"]) if "success" in a else False,
            })
        scene_row["n_episodes"] = len(ep_rows)
        return scene_row, ep_rows, md


def _write_tsv(path: Path, header: tuple, rows: list) -> None:
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(_clean(row.get(k)) for k in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _number_scenes(scenes: list) -> None:
    """``ordinal`` 을 **만든 순서**로 1 부터 매긴다 (제자리 수정).

    scene ID 가 불투명해지면서(``S7QK3M2A``) 사람이 "몇 번째 scene" 을 말할
    방법이 없어졌다. 그 자리를 이 열이 맡는다 -- R2R 처럼 "짧은 불투명 ID +
    별도의 넘버링 문서" 이고, 이 파일이 그 문서다.

    **순번은 표시용이지 식별자가 아니다.** 앞의 scene 을 지우면 뒤의 순번이
    당겨지고, 그래서 어디에도 저장하지 않는다 -- 저장하는 순간 2026-09-17 의
    재넘버링과 같은 것이 된다 (CLAUDE.md 의 규칙). 파일·에피소드·Hub 가
    쓰는 이름은 언제나 scene_id 다.

    created 를 모르는 파일은 뒤로 보낸다. 같은 값이면 scene_id 로 가른다 --
    무엇이든 한 가지로 정해져야 두 번 읽을 때 같은 표가 나온다.
    """
    order = sorted(
        range(len(scenes)),
        key=lambda i: (scenes[i].get("created") is None,
                       str(scenes[i].get("created") or ""),
                       str(scenes[i].get("scene_id") or "")))
    for n, i in enumerate(order, start=1):
        scenes[i]["ordinal"] = n


def build_dataset_index(root: Path,
                        out_dir: Optional[Path] = None) -> DatasetIndex:
    """``root`` 아래 scene 파일들을 돌아 3종 색인을 만들고 (``out_dir`` 가
    있으면) TSV 로 쓴다. 깨진 파일은 건어너뛰고 errors 에 남긴다."""
    root = Path(root)
    props = props_by_id()
    kind_map = _plan_kind_map(root)
    scenes, episodes, errors = [], [], []
    cells_acc: dict = {}
    for path in iter_scene_files(root):
        try:
            scene_row, ep_rows, md = _scan_file(path)
        except Exception as e:  # noqa: BLE001 -- 하나가 죽어도 나머지는 만든다
            msg = f"{type(e).__name__}: {e}"
            print(f"[색인 실패] {path.name}: {msg}", file=sys.stderr)
            errors.append((path.name, msg))
            continue
        scenes.append(scene_row)
        for r in ep_rows:
            # kind 는 파일에서 오지 않는다 — 스캔 후 계획 사상으로 붙인다.
            r["kind"] = _kind_of(kind_map, r["scene_id"], r["instruction_id"])
        episodes.extend(ep_rows)
        for r in ep_rows:
            key = (r["scene_id"], r["instruction_id"])
            cell = cells_acc.get(key)
            if cell is None:
                cell = cells_acc[key] = {
                    "scene_id": r["scene_id"],
                    "instruction_id": r["instruction_id"],
                    # 같은 칸의 instruction 은 실제로 같은 문장이다. 첫 것을
                    # 쓴다 (첫 것이 없는 경우는 attr 이 빠진 옛 파일뿐).
                    "instruction": r["_instruction"],
                    "md": md,
                    "n_episodes": 0,
                    "n_ok": 0,
                }
            cell["n_episodes"] += 1
            if r["_ok"]:
                cell["n_ok"] += 1
    cells = []
    for key in sorted(cells_acc, key=lambda k: (str(k[0]), str(k[1]))):
        cell = cells_acc[key]
        text = cell["instruction"] or ""
        skill = skill_of(text) if text else None
        obj, tgt = _instruction_roles(text, cell["md"], props)
        cells.append({
            "scene_id": cell["scene_id"],
            "instruction_id": cell["instruction_id"],
            "kind": _kind_of(kind_map, cell["scene_id"], cell["instruction_id"]),
            "skill": skill,
            "object": obj,
            "target": tgt,
            "n_episodes": cell["n_episodes"],
            "n_ok": cell["n_ok"],
        })
    _number_scenes(scenes)
    result = DatasetIndex(scenes=scenes, cells=cells, episodes=episodes,
                          errors=errors)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for kind, header in (("scenes", SCENES_HEADER),
                             ("cells", CELLS_HEADER),
                             ("episodes", EPISODES_HEADER)):
            p = out_dir / f"{kind}.tsv"
            _write_tsv(p, header, getattr(result, kind))
            result.paths[kind] = p
    return result
