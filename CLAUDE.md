# manipulation-stack (KNU FR3 + GELLO station)

One repository for the whole station: teleoperation, scene-based data
collection, dataset schema/QC/upload, and VLA policy deployment. Collection and
deployment share the same station config, crop params and image pipeline
(`mstack.data.crop.resize_rgb` is called both when the dataset is written and
when the policy client sends an observation) — that is why they live together.

## Python environment — read this first

There is NO repo-local venv and bare `python` is not on PATH.

- **Everything** (GUI, library, tests, scripts): `~/lerobot-venv/bin/python`
  — also reachable as `.venv/bin/python` (a machine-local symlink, gitignored).
- **Robot node only** (`scripts/launch/launch_nodes.py`): `~/pylibfranka-venv/bin/python`.
  Never use it for anything else.

## Layout = dependency direction (user's rule, 2026-08-31)

Dependencies must be readable from the tree without opening files.

Arrows point down only. `tests/gui/test_layer_rules.py` enforces this.

- `mstack/config/` — **Shared leaf: imports nothing.** Constants both the robot
  and the GUI must agree on, station config, episode-quality vocabulary.
- `mstack/core/` — abstractions others depend on (robot/agent/camera/env).
  Stubs (PrintRobot, DummyCamera) live here too; no vendor SDK may.
- `mstack/robots|agents|cameras|hw/` — concrete implementations. Nothing in
  `data/`, `scene/` or `gui/` may import these: a data logger must not know
  whether the arm is an FR3 or a simulator.
- `mstack/data/` — dataset schema/formats. `mstack/scene/` — scene format, rules,
  props, grammar, plans, Hub sync. `scene` may use `data`, never the reverse.
- `mstack/collect/` — the collection engine (drives leader+robot+cameras).
  Emits Qt signals but is not a widget; it is allowed to know concrete hardware.
- `mstack/gui/` — Qt only (widgets, dialogs, workers). `mstack/comm/` — zmq nodes;
  the process boundary is itself the abstraction, so `gui` may use it.
- `apps/workspace/features/<기능>/`   한 기능의 도메인·화면·대화상자
- `apps/workspace/shell/`             창 골격과 앱 수준 설정 페이지
- `apps/workspace/shared/`            여러 기능이 함께 쓰는 위젯·헬퍼
- `scripts/launch|check|calib|convert|analyze/` — purpose-split tools.
- `experiments/` — research only; nothing in mstack/ may import from it.
- `configs/stations|scenes|collection/` — config data. A station file is
  committed; the values that identify this machine (robot IP, camera serials,
  policy server) live in `<name>.local.yaml`, which `.gitignore` covers and
  `load_station()` deep-merges over the committed file. Never move an
  identifying value back into the tracked file — the repository is public.

Keep this invariant when adding files; a module's folder must announce its role.

## Safety layers — and what each one does NOT catch

There is no `mstack/safety/`: each layer lives with the mechanism it drives
(`joint_limit_wall` cannot leave `robots/` — it shares the leader's Dynamixel
port). This list is the index instead. **The right-hand column is the point.**
On 2026-09-10 the team found that the layer everyone assumed was protecting the
arm was not a safety feature at all, and that a limit named for dropped leaders
never fired on one. Never add a layer here without saying what it misses.

| Layer | Where | Catches | Does NOT catch |
|---|---|---|---|
| Collision reflex (100/40 N·m, 100 N) | `robots/franka_fr3.py` | impacts, abnormal wrist load | anything below threshold; deliberate contact is meant to pass |
| Reference filter (v 1.5 / a 6 / jerk 3000) | `robots/franka_fr3.py` | commands the arm cannot execute | *where* the command goes — it will faithfully chase a wrong pose at v_max |
| Leader joint-limit wall | `robots/joint_limit_wall.py` | leader poses the follower cannot reach; cable wind-up | anything inside the follower's range |
| Leader-drop guard (pitch L2 > 2.1 rad/s over 100 ms) | `collect/leader_guard.py` → `hold()` | a released or runaway leader during recording | roll-only motion (gravity exerts no moment there); the VLA path, which has no leader |
| `hold()` | `robots/franka_fr3.py` | — (mechanism, not a detector) | — |

Two things that look like safety layers and are not:

* **The `acceleration_discontinuity` reflex.** Until 2026-09-10 this was what
  actually stopped the arm when a leader was dropped — a side effect of
  libfranka refusing an impossible command, never a designed guard. Raising
  v/a and fixing the v_max taper removed it. The leader-drop guard replaces it.
* **Setpoint-vs-filter gap.** Tried and removed the same day: a real dropped
  leader only opened a 0.322 rad gap, while ordinary teleop has reached 0.532.
  A released leader falls *within* the follower's v_max, so the gap never
  grows. Do not reintroduce this without new evidence.

## Verification

- Grammar/rules: `python -m mstack.scene.instruction_grammar`, `python -m mstack.scene.scene_rules`
- Leader-drop safety threshold: `scripts/analyze/leader_speed.py [--sweep]`
  replays the 1 kHz raw logs through the *same* guard class the worker runs.
  `LEADER_DROP_SPEED_RAD_S` (2.1, over the pitch joints only — gravity exerts
  no moment about a roll axis) rests on one drop event and 28.9 s of normal
  teleop — re-run this as sessions accumulate and watch the normal-side
  headroom, because one false stop is enough for an operator to switch a
  safety layer off.
- Full GUI acceptance suite (offscreen, no hardware):
  `bash tests/gui/run_all.sh ~/lerobot-venv/bin/python`
- The collector is launched by desktop icon via `run_scene_collector.sh`
  (auto `git pull --ff-only`, then `apps/collect_launcher.py` — the wizard,
  which opens `collect_workspace` when it finishes) — keep the working tree
  clean/committed or the pull is skipped. Machine-local files (`.venv`,
  `configs/stations/*.local.yaml`) are gitignored precisely so editing them
  never dirties the tree.
- Entry points under `apps/` and `scripts/` must insert the repo root into
  `sys.path` **before** importing `mstack`; `tests/gui/test_script_bootstrap.py`
  enforces it for both directories. Nothing installs `mstack` into the venvs, so
  a missing bootstrap is an immediate `ModuleNotFoundError`.

## Conventions

- User-facing GUI strings: **no language toggle** (removed 2026-09-05; `tr()`
  is identity). Fix the language per string instead, by tier — Action / Status /
  Identity in English (`Save`, `RECORDING`, `cam1`, `S015`), Guide / Log in
  Korean. The full rationale is the `mstack/gui/i18n.py` docstring; read it
  before adding strings. New comments/docs/issues in English (issue #42).
- GUI layout rules live in `apps/workspace/shared/sizing.py` — field heights
  derived from the font (`roomy`), anything that can grow wrapped in
  `scrollable`, no horizontal scrolling. Wheel-over-combo is neutralised
  app-wide by `mstack/gui/wheel_guard.py`, so screens need not handle it.
  These came from real breakage in the launcher (2026-09-05~06); apply them
  when touching workspace screens rather than re-deriving them.
- Prop/scene decisions are recorded on GitHub issues (e.g. #36); props.yaml
  color tokens must be lowercase (the grammar parser lowercases phrases).
- This repository is **public**. Do not commit hostnames, serials or paths that
  identify a machine outside this lab: the policy server address belongs in
  `policy.url` of the station file or in `MSTACK_POLICY_SERVER`, never in
  source. Korean comments are fine and stay; new comments/docs/issues in
  English (issue #42, carried over).
- Two checkouts, one desktop icon each (2026-09-10):
  `~/teleop-franka/manipulation-stack` on `main` → "Scene 데이터 수집기"
  (real collection), and `~/teleop-franka/manipulation-stack-dev` on `dev` →
  "Scene 수집기 [dev]" (pre-merge checking). Each carries its own gitignored
  `configs/stations/*.local.yaml` and `.venv` symlink — a new worktree needs
  copies of both or it will not start.
  The dev icon sets `GELLO_STATE_DIR=~/libero_gui_logs_dev` so its logs,
  settings and 1 kHz raw windows stay out of the collection ones (a shared raw
  log dir means one session's rolling windows evict the other's — that
  destroyed evidence on 2026-09-10). **The dataset root is a setting, not an
  env var**, so a fresh dev state dir still defaults to the real
  `~/libero_datasets`; point it elsewhere on the dev GUI's first launch.
  The two cannot run at once: same ZMQ ports, and the FCI accepts one client.
- `dev` exists now, so systemic work goes there first and reaches `main` by
  `--ff-only` merge after the suite passes *and* the operator has run it on the
  robot. `main` is what the icon pulls for real collection. (The leader-drop
  safety layer is the exception on record: it landed on `main` on 2026-09-10
  before any hardware run, because the operator asked for it live rather than
  in shadow mode.)
- The icon runs `git pull --ff-only` before launching, so **whatever is on
  `origin/main` is what the operator gets on the next launch.** Say so before
  pushing anything they are about to test on hardware.
- History note: this repository was split out of the `gello_software` fork with
  a squashed initial commit. Anything older than that lives at
  `redEddie/gello_software`; `NOTICE` records which files came from upstream.
- Other agents may be editing the same files concurrently. Check
  `git status` before committing and stage your own paths explicitly rather
  than `git add -A`.
