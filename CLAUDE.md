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

## Verification

- Grammar/rules: `python -m mstack.scene.instruction_grammar`, `python -m mstack.scene.scene_rules`
- Setpoint-gap safety limit: `scripts/analyze/setpoint_gap.py` reads the 1 kHz
  raw logs and says how far `MAX_SETPOINT_GAP_RAD` (0.9 rad) can come down.
  It is meant to come down — 0.9 is 1.7x the measured worst case, nothing more.
  Try a candidate with `MSTACK_MAX_SETPOINT_GAP=<rad>` (inherited by the robot
  node from whatever launched it) before changing the source.
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
- One checkout for now: `~/teleop-franka/manipulation-stack` on `main`, which
  the desktop icon "Scene 데이터 수집기" runs. It carries a gitignored
  `configs/stations/*.local.yaml` and a `.venv` symlink; a second worktree would
  need its own copies of both.
- Small, verified changes may land on `main` directly while that is the only
  checkout (operator's call, 2026-09-08). For anything systemic, add the `dev`
  worktree first and merge `--ff-only` into `main` after the suite passes there
  — `main` is what the operator runs, so nothing lands there unverified.
- The icon runs `git pull --ff-only` before launching, so **whatever is on
  `origin/main` is what the operator gets on the next launch.** Say so before
  pushing anything they are about to test on hardware.
- History note: this repository was split out of the `gello_software` fork with
  a squashed initial commit. Anything older than that lives at
  `redEddie/gello_software`; `NOTICE` records which files came from upstream.
- Other agents may be editing the same files concurrently. Check
  `git status` before committing and stage your own paths explicitly rather
  than `git add -A`.
