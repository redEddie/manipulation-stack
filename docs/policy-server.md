# Policy server (`apps/policy_server.py`)

Serves one trained checkpoint over HTTP on the GPU machine. The FR3 controller
computer runs [`apps/fr3_policy_client.py`](policy-client.md), which sends
observations and executes the returned action chunks. The wire format is
defined once, in [`mstack/comm/policy_protocol.py`](../mstack/comm/policy_protocol.py),
and both ends import it.

## Layers

```
 GPU machine                       FR3 controller computer
┌───────────────────┐  HTTP   ┌────────────────────────┐  ZMQ    ┌──────────────┐  FCI 1 kHz  ┌─────┐
│ policy server     │ ──────▶ │ policy client          │ ──────▶ │ robot node   │ ──────────▶ │ FR3 │
│ apps/policy_      │ ◀────── │ apps/fr3_policy_       │ ◀────── │ pylibfranka  │             └─────┘
│   server.py       │         │   client.py            │         └──────────────┘
└───────────────────┘         └────────────────────────┘  ZMQ    ┌──────────────┐
                                          ▲  ────────────────────│ camera node  │◀── RealSense
                                          └──────────────────────└──────────────┘
```

| Layer | Owns | Does not know |
|---|---|---|
| Policy server | weights, the checkpoint's own pre/post-processing (camera renames, normalization, relative→absolute decoding), inference | which robot or cameras exist |
| Policy client | control rate, chunk scheduling (`CHUNK_LEAD`), image crop/resize, EE-delta IK, the per-step safety clamp | the model architecture |
| Robot / camera nodes | 1 kHz control, FCI, gripper, camera drivers ([architecture.md](architecture.md)) | that a policy is running at all |

A change stays in one layer: a new model touches only the server, a new arm only
the node. The 1 kHz loop never waits on the network.

## Protocol

| Endpoint | Body | Answer |
|---|---|---|
| `GET /info` | — | `PolicyInfo`: what the checkpoint expects (below) |
| `POST /reset` | `{"instruction": str}` | `{"status": "ok", "instruction": str}` |
| `POST /predict` | observation | `{"actions": [[action_dim] × chunk_size]}` |
| `POST /step` | `{"observation.state": [8]}` | `{"action": [8]}` — server-side step, for clients that do not re-anchor locally |
| `POST /infer` | observation | legacy combined path, kept for old clients |

Observation: `observation.state` (8 = joint1-7 rad + gripper 0..1) and
`observation.images.<camera>` as `encode_image(...)` — uint8, `image_size`²,
already square-cropped by the client. Errors come back as `{"error": "..."}`
with HTTP 400 (bad request: wrong image size, no instruction yet) or 500; the
server keeps serving either way.

### `/info` — the model's own facts

```json
{"policy": "smolvla", "fps": 20, "image_size": 224, "cameras": ["agent", "wrist"],
 "state_dim": 8, "action_type": "joint_absolute", "action_dim": 8, "chunk_size": 50, ...}
```

The client takes its control rate, image size and chunk length from here, and
refuses to connect the robot if a camera or the state length does not match.

**`fps` is the rate of the training data, not a property of the model family.**
Chunk index *i* means observation time + *i*/fps. A π0.5 trained on DROID runs at
15 Hz because DROID is 15 Hz; the same π0.5 fine-tuned on fr3-tabletop runs at
20 Hz. The LeRobot backend therefore reads fps from the training dataset's
metadata, never from a default.

Servers without `/info` (mamba-embeddingvla `real_deploy`) answer 404/405/501;
the client then assumes `LEGACY_INFO` — 20 Hz, 256², 10-step chunks — which is
what those servers were built for.

## Running

### LeRobot checkpoints (SmolVLA, π0, π0-FAST, GR00T)

Run inside a venv that can load the checkpoint (for GR00T and the π family, one
with `lerobot[pi,groot]`):

```bash
python apps/policy_server.py lerobot \
    --ckpt <run>/checkpoints/last/pretrained_model --port 8080
```

| Option | Meaning |
|---|---|
| `--n-action-steps N` | actions returned per `/predict` (default: the checkpoint's `n_action_steps`) |
| `--fill-from-state KEY ...` | feed the measured state into an input the client cannot send (see below) |

Everything model-specific comes from the checkpoint directory
(`config.json`, `train_config.json`, the saved processors): the request is built
like a LeRobotDataset sample and goes through the same
preprocessor → `predict_action_chunk` → postprocessor as offline evaluation.
That is why there is no per-model code here:

| Model | What the checkpoint's processors do for you |
|---|---|
| SmolVLA | applies the training `--rename_map` (e.g. `agent`/`wrist` → `camera1`/`camera2`); skips the absent `camera3` |
| π0 / π0-FAST | applies the training `--rename_map` (e.g. → `base_0_rgb`/`left_wrist_0_rgb`); feeds the absent `right_wrist_0_rgb` as a masked blank image |
| GR00T N1.7 | caches the state at prediction time and adds it back to the relative chunk — the robot always receives absolute joints. The whole chunk is decoded at once (the per-step `select_action` queue only approximates this and is not used) |

**Inputs the robot cannot provide.** Training with `--policy.type=<x>` declares
*every* dataset feature as an input, so fr3-tabletop checkpoints list
`observation.commanded_state` — the GELLO leader command, which does not exist
when no one is teleoperating. Declared is not the same as read: SmolVLA, π0,
π0-FAST and GR00T read only `observation.state`. The server settles it at start-up
by running the warm-up inference without those inputs:

- it succeeds → the inputs are unused; the server says so and lists them in
  `/info` under `extra.unused_inputs` (seen on the π0 10k checkpoint, 2026-10-07);
- it fails → the model really needs them; the server refuses to start. Retrain
  without them, or pass `--fill-from-state observation.commanded_state` to feed
  the measured state instead — an approximation that must be reported with any
  result.

**Same numbers as offline evaluation.** The request is turned into tensors exactly
as `LeRobotDataset` does, down to dividing pixels by 255 on the CPU: CUDA divides
by a scalar through its reciprocal, which moves about a quarter of the pixels by
one ulp, and flow-matching integration amplifies that to chunk differences of up
to ~1e-2 rad. `--seed N` reseeds before every `/predict` so the sampling noise
repeats; `check_policy_server.py --parity` then compares the server's chunk with
the offline path and expects zero difference.

### mamba-embeddingvla checkpoints

```bash
python apps/policy_server.py mamba --ckpt <ckpt>.pt --mamba-root <mamba-embeddingvla checkout> \
    --port 8080 --cuda-graph --tf32
```

Ported from `real_deploy/fr3_policy_server.py`; preprocessing is unchanged
(256² → bilinear 224 → ImageNet normalize, raw 480×640 also accepted). `/info`
reports 256² so the client keeps sending what the model was trained on.
`--fps` (default 20) must match the training data.

## Checking a server before the robot

1. **Recorded frames** — sends the first frame of real episodes and compares the
   chunk with what was recorded. Also confirms the server rejects a wrong-size
   image. Runs on the GPU machine (it reads the dataset):

   ```bash
   scripts/check/check_policy_server.py --server http://127.0.0.1:8080 --manifest <eval>.json --limit 10
   ```

   With the server started with `--seed 7`, add `--parity --parity-tol 0`: the
   script also runs the checkpoint the way offline evaluation does and requires
   the server's chunk to be identical. Results on 2026-10-07:

   | Checkpoint | `/info` | Parity (max \|server − offline\|) | Round-trip |
   |---|---|---|---|
   | SmolVLA 10k (`jeo65b_smolvla10k_biased_s0`) | 20 Hz, 224², joint_absolute[8] × 50 | 0.0 over 12 chunks | ~120 ms |
   | π0 10k (`jeo65b_pi0_10k_biased_s0`) | same; `unused_inputs: [observation.commanded_state]` | 0.0 over 8 chunks | ~150 ms |

2. **Client path** — from the robot computer, no robot or camera needed:

   ```bash
   python apps/fr3_policy_client.py --server http://<gpu>:8080 --dry-run
   ```

   It prints the worst round-trip and the `--lead-ticks` it needs. On
   2026-10-06 SmolVLA measured 280–350 ms while the GPU was shared with three
   training runs — that needs `--lead-ticks 7` at 20 Hz, not the default 2.
   On 2026-10-07 with one other job on the GPU: SmolVLA ~120 ms, π0 ~150 ms
   (`--lead-ticks 3`). Measure on the day before choosing.

## What these checks do NOT catch

| Check | Catches | Does not catch |
|---|---|---|
| `/info` match in the client | wrong rate, image size, camera set, state length, action type | a wrong *crop*: `crop_params.json` on the robot computer must be the one the data was recorded with |
| Server image-size guard (400) | a client sending 256² to a 224² model | a correctly sized image from the wrong camera |
| `check_policy_server.py` (+ `--parity`) | broken normalization, relative actions left relative, swapped cameras (error jumps); with `--parity`, any difference between the deploy path and offline evaluation | closed-loop behaviour — the policy never sees the effect of its own actions |
| Client safety clamp (`MAX_STEP_RAD`) | runaway targets far from the measured pose | a wrong but smooth trajectory (see the safety table in `CLAUDE.md`) |
