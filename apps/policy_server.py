"""Policy server -- runs on the GPU machine, serves one checkpoint over HTTP.

The FR3 controller computer runs apps/fr3_policy_client.py, which sends
observations (agent/wrist images + 8-dim joint state) and executes the returned
action chunk. Everything model-specific stays on this side: weights, camera-name
mapping, normalization, relative->absolute action decoding. The client only
learns what it must obey -- fps, image size, chunk length, action type -- from
GET /info. Wire format: mstack/comm/policy_protocol.py.

Backends:

  lerobot  any LeRobot checkpoint (SmolVLA, pi0, pi0-FAST, GR00T, ...). The
           checkpoint's own pre/post-processors run unchanged, so what the robot
           gets is exactly what offline evaluation scored. fps, cameras and image
           size come from the training dataset's metadata.
  mamba    mamba-embeddingvla VLA-JEPA checkpoints (ported from its
           real_deploy/fr3_policy_server.py). Needs that repository on disk.

Run (GPU machine, inside the venv that can load the checkpoint):

  python apps/policy_server.py lerobot --ckpt <run>/checkpoints/last/pretrained_model --port 8080
  python apps/policy_server.py mamba --ckpt <ckpt>.pt --mamba-root <mamba-embeddingvla> \\
      --port 8080 --cuda-graph --tf32

See docs/policy-server.md.
"""
from __future__ import annotations

import os
import sys

# Repo root first -- see tests/gui/test_script_bootstrap.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from mstack.comm.policy_protocol import (  # noqa: E402
    EE_DELTA,
    IMAGE_KEY_PREFIX,
    JOINT_ABSOLUTE,
    STATE_KEY,
    PolicyInfo,
    decode_observation,
)


class RequestError(Exception):
    """A malformed or incompatible request -- answered with HTTP 400, not 500."""


# ───────────────────────────── LeRobot backend ─────────────────────────────

class LeRobotBackend:
    """Serves a LeRobot `pretrained_model` directory.

    The request is built exactly like a LeRobotDataset sample (float CHW images in
    [0, 1], dataset feature names, a task string), then goes through the
    checkpoint's saved preprocessor -> predict_action_chunk -> postprocessor.
    That path is shared with offline evaluation, so camera renames (e.g. agent ->
    camera1), normalization and GR00T relative->absolute decoding are the
    checkpoint's own, not re-implemented here.
    """

    def __init__(self, ckpt: str, device: str = "cuda", n_action_steps: int | None = None,
                 fill_from_state: list[str] | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self._torch = torch
        self.ckpt = Path(ckpt)
        self.device = device
        train_cfg = json.loads((self.ckpt / "train_config.json").read_text())

        cfg = PreTrainedConfig.from_pretrained(self.ckpt)
        cfg.pretrained_path = self.ckpt
        cfg.device = device
        self.cfg = cfg
        print(f"[server] loading {cfg.type} from {self.ckpt}", flush=True)
        self.policy = get_policy_class(cfg.type).from_pretrained(self.ckpt).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=self.ckpt,
            preprocessor_overrides={"device_processor": {"device": device}})

        ds = train_cfg["dataset"]
        self.meta = LeRobotDatasetMetadata(ds["repo_id"], root=ds.get("root"), revision=ds.get("revision"))
        self.cameras = [k[len(IMAGE_KEY_PREFIX):] for k in self.meta.camera_keys]
        shapes = {tuple(self.meta.features[k]["shape"]) for k in self.meta.camera_keys}
        if len(shapes) != 1:
            raise SystemExit(f"cameras have different shapes {shapes}; one image_size cannot describe them")
        h, w, _c = shapes.pop()
        if h != w:
            raise SystemExit(f"training images are {h}x{w}; the client only sends squares")
        self.image_size = h
        self.state_dim = self.meta.features[STATE_KEY]["shape"][0]
        self.fill_from_state = list(fill_from_state or [])
        self._check_inputs(train_cfg.get("rename_map") or {})

        action_shape = cfg.output_features["action"].shape
        self.action_dim = int(action_shape[0])
        self.n_action_steps = min(n_action_steps or cfg.n_action_steps, cfg.chunk_size)
        self.instruction: str | None = None

    def _check_inputs(self, rename_map: dict) -> None:
        """Every non-image input the policy needs must be something the client sends.

        Images the policy declares but the dataset never had (smolvla_base's camera3,
        pi0's right_wrist) are padded by the policy itself. A missing *state-like*
        input is different: a GR00T trained on this dataset without an explicit
        input list also conditions on observation.commanded_state -- the GELLO leader
        command, which does not exist when no one is teleoperating.
        """
        from lerobot.configs.types import FeatureType

        inverse = {v: k for k, v in rename_map.items()}
        sent = {STATE_KEY, *self.meta.camera_keys}
        missing = []
        for name, ft in self.cfg.input_features.items():
            src = inverse.get(name, name)
            if src in sent or ft.type is FeatureType.VISUAL:
                continue
            if src in self.fill_from_state:
                print(f"[server] WARNING: {src} is fed the measured state (--fill-from-state); "
                      f"training saw a different signal there", flush=True)
                continue
            missing.append(src)
        if missing:
            raise SystemExit(
                f"checkpoint needs inputs the client does not send: {missing}. Retrain without "
                f"them, or pass --fill-from-state {' '.join(missing)} to feed the measured joint "
                f"state in their place (an approximation -- say so in any result).")

    def info(self) -> PolicyInfo:
        return PolicyInfo(
            policy=self.cfg.type, checkpoint=str(self.ckpt), fps=int(self.meta.fps),
            image_size=self.image_size, cameras=self.cameras, state_dim=self.state_dim,
            action_type=JOINT_ABSOLUTE, action_dim=self.action_dim, chunk_size=self.n_action_steps,
            extra={"dataset": self.meta.repo_id, "model_chunk_size": self.cfg.chunk_size,
                   "fill_from_state": self.fill_from_state})

    def reset(self, instruction: str | None) -> str:
        if instruction:
            self.instruction = instruction
        if not self.instruction:
            raise RequestError("no instruction yet: POST /reset {\"instruction\": ...} first")
        self.policy.reset()
        return self.instruction

    def predict(self, obs: dict) -> np.ndarray:
        torch = self._torch
        if not self.instruction:
            raise RequestError("no instruction yet: POST /reset {\"instruction\": ...} first")
        state = np.asarray(obs.get(STATE_KEY, []), dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise RequestError(f"{STATE_KEY} must have {self.state_dim} values, got {state.shape}")
        batch = {STATE_KEY: torch.from_numpy(state)[None].to(self.device)}
        for key in self.fill_from_state:
            batch[key] = batch[STATE_KEY]
        for cam in self.cameras:
            key = IMAGE_KEY_PREFIX + cam
            if key not in obs:
                raise RequestError(f"missing {key}")
            img = np.asarray(obs[key])
            if img.shape != (self.image_size, self.image_size, 3) or img.dtype != np.uint8:
                raise RequestError(
                    f"{key} is {img.dtype}{list(img.shape)}; this checkpoint was trained on uint8 "
                    f"[{self.image_size}, {self.image_size}, 3] -- resize on the client with the "
                    f"same resize_rgb the dataset was written with (GET /info gives the size)")
            t = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)
            batch[key] = t.permute(2, 0, 1).float().div_(255.0)[None]
        batch["task"] = self.instruction
        with torch.inference_mode():
            chunk = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        chunk = chunk.squeeze(0).float().cpu().numpy()
        return chunk[: self.n_action_steps]

    def warmup(self) -> None:
        """One throwaway inference so the first real request does not pay for CUDA init."""
        s = self.image_size
        obs = {STATE_KEY: np.zeros(self.state_dim, np.float32),
               **{IMAGE_KEY_PREFIX + c: np.zeros((s, s, 3), np.uint8) for c in self.cameras}}
        prev, self.instruction = self.instruction, "warmup"
        try:
            self.policy.reset()
            self.predict(obs)
        finally:
            self.instruction = prev
            self.policy.reset()


# ───────────────────────────── mamba backend ─────────────────────────────

class MambaBackend:
    """mamba-embeddingvla VLA-JEPA (DINOv3 + Gemma). Ported from its
    real_deploy/fr3_policy_server.py; the preprocessing below is that file's,
    unchanged, because it mirrors the cache builder the model was trained on:
    256^2 uint8 -> bilinear 224 -> ImageNet normalize. Raw 480x640 is still
    accepted (center square crop, area resize to 256 first).
    """

    def __init__(self, ckpt: str, mamba_root: str, device: str = "cuda", n_inference_steps: int = 10,
                 instruction: str | None = None, cuda_graph: bool = False, fps: int = 20,
                 debug_dump: str | None = None):
        root = Path(mamba_root).resolve()
        sys.path[1:1] = [str(root), str(root / "real_deploy")]   # after the repo root
        from argparse import Namespace

        import torch

        from mamba_embeddingvla.config import IMAGENET_MEAN, IMAGENET_STD
        from mamba_embeddingvla.eval.eval_vlajepa import load_dinov3
        from mamba_embeddingvla.train.builders import build_model
        from mamba_embeddingvla.utils.gemma import load_gemma

        self._torch = torch
        self.ckpt = ckpt
        self.device = device
        self.fps = fps
        self.n_inference_steps = n_inference_steps
        self.instruction = instruction
        self._graph = None
        self._dump_dir = debug_dump
        self._dump_n = 0
        if debug_dump:
            os.makedirs(debug_dump, exist_ok=True)

        print(f"[server] loading ckpt: {ckpt}", flush=True)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        args = state["args"]
        args = Namespace(**args) if isinstance(args, dict) else args
        self.chunk_size = args.chunk_size
        self.text_max_tokens = getattr(args, "text_max_tokens", 24)

        self.model = build_model(args, device)
        missing, unexpected = self.model.load_state_dict(state["model_state"], strict=False)
        print(f"[server] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval()
        assert self.model.action_scaler.fitted, "action_scaler not fitted in ckpt!"
        self.action_dim = self.model.action_emb.weight.shape[1]
        # 7-dim = EE-frame delta policy; the client re-anchors to the measured pose + IK.
        print(f"[server] action_dim={self.action_dim} gripper_dims={self.model.gripper_dims} "
              f"chunk={self.chunk_size} mode={'EE-delta' if self.action_dim == 7 else 'joint-absolute'}")

        self.text_encoder, self.tokenizer = load_gemma(device)
        self.text_encoder.eval()
        self.vision_encoder = load_dinov3(device, d_dino=getattr(args, "d_dino", 768))
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        if hasattr(self.model, "set_history_stride"):
            self.model.set_history_stride(1)
        if self.instruction:
            self.reset(self.instruction)
        if cuda_graph:
            from graph_runner import CudaGraphPolicyRunner
            print("[server] capturing CUDA graphs (encode + flow)...")
            self._graph = CudaGraphPolicyRunner(self.model, self.vision_encoder, self.text_encoder,
                                                n_inference_steps=self.n_inference_steps)
            if self.instruction:
                self.reset(self.instruction)

    def info(self) -> PolicyInfo:
        return PolicyInfo(
            policy="mamba-vlajepa", checkpoint=str(self.ckpt), fps=self.fps, image_size=256,
            cameras=["agent", "wrist"], state_dim=8,
            action_type=EE_DELTA if self.action_dim == 7 else JOINT_ABSOLUTE,
            action_dim=int(self.action_dim), chunk_size=int(self.chunk_size),
            extra={"n_inference_steps": self.n_inference_steps})

    def _preprocess(self, img_uint8: np.ndarray):
        F = self._torch.nn.functional
        h, w = img_uint8.shape[:2]
        s = min(h, w)
        t, left = (h - s) // 2, (w - s) // 2
        x = img_uint8[t:t + s, left:left + s]
        x = self._torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        x = x.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        if s > 256:
            # collection used cv2 INTER_AREA to 256; area pooling is the closest torch analog
            x = F.interpolate(x, size=(256, 256), mode="area")
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self._mean) / self._std

    def reset(self, instruction: str | None) -> str:
        torch = self._torch
        if instruction:
            self.instruction = instruction
        if not self.instruction:
            raise RequestError("no instruction yet: POST /reset {\"instruction\": ...} first")
        with torch.no_grad():
            self.tokenizer.padding_side = "left"
            tok = self.tokenizer([self.instruction], return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=self.text_max_tokens).to(self.device)
            task_seq = self.text_encoder(**tok).last_hidden_state.float()
            if self._graph is not None:
                self._graph.reset(task_seq, tok["attention_mask"])
            else:
                self.model.reset_recurrent_state(1)
                self.model.set_text_prefix(task_seq, tok["attention_mask"])
        return self.instruction

    def predict(self, obs: dict) -> np.ndarray:
        torch = self._torch
        if not self.instruction:
            raise RequestError("no instruction yet: POST /reset {\"instruction\": ...} first")
        state = np.asarray(obs[STATE_KEY], dtype=np.float32)
        if self._dump_dir and self._dump_n % 10 == 0:
            from PIL import Image
            for k in ("agent", "wrist"):
                Image.fromarray(obs[IMAGE_KEY_PREFIX + k]).save(f"{self._dump_dir}/req{self._dump_n:05d}_{k}.png")
        self._dump_n += 1
        with torch.no_grad():
            agent_t = self._preprocess(obs[IMAGE_KEY_PREFIX + "agent"])
            wrist_t = self._preprocess(obs[IMAGE_KEY_PREFIX + "wrist"])
            gripper = torch.from_numpy(state[-self.model.gripper_dims:]).to(self.device).unsqueeze(0)
            if self._graph is not None:
                return self._graph.infer(agent_t, wrist_t, gripper).cpu().numpy()
            vis, text_out = self.model.encode_vision_step(
                agent_t, wrist_t, self.vision_encoder, gripper_qpos=gripper, text_encoder=self.text_encoder)
            return self.model.generate_actions(vis, text_out, self.n_inference_steps)[0].cpu().numpy()

    def warmup(self) -> None:
        pass   # CUDA-graph capture already ran the model once


# ───────────────────────────── HTTP layer ─────────────────────────────

class PolicyServer:
    """Backend-agnostic endpoints. Single-threaded on purpose: one GPU, one robot,
    and requests must not interleave with /reset."""

    def __init__(self, backend):
        self.backend = backend
        self.info = backend.info()
        self.info.validate()
        self._chunk: np.ndarray | None = None   # cached for /step
        self._idx = 0
        self.predict_ms: list[float] = []

    def predict(self, req: dict) -> dict:
        t0 = time.perf_counter()
        chunk = np.asarray(self.backend.predict(decode_observation(req)), dtype=float)
        self.predict_ms.append((time.perf_counter() - t0) * 1000)
        if len(self.predict_ms) % 50 == 1:
            p50 = float(np.median(self.predict_ms[-50:]))
            print(f"[server] /predict #{len(self.predict_ms)}: chunk {chunk.shape}, "
                  f"{self.predict_ms[-1]:.0f} ms (median {p50:.0f})", flush=True)
        self._chunk, self._idx = chunk, 0
        return {"actions": chunk.tolist()}

    def step(self, req: dict) -> dict:
        """One absolute joint action from the cached chunk, re-anchored to the given
        measured state in EE mode. The client normally does this locally."""
        if self._chunk is None:
            raise RequestError("/step before /predict")
        d = self._chunk[min(self._idx, len(self._chunk) - 1)]
        self._idx += 1
        if self.info.action_type == EE_DELTA:
            from mstack.robots.fr3_kinematics import ee_step_to_joint
            d = ee_step_to_joint(d, np.asarray(req[STATE_KEY], dtype=float)[:7])
        return {"action": np.asarray(d, dtype=float).tolist()}

    def infer(self, req: dict) -> dict:
        """Legacy combined path (policy + fixed-anchor whole-chunk IK). Prefer /predict."""
        out = self.predict(req)
        if self.info.action_type == EE_DELTA:
            from mstack.robots.fr3_kinematics import ee_chunk_to_joint_chunk
            q = np.asarray(req[STATE_KEY], dtype=float)[:7]
            out = {"actions": ee_chunk_to_joint_chunk(np.asarray(out["actions"]), q).tolist()}
        return out

    def run(self, host: str, port: int) -> None:
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/info":
                    self._send(200, srv.info.to_dict())
                else:
                    self._send(404, {"error": f"unknown endpoint {self.path}"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                try:
                    req = json.loads(self.rfile.read(n) if n else b"{}")
                    if self.path == "/reset":
                        resp = {"status": "ok", "instruction": srv.backend.reset(req.get("instruction"))}
                        srv._chunk = None
                    elif self.path == "/predict":
                        resp = srv.predict(req)
                    elif self.path == "/step":
                        resp = srv.step(req)
                    elif self.path == "/infer":
                        resp = srv.infer(req)
                    else:
                        self._send(404, {"error": f"unknown endpoint {self.path}"})
                        return
                except RequestError as e:
                    print(f"[server] 400 {self.path}: {e}", flush=True)
                    self._send(400, {"error": str(e)})
                    return
                except Exception as e:   # keep serving; the client shows the message
                    traceback.print_exc()
                    self._send(500, {"error": f"{type(e).__name__}: {e}"})
                    return
                self._send(200, resp)

            def log_message(self, *a):
                pass

        i = self.info
        print(f"[server] {i.policy}: {i.fps} Hz, {i.image_size}^2 {i.cameras}, "
              f"{i.action_type}[{i.action_dim}] x {i.chunk_size}", flush=True)
        print(f"[server] listening on {host}:{port}", flush=True)
        HTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="backend", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", required=True)
    common.add_argument("--host", default="0.0.0.0")
    common.add_argument("--port", type=int, default=8080)
    common.add_argument("--device", default="cuda")

    lr = sub.add_parser("lerobot", parents=[common], help="LeRobot pretrained_model directory")
    lr.add_argument("--n-action-steps", type=int, default=None,
                    help="actions returned per /predict (default: the checkpoint's n_action_steps)")
    lr.add_argument("--fill-from-state", nargs="+", default=[], metavar="KEY",
                    help="feed the measured state into these inputs (e.g. observation.commanded_state)")

    mb = sub.add_parser("mamba", parents=[common], help="mamba-embeddingvla .pt checkpoint")
    mb.add_argument("--mamba-root", required=True, help="mamba-embeddingvla checkout")
    mb.add_argument("--fps", type=int, default=20, help="rate the training data was recorded at")
    mb.add_argument("--n-inference-steps", type=int, default=10)
    mb.add_argument("--instruction", default=None)
    mb.add_argument("--cuda-graph", action="store_true", help="capture encode/flow CUDA graphs")
    mb.add_argument("--tf32", action="store_true", help="TF32 matmul (action delta ~3e-4 vs fp32)")
    mb.add_argument("--debug-dump", default=None, help="save every 10th request's images here")

    args = ap.parse_args()
    if args.backend == "lerobot":
        backend = LeRobotBackend(args.ckpt, args.device, args.n_action_steps, args.fill_from_state)
    else:
        if args.tf32:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        backend = MambaBackend(args.ckpt, args.mamba_root, args.device, args.n_inference_steps,
                               args.instruction, args.cuda_graph, args.fps, args.debug_dump)
    t0 = time.perf_counter()
    backend.warmup()
    print(f"[server] ready ({time.perf_counter() - t0:.1f} s warmup)", flush=True)
    PolicyServer(backend).run(args.host, args.port)


if __name__ == "__main__":
    main()
