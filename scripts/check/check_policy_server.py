#!/usr/bin/env python3
"""Check a running policy server against recorded episodes -- no robot needed.

Why this exists
---------------
A policy that scores well offline can still fail on the robot because the
deploy path feeds it something different: wrong image size, a different crop,
a camera swapped, an action left relative, a missing normalization. Each of
those looks the same on the arm ("it moves, but badly"). This sends recorded
frames of the policy's own training dataset through the real HTTP path and
compares the returned chunk with the actions that were recorded at that time.

It checks, in order:

  1. GET /info answers and describes a protocol the client can execute.
  2. The server rejects an image of the wrong size (the size guard is live).
  3. For N episodes: /reset with the episode's instruction, /predict on its
     first frame, then the chunk-vs-recorded L2 per step and per dimension.
     Every predicted value must lie inside the dataset's action range.

Pass a held-out episode list (e.g. a training manifest's eval split) to see
held-out error; training episodes show how well the policy fits.

Usage (GPU machine, a venv with lerobot -- it reads the dataset):
    scripts/check/check_policy_server.py --server http://127.0.0.1:8080 --episodes 6 13 15
    scripts/check/check_policy_server.py --server ... --manifest eval.json --limit 10

Exit status is 0 only if every check passed.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import requests  # noqa: E402

from mstack.comm.policy_protocol import (  # noqa: E402
    JOINT_ABSOLUTE,
    STATE_KEY,
    PolicyInfo,
    encode_image,
    image_key,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=[])
    ap.add_argument("--manifest", help="JSON with an 'episode_indices' list")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--parity", action="store_true",
                    help="also run the checkpoint locally, on the dataset sample exactly as offline "
                         "evaluation does, and require the server's chunk to match it (server must be "
                         "started with --seed; same machine, it loads /info's checkpoint path)")
    ap.add_argument("--parity-tol", type=float, default=1e-3, help="max |server - offline| allowed")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ok = True
    info = PolicyInfo.from_dict(requests.get(f"{args.server}/info", timeout=10).json())
    print(f"1. /info: {info.policy} {info.fps} Hz, {info.image_size}^2 {info.cameras}, "
          f"{info.action_type}[{info.action_dim}] x {info.chunk_size}")
    if info.action_type != JOINT_ABSOLUTE:
        print("   (EE-delta policy: recorded joint actions are not comparable; stopping after 2.)")
    repo = info.extra.get("dataset")
    if not repo:
        print("   /info names no dataset; this check needs the training dataset")
        return 1

    episodes = list(args.episodes)
    if args.manifest:
        episodes += json.loads(open(args.manifest).read())["episode_indices"]
    episodes = sorted(set(episodes))[: args.limit] or [0]
    ds = LeRobotDataset(repo, episodes=episodes)

    # 2. size guard
    wrong = info.image_size + 32
    bad = {STATE_KEY: [0.0] * info.state_dim,
           **{image_key(c): encode_image(np.zeros((wrong, wrong, 3), np.uint8)) for c in info.cameras}}
    requests.post(f"{args.server}/reset", json={"instruction": "size check"}, timeout=30).raise_for_status()
    r = requests.post(f"{args.server}/predict", json=bad, timeout=30)
    guard = r.status_code == 400
    ok &= guard
    print(f"2. {wrong}^2 image -> HTTP {r.status_code} ({'rejected, OK' if guard else 'NOT rejected'})")
    if info.action_type != JOINT_ABSOLUTE:
        return 0 if ok else 1

    # 3. recorded frames
    cols = ds.hf_dataset.select_columns(["episode_index", "action"])
    first_row: dict[int, int] = {}
    for row, ep in enumerate(cols["episode_index"]):
        first_row.setdefault(int(ep), row)
    stats = ds.meta.stats["action"]
    lo, hi = np.asarray(stats["min"]), np.asarray(stats["max"])
    margin = 0.1 * (hi - lo)

    offline = None
    if args.parity:
        seed = info.extra.get("seed")
        if seed is None:
            print("   --parity needs the server started with --seed")
            return 1
        offline = _OfflinePolicy(info.checkpoint, int(seed))

    errs, per_dim, ms, parity = [], [], [], []
    print(f"3. {len(episodes)} episodes from {repo}")
    for ep in episodes:
        row = first_row[ep]
        sample = ds[row]
        obs = {STATE_KEY: sample[STATE_KEY].tolist()}
        for cam in info.cameras:
            x = sample[image_key(cam)]
            img = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            # The wire carries uint8; that is lossless only if the decoded frame was uint8/255.
            lost = float((x.permute(1, 2, 0).numpy() - img / 255.0).__abs__().max())
            if lost > 1e-6:
                print(f"   ep {ep}: {cam} frame is not uint8/255 (max {lost:.2e}) -- uint8 wire is lossy here")
            obs[image_key(cam)] = encode_image(img)
        requests.post(f"{args.server}/reset", json={"instruction": sample["task"]}, timeout=30).raise_for_status()
        t0 = time.perf_counter()
        r = requests.post(f"{args.server}/predict", json=obs, timeout=60)
        ms.append((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            print(f"   ep {ep}: HTTP {r.status_code} {r.text[:200]}")
            ok = False
            continue
        pred = np.asarray(r.json()["actions"], dtype=float)
        if offline is not None:
            ref = offline.predict(sample)[: len(pred)]
            d = float(np.abs(pred - ref).max())
            parity.append(d)
            ok &= d <= args.parity_tol
        n = min(len(pred), ds.meta.episodes["length"][ep])
        gt = np.stack([np.asarray(cols["action"][row + i]) for i in range(n)])
        pred = pred[:n]
        inside = bool(((pred >= lo - margin) & (pred <= hi + margin)).all())
        ok &= inside
        l2 = np.linalg.norm(pred - gt, axis=1)
        errs.append(l2.mean())
        per_dim.append(np.abs(pred - gt).mean(0))
        print(f"   ep {ep:5d}: chunk L2 mean {l2.mean():.3f} (step0 {l2[0]:.3f}, last {l2[-1]:.3f}), "
              f"{'in range' if inside else 'OUT OF RANGE'}, {ms[-1]:.0f} ms  {sample['task']!r}")
    if errs:
        print(f"   mean chunk L2 {np.mean(errs):.3f}; per-dim |err| {np.round(np.mean(per_dim, 0), 3).tolist()}")
        print(f"   round-trip median {np.median(ms):.0f} ms (budget at {info.fps} Hz with lead 2: "
              f"{2000 // info.fps} ms)")
    if parity:
        worst = max(parity)
        print(f"4. parity with offline evaluation (seed {info.extra['seed']}): max |server - offline| "
              f"{worst:.2e} over {len(parity)} chunks -- "
              f"{'identical path, OK' if worst <= args.parity_tol else 'DIFFERENT'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


class _OfflinePolicy:
    """The checkpoint run the way offline evaluation runs it
    (lerobot experiments/subset_bias/eval_action_following.py): dataset sample ->
    unsqueeze -> task string -> preprocessor -> predict_action_chunk -> postprocessor."""

    def __init__(self, ckpt: str, seed: int):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch, self.seed = torch, seed
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.pretrained_path = ckpt
        cfg.device = "cuda"
        self.policy = get_policy_class(cfg.type).from_pretrained(ckpt).to("cuda").eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": "cuda"}})

    def predict(self, sample: dict) -> np.ndarray:
        torch = self.torch
        batch = {k: v.unsqueeze(0).to("cuda") for k, v in sample.items() if isinstance(v, torch.Tensor)}
        batch["task"] = sample["task"]
        self.policy.reset()
        torch.manual_seed(self.seed)
        with torch.inference_mode():
            out = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        return out.squeeze(0).float().cpu().numpy()


if __name__ == "__main__":
    sys.exit(main())
