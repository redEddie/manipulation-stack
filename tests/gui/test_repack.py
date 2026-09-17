"""Repack copies already-packed episodes raw and re-encodes only the rest.

Pinned here (2026-09-17):
  * a dataset already stored as gzip-4 + shuffle is copied chunk-for-chunk --
    its stored bytes are identical afterwards, nothing was decoded;
  * an lzf dataset (a newly collected episode) is re-encoded to gzip;
  * every value survives, attrs included, and the marker is written.

Synthetic file only; the operator's .hdf5 files are never opened.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
from mstack.data.dataset_schema import REPACK_MARKER_ATTR  # noqa: E402

d = Path(tempfile.mkdtemp(prefix="repack_"))
path = d / "scene_000.hdf5"
rng = np.random.default_rng(0)
imgs = {}
with h5py.File(path, "w") as f:
    f.create_group("metadata").attrs["scene_id"] = "S000"
    for name, comp in (("episode_000", "gzip"), ("episode_001", "lzf")):
        obs = f.create_group(name).create_group("obs")
        f[name].attrs["episode_uid"] = f"EP-S000-I000-E00{name[-1]}"
        a = np.zeros((20, 48, 64, 3), np.uint8)
        a[:, 10:30, 5:40] = rng.integers(0, 255, (20, 20, 35, 3), dtype=np.uint8)
        imgs[name] = a
        # The gzip episode gets a chunk shape h5py would never pick on its own,
        # so a re-encode (which re-chunks) is told apart from a raw copy --
        # zlib is deterministic, so identical bytes alone would not prove it.
        kw = ({"compression": "gzip", "compression_opts": 4, "shuffle": True,
               "chunks": (5, 48, 64, 3)}
              if comp == "gzip" else {"compression": "lzf"})
        obs.create_dataset("agentview_rgb", data=a, **kw)
        obs.create_dataset("joint_states", data=rng.standard_normal((20, 7)).astype(np.float32))
        obs["agentview_rgb"].attrs["note"] = name
    # bytes of the gzip episode's first chunk as stored on disk
    ds = f["episode_000/obs/agentview_rgb"]
    raw_before = ds.id.read_direct_chunk((0, 0, 0, 0))[1]
    size_before = ds.id.get_storage_size()

r = subprocess.run([sys.executable, str(WT / "scripts/convert/repack_hdf5.py"), str(path)],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stdout + r.stderr

with h5py.File(path, "r") as f:
    g = f["episode_000/obs/agentview_rgb"]
    lz = f["episode_001/obs/agentview_rgb"]
    assert g.chunks == (5, 48, 64, 3), f"gzip episode was re-encoded (chunks {g.chunks})"
    assert g.id.read_direct_chunk((0, 0, 0, 0))[1] == raw_before
    assert g.id.get_storage_size() == size_before
    print("1. already-packed episode copied raw (stored bytes identical) OK")
    assert (lz.compression, lz.compression_opts, lz.shuffle) == ("gzip", 4, True)
    print("2. lzf episode re-encoded to gzip-4 + shuffle OK")
    for name, a in imgs.items():
        assert np.array_equal(f[f"{name}/obs/agentview_rgb"][()], a), name
        assert f[f"{name}/obs/agentview_rgb"].attrs["note"] == name
        assert f[name].attrs["episode_uid"].endswith(name[-1])
    assert REPACK_MARKER_ATTR in f["metadata"].attrs
    print("3. contents, attrs and repack marker intact OK")

import json  # noqa: E402

log = [json.loads(line) for line in (d / "repack_log.jsonl").read_text().splitlines()]
assert len(log) == 1 and log[0]["file"] == "scene_000.hdf5", log
e = log[0]
assert e["outcome"] in ("replaced", "not_smaller"), e
assert e["episodes"] == 2 and e["size_before"] > 0 and e["size_after"] > 0
# the gzip episode was copied raw, only the lzf one was encoded
assert 0 < e["raw_copied_bytes"] < e["uncompressed_bytes"], e
assert e["encoded_bytes"] == e["uncompressed_bytes"] - e["raw_copied_bytes"]
assert e["seconds_total"] >= e["seconds_write"] >= 0
print("3b. repack_log.jsonl records sizes, encoded bytes and timings OK")

# ---- 4. parallel encode: several lzf episodes, edge chunks, a float image
# Each worker gzips chunks itself and hands raw chunks to the parent, so the
# result must read back through HDF5's own filter pipeline identically --
# including edge chunks (shape not divisible by the chunk) and the shuffle
# filter on a multi-byte dtype.
p2 = d / "scene_001.hdf5"
arrays = {}
with h5py.File(p2, "w") as f:
    f.create_group("metadata").attrs["scene_id"] = "S001"
    for i in range(4):
        obs = f.create_group(f"episode_{i:03d}").create_group("obs")
        a = rng.integers(0, 40, (23 + i, 50, 70, 3), dtype=np.uint8)
        dep = rng.standard_normal((23 + i, 50, 70)).astype(np.float32)
        obs.create_dataset("agentview_rgb", data=a, compression="lzf")
        obs.create_dataset("depth", data=dep, compression="lzf")
        arrays[f"episode_{i:03d}/obs/agentview_rgb"] = a
        arrays[f"episode_{i:03d}/obs/depth"] = dep
p1 = d / "scene_002.hdf5"
import shutil  # noqa: E402
shutil.copy(p2, p1)
for target, jobs in ((p2, "3"), (p1, "1")):
    r = subprocess.run([sys.executable, str(WT / "scripts/convert/repack_hdf5.py"),
                        "--jobs", jobs, str(target)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
with h5py.File(p2, "r") as par, h5py.File(p1, "r") as ser:
    for name, a in arrays.items():
        got = par[name]
        assert (got.compression, got.compression_opts, got.shuffle) == ("gzip", 4, True)
        assert np.array_equal(got[()], a), f"parallel encode changed {name}"
        assert got.chunks == ser[name].chunks
print("4. parallel encode reads back identically (edge chunks, float shuffle) OK")
log = [json.loads(line) for line in (d / "repack_log.jsonl").read_text().splitlines()]
assert [x["file"] for x in log] == ["scene_000.hdf5", "scene_001.hdf5", "scene_002.hdf5"], log
assert [x["jobs"] for x in log[1:]] == [3, 1]
print("4b. one log line per run, job count recorded OK")

print("test_repack 통과")
