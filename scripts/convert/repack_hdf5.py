"""Repack a LIBERO-format HDF5: reclaim deleted space and (re)compress images.

Why
---
Deleting a ``demo_N`` group with h5py frees the space *inside* the file for
reuse but never shrinks the file on disk -- HDF5 has no hole-punching. A
dataset that has had episodes curated out therefore keeps paying for them.
Rewriting the file into a fresh one is the only way to actually reclaim it,
which is what ``h5repack`` does; that tool is not installed here, so this is
the same operation in h5py (no external dependency).

It also re-applies compression: the image datasets dominate the file, and
until 2026-09-22 the collector wrote them with ``lzf`` (fast, chosen so the
background save never stalls the operator). Once collection is over that
trade-off no longer applies, so gzip can be spent instead. Since that date
the collector writes gzip-4 from birth, so for files collected wholly after
the switch this half is already done and the remaining job is only the
dead-space reclamation above -- old ``lzf`` files still need both.

Safety
------
The rewrite goes to a sibling temp file and is verified (every dataset's
shape/dtype and a content checksum are compared against the source) before
the original is replaced. A failure anywhere leaves the original untouched.

Log
---
Every attempt appends one JSON line to ``repack_log.jsonl`` in the file's
directory: sizes before/after, uncompressed and actually re-encoded bytes,
write/verify seconds, compression, job count and outcome.

Usage:
    python scripts/convert/repack_hdf5.py <file.hdf5> [more.hdf5 ...]
        [--compression gzip|lzf|none] [--level 4] [--dry-run] [--keep-original]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from h5py._hl.filters import guess_chunk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstack.data.dataset_schema import (  # noqa: E402
    REPACK_COUNT_ATTR,
    REPACK_MARKER_ATTR,
)

# Images are the only datasets big enough for compression to matter; the
# rest are (T, <=8) float arrays where chunking overhead can exceed the win.
_MIN_COMPRESS_BYTES = 1 << 16  # 64 KiB


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _checksum(ds: h5py.Dataset) -> tuple:
    """Cheap content fingerprint: shape, dtype, and a strided sample sum."""
    a = ds[()]
    arr = np.asarray(a)
    if arr.dtype.kind in "fc":
        flat = arr.ravel()
        s = float(np.nansum(flat[::max(1, flat.size // 4096)]))
    elif arr.dtype.kind in "iub":
        flat = arr.ravel()
        s = int(flat[::max(1, flat.size // 4096)].astype(np.int64).sum())
    else:
        s = 0
    return (arr.shape, str(arr.dtype), s)


def _walk_datasets(g, prefix=""):
    for key in g:
        item = g[key]
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Group):
            yield from _walk_datasets(item, path)
        else:
            yield path, item


class _Progress:
    """Byte-based progress, throttled so a multi-GB file doesn't spam the log.

    stdout is a pipe when the GUI runs this (QProcess), so Python block-buffers
    it -- without flush=True nothing appears until the process exits, which is
    exactly the "no feedback for several minutes" this exists to fix.
    """

    def __init__(self, total: int, every_s: float = 2.0) -> None:
        self.total = max(total, 1)
        self.done = 0
        self.t0 = time.monotonic()
        self._last = 0.0
        self._every = every_s

    def add(self, n: int) -> None:
        self.done += n
        now = time.monotonic()
        el = now - self.t0
        # Skip the first tick: with ~no bytes and ~no elapsed time behind it,
        # the rate and ETA are meaningless (and alarmingly large).
        if self.done < self.total * 0.02 or el < self._every:
            return
        if now - self._last < self._every:
            return
        self._last = now
        pct = 100.0 * self.done / self.total
        rate = self.done / el / 1e6 if el > 0 else 0.0
        eta = (self.total - self.done) / (self.done / el) if self.done and el > 0 else 0.0
        print(f"  진행 {pct:5.1f}%  ({self.done/1e6:.0f}/{self.total/1e6:.0f} MB)  "
              f"{rate:.0f} MB/s  경과 {el:.0f}s  남은시간 ~{eta:.0f}s", flush=True)


def _target_filter(compression: str, level: int) -> tuple:
    """(compression, opts, shuffle) a large dataset should end up with."""
    if compression == "none":
        return (None, None, False)
    return (compression, level if compression == "gzip" else None, True)


def _already_packed(ds: h5py.Dataset, compression: str, level: int) -> bool:
    """True when the dataset is already stored exactly as a repack would write it.

    Such a dataset is copied chunk-for-chunk (H5Ocopy) instead of being
    decoded and re-encoded. After a repack, collecting more into the file adds
    lzf episodes next to the gzip ones, and re-encoding the gzip majority was
    nearly all of the time: scene_009 had 139 gzip episodes and 1 lzf, and a
    full rewrite ran at ~30 MB/s over 34 GB (2026-09-17). A raw copy of one
    episode takes 0.2 s.
    """
    if ds.chunks is None:
        return False
    return (ds.compression, ds.compression_opts, bool(ds.shuffle)) == \
        _target_filter(compression, level)


def _default_jobs() -> int:
    """All cores but two -- the GUI and the rest of the station keep a margin."""
    return max(1, (os.cpu_count() or 2) - 2)


def _shuffle(buf: bytes, itemsize: int) -> bytes:
    """HDF5 shuffle filter: byte k of every element, for k = 0..itemsize-1."""
    if itemsize == 1:
        return buf
    return np.frombuffer(buf, np.uint8).reshape(-1, itemsize).T.tobytes()


def _encode_group(job: tuple) -> list:
    """Worker: gzip+shuffle every listed dataset of one group, chunk by chunk.

    Returns ``[(dataset path, [(chunk offset, bytes), ...]), ...]``. Each chunk
    is decompressed again and compared with its source before it is returned,
    so what the parent writes is already checked in full -- not a sample.

    The HDF5 filter pipeline runs in a single thread, which held a full rewrite
    at ~35 MB/s (2026-09-17). zlib here runs in one process per group, so the
    encode scales with cores.
    """
    src_path, items, level = job
    out = []
    with h5py.File(src_path, "r") as f:
        for name, chunks in items:
            a = np.ascontiguousarray(f[name][()])
            itemsize = a.dtype.itemsize
            parts = []
            for off in itertools.product(*(range(0, n, c) for n, c in zip(a.shape, chunks))):
                sl = tuple(slice(o, o + c) for o, c in zip(off, chunks))
                blk = a[sl]
                if blk.shape != tuple(chunks):
                    # HDF5 stores edge chunks at full size, padded with the fill value.
                    full = np.zeros(chunks, a.dtype)
                    full[tuple(slice(0, n) for n in blk.shape)] = blk
                    blk = full
                raw = np.ascontiguousarray(blk).tobytes()
                comp = zlib.compress(_shuffle(raw, itemsize), level)
                if _shuffle_back(zlib.decompress(comp), itemsize) != raw:
                    raise RuntimeError(f"encode round-trip mismatch at {name} {off}")
                parts.append((off, comp))
            out.append((name, parts))
    return out


def _shuffle_back(buf: bytes, itemsize: int) -> bytes:
    if itemsize == 1:
        return buf
    return np.frombuffer(buf, np.uint8).reshape(itemsize, -1).T.tobytes()


def _total_bytes(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return sum(int(ds.size) * ds.dtype.itemsize for _, ds in _walk_datasets(f))


def _rewrite(src_path: Path, dst_path: Path, compression: str, level: int,
             progress: "Optional[_Progress]" = None, jobs: int = 1,
             stats: "Optional[dict]" = None) -> set:
    """Writes the repacked file. Returns the dataset paths encoded (and checked)
    by the worker pool, which :func:`_verify` need not decode again.

    ``stats["raw_copied_bytes"]`` counts uncompressed bytes of datasets copied
    chunk-for-chunk -- encoding time only depends on the rest."""
    if stats is not None:
        stats.setdefault("raw_copied_bytes", 0)
    pooled: dict = {}          # top-level group -> [(dataset path, chunks)]
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        _copy_attrs(src, dst)

        def rec(sg, dg):
            _copy_attrs(sg, dg)
            for key in sg:
                item = sg[key]
                if isinstance(item, h5py.Group):
                    rec(item, dg.create_group(key))
                    continue
                nbytes = int(item.size) * item.dtype.itemsize
                if _already_packed(item, compression, level):
                    sg.copy(item, dg, name=key)          # raw chunks, attrs included
                    if stats is not None:
                        stats["raw_copied_bytes"] += nbytes
                    if progress is not None:
                        progress.add(nbytes)
                    continue
                big = compression != "none" and nbytes >= _MIN_COMPRESS_BYTES
                if big and compression == "gzip" and jobs > 1:
                    # Create it empty with the final filters; the pool fills it.
                    chunks = guess_chunk(item.shape, None, item.dtype.itemsize)
                    out = dg.create_dataset(key, shape=item.shape, dtype=item.dtype,
                                            chunks=chunks, compression="gzip",
                                            compression_opts=level, shuffle=True)
                    _copy_attrs(item, out)
                    top = item.name.strip("/").split("/")[0]
                    pooled.setdefault(top, []).append((item.name, chunks))
                    continue
                data = item[()]
                kw = {}
                if big:
                    if compression == "gzip":
                        kw = {"compression": "gzip", "compression_opts": level}
                    else:
                        kw = {"compression": compression}
                    kw["shuffle"] = True
                out = dg.create_dataset(key, data=data, **kw)
                _copy_attrs(item, out)
                if progress is not None:
                    progress.add(nbytes)

        rec(src, dst)

        if pooled:
            tasks = [(str(src_path), items, level) for items in pooled.values()]
            with ProcessPoolExecutor(max_workers=min(jobs, len(tasks)),
                                     mp_context=get_context("spawn"),
                                     initializer=os.nice, initargs=(10,)) as ex:
                for results in ex.map(_encode_group, tasks):
                    for name, parts in results:
                        ds = dst[name]
                        for off, comp in parts:
                            ds.id.write_direct_chunk(off, comp)
                        if progress is not None:
                            progress.add(int(ds.size) * ds.dtype.itemsize)
    return {name for items in pooled.values() for name, _c in items}


def _verify(src_path: Path, dst_path: Path, checked: "set | None" = None) -> None:
    with h5py.File(src_path, "r") as a, h5py.File(dst_path, "r") as b:
        sa = dict(_walk_datasets(a))
        sb = dict(_walk_datasets(b))
        if set(sa) != set(sb):
            missing = (set(sa) ^ set(sb))
            raise RuntimeError(f"dataset set differs: {sorted(missing)[:5]}")
        for path, ds in sa.items():
            db = sb[path]
            if (ds.shape, ds.dtype) != (db.shape, db.dtype):
                raise RuntimeError(f"shape/dtype mismatch at {path}")
            if (ds.chunks is not None and ds.chunks == db.chunks
                    and ds.compression == db.compression
                    and ds.compression_opts == db.compression_opts
                    and ds.shuffle == db.shuffle
                    and ds.id.get_storage_size() == db.id.get_storage_size()):
                # Raw-copied: same filters and the same stored bytes count, so
                # decoding both sides would only re-read what H5Ocopy moved.
                continue
            if checked and ds.name in checked:
                continue            # every chunk round-tripped in the worker
            ca, cb = _checksum(ds), _checksum(db)
            if ca != cb:
                raise RuntimeError(f"content mismatch at {path}: {ca} vs {cb}")
        if dict(a.attrs) .keys() != dict(b.attrs).keys():
            raise RuntimeError("root attrs differ")


#: One JSON line per repack attempt, in the dataset directory next to the
#: files. Size and time are what a storage-format comparison needs, and the
#: console output that used to carry them is gone once the GUI job closes.
REPACK_LOG_NAME = "repack_log.jsonl"


def _log_run(path: Path, entry: dict) -> None:
    try:
        with (path.parent / REPACK_LOG_NAME).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"  (경고) 공간 회수 기록 실패: {e}", flush=True)


def process(path: Path, compression: str, level: int, dry_run: bool,
            keep_original: bool, jobs: int = 1) -> bool:
    if not path.exists():
        print(f"  [skip] {path}: 파일 없음", flush=True)
        return False
    before = path.stat().st_size
    tmp = path.with_suffix(path.suffix + ".repack.tmp")
    t_start = time.monotonic()
    entry = {"file": path.name, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "compression": compression, "level": level if compression == "gzip" else None,
             "jobs": jobs, "cpu_count": os.cpu_count(), "dry_run": dry_run,
             "size_before": before, "episodes": _episode_count(path)}
    stats: dict = {}
    print(f"\n=== {path.name}", flush=True)
    print(f"  현재 크기 : {before/1e6:>9.1f} MB", flush=True)
    try:
        total = _total_bytes(path)
        entry["uncompressed_bytes"] = total
        print(f"  압축 해제 기준 {total/1e6:.0f} MB 처리 예정", flush=True)
        prog = _Progress(total)
        checked = _rewrite(path, tmp, compression, level, progress=prog, jobs=jobs,
                           stats=stats)
        t_written = time.monotonic() - t_start
        print(f"  쓰기 완료 ({t_written:.1f}s) -- 내용 검증 중...", flush=True)
        _verify(path, tmp, checked)
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        print(f"  [실패] {type(e).__name__}: {e}", flush=True)
        print("  원본은 그대로입니다.", flush=True)
        _log_run(path, {**entry, "outcome": "failed",
                        "error": f"{type(e).__name__}: {e}",
                        "seconds_total": round(time.monotonic() - t_start, 3)})
        return False
    elapsed = time.monotonic() - t_start
    after = tmp.stat().st_size
    entry.update({"size_after": after, "ratio": round(after / before, 4) if before else None,
                  "raw_copied_bytes": stats.get("raw_copied_bytes", 0),
                  "encoded_bytes": total - stats.get("raw_copied_bytes", 0),
                  "seconds_write": round(t_written, 3),
                  "seconds_verify": round(elapsed - t_written, 3),
                  "seconds_total": round(elapsed, 3)})
    saved = before - after
    print(f"  회수 후   : {after/1e6:>9.1f} MB  "
          f"({100*after/before:.1f}%, {saved/1e6:+.1f} MB)  소요 {elapsed:.1f}s", flush=True)
    if dry_run:
        tmp.unlink(missing_ok=True)
        print("  [dry-run] 원본 유지, 임시 파일 삭제", flush=True)
        _log_run(path, {**entry, "outcome": "dry_run"})
        return True
    if saved <= 0:
        tmp.unlink(missing_ok=True)
        print("  줄어들지 않아 교체하지 않았습니다 (원본 유지)", flush=True)
        _log_run(path, {**entry, "outcome": "not_smaller"})
        return True
    if keep_original:
        backup = path.with_suffix(path.suffix + ".orig")
        shutil.move(str(path), str(backup))
        print(f"  원본 보관: {backup.name}", flush=True)
    else:
        path.unlink()
    shutil.move(str(tmp), str(path))
    # Record that this file has been repacked, so a later run (or the GUI's
    # selection list) can skip it without inferring from the compressor.
    try:
        with h5py.File(path, "a") as f:
            # legacy 는 data/, scene-v1 은 metadata 그룹이 마커 자리다
            # (mstack/data/libero_format.py 의 hdf5_repack_status 와 대칭).
            if "data" in f:
                anchor = f["data"]
                n_eps = len(f["data"].keys())
            else:
                anchor = f["metadata"]
                n_eps = sum(1 for k in f.keys() if k.startswith("episode_"))
            anchor.attrs[REPACK_MARKER_ATTR] = (
                f"{time.strftime('%Y-%m-%d %H:%M')} {compression}"
                + (f"-{level}" if compression == "gzip" else "")
            )
            # Episode count at this moment: collecting into the file again
            # appends lzf episodes next to these gzip ones, and the count is
            # what lets a later run report "N added since" instead of just
            # trusting a marker that has gone stale.
            anchor.attrs[REPACK_COUNT_ATTR] = n_eps
    except Exception as e:  # noqa: BLE001
        print(f"  (경고) repacked 표시 기록 실패: {e}", flush=True)
    _log_run(path, {**entry, "outcome": "replaced"})
    print("  교체 완료", flush=True)
    return True


def _episode_count(path: Path) -> "int | None":
    try:
        with h5py.File(path, "r") as f:
            if "data" in f:
                return len(f["data"].keys())
            return sum(1 for k in f.keys() if k.startswith("episode_"))
    except OSError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"],
                    help="이미지 압축 방식 (기본 gzip -- 옛 lzf 파일은 이 방식으로 다시 압축)")
    ap.add_argument("--level", type=int, default=4,
                    help="gzip 레벨 1-9 (기본 4; 높을수록 느리고 조금 더 작음)")
    ap.add_argument("--dry-run", action="store_true",
                    help="줄어드는 크기만 확인하고 원본은 건드리지 않음")
    ap.add_argument("--keep-original", action="store_true",
                    help="교체 대신 원본을 .orig 로 남김")
    ap.add_argument("--jobs", type=int, default=_default_jobs(),
                    help="gzip 인코딩에 쓸 프로세스 수 (기본: 코어 수 - 2, 1 이면 병렬 끔)")
    ap.add_argument("--skip-repacked", action="store_true",
                    help="회수할 것이 없는 파일은 건너뜀 (repacked 표시 + 죽은 공간이 적음)")
    args = ap.parse_args()

    total_before = total_after = 0
    ok = True
    t_all = time.monotonic()
    for p in args.files:
        if args.skip_repacked:
            from mstack.data.libero_format import hdf5_repack_status

            st = hdf5_repack_status(p)
            if st["repacked"]:
                print(f"\n=== {p.name}\n  [건너뜀] 회수할 것이 없음 "
                      f"(압축={st['compression']}"
                      + (f", {st['marker']}" if st["marker"] else "") + ")", flush=True)
                continue
        b = p.stat().st_size if p.exists() else 0
        ok &= process(p, args.compression, args.level, args.dry_run, args.keep_original,
                      jobs=args.jobs)
        a = p.stat().st_size if p.exists() else 0
        total_before += b
        total_after += a
    dt = time.monotonic() - t_all
    if len(args.files) > 1:
        print(f"\n합계: {total_before/1e6:.1f} MB -> {total_after/1e6:.1f} MB "
              f"({total_before and 100*total_after/total_before:.1f}%), "
              f"총 소요 {dt:.1f}s", flush=True)
    else:
        print(f"\n총 소요 {dt:.1f}s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
