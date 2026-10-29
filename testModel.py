from __future__ import annotations

import os
import sys
import gc
import csv
import json
import math
import time
import random
import shutil
import tempfile
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import h5py
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.nn.utils.parametrizations import weight_norm

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    matthews_corrcoef,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")

_LOGGER = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = "v5.0.2-cache-v2"

DEFAULT_CACHE_DIR = (
    r"E:\Neuroscience_Lab\Article reviews final codes and check\DBTR_AGF_Cache_fp16"
)

def _print_progress(prefix: str, current: int, total: int, t_start: float) -> None:
    pct = 100.0 * current / max(total, 1)
    elapsed = time.time() - t_start
    eta = (elapsed / pct * 100 - elapsed) if pct > 0.5 else 0.0
    print(
        f"\r{prefix} {current:,}/{total:,} ({pct:5.1f}%) | "
        f"elapsed {elapsed:6.1f}s | eta {eta:6.1f}s",
        end="", flush=True,
    )

def _load_cache_dict(
    meta_path: str, metadata_path: str,
    time_mmap_path: str, freq_mmap_path: str,
) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with np.load(metadata_path, allow_pickle=False) as npz:
        metadata = {k: npz[k] for k in npz.files}

    return {
        "time_mmap_path": time_mmap_path,
        "freq_mmap_path": freq_mmap_path,
        "metadata_path": metadata_path,
        "meta_path": meta_path,
        "shape_time": tuple(meta["time_shape"]),
        "shape_freq": tuple(meta["freq_shape"]),
        "dtype": np.dtype(meta["cache_dtype"]),
        "n_samples": int(meta["n_samples"]),
        "n_steps": int(meta["n_steps"]),
        "t_dim": int(meta["t_dim"]),
        "f_dim": int(meta["f_dim"]),
        "active_freq_idx": metadata["active_freq_idx"],
        "freqs_hz": metadata["freqs_hz"],
        "metadata": metadata,
        "h5_mtime": int(meta["h5_mtime"]),
        "h5_size": int(meta["h5_size_bytes"]),
        "schema_version": meta["schema_version"],
        "cache_hash": meta["cache_hash"],
        "h5_path": meta.get("h5_path"),
    }

def build_fp16_mmap_cache(
    h5_path: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    cache_dtype: str = "float16",
    chunk_size: int = 512,
    verbose: bool = True,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    valid_dtypes = {"float16": np.float16, "float32": np.float32}
    if cache_dtype not in valid_dtypes:
        raise ValueError(
            f"cache_dtype must be one of {list(valid_dtypes.keys())}, "
            f"got {cache_dtype!r}"
        )
    np_dtype = valid_dtypes[cache_dtype]
    dtype_bytes = np.dtype(np_dtype).itemsize

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    os.makedirs(cache_dir, exist_ok=True)

    h5_stat = os.stat(h5_path)
    hash_input = (
        f"{CACHE_SCHEMA_VERSION}|{os.path.abspath(h5_path)}|"
        f"{h5_stat.st_size}|{h5_stat.st_mtime_ns}|{cache_dtype}"
    )
    cache_hash = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:16]

    meta_path      = os.path.join(cache_dir, "cache_meta.json")
    metadata_path  = os.path.join(cache_dir, "metadata.npz")
    time_mmap_path = os.path.join(cache_dir, "time_matrix.mmap")
    freq_mmap_path = os.path.join(cache_dir, "freq_matrix.mmap")

    if not force_rebuild and all(
        os.path.exists(p)
        for p in (meta_path, metadata_path, time_mmap_path, freq_mmap_path)
    ):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if cached_meta.get("cache_hash") == cache_hash:
                time_ok = (
                    os.path.getsize(time_mmap_path)
                    == int(cached_meta["time_bytes"])
                )
                freq_ok = (
                    os.path.getsize(freq_mmap_path)
                    == int(cached_meta["freq_bytes"])
                )
                if time_ok and freq_ok:
                    if verbose:
                        print(
                            f"[cache] Reusing existing cache "
                            f"(hash={cache_hash}, dtype={cache_dtype}, "
                            f"time={cached_meta['time_bytes'] / 1e9:.2f} GB, "
                            f"freq={cached_meta['freq_bytes'] / 1e9:.2f} GB)"
                        )
                    return _load_cache_dict(
                        meta_path, metadata_path,
                        time_mmap_path, freq_mmap_path,
                    )
                if verbose:
                    print("[cache] On-disk size mismatch; rebuilding.")
            elif verbose:
                print(
                    f"[cache] Hash mismatch "
                    f"(cached={cached_meta.get('cache_hash')}, "
                    f"current={cache_hash}); rebuilding."
                )
        except Exception as exc:
            if verbose:
                print(f"[cache] Failed to reuse cache ({exc}); rebuilding.")

    if verbose:
        print(f"[cache] Building mmap cache in: {cache_dir}")

    with h5py.File(h5_path, "r") as hf:
        n_time = int(hf["time_matrix"].shape[0])
        n_freq = int(hf["freq_matrix"].shape[0])
        if n_time != n_freq:
            raise ValueError(
                f"Row count mismatch: time_matrix={n_time}, "
                f"freq_matrix={n_freq}. HDF5 is inconsistent."
            )
        n_samples = n_time

        n_steps = int(hf["time_matrix"].shape[1])
        t_dim   = int(hf["time_matrix"].shape[2])
        n_fb    = int(hf["freq_matrix"].shape[1])
        tc      = int(hf["freq_matrix"].shape[2])

        if tc != 18 * n_steps:
            raise ValueError(
                f"freq_matrix second dim {tc} != 18 * n_steps ({18 * n_steps})"
            )
        f_dim = 18 * n_fb

        stft_spacing = float(hf.attrs.get("stft_bin_spacing_hz", 0.2))
        fs_recovered = stft_spacing * (n_fb - 1) * 2.0
        if fs_recovered <= 0:
            raise ValueError(
                f"Cannot recover sampling rate from "
                f"stft_bin_spacing_hz={stft_spacing}"
            )
        n_fft = (n_fb - 1) * 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs_recovered).astype(np.float32)
        active_freq_idx = np.where(
            (freqs >= 0.5) & (freqs <= 100.0)
        )[0].astype(np.int32)

        metadata: Dict[str, np.ndarray] = {
            "labels":       hf["labels"][:].astype(np.int32),
            "block_ids":    hf["block_ids"][:].astype(np.int32),
            "edf_file_ids": hf["edf_file_ids"][:].astype(np.int32),
            "onsets":       hf["window_onset_seconds"][:].astype(np.float32),
            "active_freq_idx": active_freq_idx,
            "freqs_hz":     freqs,
        }
        if "raw_subject_ids" in hf:
            metadata["raw_sids"] = hf["raw_subject_ids"][:].astype(np.int32)
        else:
            metadata["raw_sids"] = hf["subject_ids"][:].astype(np.int32)

        if "subject_ids" in hf:
            metadata["subject_ids"] = hf["subject_ids"][:].astype(np.int32)
        else:
            metadata["subject_ids"] = metadata["raw_sids"].copy()

        if "effective_step_seconds" in hf:
            metadata["durs"] = hf["effective_step_seconds"][:].astype(np.float32)
        elif "interictal_window_seconds" in hf:
            metadata["durs"] = hf["interictal_window_seconds"][:].astype(np.float32)
        else:
            if verbose:
                print(
                    "[cache][WARN] HDF5 lacks effective_step_seconds; "
                    "using defaults (interictal=31.93s, preictal=2.5s)."
                )
            durs = np.full(n_samples, 31.93, dtype=np.float32)
            durs[metadata["labels"] == 1] = 2.5
            metadata["durs"] = durs

        if "window_duration_seconds" in hf:
            metadata["win_durs"] = hf[
                "window_duration_seconds"
            ][:].astype(np.float32)
        else:
            if verbose:
                print(
                    "[cache][WARN] HDF5 lacks window_duration_seconds; "
                    "defaulting to 5.0s."
                )
            metadata["win_durs"] = np.full(
                n_samples, 5.0, dtype=np.float32
            )

        time_shape = (n_samples, n_steps, t_dim)
        freq_shape = (n_samples, n_steps, f_dim)

        time_tmp = f"{time_mmap_path}.{os.getpid()}.tmp"
        freq_tmp = f"{freq_mmap_path}.{os.getpid()}.tmp"
        npz_tmp  = f"{metadata_path}.{os.getpid()}.tmp"
        meta_tmp = f"{meta_path}.{os.getpid()}.tmp"

        try:
            time_mm = np.memmap(
                time_tmp, dtype=np_dtype, mode="w+", shape=time_shape
            )
            freq_mm = np.memmap(
                freq_tmp, dtype=np_dtype, mode="w+", shape=freq_shape
            )

            t_ds = hf["time_matrix"]
            f_ds = hf["freq_matrix"]

            t_start = time.time()
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                b = end - start

                time_mm[start:end] = t_ds[start:end].astype(np_dtype)

                freq_chunk = (
                    f_ds[start:end]
                    .astype(np.float32)
                    .reshape(b, n_fb, 18, n_steps)
                    .transpose(0, 3, 2, 1)
                    .reshape(b, n_steps, f_dim)
                )
                freq_mm[start:end] = freq_chunk.astype(np_dtype)

                if verbose and (start // chunk_size) % 8 == 0:
                    _print_progress(
                        "[cache] building", end, n_samples, t_start
                    )

            if verbose:
                _print_progress(
                    "[cache] building", n_samples, n_samples, t_start
                )
                print()

            time_mm.flush()
            freq_mm.flush()
            del time_mm, freq_mm
            gc.collect()

            os.replace(time_tmp, time_mmap_path)
            os.replace(freq_tmp, freq_mmap_path)

            with open(npz_tmp, "wb") as f:
                np.savez(f, **metadata)
            os.replace(npz_tmp, metadata_path)

            time_bytes = int(np.prod(time_shape) * dtype_bytes)
            freq_bytes = int(np.prod(freq_shape) * dtype_bytes)
            meta = {
                "cache_hash": cache_hash,
                "schema_version": CACHE_SCHEMA_VERSION,
                "h5_path": os.path.abspath(h5_path),
                "h5_mtime": int(h5_stat.st_mtime),
                "h5_size_bytes": int(h5_stat.st_size),
                "cache_dtype": cache_dtype,
                "chunk_size": int(chunk_size),
                "n_samples": int(n_samples),
                "n_steps": int(n_steps),
                "t_dim": int(t_dim),
                "f_dim": int(f_dim),
                "n_freq_bins_total": int(n_fb),
                "n_active_freq_bins": int(len(active_freq_idx)),
                "recovered_fs_hz": float(fs_recovered),
                "time_shape": list(time_shape),
                "freq_shape": list(freq_shape),
                "time_bytes": time_bytes,
                "freq_bytes": freq_bytes,
                "metadata_keys": sorted(metadata.keys()),
            }
            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            os.replace(meta_tmp, meta_path)

        except Exception:
            for p in (time_tmp, freq_tmp, npz_tmp, meta_tmp):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            raise

    if verbose:
        print(
            f"[cache] Ready (hash={cache_hash}, dtype={cache_dtype}, "
            f"time={meta['time_bytes'] / 1e9:.2f} GB, "
            f"freq={meta['freq_bytes'] / 1e9:.2f} GB, "
            f"active freq bins={len(active_freq_idx)}/{n_fb})"
        )

    return _load_cache_dict(
        meta_path, metadata_path, time_mmap_path, freq_mmap_path
    )

SEED_DEFAULT = 42

def set_global_seed(seed: int, deterministic: bool = False) -> None:
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    if os.environ.get("PYTHONHASHSEED") != str(seed):
        _LOGGER.warning(
            "PYTHONHASHSEED=%s != seed=%d; launch python with "
            "PYTHONHASHSEED=%d for full hash reproducibility.",
            os.environ.get("PYTHONHASHSEED"), seed, seed,
        )
        os.environ["PYTHONHASHSEED"] = str(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except AttributeError:
            _LOGGER.warning(
                "torch.use_deterministic_algorithms unavailable; "
                "cudnn flags still apply."
            )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except AttributeError:
            pass

def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MasterDataStore:
    def __init__(self, cache: Dict[str, Any], h5_path: Optional[str] = None):
        self.cache = cache
        self.h5_path = h5_path or cache.get("h5_path")

        self.time_mm = np.memmap(
            cache["time_mmap_path"], dtype=cache["dtype"], mode="r",
            shape=cache["shape_time"],
        )
        self.freq_mm = np.memmap(
            cache["freq_mmap_path"], dtype=cache["dtype"], mode="r",
            shape=cache["shape_freq"],
        )

        m = cache["metadata"]
        self.labels = m["labels"]
        self.block_ids = m["block_ids"]
        self.edf_file_ids = m["edf_file_ids"]
        self.onsets = m["onsets"]
        self.durs = m["durs"]
        self.win_durs = m["win_durs"]
        self.raw_sids = m["raw_sids"]
        self.active_freq_idx = m["active_freq_idx"]
        self.freqs_hz = m["freqs_hz"]

        self.n_samples = int(cache["n_samples"])
        self.n_steps = int(cache["n_steps"])
        self.t_dim = int(cache["t_dim"])
        self.f_dim = int(cache["f_dim"])
        self.dtype = cache["dtype"]

        self.n_channels = 18
        self.freq_layout = "channel_major"

        if self.f_dim % self.n_channels != 0:
            raise RuntimeError(
                f"Cache f_dim={self.f_dim} is not divisible by "
                f"n_channels={self.n_channels}; cache is inconsistent with "
                f"the preprocessing contract."
            )

        self.subject_ids = self._load_subject_ids(cache)
        self.subject_indices_map = {
            int(s): np.where(self.subject_ids == s)[0].astype(np.int64)
            for s in np.unique(self.subject_ids)
        }

        self._validate_identity_merge()
        self._validate_subject_99_contents()
        self._validate_block_global_uniqueness()
        self._build_subject_lists()

        _LOGGER.info(
            "MasterDataStore: %d samples | all=%d curated=%d main_cohort=%d "
            "| has_merged_99=%s | n_channels=%d | freq_layout=%s",
            self.n_samples,
            len(self.all_subjects),
            len(self.curated_subjects),
            len(self.main_cohort_subjects),
            self.has_merged_99,
            self.n_channels,
            self.freq_layout,
        )

    def _load_subject_ids(self, cache: Dict[str, Any]) -> np.ndarray:
        if "subject_ids" not in cache["metadata"]:
            raise RuntimeError(
                "subject_ids not available in cache metadata; rebuild the "
                "cache with CACHE_SCHEMA_VERSION >= v5.0.2-cache-v2 to "
                "apply the chb01+chb21 -> Subject 99 merge."
            )
        return cache["metadata"]["subject_ids"].astype(np.int32)

    def _validate_identity_merge(self) -> None:
        self.has_merged_99 = 99 in self.subject_indices_map
        self._excluded_separate = []
        if self.has_merged_99:
            for s in (1, 21):
                if s in self.subject_indices_map:
                    self._excluded_separate.append(s)

    def _validate_subject_99_contents(self) -> None:
        if not self.has_merged_99:
            return
        raw_in_99 = set(
            np.unique(self.raw_sids[self.subject_indices_map[99]]).tolist()
        )
        unexpected = raw_in_99 - {1, 21}
        if unexpected:
            raise RuntimeError(
                f"Subject 99 contains raw patients {sorted(unexpected)} "
                f"beyond chb01+chb21; preprocessing is inconsistent."
            )

    def _validate_block_global_uniqueness(self) -> None:
        pairs = np.column_stack((self.block_ids, self.raw_sids))
        unique_pairs = np.unique(pairs, axis=0)
        u_blocks, counts = np.unique(unique_pairs[:, 0], return_counts=True)
        if np.any(counts > 1):
            bad_b = int(u_blocks[np.where(counts > 1)[0][0]])
            bad_r = unique_pairs[unique_pairs[:, 0] == bad_b, 1].tolist()
            raise RuntimeError(
                f"block_ids are not globally unique across subjects: "
                f"block_id={bad_b} maps to raw patients {bad_r}. "
                f"Regenerate the cache with global block_id assignment."
            )

    def _build_subject_lists(self) -> None:
        all_sids = sorted(self.subject_indices_map.keys())
        self.all_subjects = all_sids
        self.curated_subjects = [s for s in all_sids if s != 99]
        self.main_cohort_subjects = [
            s for s in all_sids if s not in self._excluded_separate
        ]

    def get_indices(self, sid: int) -> np.ndarray:
        sid = int(sid)
        if sid not in self.subject_indices_map:
            raise KeyError(
                f"Subject {sid} not in store. Available: {self.all_subjects}"
            )
        return self.subject_indices_map[sid]

    def get_indices_for(self, sids: Sequence[int]) -> np.ndarray:
        if not sids:
            raise ValueError("sids must be non-empty")
        parts = [self.get_indices(int(s)) for s in sids]
        return np.sort(np.concatenate(parts))

    def filter_indices_by_raw_subject(
        self, indices: np.ndarray, raw_sid: int
    ) -> np.ndarray:
        idx_arr = np.asarray(indices, dtype=np.int64)
        mask = self.raw_sids[idx_arr] == int(raw_sid)
        return idx_arr[mask]

    def raw_patients_of(self, indices: np.ndarray) -> set:
        return set(np.unique(self.raw_sids[indices]).tolist())

    def assert_no_leakage(
        self, a_idx: np.ndarray, b_idx: np.ndarray, context: str = ""
    ) -> None:
        overlap_raw = self.raw_patients_of(a_idx) & self.raw_patients_of(b_idx)
        if overlap_raw:
            raise RuntimeError(
                f"[leak:{context}] Raw patients {sorted(overlap_raw)} "
                f"appear in both sets."
            )
        overlap_blk = (
            set(np.unique(self.block_ids[a_idx]).tolist())
            & set(np.unique(self.block_ids[b_idx]).tolist())
        )
        if overlap_blk:
            raise RuntimeError(
                f"[leak:{context}] {len(overlap_blk)} block(s) overlap "
                f"between sets."
            )

    def interictal_hours(self, indices: np.ndarray) -> float:
        lbl = self.labels[indices]
        return float(self.durs[indices][lbl == 0].sum()) / 3600.0

    def total_recording_hours(self, indices: np.ndarray) -> float:
        return float(self.durs[indices].sum()) / 3600.0

    def n_preictal_blocks(self, indices: np.ndarray) -> int:
        lbl = self.labels[indices]
        if not np.any(lbl == 1):
            return 0
        return int(len(np.unique(self.block_ids[indices][lbl == 1])))

    def class_counts(self, indices: np.ndarray) -> Tuple[int, int]:
        lbl = self.labels[indices]
        return int((lbl == 0).sum()), int((lbl == 1).sum())

    def close(self) -> None:
        self.time_mm = None
        self.freq_mm = None
        gc.collect()

    def __len__(self) -> int:
        return self.n_samples

@dataclass
class Fold:
    fold_id: int
    fold_seed: int
    protocol: str
    train_sids: List[int]
    val_sids: List[int]
    test_sids: List[int]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

def _assert_no_block_overlap(
    store: MasterDataStore,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    context: str = "",
) -> None:
    overlap = (
        set(np.unique(store.block_ids[a_idx]).tolist())
        & set(np.unique(store.block_ids[b_idx]).tolist())
    )
    if overlap:
        raise RuntimeError(
            f"[block-leak:{context}] {len(overlap)} block(s) overlap "
            f"between sets."
        )

_PROTOCOL_SEED_OFFSET = {
    "one_subject": 0,
    "loso": 1_000_000,
    "holdout_70_30": 2_000_000,
    "low_data_50": 3_000_000,
    "low_data_30": 3_500_000,
    "low_data_70": 4_000_000,
}

def _fold_seed(protocol: str, subject_id: int) -> int:
    offset = _PROTOCOL_SEED_OFFSET.get(protocol, 5_000_000)
    return SEED_DEFAULT + offset + int(subject_id)

def _trim_boundary_windows_between_splits(
    store: MasterDataStore,
    primary_idx: np.ndarray,
    reference_idx: np.ndarray,
    guard_sec: float = 5.0,
) -> np.ndarray:
    if guard_sec <= 0.0 or len(primary_idx) == 0 or len(reference_idx) == 0:
        return primary_idx

    shared_fids = np.intersect1d(
        store.edf_file_ids[primary_idx],
        store.edf_file_ids[reference_idx],
    )
    if len(shared_fids) == 0:
        return primary_idx

    keep_mask = np.ones(len(primary_idx), dtype=bool)

    p_fids   = store.edf_file_ids[primary_idx]
    p_starts = store.onsets[primary_idx].astype(np.float64)
    p_durs   = store.win_durs[primary_idx].astype(np.float64)
    p_ends   = p_starts + p_durs
    max_p_dur = float(p_durs.max()) if p_durs.size > 0 else 0.0

    r_fids   = store.edf_file_ids[reference_idx]
    r_starts = store.onsets[reference_idx].astype(np.float64)
    r_ends   = r_starts + store.win_durs[reference_idx].astype(np.float64)

    for fid in shared_fids:
        p_loc = np.where(p_fids == fid)[0]
        if p_loc.size == 0:
            continue
        r_mask = r_fids == fid
        if not np.any(r_mask):
            continue

        p_s = p_starts[p_loc]
        p_e = p_ends[p_loc]

        r_s = r_starts[r_mask]
        r_e = r_ends[r_mask]
        order = np.argsort(r_s, kind="stable")
        r_s = r_s[order]
        r_e = r_e[order]

        p_order = np.argsort(p_s, kind="stable")
        p_s_sorted = p_s[p_order]

        lo = r_s - guard_sec
        hi = r_e + guard_sec

        left  = np.searchsorted(p_s_sorted, lo - max_p_dur, side="right")
        right = np.searchsorted(p_s_sorted, hi,           side="left")

        drop_local = np.zeros(p_loc.size, dtype=bool)
        for k in range(r_s.size):
            a = left[k]
            b = right[k]
            if a >= b:
                continue
            cand_idx = p_order[a:b]
            mask = (p_s[cand_idx] < hi[k]) & (p_e[cand_idx] > lo[k])
            if np.any(mask):
                drop_local[cand_idx[mask]] = True

        keep_mask[p_loc[drop_local]] = False

    return primary_idx[keep_mask]

def _subsample_train_indices_stratified(
    store: MasterDataStore,
    train_idx: np.ndarray,
    train_fraction: float,
    seed: int,
) -> np.ndarray:
    if not (0.0 < float(train_fraction) <= 1.0):
        raise ValueError(
            f"train_fraction must be in (0, 1], got {train_fraction}"
        )
    if float(train_fraction) >= 1.0 - 1e-9 or len(train_idx) == 0:
        return train_idx

    rng = np.random.default_rng(seed)
    lbl = store.labels[train_idx]
    kept_parts: List[np.ndarray] = []

    for c in (0, 1):
        c_idx = train_idx[lbl == c]
        if len(c_idx) == 0:
            continue
        k = max(1, int(round(len(c_idx) * float(train_fraction))))
        chosen = rng.choice(c_idx, size=k, replace=False)
        kept_parts.append(np.sort(chosen))

    return np.sort(np.concatenate(kept_parts)) if kept_parts else train_idx

def split_blocks_into_train_val(
    store: MasterDataStore,
    sids: Sequence[int],
    val_ratio: float = 0.15,
    block_guard: int = 1,
    seed: int = SEED_DEFAULT,
    min_val_preictal: int = 1,
    min_val_interictal: int = 1,
    min_train_preictal_windows: int = 100,
    min_train_interictal_windows: int = 50,
    n_retries: int = 10,
    boundary_guard_sec: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng_base = int(seed)
    tr_parts: List[np.ndarray] = []
    va_parts: List[np.ndarray] = []
    min_blocks = 4

    for sid in sids:
        idx = np.sort(store.get_indices(int(sid)))
        blocks = np.unique(store.block_ids[idx])

        if len(blocks) < min_blocks:
            _LOGGER.warning(
                "Subject %d: %d blocks < %d; aborting.",
                int(sid), len(blocks), min_blocks,
            )
            empty = np.array([], dtype=np.int64)
            return empty, empty

        block_cls: Dict[int, int] = {}
        for b in blocks:
            b_mask = store.block_ids[idx] == b
            lbl_set = set(np.unique(store.labels[idx][b_mask]).tolist())
            if len(lbl_set) != 1:
                _LOGGER.warning(
                    "Subject %d block %d not class-pure.", int(sid), int(b)
                )
                empty = np.array([], dtype=np.int64)
                return empty, empty
            block_cls[int(b)] = int(lbl_set.pop())

        pre_blocks_base = np.array([b for b in blocks if block_cls[int(b)] == 1])
        int_blocks_base = np.array([b for b in blocks if block_cls[int(b)] == 0])

        if len(pre_blocks_base) < 2 or len(int_blocks_base) < 2:
            _LOGGER.warning(
                "Subject %d: too few per-class blocks (pre=%d, int=%d).",
                int(sid), len(pre_blocks_base), len(int_blocks_base),
            )
            empty = np.array([], dtype=np.int64)
            return empty, empty

        success = False
        effective_guard_sec = boundary_guard_sec if block_guard > 0 else 0.0

        for retry_i in range(max(1, n_retries)):
            rng = np.random.default_rng(rng_base + retry_i * 1000)

            pre_b = pre_blocks_base.copy()
            int_b = int_blocks_base.copy()

            n_val_pre = max(min_val_preictal, int(round(len(pre_b) * val_ratio)))
            n_val_int = max(min_val_interictal, int(round(len(int_b) * val_ratio)))
            n_val_pre = min(n_val_pre, len(pre_b) - 1)
            n_val_int = min(n_val_int, len(int_b) - 1)

            rng.shuffle(pre_b)
            rng.shuffle(int_b)

            va_set = set(
                int(b) for b in (pre_b[:n_val_pre].tolist() + int_b[:n_val_int].tolist())
            )
            tr_set = set(
                int(b) for b in (pre_b[n_val_pre:].tolist() + int_b[n_val_int:].tolist())
            )

            va = idx[np.isin(store.block_ids[idx], list(va_set))]
            tr = idx[np.isin(store.block_ids[idx], list(tr_set))]

            if effective_guard_sec > 0.0:
                tr = _trim_boundary_windows_between_splits(
                    store, tr, va, guard_sec=effective_guard_sec
                )

            va_lbl = store.labels[va]
            tr_lbl = store.labels[tr]
            n_va_pre = int((va_lbl == 1).sum())
            n_va_int = int((va_lbl == 0).sum())
            n_tr_pre = int((tr_lbl == 1).sum())
            n_tr_int = int((tr_lbl == 0).sum())

            if (
                n_va_pre > 0
                and n_va_int > 0
                and n_tr_pre >= min_train_preictal_windows
                and n_tr_int >= min_train_interictal_windows
            ):
                tr_parts.append(tr)
                va_parts.append(va)
                success = True
                break

        if not success:
            _LOGGER.warning(
                "Subject %d: no valid split found after %d retries.",
                int(sid), n_retries,
            )
            empty = np.array([], dtype=np.int64)
            return empty, empty

    tr_idx = (
        np.sort(np.concatenate(tr_parts))
        if tr_parts else np.array([], dtype=np.int64)
    )
    va_idx = (
        np.sort(np.concatenate(va_parts))
        if va_parts else np.array([], dtype=np.int64)
    )
    return tr_idx, va_idx

def enumerate_one_subject_folds(
    store: MasterDataStore,
    val_ratio: float = 0.15,
    block_guard: int = 1,
    train_fraction: float = 1.0,
) -> List[Fold]:
    if not (0.0 < float(train_fraction) <= 1.0):
        raise ValueError(
            f"train_fraction must be in (0, 1], got {train_fraction}"
        )
    protocol_name = (
        "one_subject"
        if abs(float(train_fraction) - 1.0) < 1e-6
        else f"low_data_{int(round(train_fraction * 100))}"
    )
    seed_key = protocol_name
    subjects = list(store.main_cohort_subjects)
    folds: List[Fold] = []

    for fold_id, target in enumerate(subjects):
        fold_seed = _fold_seed(seed_key, int(target))
        train_sids = [int(target)]
        test_sids = [int(s) for s in subjects if int(s) != int(target)]

        tr_idx, va_idx = split_blocks_into_train_val(
            store, train_sids,
            val_ratio=val_ratio, block_guard=block_guard, seed=fold_seed,
        )
        if len(tr_idx) == 0 or len(va_idx) == 0:
            _LOGGER.warning(
                "%s fold %d (target=%d) skipped: empty split.",
                protocol_name, fold_id, int(target),
            )
            continue

        if float(train_fraction) < 1.0 - 1e-6:
            tr_idx = _subsample_train_indices_stratified(
                store, tr_idx, train_fraction=train_fraction, seed=fold_seed + 77
            )

        test_idx = store.get_indices_for(test_sids)

        _assert_no_block_overlap(store, tr_idx, va_idx, f"os_ft{fold_id}_tr_va")
        store.assert_no_leakage(tr_idx, test_idx, f"os_ft{fold_id}_tr_te")
        store.assert_no_leakage(va_idx, test_idx, f"os_ft{fold_id}_va_te")

        folds.append(Fold(
            fold_id=fold_id,
            fold_seed=fold_seed,
            protocol=protocol_name,
            train_sids=train_sids,
            val_sids=train_sids,
            test_sids=test_sids,
            train_idx=tr_idx,
            val_idx=va_idx,
            test_idx=test_idx,
            meta={
                "train_subject": int(target),
                "train_fraction": float(train_fraction),
                "val_source": "held_out_blocks_of_train_subject",
                "n_test_subjects": len(test_sids),
            },
        ))

    return folds

def enumerate_loso_folds(
    store: MasterDataStore,
    n_val_subjects: int = 2,
    train_fraction: float = 1.0,
) -> List[Fold]:
    if not (0.0 < float(train_fraction) <= 1.0):
        raise ValueError(
            f"train_fraction must be in (0, 1], got {train_fraction}"
        )
    subjects = list(store.main_cohort_subjects)
    if len(subjects) <= n_val_subjects + 1:
        raise ValueError(
            f"Not enough subjects for LOSO: {len(subjects)} total, "
            f"need > {n_val_subjects + 1}."
        )
    folds: List[Fold] = []

    for fold_id, target in enumerate(subjects):
        fold_seed = _fold_seed("loso", int(target))
        others = [int(s) for s in subjects if int(s) != int(target)]
        rng = np.random.default_rng(fold_seed)
        val_sids = sorted(
            rng.choice(others, size=n_val_subjects, replace=False).tolist()
        )
        train_sids = [s for s in others if s not in val_sids]
        test_sids = [int(target)]

        tr_idx = store.get_indices_for(train_sids)
        if float(train_fraction) < 1.0 - 1e-6:
            tr_idx = _subsample_train_indices_stratified(
                store, tr_idx, train_fraction=train_fraction, seed=fold_seed + 77
            )
        va_idx = store.get_indices_for(val_sids)
        test_idx = store.get_indices_for(test_sids)

        store.assert_no_leakage(tr_idx, va_idx, f"loso_ft{fold_id}_tr_va")
        store.assert_no_leakage(tr_idx, test_idx, f"loso_ft{fold_id}_tr_te")
        store.assert_no_leakage(va_idx, test_idx, f"loso_ft{fold_id}_va_te")

        folds.append(Fold(
            fold_id=fold_id,
            fold_seed=fold_seed,
            protocol="loso" if train_fraction == 1.0 else f"loso_low_data_{int(round(train_fraction*100))}",
            train_sids=train_sids,
            val_sids=val_sids,
            test_sids=test_sids,
            train_idx=tr_idx,
            val_idx=va_idx,
            test_idx=test_idx,
            meta={
                "test_subject": int(target),
                "n_train_subjects": len(train_sids),
                "n_val_subjects": len(val_sids),
                "train_fraction": float(train_fraction),
            },
        ))

    return folds

def enumerate_holdout_folds(
    store: MasterDataStore,
    test_ratio: float = 0.30,
    val_ratio: float = 0.15,
    block_guard: int = 2,
    n_retries: int = 10,
    boundary_guard_sec: float = 5.0,
) -> List[Fold]:
    subjects = list(store.main_cohort_subjects)
    folds: List[Fold] = []
    effective_guard_sec = boundary_guard_sec if block_guard > 0 else 0.0

    for fold_id, target in enumerate(subjects):
        fold_seed = _fold_seed("holdout_70_30", int(target))
        idx = np.sort(store.get_indices(int(target)))
        blocks = np.unique(store.block_ids[idx])

        if len(blocks) < 6:
            _LOGGER.warning(
                "holdout fold %d (target=%d): only %d blocks; skipping.",
                fold_id, int(target), len(blocks),
            )
            continue

        block_cls: Dict[int, int] = {}
        for b in blocks:
            b_mask = store.block_ids[idx] == b
            lbl_set = set(np.unique(store.labels[idx][b_mask]).tolist())
            if len(lbl_set) != 1:
                _LOGGER.warning(
                    "holdout: subject %d block %d not class-pure", int(target), int(b)
                )
                block_cls = {}
                break
            block_cls[int(b)] = int(lbl_set.pop())

        if not block_cls:
            continue

        pre_base = np.array([b for b in blocks if block_cls[int(b)] == 1])
        int_base = np.array([b for b in blocks if block_cls[int(b)] == 0])

        if len(pre_base) < 3 or len(int_base) < 3:
            _LOGGER.warning(
                "holdout fold %d (target=%d): need >=3 blocks per class for "
                "3-way split (pre=%d, int=%d); skipping.",
                fold_id, int(target), len(pre_base), len(int_base),
            )
            continue

        success = False
        for retry_i in range(max(1, n_retries)):
            rng = np.random.default_rng(fold_seed + retry_i * 1000)

            pre_b = pre_base.copy()
            int_b = int_base.copy()
            rng.shuffle(pre_b)
            rng.shuffle(int_b)

            n_test_pre = max(1, int(round(len(pre_b) * test_ratio)))
            n_test_int = max(1, int(round(len(int_b) * test_ratio)))
            n_test_pre = min(n_test_pre, len(pre_b) - 2)
            n_test_int = min(n_test_int, len(int_b) - 2)

            test_set = set(
                int(b) for b in (pre_b[:n_test_pre].tolist() + int_b[:n_test_int].tolist())
            )

            rem_pre = pre_b[n_test_pre:]
            rem_int = int_b[n_test_int:]

            n_val_pre = max(1, int(round(len(rem_pre) * val_ratio)))
            n_val_int = max(1, int(round(len(rem_int) * val_ratio)))
            n_val_pre = min(n_val_pre, len(rem_pre) - 1)
            n_val_int = min(n_val_int, len(rem_int) - 1)

            val_set = set(
                int(b) for b in (rem_pre[:n_val_pre].tolist() + rem_int[:n_val_int].tolist())
            )
            tr_set = set(
                int(b) for b in (rem_pre[n_val_pre:].tolist() + rem_int[n_val_int:].tolist())
            )

            tr = idx[np.isin(store.block_ids[idx], list(tr_set))]
            va = idx[np.isin(store.block_ids[idx], list(val_set))]
            te = idx[np.isin(store.block_ids[idx], list(test_set))]

            if effective_guard_sec > 0.0:
                tr = _trim_boundary_windows_between_splits(
                    store, tr, np.concatenate([va, te]), guard_sec=effective_guard_sec
                )
                va = _trim_boundary_windows_between_splits(
                    store, va, te, guard_sec=effective_guard_sec
                )

            if all(
                (store.labels[arr] == 0).sum() > 0 and (store.labels[arr] == 1).sum() > 0
                for arr in (tr, va, te)
            ):
                _assert_no_block_overlap(store, tr, va, f"ho_ft{fold_id}_tr_va")
                _assert_no_block_overlap(store, tr, te, f"ho_ft{fold_id}_tr_te")
                _assert_no_block_overlap(store, va, te, f"ho_ft{fold_id}_va_te")

                folds.append(Fold(
                    fold_id=fold_id,
                    fold_seed=fold_seed,
                    protocol="holdout_70_30",
                    train_sids=[int(target)],
                    val_sids=[int(target)],
                    test_sids=[int(target)],
                    train_idx=tr,
                    val_idx=va,
                    test_idx=te,
                    meta={
                        "subject": int(target),
                        "test_ratio": test_ratio,
                        "val_ratio_in_remaining": val_ratio,
                        "boundary_guard_sec": effective_guard_sec,
                        "test_split": "block_shuffle_stratified",
                    },
                ))
                success = True
                break

        if not success:
            _LOGGER.warning(
                "holdout fold %d (target=%d): no valid 3-way split found.",
                fold_id, int(target),
            )

    return folds

class DBTRDataset(Dataset):
    def __init__(
        self,
        store: MasterDataStore,
        indices: np.ndarray,
        *,
        active_freq_idx: Optional[np.ndarray] = None,
        return_metadata: bool = True,
        use_active_freq_slice: bool = True,
    ):
        super().__init__()

        if not isinstance(store, MasterDataStore):
            raise TypeError(
                f"store must be a MasterDataStore, got {type(store).__name__}"
            )
        if store.time_mm is None or store.freq_mm is None:
            raise RuntimeError(
                "store has been closed; re-open it before constructing a Dataset."
            )

        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1:
            raise ValueError(
                f"indices must be 1-D, got shape {self.indices.shape}."
            )

        self.store = store
        self.return_metadata = bool(return_metadata)
        self.use_active_freq_slice = bool(use_active_freq_slice)

        if not hasattr(store, "n_channels") or not hasattr(store, "freq_layout"):
            raise RuntimeError(
                "MasterDataStore does not expose n_channels/freq_layout; "
                "update your store (see Section 8 of the pipeline spec)."
            )
        self.n_channels = int(store.n_channels)
        if store.freq_layout != "channel_major":
            raise RuntimeError(
                f"Unsupported freq layout '{store.freq_layout}'; "
                f"this Dataset assumes channel-major per-timestep flattening."
            )

        self.n_steps = int(store.n_steps)
        self.t_dim   = int(store.t_dim)
        self.f_dim   = int(store.f_dim)
        if self.f_dim % self.n_channels != 0:
            raise ValueError(
                f"f_dim={self.f_dim} is not divisible by "
                f"n_channels={self.n_channels}; cache is inconsistent."
            )
        self.n_fb = self.f_dim // self.n_channels

        if active_freq_idx is None:
            active_freq_idx = store.active_freq_idx
        self.active_freq_idx = np.asarray(active_freq_idx, dtype=np.int64)

        if self.use_active_freq_slice:
            if self.active_freq_idx.ndim != 1:
                raise ValueError("active_freq_idx must be 1-D.")
            if self.active_freq_idx.size == 0:
                raise ValueError("active_freq_idx must be non-empty.")
            if int(self.active_freq_idx.min()) < 0:
                raise ValueError(
                    f"active_freq_idx min={int(self.active_freq_idx.min())} is negative."
                )
            if int(self.active_freq_idx.max()) >= self.n_fb:
                raise ValueError(
                    f"active_freq_idx max={int(self.active_freq_idx.max())} "
                    f"out of range for n_fb={self.n_fb}."
                )
            if not np.all(np.diff(self.active_freq_idx) > 0):
                raise ValueError(
                    "active_freq_idx must be strictly increasing."
                )
            self.n_fb_active = int(self.active_freq_idx.size)
            self.f_dim_out = self.n_fb_active * self.n_channels
        else:
            self.n_fb_active = self.n_fb
            self.f_dim_out = self.f_dim

        _LOGGER.debug(
            "DBTRDataset: n=%d | n_steps=%d | t_dim=%d | "
            "freq_in=%d | freq_out=%d (%d active bins / %d) | slice=%s",
            len(self.indices), self.n_steps, self.t_dim,
            self.f_dim, self.f_dim_out,
            self.n_fb_active, self.n_fb, self.use_active_freq_slice,
        )

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(
                f"index {i} out of range for dataset of size {len(self)}"
            )

        row = int(self.indices[i])

        time_np = np.ascontiguousarray(
            np.asarray(self.store.time_mm[row], dtype=np.float32)
        )

        freq_flat = np.asarray(self.store.freq_mm[row], dtype=np.float32)

        if self.use_active_freq_slice:
            freq_3d = freq_flat.reshape(
                self.n_steps, self.n_channels, self.n_fb
            )
            freq_sliced = freq_3d[:, :, self.active_freq_idx]
            freq_np = np.ascontiguousarray(
                freq_sliced.reshape(self.n_steps, self.f_dim_out)
            )
        else:
            freq_np = freq_flat

        item: Dict[str, Any] = {
            "time":  torch.from_numpy(time_np),
            "freq":  torch.from_numpy(freq_np),
            "label": int(self.store.labels[row]),
        }

        if self.return_metadata:
            item.update({
                "subject_id":  int(self.store.subject_ids[row]),
                "block_id":    int(self.store.block_ids[row]),
                "edf_file_id": int(self.store.edf_file_ids[row]),
                "onset":       float(self.store.onsets[row]),
                "dur":         float(self.store.durs[row]),
                "win_dur":     float(self.store.win_durs[row]),
                "index":       row,
            })

        return item

    def class_balance(self) -> Tuple[int, int]:
        lbl = self.store.labels[self.indices]
        return int((lbl == 0).sum()), int((lbl == 1).sum())

    def interictal_hours(self) -> float:
        return float(self.store.interictal_hours(self.indices))

def _sample_beta_lambda(alpha: float, device: torch.device) -> float:
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    dist = torch.distributions.Beta(
        torch.tensor(float(alpha), device=device),
        torch.tensor(float(alpha), device=device),
    )
    return float(dist.sample().item())

class LabelSmoothingCrossEntropy(nn.Module):
    _REDUCTIONS = frozenset({"none", "sum", "mean"})
    _SMOOTHING_MODES = frozenset({"uniform", "manuscript_binary"})

    def __init__(
        self,
        epsilon: float = 0.1,
        reduction: str = "mean",
        smoothing_mode: str = "uniform",
    ) -> None:
        super().__init__()

        if not (0.0 <= float(epsilon) < 1.0):
            raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
        if reduction not in self._REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {sorted(self._REDUCTIONS)}, "
                f"got {reduction!r}"
            )
        if smoothing_mode not in self._SMOOTHING_MODES:
            raise ValueError(
                f"smoothing_mode must be one of "
                f"{sorted(self._SMOOTHING_MODES)}, got {smoothing_mode!r}"
            )

        self.epsilon = float(epsilon)
        self.reduction = reduction
        self.smoothing_mode = smoothing_mode

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(
                f"logits must be 2-D (N, C), got shape {tuple(logits.shape)}"
            )

        n, c = logits.shape
        log_probs = F.log_softmax(logits, dim=1)

        if targets.dim() == 1:
            if targets.shape[0] != n:
                raise ValueError(
                    f"batch size mismatch: logits={n}, "
                    f"targets={targets.shape[0]}"
                )

            one_hot = F.one_hot(targets.long(), num_classes=c).to(log_probs.dtype)

            if self.epsilon > 0.0 and c > 1:
                if self.smoothing_mode == "uniform":
                    one_hot = (
                        one_hot * (1.0 - self.epsilon)
                        + self.epsilon / float(c)
                    )
                elif self.smoothing_mode == "manuscript_binary":
                    if c != 2:
                        raise ValueError(
                            f"smoothing_mode='manuscript_binary' is defined "
                            f"only for binary classification (C=2), got C={c}. "
                            f"Use smoothing_mode='uniform' for multi-class."
                        )
                    one_hot = one_hot * (1.0 - 2.0 * self.epsilon) + self.epsilon

            target_probs = one_hot

        elif targets.dim() == 2:
            if targets.shape != logits.shape:
                raise ValueError(
                    f"soft-target shape mismatch: "
                    f"logits={tuple(logits.shape)}, "
                    f"targets={tuple(targets.shape)}"
                )
            target_probs = targets.to(log_probs.dtype)

        else:
            raise ValueError(
                f"targets must be 1-D (hard) or 2-D (soft), "
                f"got {targets.dim()}-D"
            )

        per_sample = -(target_probs * log_probs).sum(dim=1)

        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample

    def extra_repr(self) -> str:
        return (
            f"epsilon={self.epsilon}, reduction={self.reduction!r}, "
            f"smoothing_mode={self.smoothing_mode!r}"
        )

def mixup_batch_data(
    batch: Dict[str, Any],
    *,
    alpha: float = 0.2,
    n_classes: int = 2,
) -> Dict[str, Any]:
    if "time" not in batch or "freq" not in batch or "label" not in batch:
        raise KeyError(
            "batch must contain 'time', 'freq', and 'label'; got "
            f"{sorted(batch.keys())}"
        )

    time = batch["time"]
    freq = batch["freq"]
    label = batch["label"]
    n = time.shape[0]

    if time.shape[0] != freq.shape[0] or time.shape[0] != label.shape[0]:
        raise ValueError(
            f"batch size mismatch: time={time.shape[0]}, "
            f"freq={freq.shape[0]}, label={label.shape[0]}"
        )

    if n < 2:
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        one_hot = F.one_hot(label.long(), num_classes=n_classes).to(time.dtype)
        out = dict(batch)
        out["label_soft"] = one_hot
        out["is_mixed"] = False
        return out

    device = time.device

    lam = _sample_beta_lambda(alpha, device)

    perm = torch.randperm(n, device=device)

    mixed_time = lam * time + (1.0 - lam) * time[perm]
    mixed_freq = lam * freq + (1.0 - lam) * freq[perm]

    one_hot = F.one_hot(label.long(), num_classes=n_classes).to(time.dtype)
    one_hot_perm = one_hot[perm]
    label_soft = lam * one_hot + (1.0 - lam) * one_hot_perm

    out = dict(batch)
    out["time"] = mixed_time
    out["freq"] = mixed_freq
    out["label"] = label
    out["label_soft"] = label_soft
    out["is_mixed"] = True
    return out

def mixup_collate_fn(
    batch: List[Dict[str, Any]],
    *,
    alpha: float = 0.2,
    use_mixup: bool = False,
    p_mixup: float = 1.0,
    n_classes: int = 2,
) -> Dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("mixup_collate_fn received an empty batch")
    if not (0.0 <= p_mixup <= 1.0):
        raise ValueError(f"p_mixup must be in [0, 1], got {p_mixup}")
    if use_mixup and p_mixup > 0.0 and alpha <= 0.0:
        raise ValueError(
            f"alpha must be > 0 when use_mixup=True and p_mixup>0, got {alpha}"
        )

    collated = default_collate(batch)

    collated["is_mixed"] = False

    if "label_soft" not in collated:
        label = collated["label"]
        collated["label_soft"] = F.one_hot(
            label.long(), num_classes=n_classes,
        ).to(collated["time"].dtype)

    if not use_mixup:
        return collated

    gate = torch.rand(1, device=collated["time"].device)
    if float(gate.item()) >= p_mixup:
        return collated

    return mixup_batch_data(collated, alpha=alpha, n_classes=n_classes)

def _sanity_check_batch_pipeline(
    store: MasterDataStore,
    fold: Fold,
    *,
    batch_size: int = 8,
    n_mixup_batches: int = 4,
    n_classes: int = 2,
    epsilon: float = 0.1,
) -> None:
    from torch.utils.data import DataLoader

    train_ds = DBTRDataset(store, fold.train_idx, return_metadata=True)
    val_ds = DBTRDataset(store, fold.val_idx, return_metadata=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_MixupCollateCallable(
            alpha=0.2, use_mixup=True, p_mixup=1.0, n_classes=n_classes,
        ),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_MixupCollateCallable(use_mixup=False, n_classes=n_classes),
        num_workers=0,
    )

    loss_fn = LabelSmoothingCrossEntropy(epsilon=epsilon)
    loss_fn_no_smooth = LabelSmoothingCrossEntropy(epsilon=0.0)

    n_mixed_expected = min(n_mixup_batches, len(train_loader))
    if n_mixed_expected == 0:
        raise RuntimeError(
            "_sanity_check_batch_pipeline: train_loader is empty; "
            "check fold.train_idx."
        )

    n_seen = 0
    saw_mixed = False
    for i, batch in enumerate(train_loader):
        if i >= n_mixed_expected:
            break
        n_seen += 1

        assert "time" in batch and "freq" in batch, "missing time/freq"
        assert "label" in batch, "missing hard label"
        assert "label_soft" in batch, "missing label_soft"
        assert "is_mixed" in batch, "missing is_mixed flag"
        assert isinstance(batch["is_mixed"], bool), type(batch["is_mixed"])
        assert batch["time"].dtype == torch.float32
        assert batch["freq"].dtype == torch.float32
        assert batch["label"].dtype in (torch.int32, torch.int64)
        assert batch["label_soft"].shape[-1] == n_classes
        assert torch.isfinite(batch["time"]).all()
        assert torch.isfinite(batch["freq"]).all()

        n_batch = batch["time"].shape[0]
        expected_mixed = n_batch >= 2
        assert batch["is_mixed"] is expected_mixed, (
            f"batch {i}: N={n_batch} but is_mixed={batch['is_mixed']} "
            f"(expected {expected_mixed})"
        )
        if expected_mixed:
            saw_mixed = True

        row_sums = batch["label_soft"].sum(dim=1)
        assert torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-5,
        ), f"label_soft row sums != 1 on train batch {i}"

        logits = torch.randn(n_batch, n_classes, requires_grad=True)
        targets = batch["label_soft"] if batch["is_mixed"] else batch["label"]
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss), f"non-finite loss on train batch {i}"

    assert n_seen == n_mixed_expected, (
        f"expected {n_mixed_expected} train batches, got {n_seen}"
    )
    if len(train_ds) >= 2 and batch_size >= 2 and n_seen >= 1:
        assert saw_mixed, (
            "no train batch with N>=2 was tested; the Mixup loss path "
            "was never exercised. Increase batch_size or inspect train_ds."
        )

    n_val_seen = 0
    for i, batch in enumerate(val_loader):
        if i >= 1:
            break
        n_val_seen += 1

        assert "label_soft" in batch, "missing label_soft in val batch"
        assert "is_mixed" in batch, "missing is_mixed flag in val batch"
        assert isinstance(batch["is_mixed"], bool)
        assert batch["is_mixed"] is False, (
            f"expected is_mixed=False on val batch {i}, "
            f"got {batch['is_mixed']}"
        )
        assert batch["label_soft"].shape[-1] == n_classes

        assert torch.allclose(
            batch["label_soft"].max(dim=1).values,
            torch.ones(batch["label_soft"].shape[0]),
            atol=1e-6,
        ), "val label_soft is not one-hot (max != 1)"
        assert torch.allclose(
            batch["label_soft"].sum(dim=1),
            torch.ones(batch["label_soft"].shape[0]),
            atol=1e-6,
        ), "val label_soft is not one-hot (rows do not sum to 1)"

        logits = torch.randn(batch["time"].shape[0], n_classes, requires_grad=True)
        loss_smoothed = loss_fn(logits, batch["label"])
        loss_unsmoothed = loss_fn_no_smooth(logits, batch["label"])
        assert torch.isfinite(loss_smoothed), (
            f"non-finite loss on val batch {i}"
        )
        if epsilon > 0.0:
            assert not torch.allclose(
                loss_smoothed, loss_unsmoothed, atol=1e-6,
            ), (
                f"label smoothing (ε={epsilon}) did not change the loss; "
                f"it is likely not being applied. "
                f"smoothed={loss_smoothed.item():.6f}, "
                f"unsmoothed={loss_unsmoothed.item():.6f}"
            )

    assert n_val_seen == 1, f"expected 1 val batch, got {n_val_seen}"

    print(
        f"[batch-sanity] PASS | train_batches={n_seen}/{n_mixed_expected} | "
        f"batch_size={batch_size} | n_classes={n_classes} | ε={epsilon} | "
        f"saw_mixed={saw_mixed}"
    )

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
        learnable: bool = False,
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}"
            )
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")

        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.dropout = nn.Dropout(p=float(dropout))
        self.learnable = bool(learnable)
        self.scale = float(scale)

        pe = self._build_sinusoidal_table(self.max_len, self.d_model) * self.scale

        if self.learnable:
            self.pe = nn.Parameter(pe)
        else:
            self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_sinusoidal_table(
        max_len: int,
        d_model: int,
    ) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / float(d_model))
        )
        pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        n_odd = d_model // 2
        pe[0, :, 1::2] = torch.cos(position * div_term[:n_odd])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(
                f"x must be a floating-point tensor, got {x.dtype}. "
                f"PE addition is undefined for categorical/integer data."
            )

        if x.dim() != 3:
            raise ValueError(
                f"x must be 3-D (B, T, d_model), got {x.dim()}-D with "
                f"shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"x.shape[-1]={x.shape[-1]} does not match "
                f"d_model={self.d_model}"
            )
        T = int(x.shape[1])
        if T > self.max_len:
            raise ValueError(
                f"sequence length T={T} exceeds max_len={self.max_len}"
            )

        if x.device != self.pe.device:
            raise ValueError(
                f"x is on device {x.device} but the PE table is on "
                f"{self.pe.device}. Move the model with `.to(device)` "
                f"and ensure the input is on the same device."
            )

        pe_slice = self.pe[:, :T, :].to(dtype=x.dtype)
        return self.dropout(x + pe_slice)

def _sanity_check_positional_encoding() -> None:
    pe = PositionalEncoding(d_model=64, max_len=32, dropout=0.0)
    pe.eval()
    x = torch.zeros(4, 14, 64)

    y = pe(x)
    assert y.shape == (4, 14, 64), f"shape mismatch: {tuple(y.shape)}"

    assert torch.allclose(y[0, 0, 0::2], torch.zeros(32)), "sin(0) != 0"
    assert torch.allclose(y[0, 0, 1::2], torch.ones(32)), "cos(0) != 1"

    y2 = pe(x)
    assert torch.equal(y, y2), "sinusoidal PE is not deterministic in eval"

    assert isinstance(pe.pe, torch.Tensor)
    assert not pe.pe.requires_grad, "sinusoidal PE should not require grad"

    state_keys = set(pe.state_dict().keys())
    assert "pe" not in state_keys, (
        f"sinusoidal PE should be non-persistent, but 'pe' is in state_dict "
        f"keys: {state_keys}"
    )

    try:
        pe(torch.zeros(1, 33, 64))
        raise AssertionError("expected ValueError for T > max_len")
    except ValueError:
        pass

    try:
        pe(torch.zeros(14, 64))
        raise AssertionError("expected ValueError for non-3-D input")
    except ValueError:
        pass

    try:
        pe(torch.zeros(1, 14, 32))
        raise AssertionError("expected ValueError for d_model mismatch")
    except ValueError:
        pass

    try:
        pe(torch.zeros(1, 14, 64, dtype=torch.long))
        raise AssertionError("expected TypeError for integer input")
    except TypeError:
        pass

    pe_l = PositionalEncoding(
        d_model=64, max_len=32, dropout=0.0, learnable=True,
    )
    pe_l.eval()
    assert pe_l.pe.requires_grad, "learnable PE should require grad"
    assert "pe" in set(pe_l.state_dict().keys()), (
        "learnable PE should be persistent in state_dict"
    )

    x_grad = torch.randn(4, 14, 64, requires_grad=True)
    y_grad = pe_l(x_grad)
    assert y_grad.shape == (4, 14, 64)
    y_grad.sum().backward()
    assert pe_l.pe.grad is not None, "no gradient on PE parameter"
    assert torch.isfinite(pe_l.pe.grad).all(), (
        "non-finite gradient on PE parameter"
    )
    assert x_grad.grad is not None, "no gradient on input"

    p = 0.5
    pe_d = PositionalEncoding(
        d_model=64, max_len=32, dropout=p,
    )
    x_rand = torch.randn(4, 14, 64)

    pe_d.train()
    y_train = pe_d(x_rand)
    pe_d.eval()
    y_eval = pe_d(x_rand)

    assert y_train.shape == y_eval.shape == (4, 14, 64)
    assert not torch.equal(y_train, y_eval), (
        "dropout had no effect in train mode (train == eval)"
    )
    mask = y_train != 0
    assert mask.any(), "all elements were zeroed by dropout"
    assert torch.allclose(
        y_train[mask] * (1.0 - p), y_eval[mask], atol=1e-6,
    ), "dropout scaling (1/(1-p)) is incorrect"

    pe_odd = PositionalEncoding(d_model=63, max_len=16, dropout=0.0)
    pe_odd.eval()
    y_odd = pe_odd(torch.zeros(2, 14, 63))
    assert y_odd.shape == (2, 14, 63)

    pe_s1 = PositionalEncoding(
        d_model=8, max_len=4, dropout=0.0, scale=1.0,
    )
    pe_s2 = PositionalEncoding(
        d_model=8, max_len=4, dropout=0.0, scale=2.0,
    )
    pe_s1.eval(); pe_s2.eval()
    y1 = pe_s1(torch.zeros(1, 4, 8))
    y2 = pe_s2(torch.zeros(1, 4, 8))
    assert torch.allclose(y2, 2.0 * y1), "scale not applied"
    assert torch.allclose(y2[0, 0, 1::2], torch.full((4,), 2.0))

    if torch.cuda.is_available():
        pe_cuda = PositionalEncoding(
            d_model=8, max_len=4, dropout=0.0,
        ).cuda()
        pe_cuda.eval()
        try:
            pe_cuda(torch.zeros(1, 4, 8))
            raise AssertionError(
                "expected ValueError for CPU input to CUDA PE"
            )
        except ValueError as exc:
            assert "device" in str(exc).lower(), (
                f"device-mismatch error message unclear: {exc}"
            )
    else:
        pe_meta = PositionalEncoding(
            d_model=8, max_len=4, dropout=0.0,
        )
        pe_meta.pe = pe_meta.pe.to("meta")
        try:
            pe_meta(torch.zeros(1, 4, 8))
            raise AssertionError(
                "expected ValueError for CPU input to meta-device PE"
            )
        except ValueError as exc:
            assert "device" in str(exc).lower(), (
                f"device-mismatch error message unclear: {exc}"
            )

    print("[pe-sanity] PASS")

class EnergyPreservingChannelEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int,
        dropout: float = 0.1,
        use_input_norm: bool = True,
        use_output_norm: bool = True,
        use_orthogonal_init: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(
                f"in_features must be a positive int, got {in_features!r}"
            )
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(
                f"d_model must be a positive int, got {d_model!r}"
            )
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}"
            )

        self.in_features = int(in_features)
        self.d_model = int(d_model)
        self.use_input_norm = bool(use_input_norm)
        self.use_output_norm = bool(use_output_norm)
        self.use_orthogonal_init = bool(use_orthogonal_init)

        self.in_norm: Optional[nn.LayerNorm] = (
            nn.LayerNorm(self.in_features)
            if self.use_input_norm else None
        )

        self.proj = nn.Linear(self.in_features, self.d_model, bias=True)
        if self.use_orthogonal_init:
            gain = math.sqrt(
                self.in_features / min(self.d_model, self.in_features)
            )
            nn.init.orthogonal_(self.proj.weight, gain=gain)
            nn.init.zeros_(self.proj.bias)

        self.out_norm: Optional[nn.LayerNorm] = (
            nn.LayerNorm(self.d_model)
            if self.use_output_norm else None
        )

        self.dropout = nn.Dropout(p=float(dropout))

        if _LOGGER is not None:
            _LOGGER.debug(
                "EnergyPreservingChannelEncoder: in=%d -> d_model=%d | "
                "input_norm=%s | output_norm=%s | orthogonal_init=%s | "
                "dropout=%.3f",
                self.in_features, self.d_model,
                self.use_input_norm, self.use_output_norm,
                self.use_orthogonal_init, float(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(
                f"x must be a floating-point tensor, got {x.dtype}. "
                f"Channel encoder projection is undefined for "
                f"categorical/integer data."
            )

        if x.dim() != 3:
            raise ValueError(
                f"x must be 3-D (B, T, in_features), got {x.dim()}-D "
                f"with shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"x.shape[-1]={x.shape[-1]} does not match "
                f"in_features={self.in_features}"
            )

        if self.in_norm is not None:
            x = self.in_norm(x)

        x = self.proj(x)

        if self.out_norm is not None:
            x = self.out_norm(x)

        return self.dropout(x)

def _sanity_check_channel_encoder() -> None:
    enc_t = EnergyPreservingChannelEncoder(
        in_features=576, d_model=128, dropout=0.0,
    )
    enc_t.eval()
    x_t = torch.rand(4, 14, 576)
    y_t = enc_t(x_t)
    assert y_t.shape == (4, 14, 128), f"shape mismatch: {tuple(y_t.shape)}"
    assert y_t.dtype == torch.float32
    assert torch.isfinite(y_t).all()

    enc_s = EnergyPreservingChannelEncoder(
        in_features=8964, d_model=128, dropout=0.0,
    )
    enc_s.eval()
    x_s = torch.rand(2, 14, 8964)
    y_s = enc_s(x_s)
    assert y_s.shape == (2, 14, 128), f"shape mismatch: {tuple(y_s.shape)}"

    target_norm = math.sqrt(128)
    norms_t = y_t.norm(dim=-1)
    assert torch.allclose(
        norms_t, torch.full_like(norms_t, target_norm), atol=1e-3,
    ), (
        f"CWT output L2 norms deviate from sqrt(d_model)={target_norm:.4f}: "
        f"min={norms_t.min().item():.4f}, max={norms_t.max().item():.4f}"
    )
    norms_s = y_s.norm(dim=-1)
    assert torch.allclose(
        norms_s, torch.full_like(norms_s, target_norm), atol=1e-3,
    ), "STFT output L2 norms deviate from sqrt(d_model)"

    x_small = torch.randn(2, 5, 576) * 1e-3
    x_large = torch.randn(2, 5, 576) * 1e3
    y_small = enc_t(x_small)
    y_large = enc_t(x_large)
    norm_diff = (
        y_small.norm(dim=-1) - y_large.norm(dim=-1)
    ).abs().max().item()
    assert norm_diff < 1e-3, (
        f"output norm depends on input scale: max diff={norm_diff:.4f}"
    )

    enc_unit = EnergyPreservingChannelEncoder(
        in_features=576, d_model=128, dropout=0.0,
        use_input_norm=True, use_output_norm=False,
        use_orthogonal_init=True,
    )
    enc_unit.eval()
    x_rand = torch.randn(128, 5, 576)
    y_unit = enc_unit(x_rand)
    x_norm = enc_unit.in_norm(x_rand).norm(dim=-1).mean().item()
    y_norm = y_unit.norm(dim=-1).mean().item()
    ratio = y_norm / x_norm
    assert 0.9 < ratio < 1.1, (
        f"orthogonal init (out<in) energy ratio = {ratio:.3f}, "
        f"expected ~1.0 (x_norm={x_norm:.2f}, y_norm={y_norm:.2f})"
    )

    enc_tall = EnergyPreservingChannelEncoder(
        in_features=16, d_model=64, dropout=0.0,
        use_input_norm=True, use_output_norm=False,
        use_orthogonal_init=True,
    )
    enc_tall.eval()
    x_tall = torch.randn(128, 1, 16)
    y_tall = enc_tall(x_tall)
    x_norm_t = enc_tall.in_norm(x_tall).norm(dim=-1).mean().item()
    y_norm_t = y_tall.norm(dim=-1).mean().item()
    ratio_t = y_norm_t / x_norm_t
    assert 0.9 < ratio_t < 1.1, (
        f"orthogonal init (out>in) energy ratio = {ratio_t:.3f}, "
        f"expected ~1.0 (x_norm={x_norm_t:.2f}, y_norm={y_norm_t:.2f})"
    )

    enc_grad = EnergyPreservingChannelEncoder(
        in_features=576, d_model=128, dropout=0.0,
    )
    x_grad = torch.randn(2, 14, 576, requires_grad=True)
    y_grad = enc_grad(x_grad)
    y_grad.sum().backward()
    assert enc_grad.proj.weight.grad is not None
    assert torch.isfinite(enc_grad.proj.weight.grad).all()
    assert x_grad.grad is not None

    try:
        enc_t(torch.zeros(14, 576))
        raise AssertionError("expected ValueError for non-3-D input")
    except ValueError:
        pass

    try:
        enc_t(torch.zeros(1, 14, 32))
        raise AssertionError("expected ValueError for in_features mismatch")
    except ValueError:
        pass

    try:
        enc_t(torch.zeros(1, 14, 576, dtype=torch.long))
        raise AssertionError("expected TypeError for integer input")
    except TypeError:
        pass

    enc_d = EnergyPreservingChannelEncoder(
        in_features=64, d_model=32, dropout=0.5,
    )
    x_d = torch.randn(4, 5, 64)
    enc_d.train()
    y_train = enc_d(x_d)
    enc_d.eval()
    y_eval = enc_d(x_d)
    assert y_train.shape == y_eval.shape == (4, 5, 32)
    assert not torch.equal(y_train, y_eval), (
        "dropout had no effect in train mode (train == eval)"
    )

    enc_kaiming = EnergyPreservingChannelEncoder(
        in_features=64, d_model=32, dropout=0.0,
        use_orthogonal_init=False,
    )
    enc_kaiming.eval()
    assert enc_kaiming(torch.randn(2, 5, 64)).shape == (2, 5, 32)

    enc_noin = EnergyPreservingChannelEncoder(
        in_features=64, d_model=32, dropout=0.0,
        use_input_norm=False, use_output_norm=False,
        use_orthogonal_init=False,
    )
    enc_noin.eval()
    y_noin = enc_noin(torch.randn(2, 5, 64) * 50.0)
    assert y_noin.shape == (2, 5, 32)
    assert torch.isfinite(y_noin).all()

    print("[channel-encoder-sanity] PASS")

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not (0.0 <= float(drop_prob) < 1.0):
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        causal: bool = False,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError(
                f"n_heads={n_heads} must be positive and divide d_model={d_model}"
            )
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not (0.0 <= float(drop_path) < 1.0):
            raise ValueError(f"drop_path must be in [0, 1), got {drop_path}")
        if activation not in {"gelu", "relu"}:
            raise ValueError(
                f"activation must be 'gelu' or 'relu', got {activation!r}"
            )

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.causal = bool(causal)
        self.activation = activation

        self.norm1 = nn.LayerNorm(self.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(self.d_model)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            act,
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_ff, self.d_model),
        )
        self.drop_path2 = DropPath(drop_path)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((T, T), device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(f"x must be floating-point, got {x.dtype}")
        if x.dim() != 3:
            raise ValueError(
                f"x must be 3-D (B, T, d_model), got {x.dim()}-D "
                f"with shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"x.shape[-1]={x.shape[-1]} does not match d_model={self.d_model}"
            )
        B, T, _ = x.shape

        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, T):
                raise ValueError(
                    f"key_padding_mask must be ({B}, {T}), got "
                    f"{tuple(key_padding_mask.shape)}"
                )
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(torch.bool)

        attn_mask = self._causal_mask(T, x.device) if self.causal else None

        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_path1(attn_out)

        x_norm = self.norm2(x)
        x = x + self.drop_path2(self.ff(x_norm))
        return x

class TransformerStack(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        causal: bool = False,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if not (0.0 <= float(drop_path_rate) < 1.0):
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {drop_path_rate}"
            )

        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.drop_path_rate = float(drop_path_rate)
        self.causal = bool(causal)
        self.activation = activation

        if self.n_layers == 1 and self.drop_path_rate > 0.0:
            warnings.warn(
                f"drop_path_rate={self.drop_path_rate} ignored for "
                f"n_layers=1; no schedule to interpolate.",
                RuntimeWarning,
            )
        if self.n_layers > 1 and self.drop_path_rate > 0.0:
            drop_paths = [
                self.drop_path_rate * i / (self.n_layers - 1)
                for i in range(self.n_layers)
            ]
        else:
            drop_paths = [0.0] * self.n_layers

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                drop_path=dp,
                causal=self.causal,
                activation=self.activation,
            )
            for dp in drop_paths
        ])
        self.norm = nn.LayerNorm(self.d_model)

        if _LOGGER is not None:
            _LOGGER.debug(
                "TransformerStack: n_layers=%d | d_model=%d | n_heads=%d | "
                "d_ff=%d | dropout=%.3f | drop_path_rate=%.3f | "
                "causal=%s | activation=%s | per_block_drop_path=%s",
                self.n_layers, self.d_model, self.n_heads, self.d_ff,
                self.dropout, self.drop_path_rate, self.causal,
                self.activation, [f"{dp:.3f}" for dp in drop_paths],
            )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.norm(x)

def _sanity_check_transformer() -> None:
    d_model = 128
    n_heads = 4
    d_ff = 512

    blk = TransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=0.0,
    )
    blk.eval()

    for x in (
        torch.rand(4, 14, d_model),
        torch.randn(4, 14, d_model),
        torch.zeros(2, 14, d_model),
        torch.ones(2, 14, d_model),
    ):
        y = blk(x)
        assert y.shape == x.shape, (
            f"shape mismatch: input {tuple(x.shape)}, "
            f"output {tuple(y.shape)}"
        )
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all(), (
            f"non-finite output for input distribution "
            f"min={x.min().item():.4f}, max={x.max().item():.4f}"
        )

    blk_c = TransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        dropout=0.0, causal=True,
    )
    blk_c.eval()
    x_c = torch.randn(1, 14, d_model)
    x_mod = x_c.clone()
    _single_block_perturbation = torch.randn(d_model) * 5.0
    x_mod[:, 7, :] += _single_block_perturbation

    y_orig = blk_c(x_c)
    y_mod = blk_c(x_mod)

    assert torch.allclose(y_orig[:, 0, :], y_mod[:, 0, :], atol=1e-6), (
        "single-block causal mask violated: position 0 changed when "
        "position 7 was perturbed"
    )
    assert not torch.allclose(y_orig[:, 7, :], y_mod[:, 7, :], atol=1e-6), (
        "position 7 did not change when itself was perturbed"
    )

    _stack_test_seed = 1234
    _stack_gen = torch.Generator(device="cpu")
    _stack_gen.manual_seed(_stack_test_seed)

    stack_c = TransformerStack(
        n_layers=3, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        dropout=0.0, drop_path_rate=0.0, causal=True,
    )
    stack_c.eval()

    xa = torch.randn(1, 14, d_model, generator=_stack_gen)
    xb = xa.clone()
    _perturbation = torch.randn(d_model, generator=_stack_gen) * 5.0
    xb[:, 10, :] += _perturbation

    ya = stack_c(xa)
    yb = stack_c(xb)

    assert torch.allclose(
        ya[:, :10, :], yb[:, :10, :], atol=1e-6,
    ), (
        "multi-block causal mask violated: positions < 10 changed when "
        "position 10 was perturbed"
    )

    _pos10_delta = float(
        (ya[:, 10, :] - yb[:, 10, :]).abs().max().item()
    )
    assert _pos10_delta > 1e-4, (
        f"position 10 did not change when itself was perturbed: "
        f"max|Δ|={_pos10_delta:.3e} (seed={_stack_test_seed}, "
        f"perturbation=randn*5). If this fails, verify the causal mask "
        f"in TransformerBlock._causal_mask uses the boolean upper-"
        f"triangular convention (True=masked)."
    )

    blk_nc = TransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        dropout=0.0, causal=False,
    )
    blk_nc.eval()
    y_orig_nc = blk_nc(x_c)
    y_mod_nc = blk_nc(x_mod)
    assert not torch.allclose(
        y_orig_nc[:, 0, :], y_mod_nc[:, 0, :], atol=1e-6,
    ), "non-causal mask: position 0 should change when position 7 changes"

    blk_pad = TransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=0.0,
    )
    blk_pad.eval()
    x_p = torch.randn(2, 14, d_model)
    kpm = torch.zeros(2, 14, dtype=torch.bool)
    kpm[:, -3:] = True
    y_p1 = blk_pad(x_p, key_padding_mask=kpm)
    x_p_mod = x_p.clone()
    x_p_mod[:, -3:, :] += 1.0
    y_p2 = blk_pad(x_p_mod, key_padding_mask=kpm)
    assert torch.allclose(y_p1[:, :-3, :], y_p2[:, :-3, :], atol=1e-5), (
        "key_padding_mask violated: unmasked positions changed when only "
        "padding positions were perturbed"
    )

    stack = TransformerStack(
        n_layers=2, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        dropout=0.0, drop_path_rate=0.0,
    )
    stack.train()
    x_train = torch.randn(2, 14, d_model, requires_grad=True)
    y_train = stack(x_train)
    y_train.sum().backward()
    assert x_train.grad is not None
    for name, p in stack.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    stack_dp = TransformerStack(
        n_layers=4, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        dropout=0.0, drop_path_rate=0.3,
    )
    expected = [0.0, 0.1, 0.2, 0.3]
    actual = [b.drop_path1.drop_prob for b in stack_dp.blocks]
    assert all(abs(e - a) < 1e-9 for e, a in zip(expected, actual)), (
        f"DropPath schedule mismatch: expected {expected}, got {actual}"
    )

    dp = DropPath(drop_prob=0.5)
    z = torch.ones(100, 4)
    dp.eval()
    assert torch.equal(dp(z), z)
    dp.train()
    assert (dp(z) != z).any()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TransformerStack(
            n_layers=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            drop_path_rate=0.3,
        )
        assert any("drop_path_rate" in str(wi.message) for wi in w), (
            "expected RuntimeWarning for n_layers=1 + drop_path_rate>0"
        )

    if torch.cuda.is_available():
        blk_amp = TransformerBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=0.0, causal=True,
        ).cuda()
        blk_amp.eval()
        x_amp = torch.randn(2, 14, d_model, device="cuda", dtype=torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y_amp = blk_amp(x_amp)
        assert y_amp.dtype in (torch.bfloat16, torch.float32), (
            f"unexpected AMP output dtype: {y_amp.dtype}"
        )
        assert torch.isfinite(y_amp).all(), (
            "bf16 Transformer produced non-finite"
        )

    for bad_input, exc_type in [
        (torch.zeros(14, d_model), ValueError),
        (torch.zeros(1, 14, d_model // 2), ValueError),
        (torch.zeros(1, 14, d_model, dtype=torch.long), TypeError),
    ]:
        try:
            blk(bad_input)
            raise AssertionError(f"expected {exc_type} for {bad_input.shape}")
        except exc_type:
            pass

    try:
        TransformerStack(n_layers=2, d_model=d_model, n_heads=3, d_ff=d_ff)
        raise AssertionError("expected ValueError for d_model % n_heads")
    except ValueError:
        pass

    n_params = sum(p.numel() for p in stack.parameters())
    print(
        f"[transformer-sanity] PASS | "
        f"d_model={d_model} | n_heads={n_heads} | d_ff={d_ff} | "
        f"stack_params={n_params:,} | "
        f"drop_path_schedule={actual} | "
        f"cuda_amp={'yes' if torch.cuda.is_available() else 'no'}"
    )

class BiLSTMBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise ValueError(
                f"d_model must be an int, got {type(d_model).__name__}"
                f"({d_model!r})"
            )
        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}"
            )
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model={d_model} must be even for a bidirectional "
                f"LSTM with 2*(d_model//2) output."
            )
        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise ValueError(
                f"num_layers must be an int, got "
                f"{type(num_layers).__name__}({num_layers!r}); "
                f"floats are rejected to avoid silent truncation."
            )
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}"
            )
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}"
            )

        self.d_model = int(d_model)
        self.hidden_size = int(d_model) // 2
        self.num_layers = int(num_layers)
        self.use_residual = bool(use_residual)

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=float(dropout))

        if _LOGGER is not None:
            n_lstm = sum(p.numel() for p in self.lstm.parameters())
            _LOGGER.debug(
                "BiLSTMBlock: d_model=%d | hidden=%d | num_layers=%d | "
                "bidirectional=True | use_residual=%s | dropout=%.3f | "
                "lstm_params=%d | output_dim=%d",
                self.d_model, self.hidden_size, self.num_layers,
                self.use_residual, float(dropout),
                n_lstm, 2 * self.hidden_size,
            )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(
                f"x must be floating-point, got {x.dtype}"
            )
        if x.dim() != 3:
            raise ValueError(
                f"x must be 3-D (B, T, d_model), got {x.dim()}-D "
                f"with shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"x.shape[-1]={x.shape[-1]} does not match "
                f"d_model={self.d_model}"
            )

        if hx is not None:
            if not (isinstance(hx, tuple) and len(hx) == 2):
                raise ValueError(
                    f"hx must be a tuple (h_0, c_0), got "
                    f"{type(hx).__name__}"
                )
            h, c = hx
            expect = (2 * self.num_layers, x.shape[0], self.hidden_size)
            if tuple(h.shape) != expect or tuple(c.shape) != expect:
                raise ValueError(
                    f"hx shapes must each be {expect}, got "
                    f"h={tuple(h.shape)}, c={tuple(c.shape)}"
                )

        lstm_out, _ = self.lstm(x, hx)

        if self.use_residual:
            out = self.norm(x + lstm_out)
        else:
            out = self.norm(lstm_out)
        return self.dropout(out)

def _sanity_check_bilstm() -> None:
    d_model = 128

    blk = BiLSTMBlock(
        d_model=d_model, num_layers=1, dropout=0.0, use_residual=True,
    )
    blk.eval()

    for x in (
        torch.rand(4, 14, d_model),
        torch.randn(4, 14, d_model),
        torch.zeros(2, 14, d_model),
        torch.ones(2, 14, d_model),
    ):
        y = blk(x)
        assert y.shape == x.shape, (
            f"shape mismatch for input {tuple(x.shape)}: "
            f"output {tuple(y.shape)}"
        )
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all(), (
            f"non-finite output for input distribution "
            f"min={x.min().item():.4f}, max={x.max().item():.4f}"
        )

    xa = torch.randn(1, 14, d_model)
    xb = xa.clone()
    xb[:, 7, :] += 1.0

    ya = blk(xa)
    yb = blk(xb)

    assert not torch.allclose(ya[:, 0, :], yb[:, 0, :], atol=1e-6)
    assert not torch.allclose(ya[:, 13, :], yb[:, 13, :], atol=1e-6)
    assert not torch.allclose(ya[:, 7, :], yb[:, 7, :], atol=1e-6)

    blk_g = BiLSTMBlock(
        d_model=d_model, num_layers=1, dropout=0.0, use_residual=True,
    )
    blk_g.train()
    x_g = torch.randn(2, 14, d_model, requires_grad=True)
    y_g = blk_g(x_g)
    y_g.sum().backward()
    assert x_g.grad is not None
    for name, p in blk_g.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    blk_nr = BiLSTMBlock(
        d_model=d_model, num_layers=1, dropout=0.0, use_residual=False,
    )
    blk_nr.eval()
    y_nr = blk_nr(torch.randn(2, 14, d_model))
    assert y_nr.shape == (2, 14, d_model)
    assert torch.isfinite(y_nr).all()

    blk_l2 = BiLSTMBlock(
        d_model=d_model, num_layers=2, dropout=0.1, use_residual=True,
    )
    blk_l2.eval()
    y_l2 = blk_l2(torch.randn(2, 14, d_model))
    assert y_l2.shape == (2, 14, d_model)
    assert torch.isfinite(y_l2).all()

    blk_d = BiLSTMBlock(
        d_model=64, num_layers=1, dropout=0.5, use_residual=True,
    )
    x_d = torch.randn(4, 14, 64)
    blk_d.train()
    y_train = blk_d(x_d)
    blk_d.eval()
    y_eval = blk_d(x_d)
    assert not torch.equal(y_train, y_eval)

    B, T = 2, 14
    n_dir = 2
    hx_ok = (
        torch.zeros(n_dir * blk.num_layers, B, blk.hidden_size),
        torch.zeros(n_dir * blk.num_layers, B, blk.hidden_size),
    )
    y_hx = blk(torch.randn(B, T, d_model), hx=hx_ok)
    assert y_hx.shape == (B, T, d_model)

    hx_bad = (
        torch.zeros(n_dir * blk.num_layers, B, blk.hidden_size + 1),
        torch.zeros(n_dir * blk.num_layers, B, blk.hidden_size),
    )
    try:
        blk(torch.randn(B, T, d_model), hx=hx_bad)
        raise AssertionError("expected ValueError for bad hx shape")
    except ValueError:
        pass

    try:
        BiLSTMBlock(d_model=127)
        raise AssertionError("expected ValueError for odd d_model")
    except ValueError:
        pass
    try:
        BiLSTMBlock(d_model=True)
        raise AssertionError("expected ValueError for d_model=True")
    except ValueError:
        pass
    try:
        BiLSTMBlock(d_model=d_model, num_layers=1.5)
        raise AssertionError("expected ValueError for float num_layers")
    except ValueError:
        pass
    try:
        BiLSTMBlock(d_model=d_model, num_layers=0)
        raise AssertionError("expected ValueError for num_layers=0")
    except ValueError:
        pass

    blk_guard = BiLSTMBlock(d_model=d_model, dropout=0.0)
    blk_guard.eval()
    for bad_input, exc_type in [
        (torch.zeros(14, d_model), ValueError),
        (torch.zeros(1, 14, d_model // 2), ValueError),
        (torch.zeros(1, 14, d_model, dtype=torch.long), TypeError),
    ]:
        try:
            blk_guard(bad_input)
            raise AssertionError(
                f"expected {exc_type} for {bad_input.shape}"
            )
        except exc_type:
            pass

    if torch.cuda.is_available():
        blk_amp = BiLSTMBlock(
            d_model=d_model, num_layers=1, dropout=0.0, use_residual=True,
        ).cuda()
        blk_amp.eval()
        x_amp = torch.randn(2, 14, d_model, device="cuda", dtype=torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y_amp = blk_amp(x_amp)
        assert y_amp.dtype in (torch.bfloat16, torch.float32), (
            f"unexpected AMP output dtype: {y_amp.dtype}"
        )
        assert torch.isfinite(y_amp).all(), "bf16 BiLSTM produced non-finite"

    n_lstm = sum(p.numel() for p in blk.lstm.parameters())
    assert 90000 <= n_lstm <= 110000, (
        f"unexpected LSTM param count: {n_lstm} "
        f"(expected ~99,328 for d_model=128, num_layers=1)"
    )

    print(
        f"[bilstm-sanity] PASS | d_model={d_model} | "
        f"hidden={blk.hidden_size} | num_layers={blk.num_layers} | "
        f"lstm_params={n_lstm:,} | "
        f"cuda_amp={'yes' if torch.cuda.is_available() else 'no'}"
    )

class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(
                f"in_channels must be a positive int, got {in_channels!r}"
            )
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError(
                f"out_channels must be a positive int, got {out_channels!r}"
            )
        if not isinstance(kernel_size, int) or kernel_size < 1:
            raise ValueError(
                f"kernel_size must be a positive int, got {kernel_size!r}"
            )
        if not isinstance(dilation, int) or dilation < 1:
            raise ValueError(
                f"dilation must be a positive int, got {dilation!r}"
            )
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1) (1.0 would zero the main "
                f"path), got {dropout}"
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.dropout_p = float(dropout)
        self.use_residual = bool(use_residual)

        self.padding = (self.kernel_size - 1) * self.dilation

        if self.use_residual:
            if self.in_channels != self.out_channels:
                self.residual_conv = nn.Conv1d(
                    self.in_channels, self.out_channels, kernel_size=1,
                )
            else:
                self.residual_conv = nn.Identity()
        else:
            self.residual_conv = None

        self.conv1 = weight_norm(nn.Conv1d(
            self.in_channels, self.out_channels,
            kernel_size=self.kernel_size, dilation=self.dilation,
            padding=0,
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            self.out_channels, self.out_channels,
            kernel_size=self.kernel_size, dilation=self.dilation,
            padding=0,
        ))
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.relu = nn.ReLU()

        if _LOGGER is not None:
            _LOGGER.debug(
                "TCNBlock: in=%d -> out=%d | K=%d | dilation=%d | "
                "padding=%d | dropout=%.3f | use_residual=%s",
                self.in_channels, self.out_channels,
                self.kernel_size, self.dilation, self.padding,
                self.dropout_p, self.use_residual,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(
                f"x must be floating-point, got {x.dtype}"
            )
        if x.dim() != 3:
            raise ValueError(
                f"x must be 3-D (B, T, C), got {x.dim()}-D with shape "
                f"{tuple(x.shape)}"
            )
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"x.shape[-1]={x.shape[-1]} does not match "
                f"in_channels={self.in_channels}"
            )

        y = x.transpose(1, 2)

        res = self.residual_conv(y) if self.use_residual else None

        y_padded = F.pad(y, (self.padding, 0))
        y = self.conv1(y_padded)
        y = self.dropout1(y)
        y = self.relu(y)

        y_padded = F.pad(y, (self.padding, 0))
        y = self.conv2(y_padded)
        y = self.dropout2(y)
        y = self.relu(y)

        if self.use_residual:
            y = y + res

        return y.transpose(1, 2)

class TCNStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
        dilations: Optional[Sequence[int]] = None,
        use_final_norm: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(
                f"in_channels must be a positive int, got {in_channels!r}"
            )
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError(
                f"out_channels must be a positive int, got {out_channels!r}"
            )
        if not isinstance(n_layers, int) or n_layers <= 0:
            raise ValueError(
                f"n_layers must be a positive int, got {n_layers!r}"
            )
        if not isinstance(kernel_size, int) or kernel_size < 1:
            raise ValueError(
                f"kernel_size must be a positive int, got {kernel_size!r}"
            )
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        if dilations is not None:
            if not isinstance(dilations, (list, tuple)) or len(dilations) == 0:
                raise ValueError(
                    f"dilations must be a non-empty list/tuple, got {dilations!r}"
                )
            for d in dilations:
                if not isinstance(d, int) or d < 1:
                    raise ValueError(
                        f"each dilation must be a positive int, got {d!r}"
                    )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_layers = int(n_layers)
        self.kernel_size = int(kernel_size)
        self.dropout_p = float(dropout)
        self.use_residual = bool(use_residual)
        self.use_final_norm = bool(use_final_norm)

        if dilations is not None:
            self.dilations = [int(d) for d in dilations]
            self.n_layers = len(self.dilations)
            if _LOGGER is not None:
                _LOGGER.info(
                    "TCNStack: using custom dilations %s (overriding n_layers=%d)",
                    self.dilations, n_layers,
                )
        else:
            if self.n_layers > 8:
                warnings.warn(
                    f"n_layers={self.n_layers} yields dilation up to "
                    f"{2 ** (self.n_layers - 1)}; check memory.",
                    RuntimeWarning,
                )
            self.dilations = [2 ** i for i in range(self.n_layers)]

        blocks = []
        for i, d in enumerate(self.dilations):
            block_in = self.in_channels if i == 0 else self.out_channels
            blocks.append(TCNBlock(
                in_channels=block_in,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=d,
                dropout=self.dropout_p,
                use_residual=self.use_residual,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.norm = (
            nn.LayerNorm(self.out_channels) if self.use_final_norm else None
        )

        self.receptive_field = 1 + (self.kernel_size - 1) * sum(self.dilations)
        self.effective_receptive_field = (
            1 + 2 * (self.kernel_size - 1) * sum(self.dilations)
        )

        if _LOGGER is not None:
            _LOGGER.info(
                "TCNStack: in=%d -> out=%d | n_layers=%d | K=%d | "
                "dilations=%s | RF_eq8=%d | RF_effective_2conv=%d | "
                "dropout=%.3f | use_residual=%s | use_final_norm=%s",
                self.in_channels, self.out_channels, self.n_layers,
                self.kernel_size, self.dilations, self.receptive_field,
                self.effective_receptive_field, self.dropout_p,
                self.use_residual, self.use_final_norm,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(f"x must be floating-point, got {x.dtype}")
        if x.dim() != 3:
            raise ValueError(
                f"x must be 3-D (B, T, in_channels), got {x.dim()}-D "
                f"with shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"x.shape[-1]={x.shape[-1]} does not match "
                f"in_channels={self.in_channels}"
            )

        for block in self.blocks:
            x = block(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

def _sanity_check_tcn() -> None:
    d_model = 128

    blk = TCNBlock(
        in_channels=d_model, out_channels=d_model,
        kernel_size=3, dilation=1, dropout=0.0,
    )
    blk.eval()

    for x in (
        torch.rand(4, 14, d_model),
        torch.randn(4, 14, d_model),
        torch.zeros(2, 14, d_model),
        torch.ones(2, 14, d_model),
    ):
        y = blk(x)
        assert y.shape == x.shape, (
            f"block shape mismatch: input {tuple(x.shape)}, "
            f"output {tuple(y.shape)}"
        )
        assert y.dtype == x.dtype
        assert torch.isfinite(y).all()

    xa = torch.randn(1, 14, d_model)
    xb = xa.clone()
    xb[:, 7, :] += 1.0
    ya, yb = blk(xa), blk(xb)
    assert torch.allclose(ya[:, :7, :], yb[:, :7, :], atol=1e-6), (
        "block causal mask violated: positions 0-6 changed when "
        "position 7 was perturbed"
    )
    assert not torch.allclose(ya[:, 7, :], yb[:, 7, :], atol=1e-6), (
        "block: position 7 did not change when itself was perturbed"
    )

    stack = TCNStack(
        in_channels=d_model, out_channels=d_model,
        n_layers=3, kernel_size=3, dropout=0.0,
    )
    stack.eval()

    assert stack.dilations == [1, 2, 4], (
        f"default dilation schedule should be [1,2,4], "
        f"got {stack.dilations}"
    )
    assert stack.receptive_field == 15, (
        f"default RF should be 15 (K=3, dilations=[1,2,4]), "
        f"got {stack.receptive_field}"
    )

    xa = torch.randn(1, 14, d_model)
    xb = xa.clone()
    xb[:, 10, :] += 1.0
    ya, yb = stack(xa), stack(xb)
    assert torch.allclose(ya[:, :10, :], yb[:, :10, :], atol=1e-6), (
        "stack causal mask violated: positions < 10 changed when "
        "position 10 was perturbed"
    )
    assert not torch.allclose(ya[:, 10, :], yb[:, 10, :], atol=1e-6), (
        "stack: position 10 did not change when itself was perturbed"
    )

    xa = torch.randn(1, 14, d_model)
    xb = xa.clone()
    xb[:, 0, :] += 10.0
    ya, yb = stack(xa), stack(xb)
    for t in range(14):
        assert not torch.allclose(ya[:, t, :], yb[:, t, :], atol=1e-4), (
            f"receptive-field test failed at position {t}: perturbing "
            f"position 0 should propagate everywhere with RF=15 > T=14"
        )

    stack.train()
    x_g = torch.randn(2, 14, d_model, requires_grad=True)
    y_g = stack(x_g)
    y_g.sum().backward()
    assert x_g.grad is not None, "no gradient on input"
    for name, p in stack.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    blk_nr = TCNBlock(
        in_channels=d_model, out_channels=d_model,
        kernel_size=3, dilation=1, dropout=0.0, use_residual=False,
    )
    blk_nr.eval()
    y_nr = blk_nr(torch.randn(2, 14, d_model))
    assert y_nr.shape == (2, 14, d_model)
    assert torch.isfinite(y_nr).all()

    stack_nr = TCNStack(
        in_channels=d_model, out_channels=d_model,
        n_layers=3, dropout=0.0, use_residual=False,
    )
    stack_nr.eval()
    y_stack_nr = stack_nr(torch.randn(2, 14, d_model))
    assert y_stack_nr.shape == (2, 14, d_model)
    assert torch.isfinite(y_stack_nr).all()

    stack_custom = TCNStack(
        in_channels=d_model, out_channels=d_model,
        dilations=[1, 1, 1], kernel_size=3, dropout=0.0,
    )
    assert stack_custom.dilations == [1, 1, 1]
    assert stack_custom.receptive_field == 7, (
        f"custom dilations=[1,1,1] should yield RF=7, "
        f"got {stack_custom.receptive_field}"
    )
    assert stack_custom(torch.randn(2, 14, d_model)).shape == (
        2, 14, d_model,
    )

    stack_nonorm = TCNStack(
        in_channels=d_model, out_channels=d_model,
        n_layers=3, dropout=0.0, use_final_norm=False,
    )
    assert stack_nonorm.norm is None, (
        "use_final_norm=False should set norm to None"
    )
    y_nonorm = stack_nonorm(torch.randn(2, 14, d_model))
    assert y_nonorm.shape == (2, 14, d_model)
    assert torch.isfinite(y_nonorm).all()

    for bad_input, exc_type in [
        (torch.zeros(14, d_model), ValueError),
        (torch.zeros(1, 14, d_model // 2), ValueError),
        (torch.zeros(1, 14, d_model, dtype=torch.long), TypeError),
    ]:
        try:
            blk(bad_input)
            raise AssertionError(
                f"expected {exc_type} for {bad_input.shape}"
            )
        except exc_type:
            pass

    for bad_input, exc_type in [
        (torch.zeros(14, d_model), ValueError),
        (torch.zeros(1, 14, d_model // 2), ValueError),
        (torch.zeros(1, 14, d_model, dtype=torch.long), TypeError),
    ]:
        try:
            stack(bad_input)
            raise AssertionError(
                f"expected {exc_type} at stack for {bad_input.shape}"
            )
        except exc_type:
            pass

    for bad_kwargs, exc_type in [
        (dict(in_channels=128, out_channels=128, kernel_size=0),
         ValueError),
        (dict(in_channels=128, out_channels=128, dilation=0),
         ValueError),
        (dict(in_channels=0, out_channels=128), ValueError),
        (dict(in_channels=128, out_channels=0), ValueError),
        (dict(in_channels=128, out_channels=128, dropout=1.0),
         ValueError),
        (dict(in_channels=128, out_channels=128, dropout=-0.1),
         ValueError),
    ]:
        try:
            TCNBlock(**bad_kwargs)
            raise AssertionError(
                f"expected {exc_type} for TCNBlock({bad_kwargs})"
            )
        except exc_type:
            pass

    for bad_dilations in ([], [1, 0, 2], [1, 2, -1], [0], [-3]):
        try:
            TCNStack(
                in_channels=128, out_channels=128,
                dilations=bad_dilations,
            )
            raise AssertionError(
                f"expected ValueError for TCNStack(dilations={bad_dilations})"
            )
        except ValueError:
            pass

    for bad_type in (5, "1,2,4", {1: 1, 2: 2}):
        try:
            TCNStack(
                in_channels=128, out_channels=128,
                dilations=bad_type,
            )
            raise AssertionError(
                f"expected ValueError for non-sequence dilations={bad_type!r}"
            )
        except ValueError:
            pass

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        TCNStack(
            in_channels=d_model, out_channels=d_model,
            n_layers=10, dropout=0.0,
        )
    assert any(
        "n_layers" in str(w.message) and issubclass(w.category, RuntimeWarning)
        for w in captured
    ), (
        "expected a RuntimeWarning when n_layers > 8 (auto schedule); "
        f"captured={[str(w.message) for w in captured]}"
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        TCNStack(
            in_channels=d_model, out_channels=d_model,
            dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            dropout=0.0,
        )
    assert not any(
        "n_layers" in str(w.message) for w in captured
    ), (
        "custom dilations should bypass the n_layers>8 warning"
    )

    if torch.cuda.is_available():
        blk_amp = TCNBlock(
            in_channels=d_model, out_channels=d_model,
            kernel_size=3, dilation=1, dropout=0.0,
        ).cuda()
        blk_amp.eval()
        x_amp = torch.randn(
            2, 14, d_model, device="cuda", dtype=torch.bfloat16,
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y_amp = blk_amp(x_amp)
        assert y_amp.dtype in (torch.bfloat16, torch.float32), (
            f"unexpected AMP output dtype: {y_amp.dtype}"
        )
        assert torch.isfinite(y_amp).all(), (
            "bf16 TCN produced non-finite output"
        )

    n_params = sum(p.numel() for p in stack.parameters())
    assert 280_000 <= n_params <= 310_000, (
        f"unexpected TCN param count: {n_params:,} "
        f"(expected ~296,704 for the default 3-layer stack)"
    )

    print(
        f"[tcn-sanity] PASS | d_model={d_model} | "
        f"dilations={stack.dilations} | RF={stack.receptive_field} | "
        f"params={n_params:,} | "
        f"cuda_amp={'yes' if torch.cuda.is_available() else 'no'}"
    )

class AdaptiveGatingFusion(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        gate_mode: str = "per_sample",
        bias_init: float = 0.0,
        weight_init_std: float = 0.01,
        dropout: float = 0.0,
        use_entropy_reg: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive int, got {d_model!r}")
        if gate_mode not in {"per_sample", "per_token"}:
            raise ValueError(
                f"gate_mode must be 'per_sample' or 'per_token', got {gate_mode!r}"
            )
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not (float(weight_init_std) > 0.0):
            raise ValueError(
                f"weight_init_std must be positive, got {weight_init_std}"
            )
        if not (float(eps) > 0.0):
            raise ValueError(f"eps must be positive, got {eps}")

        self.d_model = int(d_model)
        self.gate_mode = str(gate_mode)
        self.bias_init = float(bias_init)
        self.weight_init_std = float(weight_init_std)
        self.use_entropy_reg = bool(use_entropy_reg)
        self.eps = float(eps)

        self.gate_proj = nn.Linear(2 * self.d_model, self.d_model, bias=True)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=self.weight_init_std)
        nn.init.constant_(self.gate_proj.bias, self.bias_init)
        self.dropout = nn.Dropout(p=float(dropout))

        if _LOGGER is not None:
            _LOGGER.debug(
                "AdaptiveGatingFusion: d_model=%d | gate_mode=%s | "
                "bias_init=%.3f | weight_init_std=%.4f | dropout=%.3f | "
                "use_entropy_reg=%s | gate_params=%d",
                self.d_model, self.gate_mode, self.bias_init,
                self.weight_init_std, float(dropout), self.use_entropy_reg,
                sum(p.numel() for p in self.gate_proj.parameters()),
            )

    def forward(
        self,
        h_temporal: torch.Tensor,
        h_spectral: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not h_temporal.is_floating_point():
            raise TypeError(
                f"h_temporal must be floating-point, got {h_temporal.dtype}"
            )
        if not h_spectral.is_floating_point():
            raise TypeError(
                f"h_spectral must be floating-point, got {h_spectral.dtype}"
            )
        if h_temporal.dtype != h_spectral.dtype:
            raise TypeError(
                f"h_temporal and h_spectral must have the same dtype, "
                f"got {h_temporal.dtype} and {h_spectral.dtype}"
            )
        if h_temporal.dim() != 3 or h_spectral.dim() != 3:
            raise ValueError("h_temporal and h_spectral must be 3-D (B, T, d_model)")
        if h_temporal.shape != h_spectral.shape:
            raise ValueError(
                f"Shape mismatch between branches: "
                f"{tuple(h_temporal.shape)} vs {tuple(h_spectral.shape)}"
            )
        if h_temporal.shape[-1] != self.d_model:
            raise ValueError(
                f"h_temporal.shape[-1]={h_temporal.shape[-1]} does not "
                f"match d_model={self.d_model}"
            )
        if h_temporal.shape[1] == 0:
            raise ValueError("zero-length time axis")

        cat = torch.cat([h_temporal, h_spectral], dim=-1)

        if self.gate_mode == "per_sample":
            pooled = cat.mean(dim=1, keepdim=True)
            g = torch.sigmoid(self.gate_proj(pooled))
        else:
            g = torch.sigmoid(self.gate_proj(cat))

        h_fused = g * h_temporal + (1.0 - g) * h_spectral

        return self.dropout(h_fused), g

    def gate_weight_l2_sq(self) -> torch.Tensor:
        return (self.gate_proj.weight ** 2).sum()

    def gate_entropy(self, g: torch.Tensor) -> torch.Tensor:
        g_c = g.clamp(self.eps, 1.0 - self.eps)
        return (
            g_c * torch.log(g_c)
            + (1.0 - g_c) * torch.log(1.0 - g_c)
        ).mean()

    def gate_saturation_kl(self, g: torch.Tensor) -> torch.Tensor:
        return math.log(2.0) + self.gate_entropy(g)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, gate_mode={self.gate_mode!r}, "
            f"bias_init={self.bias_init:.3f}, "
            f"weight_init_std={self.weight_init_std:.4f}, "
            f"use_entropy_reg={self.use_entropy_reg}"
        )

def _sanity_check_agf() -> None:
    d_model = 128
    B, T = 4, 14
    log2 = math.log(2.0)

    for bad_kwargs, exc_type in [
        (dict(d_model=0), ValueError),
        (dict(d_model=-1), ValueError),
        (dict(d_model=128, gate_mode="foo"), ValueError),
        (dict(d_model=128, gate_mode="per-token"), ValueError),
        (dict(d_model=128, dropout=1.0), ValueError),
        (dict(d_model=128, dropout=-0.1), ValueError),
        (dict(d_model=128, weight_init_std=0.0), ValueError),
        (dict(d_model=128, weight_init_std=-0.01), ValueError),
        (dict(d_model=128, eps=0.0), ValueError),
        (dict(d_model=128, eps=-1e-6), ValueError),
    ]:
        try:
            AdaptiveGatingFusion(**bad_kwargs)
            raise AssertionError(
                f"expected {exc_type.__name__} for AGF({bad_kwargs})"
            )
        except exc_type:
            pass

    agf_ps = AdaptiveGatingFusion(
        d_model=d_model, gate_mode="per_sample",
        bias_init=0.0, weight_init_std=0.01, dropout=0.0,
    )
    agf_ps.eval()

    agf_pt = AdaptiveGatingFusion(
        d_model=d_model, gate_mode="per_token",
        bias_init=0.0, weight_init_std=0.01, dropout=0.0,
    )
    agf_pt.eval()

    assert float(agf_ps.gate_proj.bias.abs().max()) == 0.0
    assert float(agf_pt.gate_proj.bias.abs().max()) == 0.0

    w_std = float(agf_ps.gate_proj.weight.std().item())
    assert 0.005 < w_std < 0.02, (
        f"gate_proj.weight std={w_std:.4f} not in expected range"
    )

    n_gate_params = sum(p.numel() for p in agf_ps.gate_proj.parameters())
    assert n_gate_params == 2 * d_model * d_model + d_model

    torch.manual_seed(0)
    h_t = torch.randn(B, T, d_model)
    h_s = torch.randn(B, T, d_model)

    with torch.no_grad():
        h_fused_ps, g_ps = agf_ps(h_t, h_s)
        h_fused_pt, g_pt = agf_pt(h_t, h_s)

    assert h_fused_ps.shape == (B, T, d_model)
    assert h_fused_pt.shape == (B, T, d_model)
    assert g_ps.shape == (B, 1, d_model), f"got g_ps shape {g_ps.shape}"
    assert g_pt.shape == (B, T, d_model), f"got g_pt shape {g_pt.shape}"

    for g in (g_ps, g_pt):
        assert torch.isfinite(g).all()
        assert (g > 0.0).all(), "gate has value <= 0"
        assert (g < 1.0).all(), "gate has value >= 1"

    g_ps_mean = float(g_ps.mean().item())
    g_pt_mean = float(g_pt.mean().item())
    assert abs(g_ps_mean - 0.5) < 0.1, (
        f"per_sample g mean={g_ps_mean:.4f} not close to 0.5 at init"
    )
    assert abs(g_pt_mean - 0.5) < 0.1, (
        f"per_token g mean={g_pt_mean:.4f} not close to 0.5 at init"
    )

    agf_ps.train()
    agf_pt.train()

    for agf in (agf_ps, agf_pt):
        hx = torch.randn(B, T, d_model, requires_grad=True)
        hy = torch.randn(B, T, d_model, requires_grad=True)
        out, _ = agf(hx, hy)
        out.sum().backward()

        assert hx.grad is not None, "no gradient to h_temporal"
        assert hy.grad is not None, "no gradient to h_spectral"
        assert torch.isfinite(hx.grad).all()
        assert torch.isfinite(hy.grad).all()
        assert agf.gate_proj.weight.grad is not None, "no grad to W_g"
        assert agf.gate_proj.bias.grad is not None, "no grad to b_g"
        assert torch.isfinite(agf.gate_proj.weight.grad).all()
        assert torch.isfinite(agf.gate_proj.bias.grad).all()

    agf_ps.eval()
    agf_pt.eval()

    l2 = agf_ps.gate_weight_l2_sq()
    assert l2.dim() == 0
    assert torch.isfinite(l2)
    assert float(l2.item()) >= 0.0

    ent_ps = float(agf_ps.gate_entropy(g_ps).item())
    ent_pt = float(agf_pt.gate_entropy(g_pt).item())
    for v in (ent_ps, ent_pt):
        assert math.isfinite(v)
        assert -log2 - 1e-5 <= v <= 1e-5, (
            f"gate_entropy={v:.6f} outside [-log 2, 0]"
        )

    g_half = torch.full((2, 4, d_model), 0.5)
    ent_half = float(agf_ps.gate_entropy(g_half).item())
    assert abs(ent_half + log2) < 1e-4, (
        f"gate_entropy(0.5)={ent_half:.6f}, expected -log 2=-{log2:.6f}"
    )

    g_sat = torch.tensor([[[1e-6] * d_model], [[1.0 - 1e-6] * d_model]])
    ent_sat = float(agf_ps.gate_entropy(g_sat).item())
    assert math.isfinite(ent_sat)
    assert ent_sat > ent_half + 1e-3, (
        f"gate_entropy should be larger for saturated g; "
        f"got {ent_sat:.6f} vs {ent_half:.6f}"
    )

    g_grad = torch.tensor(
        [[[0.3] * d_model], [[0.7] * d_model]], requires_grad=True
    )
    ent = agf_ps.gate_entropy(g_grad)
    ent.backward()
    assert g_grad.grad is not None
    assert torch.isfinite(g_grad.grad).all()
    assert (g_grad.grad.abs() > 1e-6).any(), (
        "gate_entropy has zero gradient at non-0.5 values"
    )

    for t in (0, T // 2, T - 1):
        expected = (
            g_ps[:, 0, :] * h_t[:, t, :]
            + (1.0 - g_ps[:, 0, :]) * h_s[:, t, :]
        )
        assert torch.allclose(expected, h_fused_ps[:, t, :], atol=1e-6), (
            f"per_sample fusion formula mismatch at t={t}"
        )

    assert not torch.allclose(g_pt[0, 0, :], g_pt[0, 1, :], atol=1e-6), (
        "per_token gate is identical across time"
    )

    with torch.no_grad():
        h_fused_ps_2, _ = agf_ps(h_t, h_s)
    assert torch.equal(h_fused_ps, h_fused_ps_2)

    agf_pt_c = AdaptiveGatingFusion(
        d_model=d_model, gate_mode="per_token", weight_init_std=0.1
    ).eval()
    agf_ps_c = AdaptiveGatingFusion(
        d_model=d_model, gate_mode="per_sample", weight_init_std=0.1
    ).eval()

    t_perturb = 5
    h_t_pert = h_t.clone()
    h_t_pert[:, t_perturb, :] += 5.0

    with torch.no_grad():
        h_fused_pt_c, _ = agf_pt_c(h_t, h_s)
        h_fused_pt_pert, _ = agf_pt_c(h_t_pert, h_s)
        for t in range(T):
            same = torch.allclose(
                h_fused_pt_c[:, t, :], h_fused_pt_pert[:, t, :], atol=1e-5
            )
            if t == t_perturb:
                assert not same, f"per_token: perturbed position {t} unchanged"
            else:
                assert same, (
                    f"per_token: position {t} changed (should be causal!)"
                )

        h_fused_ps_c, _ = agf_ps_c(h_t, h_s)
        h_fused_ps_pert, _ = agf_ps_c(h_t_pert, h_s)
        n_changed = sum(
            not torch.allclose(
                h_fused_ps_c[:, t, :], h_fused_ps_pert[:, t, :], atol=1e-5
            )
            for t in range(T)
        )
        assert n_changed >= T - 1, (
            f"per_sample should be non-causal; only {n_changed}/{T} changed"
        )

    try:
        agf_ps(torch.randn(B, T, d_model), torch.randn(B, T + 1, d_model))
        raise AssertionError("expected ValueError for shape mismatch")
    except ValueError:
        pass

    try:
        agf_ps(torch.randn(B, T, d_model), torch.randn(B, T, d_model // 2))
        raise AssertionError("expected ValueError for d_model mismatch")
    except ValueError:
        pass

    try:
        agf_ps(torch.randn(T, d_model), torch.randn(T, d_model))
        raise AssertionError("expected ValueError for non-3-D input")
    except ValueError:
        pass

    try:
        agf_ps(
            torch.zeros(B, T, d_model, dtype=torch.long),
            torch.zeros(B, T, d_model, dtype=torch.long),
        )
        raise AssertionError("expected TypeError for integer input")
    except TypeError:
        pass

    try:
        agf_ps(
            torch.randn(B, T, d_model, dtype=torch.float32),
            torch.randn(B, T, d_model, dtype=torch.float64),
        )
        raise AssertionError("expected TypeError for dtype mismatch")
    except TypeError:
        pass

    print(
        f"[agf-sanity] PASS | d_model={d_model} | "
        f"gate_params={n_gate_params:,} | "
        f"per_sample_g≈{g_ps_mean:.3f} | per_token_g≈{g_pt_mean:.3f} | "
        f"entropy(0.5)={ent_half:.4f} | entropy(sat)={ent_sat:.4f}"
    )

class DBTRAGF(nn.Module):
    _FUSION_MODES = {
        "agf", "concat", "fixed", "temporal_only", "spectral_only"
    }
    _POOLING_MODES = {"mean", "last", "max"}

    def __init__(
        self,
        time_in_features: int = 576,
        freq_in_features: int = 8964,
        n_steps: int = 14,
        d_model: int = 128,
        tf_temporal_layers: int = 2,
        tf_temporal_heads: int = 4,
        tf_temporal_dff: int = 512,
        bilstm_layers: int = 1,
        tf_spectral_layers: int = 2,
        tf_spectral_heads: int = 4,
        tf_spectral_dff: int = 512,
        tcn_layers: int = 3,
        tcn_kernel: int = 3,
        tcn_dilations: Optional[Sequence[int]] = None,
        agf_gate_mode: str = "per_sample",
        agf_bias_init: float = 0.0,
        agf_weight_init_std: float = 0.01,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        fusion_mode: str = "agf",
        classifier_hidden: Optional[int] = None,
        pooling: str = "mean",
        n_classes: int = 2,
    ) -> None:
        super().__init__()

        if fusion_mode not in self._FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {sorted(self._FUSION_MODES)}, "
                f"got {fusion_mode!r}"
            )
        if pooling not in self._POOLING_MODES:
            raise ValueError(
                f"pooling must be one of {sorted(self._POOLING_MODES)}, "
                f"got {pooling!r}"
            )
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps!r}")
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes!r}")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not (0.0 <= float(drop_path_rate) < 1.0):
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {drop_path_rate}"
            )
        if classifier_hidden is not None:
            if not isinstance(classifier_hidden, int) or classifier_hidden <= 0:
                raise ValueError(
                    f"classifier_hidden must be a positive int or None, "
                    f"got {classifier_hidden!r}"
                )

        self.time_in_features = int(time_in_features)
        self.freq_in_features = int(freq_in_features)
        self.n_steps = int(n_steps)
        self.d_model = int(d_model)

        self.tf_temporal_layers = int(tf_temporal_layers)
        self.tf_temporal_heads = int(tf_temporal_heads)
        self.tf_temporal_dff = int(tf_temporal_dff)
        self.bilstm_layers = int(bilstm_layers)

        self.tf_spectral_layers = int(tf_spectral_layers)
        self.tf_spectral_heads = int(tf_spectral_heads)
        self.tf_spectral_dff = int(tf_spectral_dff)
        self.tcn_layers = int(tcn_layers)
        self.tcn_kernel = int(tcn_kernel)
        self.tcn_dilations = (
            None if tcn_dilations is None else [int(d) for d in tcn_dilations]
        )

        self.agf_gate_mode = str(agf_gate_mode)
        self.agf_bias_init = float(agf_bias_init)
        self.agf_weight_init_std = float(agf_weight_init_std)

        self.dropout_p = float(dropout)
        self.drop_path_rate = float(drop_path_rate)
        self.fusion_mode = str(fusion_mode)
        self.classifier_hidden = (
            None if classifier_hidden is None else int(classifier_hidden)
        )
        self.pooling = str(pooling)
        self.n_classes = int(n_classes)

        self.dropout = self.dropout_p

        if fusion_mode != "spectral_only":
            self.temporal_encoder = EnergyPreservingChannelEncoder(
                in_features=self.time_in_features,
                d_model=self.d_model,
                dropout=self.dropout_p,
            )
            self.temporal_pe = PositionalEncoding(
                d_model=self.d_model,
                max_len=self.n_steps,
                dropout=self.dropout_p,
                learnable=False,
            )
            self.temporal_transformer = TransformerStack(
                n_layers=self.tf_temporal_layers,
                d_model=self.d_model,
                n_heads=self.tf_temporal_heads,
                d_ff=self.tf_temporal_dff,
                dropout=self.dropout_p,
                drop_path_rate=self.drop_path_rate,
                causal=False,
                activation="gelu",
            )
            self.temporal_bilstm = BiLSTMBlock(
                d_model=self.d_model,
                num_layers=self.bilstm_layers,
                dropout=self.dropout_p,
                use_residual=True,
            )
        else:
            self.temporal_encoder = None
            self.temporal_pe = None
            self.temporal_transformer = None
            self.temporal_bilstm = None

        if fusion_mode != "temporal_only":
            self.spectral_encoder = EnergyPreservingChannelEncoder(
                in_features=self.freq_in_features,
                d_model=self.d_model,
                dropout=self.dropout_p,
            )
            self.spectral_pe = PositionalEncoding(
                d_model=self.d_model,
                max_len=self.n_steps,
                dropout=self.dropout_p,
                learnable=False,
            )
            self.spectral_transformer = TransformerStack(
                n_layers=self.tf_spectral_layers,
                d_model=self.d_model,
                n_heads=self.tf_spectral_heads,
                d_ff=self.tf_spectral_dff,
                dropout=self.dropout_p,
                drop_path_rate=self.drop_path_rate,
                causal=True,
                activation="gelu",
            )
            self.spectral_tcn = TCNStack(
                in_channels=self.d_model,
                out_channels=self.d_model,
                n_layers=self.tcn_layers,
                kernel_size=self.tcn_kernel,
                dropout=self.dropout_p,
                use_residual=True,
                dilations=self.tcn_dilations,
                use_final_norm=True,
            )
        else:
            self.spectral_encoder = None
            self.spectral_pe = None
            self.spectral_transformer = None
            self.spectral_tcn = None

        if fusion_mode == "agf":
            self.agf = AdaptiveGatingFusion(
                d_model=self.d_model,
                gate_mode=self.agf_gate_mode,
                bias_init=self.agf_bias_init,
                weight_init_std=self.agf_weight_init_std,
                dropout=0.0,
                use_entropy_reg=True,
            )
            self.fusion_proj = None
        elif fusion_mode == "concat":
            self.agf = None
            self.fusion_proj = nn.Linear(2 * self.d_model, self.d_model)
        else:
            self.agf = None
            self.fusion_proj = None

        if self.classifier_hidden is None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.n_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.classifier_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.classifier_hidden, self.n_classes),
            )

        if _LOGGER is not None:
            _LOGGER.info(
                "DBTRAGF: fusion_mode=%s | d_model=%d | n_steps=%d | "
                "pooling=%s | classifier_hidden=%s | n_classes=%d | "
                "total_params=%d",
                self.fusion_mode, self.d_model, self.n_steps,
                self.pooling, self.classifier_hidden, self.n_classes,
                sum(p.numel() for p in self.parameters()),
            )

    def forward(
        self,
        time: Optional[torch.Tensor] = None,
        freq: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if self.temporal_encoder is not None:
            if time is None:
                raise ValueError(
                    "time input is required when temporal branch is built "
                    f"(fusion_mode={self.fusion_mode!r})."
                )
            if not time.is_floating_point():
                raise TypeError(f"time must be float, got {time.dtype}")
            if time.dim() != 3:
                raise ValueError(
                    f"time must be 3-D (B, n_steps, in_features), got "
                    f"{time.dim()}-D with shape {tuple(time.shape)}"
                )
            if time.shape[1] != self.n_steps:
                raise ValueError(
                    f"time.shape[1]={time.shape[1]} != n_steps={self.n_steps}"
                )

        if self.spectral_encoder is not None:
            if freq is None:
                raise ValueError(
                    "freq input is required when spectral branch is built "
                    f"(fusion_mode={self.fusion_mode!r})."
                )
            if not freq.is_floating_point():
                raise TypeError(f"freq must be float, got {freq.dtype}")
            if freq.dim() != 3:
                raise ValueError(
                    f"freq must be 3-D (B, n_steps, in_features), got "
                    f"{freq.dim()}-D with shape {tuple(freq.shape)}"
                )
            if freq.shape[1] != self.n_steps:
                raise ValueError(
                    f"freq.shape[1]={freq.shape[1]} != n_steps={self.n_steps}"
                )

        h_temporal: Optional[torch.Tensor] = None
        if self.temporal_encoder is not None:
            x_t = self.temporal_encoder(time)
            x_t = self.temporal_pe(x_t)
            x_t = self.temporal_transformer(x_t)
            h_temporal = self.temporal_bilstm(x_t)

        h_spectral: Optional[torch.Tensor] = None
        if self.spectral_encoder is not None:
            x_s = self.spectral_encoder(freq)
            x_s = self.spectral_pe(x_s)
            x_s = self.spectral_transformer(x_s)
            h_spectral = self.spectral_tcn(x_s)

        if h_temporal is not None and h_spectral is not None:
            if h_temporal.shape[0] != h_spectral.shape[0]:
                raise ValueError(
                    f"batch sizes differ: time={h_temporal.shape[0]}, "
                    f"freq={h_spectral.shape[0]}"
                )

        gate: Optional[torch.Tensor] = None
        if self.fusion_mode == "agf":
            if h_temporal is None or h_spectral is None:
                raise RuntimeError(
                    "fusion_mode='agf' requires both temporal and "
                    "spectral branches."
                )
            h_fused, gate = self.agf(h_temporal, h_spectral)
        elif self.fusion_mode == "concat":
            if h_temporal is None or h_spectral is None:
                raise RuntimeError(
                    "fusion_mode='concat' requires both branches."
                )
            h_fused = self.fusion_proj(
                torch.cat([h_temporal, h_spectral], dim=-1)
            )
        elif self.fusion_mode == "fixed":
            if h_temporal is None or h_spectral is None:
                raise RuntimeError(
                    "fusion_mode='fixed' requires both branches."
                )
            h_fused = 0.5 * h_temporal + 0.5 * h_spectral
        elif self.fusion_mode == "temporal_only":
            if h_temporal is None:
                raise RuntimeError(
                    "fusion_mode='temporal_only' requires temporal branch."
                )
            h_fused = h_temporal
        elif self.fusion_mode == "spectral_only":
            if h_spectral is None:
                raise RuntimeError(
                    "fusion_mode='spectral_only' requires spectral branch."
                )
            h_fused = h_spectral
        else:
            raise RuntimeError(
                f"Unknown fusion_mode: {self.fusion_mode!r}"
            )

        if self.pooling == "mean":
            pooled = h_fused.mean(dim=1)
        elif self.pooling == "last":
            pooled = h_fused[:, -1, :]
        elif self.pooling == "max":
            pooled = h_fused.max(dim=1).values
        else:
            raise RuntimeError(f"Unknown pooling: {self.pooling!r}")

        logits = self.classifier(pooled)

        return {
            "logits": logits,
            "h_temporal": h_temporal,
            "h_spectral": h_spectral,
            "h_fused": h_fused,
            "gate": gate,
        }

    def auxiliary_loss(
        self,
        gate: Optional[torch.Tensor],
        entropy_weight: float = 1e-2,
        l2_weight: float = 1e-4,
        non_negative_entropy: bool = True,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        zero = torch.zeros((), device=device)

        if self.fusion_mode != "agf" or gate is None or self.agf is None:
            return zero

        loss_aux = float(l2_weight) * self.agf.gate_weight_l2_sq()

        if self.agf.use_entropy_reg and float(entropy_weight) > 0.0:
            if non_negative_entropy:
                ent_term = self.agf.gate_saturation_kl(gate)
            else:
                ent_term = self.agf.gate_entropy(gate)
            loss_aux = loss_aux + float(entropy_weight) * ent_term

        return loss_aux

    def num_parameters(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}

        if self.temporal_encoder is not None:
            counts["temporal_encoder"] = sum(
                p.numel() for p in self.temporal_encoder.parameters()
            )
            counts["temporal_transformer"] = sum(
                p.numel() for p in self.temporal_transformer.parameters()
            )
            counts["temporal_bilstm"] = sum(
                p.numel() for p in self.temporal_bilstm.parameters()
            )
            counts["temporal_pe"] = 0

        if self.spectral_encoder is not None:
            counts["spectral_encoder"] = sum(
                p.numel() for p in self.spectral_encoder.parameters()
            )
            counts["spectral_transformer"] = sum(
                p.numel() for p in self.spectral_transformer.parameters()
            )
            counts["spectral_tcn"] = sum(
                p.numel() for p in self.spectral_tcn.parameters()
            )
            counts["spectral_pe"] = 0

        if self.agf is not None:
            counts["agf"] = sum(p.numel() for p in self.agf.parameters())
        if self.fusion_proj is not None:
            counts["fusion_proj"] = sum(
                p.numel() for p in self.fusion_proj.parameters()
            )

        counts["classifier"] = sum(
            p.numel() for p in self.classifier.parameters()
        )

        counts["total"] = sum(p.numel() for p in self.parameters())

        return counts

    def get_config(self) -> Dict[str, Any]:
        return {
            "time_in_features": self.time_in_features,
            "freq_in_features": self.freq_in_features,
            "n_steps": self.n_steps,
            "d_model": self.d_model,
            "tf_temporal_layers": self.tf_temporal_layers,
            "tf_temporal_heads": self.tf_temporal_heads,
            "tf_temporal_dff": self.tf_temporal_dff,
            "bilstm_layers": self.bilstm_layers,
            "tf_spectral_layers": self.tf_spectral_layers,
            "tf_spectral_heads": self.tf_spectral_heads,
            "tf_spectral_dff": self.tf_spectral_dff,
            "tcn_layers": self.tcn_layers,
            "tcn_kernel": self.tcn_kernel,
            "tcn_dilations": (
                None if self.tcn_dilations is None
                else [int(d) for d in self.tcn_dilations]
            ),
            "agf_gate_mode": self.agf_gate_mode,
            "agf_bias_init": self.agf_bias_init,
            "agf_weight_init_std": self.agf_weight_init_std,
            "dropout": self.dropout_p,
            "drop_path_rate": self.drop_path_rate,
            "fusion_mode": self.fusion_mode,
            "classifier_hidden": self.classifier_hidden,
            "pooling": self.pooling,
            "n_classes": self.n_classes,
        }

    def extra_repr(self) -> str:
        return (
            f"fusion_mode={self.fusion_mode!r}, "
            f"d_model={self.d_model}, n_steps={self.n_steps}, "
            f"pooling={self.pooling!r}, n_classes={self.n_classes}"
        )

def _sanity_check_dbtr_agf() -> None:
    B = 2
    n_steps = 14
    time_feat = 576
    freq_feat = 8964
    d_model = 64

    for bad_kwargs, exc_type in [
        (dict(fusion_mode="foo"), ValueError),
        (dict(fusion_mode="concat_learned"), ValueError),
        (dict(pooling="median"), ValueError),
        (dict(pooling="first"), ValueError),
        (dict(n_steps=0), ValueError),
        (dict(n_steps=-1), ValueError),
        (dict(n_classes=1), ValueError),
        (dict(n_classes=0), ValueError),
        (dict(dropout=1.0), ValueError),
        (dict(dropout=-0.1), ValueError),
        (dict(drop_path_rate=1.0), ValueError),
        (dict(classifier_hidden=0), ValueError),
        (dict(classifier_hidden=-32), ValueError),
    ]:
        try:
            DBTRAGF(d_model=d_model, **bad_kwargs)
            raise AssertionError(
                f"expected {exc_type.__name__} for DBTRAGF({bad_kwargs})"
            )
        except exc_type:
            pass

    def _small_model(fusion_mode: str = "agf", **overrides) -> DBTRAGF:
        return DBTRAGF(
            time_in_features=time_feat,
            freq_in_features=freq_feat,
            n_steps=n_steps,
            d_model=d_model,
            tf_temporal_layers=1,
            tf_temporal_heads=4,
            tf_temporal_dff=128,
            bilstm_layers=1,
            tf_spectral_layers=1,
            tf_spectral_heads=4,
            tf_spectral_dff=128,
            tcn_layers=2,
            tcn_kernel=3,
            tcn_dilations=[1, 2],
            dropout=0.0,
            drop_path_rate=0.0,
            fusion_mode=fusion_mode,
            **overrides,
        )

    torch.manual_seed(0)
    time = torch.rand(B, n_steps, time_feat)
    freq = torch.rand(B, n_steps, freq_feat)

    for mode in ["agf", "concat", "fixed", "temporal_only", "spectral_only"]:
        model = _small_model(fusion_mode=mode).eval()
        with torch.no_grad():
            out = model(time, freq)

        assert "logits" in out
        assert "h_fused" in out
        assert "h_temporal" in out
        assert "h_spectral" in out
        assert "gate" in out

        assert out["logits"].shape == (B, 2), (
            f"[{mode}] logits shape {out['logits'].shape}"
        )
        assert out["h_fused"].shape == (B, n_steps, d_model), (
            f"[{mode}] h_fused shape {out['h_fused'].shape}"
        )

        for key in ("logits", "h_fused"):
            assert torch.isfinite(out[key]).all(), (
                f"[{mode}] non-finite {key}"
            )

        if mode == "spectral_only":
            assert out["h_temporal"] is None, (
                f"[{mode}] h_temporal should be None"
            )
        else:
            assert out["h_temporal"].shape == (B, n_steps, d_model)
            assert torch.isfinite(out["h_temporal"]).all()

        if mode == "temporal_only":
            assert out["h_spectral"] is None, (
                f"[{mode}] h_spectral should be None"
            )
        else:
            assert out["h_spectral"].shape == (B, n_steps, d_model)
            assert torch.isfinite(out["h_spectral"]).all()

        if mode == "agf":
            assert out["gate"] is not None
            assert out["gate"].shape in [
                (B, 1, d_model), (B, n_steps, d_model),
            ], f"[agf] gate shape {out['gate'].shape}"
            assert torch.isfinite(out["gate"]).all()
            assert (out["gate"] > 0).all()
            assert (out["gate"] < 1).all()
        else:
            assert out["gate"] is None, (
                f"[{mode}] gate should be None"
            )

    model_agf = _small_model(fusion_mode="agf")
    model_agf.train()
    out_agf = model_agf(time, freq)
    aux = model_agf.auxiliary_loss(out_agf["gate"])
    assert aux.dim() == 0, f"aux loss should be scalar, got {aux.dim()}-D"
    assert torch.isfinite(aux), "aux loss must be finite"
    assert abs(float(aux.item())) > 0.0, (
        f"aux loss should be non-zero for agf after forward, "
        f"got {aux.item()}"
    )
    aux.backward(retain_graph=False)
    assert model_agf.agf.gate_proj.weight.grad is not None, (
        "aux loss did not produce gradients on gate_proj.weight"
    )
    assert torch.isfinite(model_agf.agf.gate_proj.weight.grad).all(), (
        "non-finite gradient on gate_proj.weight from aux loss"
    )

    for mode in ["concat", "fixed", "temporal_only", "spectral_only"]:
        m = _small_model(fusion_mode=mode).eval()
        with torch.no_grad():
            o = m(time, freq)
        a = m.auxiliary_loss(o["gate"])
        assert a.dim() == 0
        assert float(a.item()) == 0.0, (
            f"[{mode}] aux loss should be 0, got {a.item()}"
        )

    for mode in ["agf", "concat", "fixed", "temporal_only", "spectral_only"]:
        m = _small_model(fusion_mode=mode)
        counts = m.num_parameters()
        total_actual = sum(p.numel() for p in m.parameters())
        assert counts["total"] == total_actual, (
            f"[{mode}] num_parameters total={counts['total']} != "
            f"sum(p.numel() for p in m.parameters())={total_actual}"
        )
        component_sum = sum(
            v for k, v in counts.items()
            if k not in ("total", "temporal_pe", "spectral_pe")
        )
        assert component_sum == total_actual, (
            f"[{mode}] component sum {component_sum} != "
            f"actual {total_actual}"
        )

    counts_agf = _small_model(fusion_mode="agf").num_parameters()
    assert "agf" in counts_agf
    assert "fusion_proj" not in counts_agf
    assert "temporal_encoder" in counts_agf
    assert "spectral_encoder" in counts_agf

    counts_concat = _small_model(fusion_mode="concat").num_parameters()
    assert "fusion_proj" in counts_concat
    assert "agf" not in counts_concat

    counts_fixed = _small_model(fusion_mode="fixed").num_parameters()
    assert "agf" not in counts_fixed
    assert "fusion_proj" not in counts_fixed

    counts_t = _small_model(fusion_mode="temporal_only").num_parameters()
    assert "temporal_encoder" in counts_t
    assert "spectral_encoder" not in counts_t
    assert "spectral_tcn" not in counts_t
    assert "spectral_transformer" not in counts_t

    counts_s = _small_model(fusion_mode="spectral_only").num_parameters()
    assert "spectral_encoder" in counts_s
    assert "temporal_encoder" not in counts_s
    assert "temporal_bilstm" not in counts_s
    assert "temporal_transformer" not in counts_s

    assert counts_agf["total"] > counts_t["total"], (
        "AGF model should have more parameters than temporal_only"
    )
    assert counts_agf["total"] > counts_s["total"], (
        "AGF model should have more parameters than spectral_only"
    )

    model_grad = _small_model(fusion_mode="agf")
    model_grad.train()
    out_g = model_grad(time, freq)
    loss = out_g["logits"].sum() + model_grad.auxiliary_loss(out_g["gate"])
    loss.backward()

    for name, p in model_grad.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    model_eval = _small_model(fusion_mode="agf").eval()
    with torch.no_grad():
        o1 = model_eval(time, freq)
        o2 = model_eval(time, freq)
    assert torch.equal(o1["logits"], o2["logits"])

    model_mlp = _small_model(
        fusion_mode="agf", classifier_hidden=32,
    ).eval()
    with torch.no_grad():
        o_mlp = model_mlp(time, freq)
    assert o_mlp["logits"].shape == (B, 2)
    assert torch.isfinite(o_mlp["logits"]).all()

    for pool in ["mean", "last", "max"]:
        model_p = _small_model(fusion_mode="agf", pooling=pool).eval()
        with torch.no_grad():
            o_p = model_p(time, freq)
        assert o_p["logits"].shape == (B, 2), (
            f"[pool={pool}] logits shape {o_p['logits'].shape}"
        )
        assert torch.isfinite(o_p["logits"]).all()

    model_guard = _small_model(fusion_mode="agf").eval()

    try:
        model_guard(time[:, :5], freq[:, :5])
        raise AssertionError("expected ValueError for wrong n_steps")
    except ValueError:
        pass

    try:
        model_guard(time, freq[:1])
        raise AssertionError("expected ValueError for batch mismatch")
    except ValueError:
        pass

    try:
        model_guard(time[0], freq)
        raise AssertionError("expected ValueError for non-3-D time")
    except ValueError:
        pass

    try:
        model_guard(time, freq[0])
        raise AssertionError("expected ValueError for non-3-D freq")
    except ValueError:
        pass

    try:
        model_guard(
            torch.zeros(B, n_steps, time_feat, dtype=torch.long),
            freq,
        )
        raise AssertionError("expected TypeError for integer time")
    except TypeError:
        pass

    try:
        model_guard(
            time,
            torch.zeros(B, n_steps, freq_feat, dtype=torch.long),
        )
        raise AssertionError("expected TypeError for integer freq")
    except TypeError:
        pass

    cuda_amp = "no"
    if torch.cuda.is_available():
        cuda_amp = "yes"
        model_amp = _small_model(fusion_mode="agf").cuda().eval()
        time_amp = time.cuda().to(torch.bfloat16)
        freq_amp = freq.cuda().to(torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out_amp = model_amp(time_amp, freq_amp)
        assert out_amp["logits"].dtype in (
            torch.bfloat16, torch.float32,
        ), f"unexpected AMP logits dtype: {out_amp['logits'].dtype}"
        assert torch.isfinite(out_amp["logits"]).all()

    param_summary = {
        mode: _small_model(fusion_mode=mode).num_parameters()["total"]
        for mode in ["agf", "concat", "fixed", "temporal_only", "spectral_only"]
    }
    print(
        f"[dbtr-agf-sanity] PASS | d_model={d_model} | B={B} | "
        f"n_steps={n_steps} | cuda_amp={cuda_amp}"
    )
    print(
        "[dbtr-agf-sanity] small-config total params by mode: "
        + " | ".join(f"{m}={n:,}" for m, n in param_summary.items())
    )

from dataclasses import dataclass, field
import logging
import warnings
from typing import Dict, Any, List, Optional

import numpy as np
from scipy.special import logsumexp

_LOGGER = logging.getLogger(__name__)

def _validate_probability_vector(p: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(p < 0.0):
        raise ValueError(f"{name} contains negative values")
    if not np.isclose(float(p.sum()), 1.0, atol=1e-6):
        raise ValueError(
            f"{name} does not sum to 1 (sum={float(p.sum()):.6f})"
        )

def _safe_stochastic_log(M: np.ndarray, eps: float, name: str) -> np.ndarray:
    M_safe = np.clip(np.asarray(M, dtype=np.float64), eps, None)
    denom = M_safe.sum(axis=-1, keepdims=True)
    M_safe = M_safe / denom
    log_M = np.log(M_safe)
    if not np.all(np.isfinite(log_M)):
        raise RuntimeError(
            f"{name}: log of safe matrix produced non-finite values"
        )
    return log_M

def _gaussian_log_emission(
    logits: np.ndarray,
    means: np.ndarray,
    vars_: np.ndarray,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    means = np.asarray(means, dtype=np.float64).reshape(-1)
    vars_ = np.asarray(vars_, dtype=np.float64).reshape(-1)

    if means.shape[0] != 2 or vars_.shape[0] != 2:
        raise ValueError(
            f"means and vars_ must have length 2, got "
            f"{means.shape[0]} and {vars_.shape[0]}"
        )
    if np.any(vars_ <= 0.0):
        raise ValueError("vars_ must be strictly positive")
    if not np.all(np.isfinite(means)):
        raise ValueError("means must be finite")
    if not np.all(np.isfinite(vars_)):
        raise ValueError("vars_ must be finite")

    diff = logits[:, None] - means[None, :]
    return -0.5 * (
        diff * diff / vars_[None, :]
        + np.log(2.0 * np.pi * vars_[None, :])
    )

@dataclass
class CausalHMMPostprocessor:
    A: np.ndarray
    pi: np.ndarray
    means: np.ndarray
    vars: np.ndarray
    eps: float = 1e-8

    _log_A: np.ndarray = field(init=False, repr=False)
    _log_pi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=np.float64)
        self.pi = np.asarray(self.pi, dtype=np.float64)
        self.means = np.asarray(self.means, dtype=np.float64)
        self.vars = np.asarray(self.vars, dtype=np.float64)

        if self.A.shape != (2, 2):
            raise ValueError(f"A must have shape (2, 2), got {self.A.shape}")
        if self.pi.shape != (2,):
            raise ValueError(f"pi must have shape (2,), got {self.pi.shape}")
        if self.means.shape != (2,):
            raise ValueError(
                f"means must have shape (2,), got {self.means.shape}"
            )
        if self.vars.shape != (2,):
            raise ValueError(
                f"vars must have shape (2,), got {self.vars.shape}"
            )
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if np.any(self.vars <= 0.0):
            raise ValueError("vars must be strictly positive")
        if not np.all(np.isfinite(self.means)):
            raise ValueError("means must be finite")
        if not np.all(np.isfinite(self.vars)):
            raise ValueError("vars must be finite")

        _validate_probability_vector(self.A[0], "A[0]")
        _validate_probability_vector(self.A[1], "A[1]")
        _validate_probability_vector(self.pi, "pi")

        self._log_A = _safe_stochastic_log(self.A, self.eps, "A")
        self._log_pi = _safe_stochastic_log(self.pi, self.eps, "pi")

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        labels: np.ndarray,
        sequence_ids: np.ndarray,
        onsets: np.ndarray,
        *,
        transition_smooth: float = 1.0,
        var_floor: float = 1e-6,
        eps: float = 1e-8,
    ) -> "CausalHMMPostprocessor":
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        sequence_ids = np.asarray(sequence_ids).reshape(-1)
        onsets = np.asarray(onsets, dtype=np.float64).reshape(-1)

        if logits.shape[0] == 0:
            raise ValueError("Cannot fit HMM on empty input")
        if not (
            logits.shape[0] == labels.shape[0]
            == sequence_ids.shape[0] == onsets.shape[0]
        ):
            raise ValueError(
                f"Length mismatch: logits={logits.shape[0]}, "
                f"labels={labels.shape[0]}, "
                f"sequence_ids={sequence_ids.shape[0]}, "
                f"onsets={onsets.shape[0]}"
            )
        if not np.all(np.isin(labels, [0, 1])):
            bad = np.unique(labels[~np.isin(labels, [0, 1])])
            raise ValueError(
                f"labels must be in {{0, 1}}, got {bad.tolist()}"
            )
        if transition_smooth <= 0.0:
            raise ValueError(
                f"transition_smooth must be positive, got "
                f"{transition_smooth}"
            )
        if var_floor <= 0.0:
            raise ValueError(
                f"var_floor must be positive, got {var_floor}"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        A_counts = np.full((2, 2), float(transition_smooth), dtype=np.float64)
        pi_counts = np.full(2, float(transition_smooth), dtype=np.float64)

        for seq_id in np.unique(sequence_ids):
            idx = np.where(sequence_ids == seq_id)[0]
            if idx.shape[0] == 0:
                continue
            order = np.argsort(onsets[idx], kind="stable")
            y = labels[idx[order]]

            pi_counts[int(y[0])] += 1.0
            for t in range(y.shape[0] - 1):
                A_counts[int(y[t]), int(y[t + 1])] += 1.0

        A = A_counts / A_counts.sum(axis=1, keepdims=True)
        pi = pi_counts / pi_counts.sum()

        means = np.zeros(2, dtype=np.float64)
        vars_ = np.ones(2, dtype=np.float64)
        single_sample_states: List[int] = []
        empty_states: List[int] = []

        for s in (0, 1):
            vals = logits[labels == s]
            n_s = int(vals.shape[0])

            if n_s >= 2:
                means[s] = float(np.mean(vals))
                v = float(np.var(vals, ddof=1))
                vars_[s] = max(v, float(var_floor))
            elif n_s == 1:
                means[s] = float(vals[0])
                vars_[s] = 1.0
                single_sample_states.append(s)
            else:
                means[s] = -2.0 if s == 0 else 2.0
                vars_[s] = 1.0
                empty_states.append(s)

        if single_sample_states or empty_states:
            parts = []
            if single_sample_states:
                parts.append(
                    f"states {single_sample_states} have exactly one "
                    f"validation sample: using observed mean and "
                    f"unit variance (prior)"
                )
            if empty_states:
                parts.append(
                    f"states {empty_states} have no validation samples: "
                    f"using fallback means (-2, +2) and unit variance"
                )
            warnings.warn(
                "CausalHMMPostprocessor.fit: "
                + "; ".join(parts)
                + ". If these states are truly absent from validation, "
                "the HMM may be degenerate.",
                RuntimeWarning,
                stacklevel=2,
            )

        return cls(
            A=A, pi=pi, means=means, vars=vars_, eps=float(eps),
        )

    def filter_sequence(
        self,
        logits: np.ndarray,
        *,
        return_log_odds: bool = False,
    ) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        T = logits.shape[0]

        if not np.all(np.isfinite(logits)):
            raise ValueError("logits must be finite")

        if T == 0:
            return np.empty(0, dtype=np.float64)

        log_emit = _gaussian_log_emission(logits, self.means, self.vars)
        log_A = self._log_A
        log_pi = self._log_pi

        log_post = np.empty((T, 2), dtype=np.float64)

        alpha = log_pi + log_emit[0]
        alpha = alpha - logsumexp(alpha)
        log_post[0] = alpha

        for t in range(1, T):
            alpha_new = (
                logsumexp(alpha[:, None] + log_A, axis=0) + log_emit[t]
            )
            alpha_new = alpha_new - logsumexp(alpha_new)
            alpha = alpha_new
            log_post[t] = alpha

        if return_log_odds:
            return log_post[:, 1] - log_post[:, 0]

        return np.exp(log_post[:, 1])

    @staticmethod
    def alarms_from_filter(
        p_pre: np.ndarray,
        onsets: np.ndarray,
        *,
        threshold: float = 0.5,
        refractory_sec: float = 30.0 * 60.0,
        alarm_mode: str = "one_per_episode",
    ) -> np.ndarray:
        p_pre = np.asarray(p_pre, dtype=np.float64).reshape(-1)
        onsets = np.asarray(onsets, dtype=np.float64).reshape(-1)

        if p_pre.shape[0] != onsets.shape[0]:
            raise ValueError(
                f"Length mismatch: p_pre={p_pre.shape[0]}, "
                f"onsets={onsets.shape[0]}"
            )
        if not (0.0 < float(threshold) < 1.0):
            raise ValueError(
                f"threshold must be in (0, 1), got {threshold}"
            )
        if refractory_sec < 0.0:
            raise ValueError(
                f"refractory_sec must be non-negative, got "
                f"{refractory_sec}"
            )
        if alarm_mode not in {"one_per_episode", "periodic_while_above"}:
            raise ValueError(
                f"alarm_mode must be 'one_per_episode' or "
                f"'periodic_while_above', got {alarm_mode!r}"
            )

        if p_pre.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        order = np.argsort(onsets, kind="stable")
        p_sorted = p_pre[order]
        t_sorted = onsets[order]

        alarms: List[float] = []
        last_alarm = -np.inf

        if alarm_mode == "one_per_episode":
            above = False
            for p, t in zip(p_sorted, t_sorted):
                is_above = p >= float(threshold)
                if is_above and not above:
                    if (t - last_alarm) >= float(refractory_sec):
                        alarms.append(float(t))
                        last_alarm = float(t)
                above = is_above

        else:
            for p, t in zip(p_sorted, t_sorted):
                is_above = p >= float(threshold)
                if is_above and (t - last_alarm) >= float(refractory_sec):
                    alarms.append(float(t))
                    last_alarm = float(t)

        return np.asarray(alarms, dtype=np.float64)

    def info(self) -> Dict[str, Any]:
        return {
            "A": self.A.tolist(),
            "pi": self.pi.tolist(),
            "means": self.means.tolist(),
            "vars": self.vars.tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"CausalHMMPostprocessor("
            f"A=[[{self.A[0,0]:.4f},{self.A[0,1]:.4f}],"
            f"[{self.A[1,0]:.4f},{self.A[1,1]:.4f}]], "
            f"means=[{self.means[0]:.3f},{self.means[1]:.3f}], "
            f"vars=[{self.vars[0]:.3f},{self.vars[1]:.3f}])"
        )

def apply_hmm_to_test_set(
    hmm: CausalHMMPostprocessor,
    logits: np.ndarray,
    onsets: np.ndarray,
    sequence_ids: np.ndarray,
    *,
    edf_start_offsets: Optional[Dict[int, float]] = None,
    threshold: float = 0.5,
    refractory_sec: float = 30.0 * 60.0,
    alarm_mode: str = "one_per_episode",
) -> Dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    onsets = np.asarray(onsets, dtype=np.float64).reshape(-1)
    sequence_ids = np.asarray(sequence_ids).reshape(-1)

    if not (
        logits.shape[0] == onsets.shape[0] == sequence_ids.shape[0]
    ):
        raise ValueError(
            f"Length mismatch: logits={logits.shape[0]}, "
            f"onsets={onsets.shape[0]}, "
            f"sequence_ids={sequence_ids.shape[0]}"
        )

    if edf_start_offsets is not None and len(edf_start_offsets) > 0:
        sample_key = next(iter(edf_start_offsets))
        if not isinstance(sample_key, (int, np.integer)):
            raise TypeError(
                f"edf_start_offsets keys must be ints matching "
                f"sequence_ids; got {type(sample_key).__name__} "
                f"(example key: {sample_key!r})"
            )

    if edf_start_offsets is None:
        warnings.warn(
            "apply_hmm_to_test_set: edf_start_offsets not provided; "
            "alarm times are per-EDF relative and MUST NOT be sorted or "
            "compared across EDFs. Use event_level_metrics_per_edf for "
            "correct evaluation. Pass edf_start_offsets to obtain "
            "absolute times.",
            RuntimeWarning,
            stacklevel=2,
        )

    N = logits.shape[0]
    p_pre_all = np.zeros(N, dtype=np.float64)
    alarms_by_edf: Dict[int, np.ndarray] = {}
    alarm_relative_times: List[float] = []
    alarm_seq_ids: List[int] = []

    for seq_id in np.unique(sequence_ids):
        idx = np.where(sequence_ids == seq_id)[0]
        if idx.shape[0] == 0:
            continue

        order = np.argsort(onsets[idx], kind="stable")
        idx_sorted = idx[order]

        seq_logits = logits[idx_sorted]
        seq_onsets = onsets[idx_sorted]

        p_pre = hmm.filter_sequence(seq_logits)
        p_pre_all[idx_sorted] = p_pre

        seq_alarms = hmm.alarms_from_filter(
            p_pre,
            seq_onsets,
            threshold=threshold,
            refractory_sec=refractory_sec,
            alarm_mode=alarm_mode,
        )
        seq_id_int = int(seq_id)
        alarms_by_edf[seq_id_int] = seq_alarms

        for a in seq_alarms.tolist():
            alarm_relative_times.append(float(a))
            alarm_seq_ids.append(seq_id_int)

    alarm_relative_times_arr = np.asarray(
        alarm_relative_times, dtype=np.float64
    )
    alarm_seq_ids_arr = np.asarray(alarm_seq_ids, dtype=np.int64)

    if edf_start_offsets is not None:
        if alarm_seq_ids_arr.size > 0:
            missing = [
                int(sid) for sid in np.unique(alarm_seq_ids_arr).tolist()
                if int(sid) not in edf_start_offsets
            ]
            if missing:
                raise ValueError(
                    f"edf_start_offsets missing entries for EDFs with "
                    f"alarms: {missing}"
                )
            absolute = np.asarray(
                [
                    float(edf_start_offsets[int(sid)]) + float(t)
                    for sid, t in zip(
                        alarm_seq_ids_arr, alarm_relative_times_arr
                    )
                ],
                dtype=np.float64,
            )
        else:
            absolute = np.empty(0, dtype=np.float64)
    else:
        absolute = None

    return {
        "p_pre_all": p_pre_all,
        "alarms_by_edf": alarms_by_edf,
        "alarm_relative_times": alarm_relative_times_arr,
        "alarm_seq_ids": alarm_seq_ids_arr,
        "alarm_absolute_times": absolute,
    }

def event_level_metrics_per_edf(
    seizures_by_edf: Dict[int, np.ndarray],
    alarms_by_edf: Dict[int, np.ndarray],
    edf_durations_hours: Dict[int, float],
    *,
    preictal_horizon_sec: float = 35.0 * 60.0,
) -> Dict[str, Any]:
    if preictal_horizon_sec <= 0.0:
        raise ValueError(
            f"preictal_horizon_sec must be positive, got "
            f"{preictal_horizon_sec}"
        )

    all_edf_ids = (
        set(seizures_by_edf.keys())
        | set(alarms_by_edf.keys())
        | set(edf_durations_hours.keys())
    )

    missing_duration = sorted(
        eid for eid in all_edf_ids
        if eid not in edf_durations_hours
        and (
            len(np.asarray(seizures_by_edf.get(eid, [])).reshape(-1)) > 0
            or len(np.asarray(alarms_by_edf.get(eid, [])).reshape(-1)) > 0
        )
    )
    if missing_duration:
        warnings.warn(
            f"event_level_metrics_per_edf: EDFs {missing_duration} "
            f"have seizures or alarms but no entry in "
            f"edf_durations_hours; they contribute 0 hours to "
            f"total_hours, which inflates FPR. Provide "
            f"edf_durations_hours for all EDFs.",
            RuntimeWarning,
            stacklevel=2,
        )

    total_hours = 0.0
    n_seizures = 0
    n_detected = 0
    n_false_alarms = 0
    n_alarms_total = 0
    lead_times: List[float] = []

    for edf_id in sorted(all_edf_ids):
        sz = np.asarray(
            seizures_by_edf.get(edf_id, np.empty(0)), dtype=np.float64
        ).reshape(-1)
        al = np.asarray(
            alarms_by_edf.get(edf_id, np.empty(0)), dtype=np.float64
        ).reshape(-1)
        dur_h = float(edf_durations_hours.get(edf_id, 0.0))
        if dur_h < 0.0:
            raise ValueError(
                f"edf_durations_hours[{edf_id}] must be non-negative, "
                f"got {dur_h}"
            )
        total_hours += dur_h

        sz = np.sort(sz)
        al = np.sort(al)
        n_seizures += int(sz.shape[0])
        n_alarms_total += int(al.shape[0])

        used_alarm = np.zeros(al.shape[0], dtype=bool)
        for s in sz:
            lo = s - float(preictal_horizon_sec)
            hi = s

            cand = np.where((al >= lo) & (al <= hi) & (~used_alarm))[0]

            if cand.size > 0:
                n_detected += 1
                chosen = int(cand[0])
                lead_times.append(float(s - al[chosen]))
                used_alarm[chosen] = True

        n_false_alarms += int((~used_alarm).sum())

    event_sensitivity = (
        n_detected / n_seizures if n_seizures > 0 else float("nan")
    )
    fpr_h = (
        n_false_alarms / total_hours if total_hours > 0.0 else float("nan")
    )
    mean_lead_min = (
        float(np.mean(lead_times)) / 60.0 if lead_times else float("nan")
    )
    median_lead_min = (
        float(np.median(lead_times)) / 60.0 if lead_times else float("nan")
    )

    return {
        "event_sensitivity": float(event_sensitivity),
        "false_alarms_per_hour": float(fpr_h),
        "mean_lead_time_min": float(mean_lead_min),
        "median_lead_time_min": float(median_lead_min),
        "n_seizures": int(n_seizures),
        "n_detected": int(n_detected),
        "n_false_alarms": int(n_false_alarms),
        "n_alarms_total": int(n_alarms_total),
        "total_hours": float(total_hours),
    }

def event_level_metrics_global(
    seizure_onsets: np.ndarray,
    alarm_times: np.ndarray,
    total_monitoring_hours: float,
    *,
    preictal_horizon_sec: float = 35.0 * 60.0,
) -> Dict[str, Any]:
    if preictal_horizon_sec <= 0.0:
        raise ValueError(
            f"preictal_horizon_sec must be positive, got "
            f"{preictal_horizon_sec}"
        )
    if total_monitoring_hours < 0.0:
        raise ValueError(
            f"total_monitoring_hours must be non-negative, got "
            f"{total_monitoring_hours}"
        )

    seizure_onsets = np.sort(
        np.asarray(seizure_onsets, dtype=np.float64).reshape(-1)
    )
    alarm_times = np.sort(
        np.asarray(alarm_times, dtype=np.float64).reshape(-1)
    )

    if (
        seizure_onsets.size > 0
        and alarm_times.size > 0
        and float(np.max(alarm_times))
        < float(np.min(seizure_onsets)) - float(preictal_horizon_sec)
        and float(np.max(alarm_times)) < float(preictal_horizon_sec)
    ):
        warnings.warn(
            "event_level_metrics_global: all alarms fire before any "
            "seizure and before preictal_horizon_sec. Did you pass "
            "per-EDF relative times to event_level_metrics_global? "
            "Use event_level_metrics_per_edf for per-EDF data.",
            RuntimeWarning,
            stacklevel=2,
        )

    used_alarm = np.zeros(alarm_times.shape[0], dtype=bool)
    lead_times: List[float] = []

    for s in seizure_onsets:
        lo = s - float(preictal_horizon_sec)
        hi = s
        cand = np.where((alarm_times >= lo) & (alarm_times <= hi))[0]
        if cand.shape[0] > 0:
            lead_times.append(float(s - alarm_times[cand[0]]))
            used_alarm[cand] = True

    n_seizures = int(seizure_onsets.shape[0])
    n_detected = len(lead_times)
    n_false = int((~used_alarm).sum())

    event_sensitivity = (
        n_detected / n_seizures if n_seizures > 0 else float("nan")
    )
    fpr_h = (
        n_false / total_monitoring_hours
        if total_monitoring_hours > 0 else float("nan")
    )
    mean_lead_min = (
        float(np.mean(lead_times)) / 60.0 if lead_times else float("nan")
    )
    median_lead_min = (
        float(np.median(lead_times)) / 60.0 if lead_times else float("nan")
    )

    return {
        "event_sensitivity": float(event_sensitivity),
        "false_alarms_per_hour": float(fpr_h),
        "mean_lead_time_min": float(mean_lead_min),
        "median_lead_time_min": float(median_lead_min),
        "n_seizures": int(n_seizures),
        "n_detected": int(n_detected),
        "n_false_alarms": int(n_false),
        "n_alarms_total": int(alarm_times.shape[0]),
        "total_hours": float(total_monitoring_hours),
    }

def _sanity_check_hmm() -> None:
    rng = np.random.default_rng(0)

    n_edfs = 3
    w_per_edf = 20
    seq_ids = np.repeat(np.arange(n_edfs), w_per_edf)
    onsets = np.concatenate(
        [np.arange(w_per_edf, dtype=np.float64) * 5.0 for _ in range(n_edfs)]
    )
    labels = np.zeros(n_edfs * w_per_edf, dtype=np.int64)
    for k in range(n_edfs):
        labels[k * w_per_edf + 10: k * w_per_edf + 20] = 1
    logits = np.where(
        labels == 1,
        rng.normal(loc=2.0, scale=1.0, size=labels.shape[0]),
        rng.normal(loc=-2.0, scale=1.0, size=labels.shape[0]),
    )

    hmm = CausalHMMPostprocessor.fit(
        logits=logits,
        labels=labels,
        sequence_ids=seq_ids,
        onsets=onsets,
    )

    assert np.allclose(hmm.A.sum(axis=1), 1.0, atol=1e-9)
    assert np.isclose(hmm.pi.sum(), 1.0, atol=1e-9)
    for arr in (hmm.A, hmm.pi, hmm.means, hmm.vars):
        assert np.all(np.isfinite(arr))
    assert np.all(hmm.vars > 0.0)

    assert np.allclose(
        np.exp(hmm._log_A).sum(axis=1), 1.0, atol=1e-12
    )
    assert np.isclose(float(np.exp(hmm._log_pi).sum()), 1.0, atol=1e-12)

    p_pre = hmm.filter_sequence(logits[:w_per_edf])
    assert p_pre.shape == (w_per_edf,)
    assert np.all(np.isfinite(p_pre))
    assert np.all(p_pre >= 0.0) and np.all(p_pre <= 1.0)

    lo = hmm.filter_sequence(logits[:w_per_edf], return_log_odds=True)
    assert lo.shape == (w_per_edf,)
    assert np.all(np.isfinite(lo))

    pre_heavy = np.full(30, 2.0) + rng.normal(0.0, 0.5, size=30)
    int_heavy = np.full(30, -2.0) + rng.normal(0.0, 0.5, size=30)
    assert hmm.filter_sequence(pre_heavy)[-1] > 0.9
    assert hmm.filter_sequence(int_heavy)[-1] < 0.1

    base = rng.normal(size=12)
    lo_base = hmm.filter_sequence(base, return_log_odds=True)

    for t_perturb in range(1, len(base)):
        perturbed = base.copy()
        perturbed[t_perturb] += 5.0
        lo_perturbed = hmm.filter_sequence(perturbed, return_log_odds=True)

        assert np.array_equal(
            lo_base[:t_perturb], lo_perturbed[:t_perturb],
        ), (
            f"Causality violated at index {t_perturb}: "
            f"log-odds[0:{t_perturb}] changed when only "
            f"logit[{t_perturb}] was perturbed"
        )
        assert not np.isclose(
            lo_base[t_perturb], lo_perturbed[t_perturb], atol=1e-6,
        ), (
            f"log-odds[{t_perturb}] did not respond to its own "
            f"perturbation"
        )

    p_seq = np.array(
        [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.2, 0.1, 0.6, 0.8]
    )
    onsets_seq = np.arange(len(p_seq), dtype=np.float64) * 60.0

    alarms = CausalHMMPostprocessor.alarms_from_filter(
        p_seq, onsets_seq, threshold=0.5, refractory_sec=0.0,
        alarm_mode="one_per_episode",
    )
    assert alarms.shape[0] == 2
    assert np.isclose(alarms[0], 3 * 60.0)
    assert np.isclose(alarms[1], 9 * 60.0)

    alarms_ref = CausalHMMPostprocessor.alarms_from_filter(
        p_seq, onsets_seq, threshold=0.5, refractory_sec=10 * 60.0,
        alarm_mode="one_per_episode",
    )
    assert alarms_ref.shape[0] == 1
    assert np.isclose(alarms_ref[0], 3 * 60.0)

    always_high = np.full(30, 0.9)
    onsets_high = np.arange(30, dtype=np.float64) * 60.0
    alarms_periodic = CausalHMMPostprocessor.alarms_from_filter(
        always_high, onsets_high, threshold=0.5,
        refractory_sec=10 * 60.0, alarm_mode="periodic_while_above",
    )
    assert alarms_periodic.shape[0] == 3, (
        f"periodic_while_above: expected 3 alarms, "
        f"got {alarms_periodic.shape[0]}"
    )
    assert np.isclose(alarms_periodic[1], 10 * 60.0)

    alarms_one = CausalHMMPostprocessor.alarms_from_filter(
        always_high, onsets_high, threshold=0.5,
        refractory_sec=10 * 60.0, alarm_mode="one_per_episode",
    )
    assert alarms_one.shape[0] == 1

    for bad_kwargs in [
        dict(threshold=0.0),
        dict(threshold=1.0),
        dict(threshold=-0.1),
        dict(refractory_sec=-1.0),
        dict(alarm_mode="foo"),
    ]:
        try:
            CausalHMMPostprocessor.alarms_from_filter(
                p_seq, onsets_seq, **bad_kwargs,
            )
            raise AssertionError(f"expected ValueError for {bad_kwargs}")
        except ValueError:
            pass

    metrics = event_level_metrics_per_edf(
        seizures_by_edf={
            0: np.array([1800.0]),
            1: np.array([900.0]),
            2: np.empty(0),
        },
        alarms_by_edf={
            0: np.array([1500.0]),
            1: np.empty(0),
            2: np.array([600.0]),
        },
        edf_durations_hours={0: 1.0, 1: 1.0, 2: 1.0},
        preictal_horizon_sec=35 * 60,
    )
    assert metrics["n_seizures"] == 2
    assert metrics["n_detected"] == 1
    assert np.isclose(metrics["event_sensitivity"], 0.5)
    assert metrics["n_false_alarms"] == 1
    assert np.isclose(metrics["false_alarms_per_hour"], 1.0 / 3.0)
    assert np.isclose(metrics["mean_lead_time_min"], 5.0)
    assert np.isclose(metrics["total_hours"], 3.0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = event_level_metrics_per_edf(
            seizures_by_edf={0: np.array([1800.0])},
            alarms_by_edf={0: np.array([1500.0])},
            edf_durations_hours={},
        )
        assert any(
            issubclass(wi.category, RuntimeWarning)
            and "edf_durations_hours" in str(wi.message)
            for wi in w
        ), "expected RuntimeWarning for missing EDF duration"

    offsets = {0: 0.0, 1: 3600.0, 2: 7200.0}
    seizures_global = np.concatenate([
        offsets[0] + np.array([1800.0]),
        offsets[1] + np.array([900.0]),
    ])
    alarms_global = np.concatenate([
        offsets[0] + np.array([1500.0]),
        offsets[2] + np.array([600.0]),
    ])
    metrics_global = event_level_metrics_global(
        seizure_onsets=seizures_global,
        alarm_times=alarms_global,
        total_monitoring_hours=3.0,
        preictal_horizon_sec=35 * 60,
    )
    assert metrics_global["n_seizures"] == metrics["n_seizures"]
    assert metrics_global["n_detected"] == metrics["n_detected"]
    assert metrics_global["n_false_alarms"] == metrics["n_false_alarms"]
    assert np.isclose(
        metrics_global["event_sensitivity"], metrics["event_sensitivity"]
    )
    assert np.isclose(
        metrics_global["mean_lead_time_min"], metrics["mean_lead_time_min"]
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = event_level_metrics_global(
            seizure_onsets=np.array([10000.0, 20000.0]),
            alarm_times=np.array([100.0, 200.0]),
            total_monitoring_hours=2.0,
            preictal_horizon_sec=35 * 60,
        )
        assert any(
            issubclass(wi.category, RuntimeWarning)
            and "relative" in str(wi.message).lower()
            for wi in w
        ), "expected RuntimeWarning for relative-times heuristic"

    try:
        apply_hmm_to_test_set(
            hmm,
            logits=logits,
            onsets=onsets,
            sequence_ids=seq_ids,
            edf_start_offsets={"edf_0": 0.0, "edf_1": 3600.0},
        )
        raise AssertionError("expected TypeError for string offset keys")
    except TypeError as exc:
        assert "int" in str(exc).lower(), (
            f"error message unclear: {exc}"
        )

    offsets_dict = {int(k): float(k) * 3600.0 for k in np.unique(seq_ids)}
    out_with_offsets = apply_hmm_to_test_set(
        hmm,
        logits=logits,
        onsets=onsets,
        sequence_ids=seq_ids,
        edf_start_offsets=offsets_dict,
        threshold=0.5,
        refractory_sec=0.0,
    )
    assert out_with_offsets["alarm_absolute_times"] is not None
    n_alarms = int(out_with_offsets["alarm_relative_times"].size)
    assert n_alarms > 0, (
        "test fixture produced no alarms; the offset-conversion check "
        "was not exercised. Adjust the synthetic logits."
    )
    assert (
        out_with_offsets["alarm_absolute_times"].shape
        == out_with_offsets["alarm_relative_times"].shape
    )
    for sid, t_rel, t_abs in zip(
        out_with_offsets["alarm_seq_ids"],
        out_with_offsets["alarm_relative_times"],
        out_with_offsets["alarm_absolute_times"],
    ):
        assert np.isclose(t_abs, offsets_dict[int(sid)] + t_rel)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = CausalHMMPostprocessor.fit(
            logits=np.array([0.5, 0.6, 0.4]),
            labels=np.array([0, 0, 0]),
            sequence_ids=np.array([0, 0, 0]),
            onsets=np.array([0.0, 5.0, 10.0]),
        )
        assert any(
            issubclass(wi.category, RuntimeWarning) for wi in w
        ), "expected RuntimeWarning for single-class fit"

    for bad_kwargs in [
        dict(transition_smooth=0.0),
        dict(transition_smooth=-1.0),
        dict(var_floor=0.0),
        dict(var_floor=-1e-6),
        dict(eps=0.0),
    ]:
        try:
            CausalHMMPostprocessor.fit(
                logits=logits,
                labels=labels,
                sequence_ids=seq_ids,
                onsets=onsets,
                **bad_kwargs,
            )
            raise AssertionError(f"expected ValueError for {bad_kwargs}")
        except ValueError:
            pass

    for func in (event_level_metrics_per_edf, event_level_metrics_global):
        try:
            if func is event_level_metrics_per_edf:
                func(
                    seizures_by_edf={0: np.array([100.0])},
                    alarms_by_edf={0: np.array([50.0])},
                    edf_durations_hours={0: 1.0},
                    preictal_horizon_sec=0.0,
                )
            else:
                func(
                    seizure_onsets=np.array([100.0]),
                    alarm_times=np.array([50.0]),
                    total_monitoring_hours=1.0,
                    preictal_horizon_sec=-1.0,
                )
            raise AssertionError(
                "expected ValueError for preictal_horizon_sec<=0"
            )
        except ValueError:
            pass

    _LOGGER.info(
        "[hmm-sanity] PASS | "
        "A=[[%.3f,%.3f],[%.3f,%.3f]] | "
        "means=[%.2f,%.2f] | vars=[%.2f,%.2f] | "
        "causality=verified | per_edf_metrics=verified | "
        "global_metrics=verified | offset_conversion=verified",
        hmm.A[0, 0], hmm.A[0, 1], hmm.A[1, 0], hmm.A[1, 1],
        hmm.means[0], hmm.means[1], hmm.vars[0], hmm.vars[1],
    )

from dataclasses import dataclass, field, replace, asdict, is_dataclass
from typing import Dict, List, Optional, Any, Tuple
import os
import json
import csv
import time

import numpy as np
import torch
from scipy import stats

BIGMODEL_RESULT_DIR = (
    r"E:\Neuroscience_Lab\Article reviews final codes and check\BigModelResult"
)

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

@dataclass
class FoldPipelineResult:
    fold_id: int
    protocol: str
    train_sids: List[int]
    test_sids: List[int]
    before_hmm: Optional[WindowEvalResult] = None
    after_hmm: Optional[WindowEvalResult] = None
    hmm_info: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    train_summary: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: Optional[str] = None

@dataclass
class SubjectSummary:
    subject_id: int
    n_folds: int
    before_mean: Dict[str, float]
    before_std: Dict[str, float]
    after_mean: Dict[str, float]
    after_std: Dict[str, float]

@dataclass
class ProtocolResult:
    protocol_name: str
    n_folds: int
    n_folds_failed: int
    before_hmm_mean: Dict[str, float]
    before_hmm_std: Dict[str, float]
    after_hmm_mean: Dict[str, float]
    after_hmm_std: Dict[str, float]
    per_subject: Dict[int, SubjectSummary]
    config: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    fold_results: List[FoldPipelineResult] = field(default_factory=list)

@torch.inference_mode()
def _forward_windows(
    store: MasterDataStore,
    model: nn.Module,
    fold: Fold,
    *,
    split: str = "val",
    batch_size: int = 256,
    use_amp: bool = True,
    amp_dtype: str = "bfloat16",
    num_workers: int = 0,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split not in {"val", "test", "train"}:
        raise ValueError(
            f"_forward_windows: split must be 'val', 'test', or 'train', "
            f"got {split!r}"
        )

    indices_map = {
        "val": fold.val_idx,
        "test": fold.test_idx,
        "train": fold.train_idx,
    }
    indices = np.asarray(indices_map[split], dtype=np.int64)
    if indices.shape[0] == 0:
        raise ValueError(
            f"_forward_windows: split '{split}' is empty (n=0)"
        )

    if device is None:
        device_obj = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device_obj = torch.device(device)
    amp_dtype_obj = (
        torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    )
    _use_amp = bool(use_amp and device_obj.type == "cuda")

    ds = DBTRDataset(store, indices, return_metadata=False)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=_MixupCollateCallable(use_mixup=False, n_classes=2),
        num_workers=int(num_workers),
        pin_memory=bool(device_obj.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    was_training = model.training
    model.eval()
    try:
        all_logits: List[np.ndarray] = []
        for batch in loader:
            time_b = batch["time"].to(device_obj, non_blocking=True)
            freq_b = batch["freq"].to(device_obj, non_blocking=True)
            with torch.amp.autocast(
                device_type=device_obj.type,
                dtype=amp_dtype_obj,
                enabled=_use_amp,
            ):
                logits = model(time_b, freq_b)["logits"]
            logit_diff = logits[:, 1] - logits[:, 0]
            all_logits.append(logit_diff.float().cpu().numpy())
    finally:
        model.train(was_training)

    logits_arr = np.concatenate(all_logits, axis=0).astype(np.float32)

    y_true = store.labels[indices].astype(np.int32)
    edf_ids = store.edf_file_ids[indices].astype(np.int32)
    onsets = store.onsets[indices].astype(np.float64)
    return logits_arr, y_true, edf_ids, onsets

def _mean_std_over_dicts(
    dicts: List[Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not dicts:
        return {}, {}
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts if k in d and np.isfinite(d[k])]
        if not vals:
            continue
        means[k] = float(np.mean(vals))
        stds[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return means, stds

def _group_by_subject(
    results: List[WindowEvalResult],
) -> Dict[int, Dict[str, List[float]]]:
    out: Dict[int, Dict[str, List[float]]] = {}
    for r in results:
        for sid, metrics in r.per_subject.items():
            sid = int(sid)
            out.setdefault(sid, {})
            for k, v in metrics.items():
                if np.isfinite(v):
                    out[sid].setdefault(k, []).append(float(v))
    return out

def _build_result_from_probs(
    store: MasterDataStore,
    fold: Fold,
    split: str,
    indices: np.ndarray,
    probs: np.ndarray,
    model_logits: np.ndarray,
    *,
    threshold: float,
    refractory_sec: float,
    preictal_horizon_sec: float,
    alarm_mode: str,
    model_name: str,
    edf_durations_hours_override: Optional[Dict[int, float]] = None,
) -> WindowEvalResult:
    for name, val in (
        ("threshold", threshold),
        ("refractory_sec", refractory_sec),
        ("preictal_horizon_sec", preictal_horizon_sec),
    ):
        if not np.isfinite(val):
            raise ValueError(
                f"_build_result_from_probs: {name} must be finite, "
                f"got {val}"
            )

    if not (0.0 < float(threshold) < 1.0):
        raise ValueError(
            f"_build_result_from_probs: threshold must be in (0, 1), "
            f"got {threshold}"
        )
    if refractory_sec < 0.0:
        raise ValueError(
            f"_build_result_from_probs: refractory_sec must be >= 0, "
            f"got {refractory_sec}"
        )
    if preictal_horizon_sec <= 0.0:
        raise ValueError(
            f"_build_result_from_probs: preictal_horizon_sec must be > 0, "
            f"got {preictal_horizon_sec}"
        )

    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs).reshape(-1)
    model_logits = np.asarray(model_logits).reshape(-1)

    if not (indices.shape == probs.shape == model_logits.shape):
        raise ValueError(
            f"_build_result_from_probs: shape mismatch: "
            f"indices={indices.shape}, probs={probs.shape}, "
            f"model_logits={model_logits.shape}"
        )

    if indices.size == 0:
        raise ValueError(
            "_build_result_from_probs: empty indices; nothing to evaluate."
        )

    n_store = int(len(store.labels))
    idx_min = int(indices.min())
    idx_max = int(indices.max())
    if idx_min < 0 or idx_max >= n_store:
        raise IndexError(
            f"_build_result_from_probs: indices out of bounds: "
            f"min={idx_min}, max={idx_max}, n_store={n_store}. "
            f"Negative indices would be silently wrapped by numpy."
        )

    if not np.all(np.isfinite(probs)):
        raise ValueError(
            "_build_result_from_probs: probs contains NaN/Inf."
        )
    if not np.all(np.isfinite(model_logits)):
        raise ValueError(
            "_build_result_from_probs: model_logits contains NaN/Inf."
        )

    probs_f32 = probs.astype(np.float32, copy=False)

    p_min = float(probs_f32.min())
    p_max = float(probs_f32.max())
    if p_min < 0.0 or p_max > 1.0:
        raise ValueError(
            f"_build_result_from_probs: probs must be in [0, 1]; "
            f"got min={p_min:.3e}, max={p_max:.3e}. "
            f"Did you pass raw logits by mistake?"
        )

    y_true = store.labels[indices].astype(np.int32)
    uniq_y = np.unique(y_true)
    if not (set(uniq_y.tolist()) <= {0, 1}):
        raise ValueError(
            f"_build_result_from_probs: y_true must be in {{0,1}}, "
            f"got unique values {uniq_y.tolist()}"
        )

    y_pred = (probs >= float(threshold)).astype(np.int32)

    edf_ids = store.edf_file_ids[indices].astype(np.int32)
    subj_ids = store.subject_ids[indices].astype(np.int32)

    onsets_arr = store.onsets[indices].astype(np.float64)
    durs_arr = store.durs[indices].astype(np.float64)
    win_durs_arr = store.win_durs[indices].astype(np.float64)

    total_ih = store.interictal_hours(indices)

    window_aggregate = _metrics_for_group(
        y_true=y_true,
        y_pred=y_pred,
        y_score=probs_f32,
        interictal_hours=float(total_ih),
    )

    confusion_pooled = np.zeros((2, 2), dtype=np.int64)
    confusion_pooled[0, 0] = int(((y_true == 0) & (y_pred == 0)).sum())
    confusion_pooled[0, 1] = int(((y_true == 0) & (y_pred == 1)).sum())
    confusion_pooled[1, 0] = int(((y_true == 1) & (y_pred == 0)).sum())
    confusion_pooled[1, 1] = int(((y_true == 1) & (y_pred == 1)).sum())

    roc_fpr = roc_tpr = roc_thr = None
    roc_auc_val: Optional[float] = None
    if uniq_y.size == 2:
        try:
            roc_fpr, roc_tpr, roc_thr = roc_curve(
                y_true, probs_f32, pos_label=1,
            )
            roc_auc_val = float(auc(roc_fpr, roc_tpr))
        except ValueError as exc:
            _LOGGER.warning(
                "_build_result_from_probs: roc_curve raised "
                "ValueError (%s); roc_auc will be None.", exc,
            )

    alarms_by_edf = _compute_alarms_by_edf(
        probs=probs_f32,
        onsets=onsets_arr,
        edf_ids=edf_ids,
        threshold=float(threshold),
        refractory_sec=float(refractory_sec),
        alarm_mode=str(alarm_mode),
    )

    seizure_onsets_by_edf = _reconstruct_seizure_onsets_by_edf(
        store,
        indices,
        preictal_horizon_sec=float(preictal_horizon_sec),
    )

    if edf_durations_hours_override is not None:
        edf_durations_hours: Dict[int, float] = {
            int(eid): float(h)
            for eid, h in edf_durations_hours_override.items()
        }
        missing = [
            int(eid) for eid in np.unique(edf_ids)
            if int(eid) not in edf_durations_hours
        ]
        if missing:
            raise ValueError(
                f"_build_result_from_probs: "
                f"edf_durations_hours_override missing entries for "
                f"EDFs {missing}"
            )
    else:
        uniq_edfs, inv = np.unique(
            edf_ids.astype(np.int64, copy=False), return_inverse=True
        )
        dur_sums = np.bincount(inv, weights=durs_arr)
        edf_durations_hours = {
            int(eid): float(dur_sums[i]) / 3600.0
            for i, eid in enumerate(uniq_edfs)
        }

    alarm_metrics = event_level_metrics_per_edf(
        seizures_by_edf=seizure_onsets_by_edf,
        alarms_by_edf=alarms_by_edf,
        edf_durations_hours=edf_durations_hours,
        preictal_horizon_sec=float(preictal_horizon_sec),
    )

    aggregate = dict(window_aggregate)
    aggregate.update({
        "event_sensitivity": float(alarm_metrics["event_sensitivity"]),
        "false_alarms_per_hour": float(alarm_metrics["false_alarms_per_hour"]),
        "mean_lead_time_min": float(alarm_metrics["mean_lead_time_min"]),
        "median_lead_time_min": float(alarm_metrics["median_lead_time_min"]),
        "n_seizures": int(alarm_metrics["n_seizures"]),
        "n_detected": int(alarm_metrics["n_detected"]),
        "n_false_alarms": int(alarm_metrics["n_false_alarms"]),
        "n_alarms_total": int(alarm_metrics["n_alarms_total"]),
        "event_total_hours": float(alarm_metrics["total_hours"]),
    })

    per_subject: Dict[int, Dict[str, float]] = {}
    for sid in np.unique(subj_ids):
        mask = subj_ids == sid
        idx_sid = indices[mask]
        subj_ih = store.interictal_hours(idx_sid)

        subj_metrics = _metrics_for_group(
            y_true=y_true[mask],
            y_pred=y_pred[mask],
            y_score=probs_f32[mask],
            interictal_hours=float(subj_ih),
        )

        subj_edfs = np.unique(edf_ids[mask])
        subj_seiz = {
            int(eid): seizure_onsets_by_edf.get(
                int(eid), np.empty(0, dtype=np.float64)
            )
            for eid in subj_edfs
        }
        subj_alarms = {
            int(eid): alarms_by_edf.get(
                int(eid), np.empty(0, dtype=np.float64)
            )
            for eid in subj_edfs
        }
        subj_durs = {
            int(eid): float(edf_durations_hours.get(int(eid), 0.0))
            for eid in subj_edfs
        }

        subj_event = event_level_metrics_per_edf(
            seizures_by_edf=subj_seiz,
            alarms_by_edf=subj_alarms,
            edf_durations_hours=subj_durs,
            preictal_horizon_sec=float(preictal_horizon_sec),
        )

        subj_metrics.update({
            "event_sensitivity": float(subj_event["event_sensitivity"]),
            "false_alarms_per_hour": float(subj_event["false_alarms_per_hour"]),
            "mean_lead_time_min": float(subj_event["mean_lead_time_min"]),
            "median_lead_time_min": float(subj_event["median_lead_time_min"]),
            "n_seizures": int(subj_event["n_seizures"]),
            "n_detected": int(subj_event["n_detected"]),
            "n_false_alarms": int(subj_event["n_false_alarms"]),
            "n_alarms_total": int(subj_event["n_alarms_total"]),
            "event_total_hours": float(subj_event["total_hours"]),
        })

        per_subject[int(sid)] = subj_metrics

    macro: Dict[str, float] = {}
    if per_subject:
        first = next(iter(per_subject.values()))
        for k in first.keys():
            vals = np.asarray(
                [m.get(k, np.nan) for m in per_subject.values()],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                macro[f"{k}_mean"] = float(vals.mean())
                macro[f"{k}_std"] = float(
                    vals.std(ddof=1) if vals.size > 1 else 0.0
                )

    raw_subject_ids_arr = store.raw_sids[indices].astype(np.int32)

    per_raw_subject: Dict[int, Dict[str, float]] = {}
    for r_sid in np.unique(raw_subject_ids_arr):
        mask_r = raw_subject_ids_arr == r_sid
        idx_r = indices[mask_r]
        raw_ih = store.interictal_hours(idx_r)

        m = _metrics_for_group(
            y_true=y_true[mask_r],
            y_pred=y_pred[mask_r],
            y_score=probs_f32[mask_r],
            interictal_hours=float(raw_ih),
        )

        raw_edfs = np.unique(edf_ids[mask_r])
        raw_seiz = {
            int(eid): seizure_onsets_by_edf.get(
                int(eid), np.empty(0, dtype=np.float64)
            )
            for eid in raw_edfs
        }
        raw_alarms = {
            int(eid): alarms_by_edf.get(
                int(eid), np.empty(0, dtype=np.float64)
            )
            for eid in raw_edfs
        }
        raw_durs = {
            int(eid): float(edf_durations_hours.get(int(eid), 0.0))
            for eid in raw_edfs
        }

        raw_event = event_level_metrics_per_edf(
            seizures_by_edf=raw_seiz,
            alarms_by_edf=raw_alarms,
            edf_durations_hours=raw_durs,
            preictal_horizon_sec=float(preictal_horizon_sec),
        )

        m.update({
            "event_sensitivity": float(raw_event["event_sensitivity"]),
            "false_alarms_per_hour": float(raw_event["false_alarms_per_hour"]),
            "mean_lead_time_min": float(raw_event["mean_lead_time_min"]),
            "median_lead_time_min": float(raw_event["median_lead_time_min"]),
            "n_seizures": int(raw_event["n_seizures"]),
            "n_detected": int(raw_event["n_detected"]),
            "n_false_alarms": int(raw_event["n_false_alarms"]),
            "n_alarms_total": int(raw_event["n_alarms_total"]),
            "event_total_hours": float(raw_event["total_hours"]),
        })

        per_raw_subject[int(r_sid)] = m

    return WindowEvalResult(
        split_name=str(split),
        fold_id=int(fold.fold_id),
        protocol=str(fold.protocol),
        model_name=str(model_name),
        logits=np.asarray(model_logits, dtype=np.float32),
        probs=probs_f32,
        y_true=y_true,
        y_pred=y_pred,
        subject_ids=subj_ids,
        edf_file_ids=edf_ids,
        onsets=onsets_arr,
        durs=durs_arr,
        win_durs=win_durs_arr,
        row_indices=np.asarray(indices, dtype=np.int64),
        aggregate=aggregate,
        per_subject=per_subject,
        macro=macro,
        confusion=confusion_pooled,
        raw_subject_ids=raw_subject_ids_arr,
        per_raw_subject=per_raw_subject,
        roc_fpr=roc_fpr,
        roc_tpr=roc_tpr,
        roc_thresholds=roc_thr,
        roc_auc=roc_auc_val,
        window_fpr_per_hour=float(window_aggregate["window_fpr_per_hour"]),
        alarm_metrics=alarm_metrics,
        alarms_by_edf=alarms_by_edf,
        threshold=float(threshold),
        refractory_sec=float(refractory_sec),
        preictal_horizon_sec=float(preictal_horizon_sec),
        alarm_mode=str(alarm_mode),
    )

def evaluate_with_hmm(
    store: MasterDataStore,
    model: nn.Module,
    fold: Fold,
    *,
    batch_size: int = 256,
    use_amp: bool = True,
    amp_dtype: str = "bfloat16",
    num_workers: int = 0,
    device: Optional[str] = None,
    threshold: float = 0.5,
    refractory_sec: float = 30.0 * 60.0,
    preictal_horizon_sec: float = 35.0 * 60.0,
    alarm_mode: str = "one_per_episode",
    model_name: str = "dbtr_agf",
    hmm_transition_smooth: float = 1.0,
    hmm_var_floor: float = 1e-6,
) -> Tuple[WindowEvalResult, WindowEvalResult, CausalHMMPostprocessor]:
    val_logits, val_labels, val_edf_ids, val_onsets = _forward_windows(
        store, model, fold, split="val",
        batch_size=batch_size, use_amp=use_amp, amp_dtype=amp_dtype,
        num_workers=num_workers, device=device,
    )
    hmm = CausalHMMPostprocessor.fit(
        logits=val_logits,
        labels=val_labels,
        sequence_ids=val_edf_ids,
        onsets=val_onsets,
        transition_smooth=float(hmm_transition_smooth),
        var_floor=float(hmm_var_floor),
    )

    before = evaluate_windows(
        store, model, fold, split="test",
        batch_size=batch_size, use_amp=use_amp, amp_dtype=amp_dtype,
        num_workers=num_workers, device=device,
        threshold=threshold, refractory_sec=refractory_sec,
        preictal_horizon_sec=preictal_horizon_sec,
        alarm_mode=alarm_mode,
        model_name=model_name,
    )

    hmm_out = apply_hmm_to_test_set(
        hmm,
        logits=before.logits,
        onsets=before.onsets,
        sequence_ids=before.edf_file_ids,
        threshold=float(threshold),
        refractory_sec=float(refractory_sec),
        alarm_mode=str(alarm_mode),
    )
    p_pre_test = np.asarray(hmm_out["p_pre_all"], dtype=np.float32)

    if p_pre_test.shape[0] != before.row_indices.shape[0]:
        raise RuntimeError(
            f"evaluate_with_hmm: HMM output length ({p_pre_test.shape[0]}) "
            f"does not match test row count ({before.row_indices.shape[0]}). "
            f"This indicates a bug in apply_hmm_to_test_set's ordering."
        )

    after = _build_result_from_probs(
        store, fold, "test",
        indices=before.row_indices,
        probs=p_pre_test,
        model_logits=before.logits,
        threshold=float(threshold),
        refractory_sec=float(refractory_sec),
        preictal_horizon_sec=float(preictal_horizon_sec),
        alarm_mode=str(alarm_mode),
        model_name=f"{model_name}+hmm",
    )

    return before, after, hmm

def run_one_fold_pipeline(
    store: MasterDataStore,
    fold: Fold,
    cfg: TrainConfig,
    *,
    hmm_transition_smooth: float = 1.0,
    hmm_var_floor: float = 1e-6,
    threshold: float = 0.5,
    refractory_sec: float = 30.0 * 60.0,
    preictal_horizon_sec: float = 35.0 * 60.0,
    alarm_mode: str = "one_per_episode",
    model_name: str = "dbtr_agf",
    eval_batch_size: int = 256,
    eval_num_workers: int = 0,
    eval_use_amp: bool = True,
) -> FoldPipelineResult:
    t0 = time.perf_counter()
    fr = FoldPipelineResult(
        fold_id=int(fold.fold_id),
        protocol=str(fold.protocol),
        train_sids=[int(s) for s in fold.train_sids],
        test_sids=[int(s) for s in fold.test_sids],
    )

    model: Optional[nn.Module] = None
    try:
        train_out = train_one_fold(store, fold, cfg)
        model = train_out["model"]
        fr.history = train_out["history"]
        fr.train_summary = {
            "best_epoch": int(train_out["best_epoch"]),
            "best_val_loss": float(train_out["best_val_loss"]),
            "best_val_acc": float(train_out["best_val_acc"]),
            "n_train": int(train_out["n_train"]),
            "n_val": int(train_out["n_val"]),
            "device": str(train_out["device"]),
        }

        before, after, hmm = evaluate_with_hmm(
            store, model, fold,
            batch_size=eval_batch_size,
            use_amp=eval_use_amp,
            num_workers=eval_num_workers,
            device=train_out["device"],
            threshold=threshold,
            refractory_sec=refractory_sec,
            preictal_horizon_sec=preictal_horizon_sec,
            alarm_mode=alarm_mode,
            model_name=model_name,
            hmm_transition_smooth=hmm_transition_smooth,
            hmm_var_floor=hmm_var_floor,
        )
        fr.before_hmm = before
        fr.after_hmm = after
        fr.hmm_info = hmm.info()

    except Exception as e:
        fr.error = f"{type(e).__name__}: {e}"
        _LOGGER.exception(
            "[fold %d] pipeline failed: %s", fold.fold_id, e,
        )
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fr.duration_seconds = time.perf_counter() - t0
    return fr

def _macro_mean_std_from_per_subject(
    per_subject: Dict[int, "SubjectSummary"],
    phase: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if phase not in {"before", "after"}:
        raise ValueError(
            f"phase must be 'before' or 'after', got {phase!r}"
        )

    buckets: Dict[str, List[float]] = {}
    for s in per_subject.values():
        d = s.before_mean if phase == "before" else s.after_mean
        for k, v in d.items():
            if np.isfinite(v):
                buckets.setdefault(k, []).append(float(v))

    mean: Dict[str, float] = {
        k: float(np.mean(v)) for k, v in buckets.items() if v
    }
    std: Dict[str, float] = {
        k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        for k, v in buckets.items() if v
    }
    return mean, std

def _aggregate_folds(
    fold_results: List[FoldPipelineResult],
    protocol_name: str,
    config: Dict[str, Any],
    duration_seconds: float,
) -> ProtocolResult:
    ok = [
        f for f in fold_results
        if f.error is None
        and f.before_hmm is not None
        and f.after_hmm is not None
    ]
    n_failed = len(fold_results) - len(ok)

    subjects_before = _group_by_subject([f.before_hmm for f in ok])
    subjects_after = _group_by_subject([f.after_hmm for f in ok])
    all_sids = set(subjects_before.keys()) | set(subjects_after.keys())

    per_subject: Dict[int, SubjectSummary] = {}
    for sid in sorted(all_sids):
        b = subjects_before.get(int(sid), {})
        a = subjects_after.get(int(sid), {})

        all_lens = [len(v) for v in b.values()] + [len(v) for v in a.values()]
        n_folds = max(all_lens) if all_lens else 0

        per_subject[int(sid)] = SubjectSummary(
            subject_id=int(sid),
            n_folds=int(n_folds),
            before_mean={k: float(np.mean(v)) for k, v in b.items() if v},
            before_std={
                k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                for k, v in b.items() if v
            },
            after_mean={k: float(np.mean(v)) for k, v in a.items() if v},
            after_std={
                k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                for k, v in a.items() if v
            },
        )

    before_mean, before_std = _macro_mean_std_from_per_subject(
        per_subject, "before"
    )
    after_mean, after_std = _macro_mean_std_from_per_subject(
        per_subject, "after"
    )

    fold_before_mean, fold_before_std = _mean_std_over_dicts(
        [f.before_hmm.aggregate for f in ok]
    )
    fold_after_mean, fold_after_std = _mean_std_over_dicts(
        [f.after_hmm.aggregate for f in ok]
    )

    cfg = dict(config)
    cfg["fold_micro_before_mean"] = fold_before_mean
    cfg["fold_micro_before_std"] = fold_before_std
    cfg["fold_micro_after_mean"] = fold_after_mean
    cfg["fold_micro_after_std"] = fold_after_std
    cfg["aggregation_mode_table1"] = "subject_wise_macro"

    return ProtocolResult(
        protocol_name=str(protocol_name),
        n_folds=len(fold_results),
        n_folds_failed=int(n_failed),
        before_hmm_mean=before_mean,
        before_hmm_std=before_std,
        after_hmm_mean=after_mean,
        after_hmm_std=after_std,
        per_subject=per_subject,
        config=cfg,
        duration_seconds=float(duration_seconds),
        fold_results=fold_results,
    )

def _save_fold_result(fr: FoldPipelineResult, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if fr.before_hmm is not None:
        fr.before_hmm.save_to(out_dir, prefix="before_")
    if fr.after_hmm is not None:
        fr.after_hmm.save_to(out_dir, prefix="after_")

    meta = {
        "fold_id": fr.fold_id,
        "protocol": fr.protocol,
        "train_sids": fr.train_sids,
        "test_sids": fr.test_sids,
        "hmm_info": fr.hmm_info,
        "train_summary": fr.train_summary,
        "n_epochs": len(fr.history),
        "duration_seconds": fr.duration_seconds,
        "error": fr.error,
    }
    with open(os.path.join(out_dir, "fold_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    if fr.history:
        keys: List[str] = list(dict.fromkeys(
            k for row in fr.history for k in row.keys()
        ))
        with open(os.path.join(out_dir, "history.csv"), "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for row in fr.history:
                writer.writerow([row.get(k, "") for k in keys])

def _write_table1_row(result: ProtocolResult, path: str) -> None:
    metrics = sorted(result.before_hmm_mean.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["protocol", "n_folds_ok", "n_folds_failed"]
        for m in metrics:
            header += [
                f"{m}_before_mean", f"{m}_before_std",
                f"{m}_after_mean", f"{m}_after_std",
            ]
        writer.writerow(header)
        row = [
            result.protocol_name,
            result.n_folds - result.n_folds_failed,
            result.n_folds_failed,
        ]
        for m in metrics:
            row += [
                result.before_hmm_mean.get(m, ""),
                result.before_hmm_std.get(m, ""),
                result.after_hmm_mean.get(m, ""),
                result.after_hmm_std.get(m, ""),
            ]
        writer.writerow(row)

def _save_protocol_result(
    result: ProtocolResult,
    out_dir: str,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    summary_path = os.path.join(out_dir, "protocol_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "protocol_name": result.protocol_name,
            "n_folds": int(result.n_folds),
            "n_folds_failed": int(result.n_folds_failed),
            "before_hmm_mean": result.before_hmm_mean,
            "before_hmm_std": result.before_hmm_std,
            "after_hmm_mean": result.after_hmm_mean,
            "after_hmm_std": result.after_hmm_std,
            "duration_seconds": float(result.duration_seconds),
            "config": result.config,
        }, f, indent=2, default=_json_default)
    paths["protocol_summary"] = summary_path

    before_csv = os.path.join(out_dir, "per_subject_before.csv")
    _write_subject_csv(result.per_subject, mode="before", path=before_csv)
    paths["per_subject_before"] = before_csv

    after_csv = os.path.join(out_dir, "per_subject_after.csv")
    _write_subject_csv(result.per_subject, mode="after", path=after_csv)
    paths["per_subject_after"] = after_csv

    table1_path = os.path.join(out_dir, "table1_row.csv")
    _write_table1_row(result, table1_path)
    paths["table1_row"] = table1_path

    return paths

def _run_protocol(
    store: MasterDataStore,
    folds: List[Fold],
    cfg: TrainConfig,
    protocol_name: str,
    out_dir: str,
    *,
    save_per_fold: bool = True,
    prune_fold_results: bool = False,
    **pipeline_kwargs: Any,
) -> ProtocolResult:
    if not folds:
        raise ValueError(
            f"_run_protocol({protocol_name}): empty folds list"
        )

    fold_ids = [int(f.fold_id) for f in folds]
    if len(set(fold_ids)) != len(fold_ids):
        dup = sorted({x for x in fold_ids if fold_ids.count(x) > 1})
        raise ValueError(
            f"_run_protocol({protocol_name}): duplicate fold_id(s) "
            f"{dup} — artifacts would be silently overwritten"
        )

    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    fold_results: List[FoldPipelineResult] = []

    _LOGGER.info(
        "[%s] starting %d folds; out_dir=%s",
        protocol_name, len(folds), out_dir,
    )

    for i, fold in enumerate(folds, start=1):
        _LOGGER.info(
            "[%s] fold %d/%d | train=%s | n_test_subjects=%d",
            protocol_name, i, len(folds),
            fold.train_sids, len(fold.test_sids),
        )
        fr = run_one_fold_pipeline(
            store, fold, cfg, **pipeline_kwargs,
        )
        fold_results.append(fr)

        if save_per_fold:
            try:
                _save_fold_result(
                    fr, os.path.join(out_dir, f"fold_{fold.fold_id:03d}"),
                )
            except Exception as save_exc:
                _LOGGER.warning(
                    "[%s] fold %d: could not save artifacts (%s)",
                    protocol_name, fold.fold_id, save_exc,
                )

        if (
            fr.error is None
            and fr.before_hmm is not None
            and fr.after_hmm is not None
        ):
            b = fr.before_hmm.aggregate
            a = fr.after_hmm.aggregate
            _LOGGER.info(
                "[%s] fold %d done (%.1fs) | "
                "before: acc=%.4f sen=%.4f spe=%.4f fpr=%.4f/h | "
                "after: acc=%.4f sen=%.4f spe=%.4f fpr=%.4f/h | "
                "event_fpr=%.4f/h",
                protocol_name, fold.fold_id, fr.duration_seconds,
                b.get("accuracy", float("nan")),
                b.get("sensitivity", float("nan")),
                b.get("specificity", float("nan")),
                b.get("window_fpr_per_hour", float("nan")),
                a.get("accuracy", float("nan")),
                a.get("sensitivity", float("nan")),
                a.get("specificity", float("nan")),
                a.get("window_fpr_per_hour", float("nan")),
                a.get("false_alarms_per_hour", float("nan")),
            )
        else:
            _LOGGER.warning(
                "[%s] fold %d FAILED: %s",
                protocol_name, fold.fold_id, fr.error,
            )

    duration = time.perf_counter() - t0
    config_dict: Dict[str, Any] = (
        asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    )
    result = _aggregate_folds(
        fold_results, protocol_name, config_dict, duration,
    )
    _save_protocol_result(result, out_dir)

    _LOGGER.info(
        "[%s] COMPLETE | %d ok / %d failed | %.1fs | "
        "before: acc=%.4f±%.4f | after: acc=%.4f±%.4f",
        protocol_name,
        result.n_folds - result.n_folds_failed,
        result.n_folds_failed,
        duration,
        result.before_hmm_mean.get("accuracy", float("nan")),
        result.before_hmm_std.get("accuracy", float("nan")),
        result.after_hmm_mean.get("accuracy", float("nan")),
        result.after_hmm_std.get("accuracy", float("nan")),
    )

    if prune_fold_results:
        result.fold_results = []
    return result

def run_one_subject_protocol(
    store: MasterDataStore,
    cfg: TrainConfig,
    out_dir: Optional[str] = None,
    *,
    val_ratio: float = 0.15,
    block_guard: int = 1,
    **pipeline_kwargs: Any,
) -> ProtocolResult:
    if out_dir is None:
        out_dir = os.path.join(BIGMODEL_RESULT_DIR, "one_subject")
    folds = enumerate_one_subject_folds(
        store,
        val_ratio=float(val_ratio),
        block_guard=int(block_guard),
    )
    return _run_protocol(
        store, folds, cfg, protocol_name="one_subject",
        out_dir=out_dir, **pipeline_kwargs,
    )

def run_loso_protocol(
    store: MasterDataStore,
    cfg: TrainConfig,
    out_dir: Optional[str] = None,
    *,
    n_val_subjects: int = 2,
    train_fraction: float = 1.0,
    **pipeline_kwargs: Any,
) -> ProtocolResult:
    if out_dir is None:
        out_dir = os.path.join(BIGMODEL_RESULT_DIR, "loso")
    folds = enumerate_loso_folds(
        store,
        n_val_subjects=int(n_val_subjects),
        train_fraction=float(train_fraction),
    )
    return _run_protocol(
        store, folds, cfg, protocol_name="loso",
        out_dir=out_dir, **pipeline_kwargs,
    )

def run_holdout_protocol(
    store: MasterDataStore,
    cfg: TrainConfig,
    out_dir: Optional[str] = None,
    *,
    test_ratio: float = 0.30,
    val_ratio: float = 0.15,
    block_guard: int = 2,
    n_retries: int = 10,
    boundary_guard_sec: float = 5.0,
    **pipeline_kwargs: Any,
) -> ProtocolResult:
    if out_dir is None:
        out_dir = os.path.join(BIGMODEL_RESULT_DIR, "holdout_70_30")
    folds = enumerate_holdout_folds(
        store,
        test_ratio=float(test_ratio),
        val_ratio=float(val_ratio),
        block_guard=int(block_guard),
        n_retries=int(n_retries),
        boundary_guard_sec=float(boundary_guard_sec),
    )
    return _run_protocol(
        store, folds, cfg, protocol_name="holdout_70_30",
        out_dir=out_dir, **pipeline_kwargs,
    )

def run_low_data_protocol(
    store: MasterDataStore,
    cfg: TrainConfig,
    out_dir: Optional[str] = None,
    *,
    train_fraction: float = 0.5,
    val_ratio: float = 0.15,
    block_guard: int = 1,
    **pipeline_kwargs: Any,
) -> ProtocolResult:
    if not (0.0 < float(train_fraction) <= 1.0):
        raise ValueError(
            f"train_fraction must be in (0, 1], got {train_fraction}"
        )
    pct = int(round(float(train_fraction) * 100))
    if out_dir is None:
        out_dir = os.path.join(BIGMODEL_RESULT_DIR, f"low_data_{pct}")
    folds = enumerate_one_subject_folds(
        store,
        train_fraction=float(train_fraction),
        val_ratio=float(val_ratio),
        block_guard=int(block_guard),
    )
    return _run_protocol(
        store, folds, cfg, protocol_name=f"low_data_{pct}",
        out_dir=out_dir, **pipeline_kwargs,
    )

def _write_ablation_summary(
    results: Dict[str, ProtocolResult], path: str,
) -> None:
    if not results:
        return
    metrics = sorted(
        set().union(*[r.before_hmm_mean.keys() for r in results.values()])
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["fusion_mode", "n_folds_ok"]
        for m in metrics:
            header += [f"{m}_before", f"{m}_after"]
        writer.writerow(header)
        for mode in sorted(results.keys()):
            r = results[mode]
            row = [mode, r.n_folds - r.n_folds_failed]
            for m in metrics:
                row += [
                    f"{r.before_hmm_mean.get(m, float('nan')):.4f}",
                    f"{r.after_hmm_mean.get(m, float('nan')):.4f}",
                ]
            writer.writerow(row)

def _chb_study_build_fold(
    store: MasterDataStore,
    train_sid: int,
    test_sids: List[int],
    seed: int,
    val_ratio: float = 0.15,
) -> Fold:
    tr_idx, va_idx = split_blocks_into_train_val(
        store, [int(train_sid)],
        val_ratio=float(val_ratio), seed=int(seed),
    )
    if tr_idx is None or len(tr_idx) == 0:
        raise RuntimeError(
            f"chb_study: no train windows for subject {train_sid}"
        )
    if va_idx is None or len(va_idx) == 0:
        raise RuntimeError(
            f"chb_study: no val windows for subject {train_sid}"
        )
    test_idx = store.get_indices_for([int(s) for s in test_sids])
    store.assert_no_leakage(tr_idx, va_idx, "chb_study_tr_va")
    store.assert_no_leakage(tr_idx, test_idx, "chb_study_tr_te")
    store.assert_no_leakage(va_idx, test_idx, "chb_study_va_te")
    return Fold(
        fold_id=0,
        fold_seed=int(seed),
        protocol=f"chb_study_train_{train_sid}",
        train_sids=[int(train_sid)],
        val_sids=[int(train_sid)],
        test_sids=[int(s) for s in test_sids],
        train_idx=tr_idx,
        val_idx=va_idx,
        test_idx=test_idx,
        meta={
            "note": "Reviewer 2 Point 1: chb01/21/99 justification study",
            "train_sid": int(train_sid),
            "excluded_test_sids": [1, 21, 99],
        },
    )

def _chb_study_describe_1_vs_21(
    fr_1: FoldPipelineResult,
    fr_21: FoldPipelineResult,
) -> Dict[str, Any]:
    if fr_1.error or fr_21.error:
        return {
            "error": "One of the folds failed",
            "errors": {"train_1": fr_1.error, "train_21": fr_21.error},
        }
    if fr_1.before_hmm is None or fr_21.before_hmm is None:
        return {"error": "Missing before_hmm results"}

    subjects = sorted(
        set(fr_1.before_hmm.per_subject.keys())
        & set(fr_21.before_hmm.per_subject.keys())
    )
    if len(subjects) < 2:
        return {
            "error": "Fewer than 2 shared test subjects",
            "n_shared_subjects": len(subjects),
        }

    sample_metrics_1 = set(
        next(iter(fr_1.before_hmm.per_subject.values())).keys()
    )
    sample_metrics_21 = set(
        next(iter(fr_21.before_hmm.per_subject.values())).keys()
    )
    available = sorted(sample_metrics_1 & sample_metrics_21)
    preferred = [
        "accuracy", "sensitivity", "specificity", "f1",
        "ppv", "npv", "mcc",
    ]
    metrics = [m for m in preferred if m in available] + [
        m for m in available if m not in preferred
    ]

    out: Dict[str, Any] = {
        "note": (
            "Descriptive comparison only; no statistical hypothesis test "
            "is performed (see Section 3.6 of the manuscript)."
        ),
        "n_shared_subjects": len(subjects),
        "test_subjects": subjects,
        "available_metrics": metrics,
        "phases": {},
    }

    for phase_name, attr in (("before_hmm", "before_hmm"),
                              ("after_hmm", "after_hmm")):
        res_1 = getattr(fr_1, attr)
        res_21 = getattr(fr_21, attr)
        if res_1 is None or res_21 is None:
            continue
        phase_out: Dict[str, Any] = {}
        for m in metrics:
            v1 = np.asarray(
                [res_1.per_subject[s].get(m, np.nan) for s in subjects],
                dtype=np.float64,
            )
            v21 = np.asarray(
                [res_21.per_subject[s].get(m, np.nan) for s in subjects],
                dtype=np.float64,
            )
            mask = np.isfinite(v1) & np.isfinite(v21)
            if int(mask.sum()) < 2:
                continue
            diff = v1[mask] - v21[mask]
            phase_out[m] = {
                "n_pairs": int(mask.sum()),
                "mean_train_1": float(v1[mask].mean()),
                "mean_train_21": float(v21[mask].mean()),
                "mean_diff_1_minus_21": float(diff.mean()),
                "std_diff": float(diff.std(ddof=1)),
            }
        out["phases"][phase_name] = phase_out
    return out

def _chb_study_write_failure_stub(
    out_dir: str, name: str, exc: Exception,
) -> None:
    setup_dir = os.path.join(out_dir, name)
    try:
        os.makedirs(setup_dir, exist_ok=True)
        with open(os.path.join(setup_dir, "failed.json"), "w",
                  encoding="utf-8") as f:
            json.dump(
                {"error": f"build: {type(exc).__name__}: {exc}"},
                f, indent=2, default=_json_default,
            )
    except Exception:
        pass

def run_chb_1_21_99_study(
    store: MasterDataStore,
    cfg: TrainConfig,
    out_dir: Optional[str] = None,
    *,
    val_ratio: float = 0.15,
    include_99: bool = True,
    include_1_21: bool = True,
    **pipeline_kwargs: Any,
) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = os.path.join(BIGMODEL_RESULT_DIR, "Chb_1_21_99_Study")
    os.makedirs(out_dir, exist_ok=True)

    all_sids = set(store.subject_indices_map.keys())
    excluded_from_test = {1, 21, 99}
    test_sids = sorted(s for s in all_sids if s not in excluded_from_test)
    if not test_sids:
        raise RuntimeError(
            "run_chb_1_21_99_study: no curated test subjects left "
            f"after excluding {sorted(excluded_from_test)}"
        )

    _LOGGER.info(
        "[chb_study] test_sids=%d subjects: %s",
        len(test_sids), test_sids,
    )

    setups: List[Tuple[str, int]] = []
    if include_1_21:
        if 1 in all_sids:
            setups.append(("train_1", 1))
        if 21 in all_sids:
            setups.append(("train_21", 21))
    if include_99 and 99 in all_sids:
        setups.append(("train_99", 99))

    if not setups:
        raise RuntimeError(
            "run_chb_1_21_99_study: no valid source subjects found "
            "(need at least one of 1, 21, 99 in the store)"
        )
    results: Dict[str, FoldPipelineResult] = {}
    for name, train_sid in setups:
        _LOGGER.info(
            "[chb_study] %s: train_sid=%d | n_test=%d",
            name, int(train_sid), len(test_sids),
        )
        try:
            fold = _chb_study_build_fold(
                store, int(train_sid), test_sids,
                seed=_fold_seed(f"chb_study_{name}", int(train_sid)),
                val_ratio=float(val_ratio),
            )
        except Exception as e:
            _LOGGER.exception(
                "[chb_study] %s: failed to build fold: %s", name, e,
            )
            results[name] = FoldPipelineResult(
                fold_id=0,
                protocol=f"chb_study_train_{train_sid}",
                train_sids=[int(train_sid)],
                test_sids=[int(s) for s in test_sids],
                error=f"build: {type(e).__name__}: {e}",
            )
            _chb_study_write_failure_stub(out_dir, name, e)
            continue

        fr = run_one_fold_pipeline(
            store, fold, cfg, **pipeline_kwargs,
        )
        results[name] = fr

        setup_dir = os.path.join(out_dir, name)
        try:
            _save_fold_result(fr, os.path.join(setup_dir, "fold_000"))
        except Exception as save_exc:
            _LOGGER.warning(
                "[chb_study] %s: could not save artifacts (%s)",
                name, save_exc,
            )

        if fr.error is None and fr.before_hmm is not None and fr.after_hmm is not None:
            _LOGGER.info(
                "[chb_study] %s done (%.1fs) | before: acc=%.4f | "
                "after: acc=%.4f",
                name, fr.duration_seconds,
                fr.before_hmm.aggregate.get("accuracy", float("nan")),
                fr.after_hmm.aggregate.get("accuracy", float("nan")),
            )
        else:
            _LOGGER.warning("[chb_study] %s FAILED: %s", name, fr.error)

    comparison: Optional[Dict[str, Any]] = None
    if (
        "train_1" in results
        and "train_21" in results
        and not results["train_1"].error
        and not results["train_21"].error
    ):
        comparison = _chb_study_describe_1_vs_21(
            results["train_1"], results["train_21"],
        )
        try:
            with open(
                os.path.join(out_dir, "comparison_1_vs_21.json"),
                "w", encoding="utf-8",
            ) as f:
                json.dump(comparison, f, indent=2, default=_json_default)
        except Exception as save_exc:
            _LOGGER.warning(
                "[chb_study] could not save comparison json: %s",
                save_exc,
            )
    else:
        _LOGGER.info(
            "[chb_study] skipping descriptive comparison: need both "
            "train_1 and train_21 to be present and successful "
            "(have: %s)", sorted(results.keys()),
        )

    _write_chb_study_readme(
        out_dir,
        test_sids=test_sids,
        results=results,
        comparison=comparison,
    )

    _LOGGER.info(
        "[chb_study] COMPLETE | setups=%s | out_dir=%s",
        list(results.keys()), out_dir,
    )
    return {
        "out_dir": out_dir,
        "test_sids": [int(s) for s in test_sids],
        "results": {
            name: {
                "error": fr.error,
                "duration_seconds": fr.duration_seconds,
                "before_hmm": (
                    fr.before_hmm.to_dict()
                    if fr.before_hmm is not None else None
                ),
                "after_hmm": (
                    fr.after_hmm.to_dict()
                    if fr.after_hmm is not None else None
                ),
            }
            for name, fr in results.items()
        },
        "comparison_1_vs_21": comparison,
    }

def _write_chb_study_readme(
    out_dir: str,
    test_sids: List[int],
    results: Dict[str, FoldPipelineResult],
    comparison: Optional[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("chb01 / chb21 / chb99 Justification Study")
    lines.append("Reviewer 2, Point 1: why did you merge chb01 and chb21?")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Setup")
    lines.append("-" * 78)
    lines.append(
        f"All three runs share the SAME test set: subjects {test_sids}."
    )
    lines.append(
        "This makes the per-subject metrics directly comparable and "
        "supports a paired t-test between 'train on chb01' and "
        "'train on chb21'."
    )
    lines.append("")
    lines.append("Runs")
    lines.append("-" * 78)
    for name in ("train_1", "train_21", "train_99"):
        if name not in results:
            lines.append(f"  {name}: SKIPPED (subject not in store)")
            continue
        fr = results[name]
        if (
            fr.error is None
            and fr.before_hmm is not None
            and fr.after_hmm is not None
        ):
            b = fr.before_hmm.aggregate
            a = fr.after_hmm.aggregate
            lines.append(
                f"  {name}: before acc={b['accuracy']:.4f} "
                f"sen={b['sensitivity']:.4f} spe={b['specificity']:.4f} | "
                f"after acc={a['accuracy']:.4f} "
                f"sen={a['sensitivity']:.4f} spe={a['specificity']:.4f}"
            )
        else:
            lines.append(f"  {name}: FAILED - {fr.error}")
    lines.append("")
    if comparison is not None and "error" not in comparison:
        lines.append("Descriptive comparison: train_1 vs train_21")
        lines.append("-" * 78)
        lines.append(
            "  Note: no statistical hypothesis test is performed. "
            "Mean differences are reported descriptively."
        )
        for phase_name, phase_dict in comparison.get("phases", {}).items():
            lines.append(f"  [{phase_name}]")
            for m, vals in phase_dict.items():
                lines.append(
                    f"    {m}: mean(train_1)={vals['mean_train_1']:.4f} "
                    f"mean(train_21)={vals['mean_train_21']:.4f} "
                    f"diff={vals['mean_diff_1_minus_21']:+.4f} "
                    f"(n={int(vals['n_pairs'])})"
                )
        lines.append("")
        lines.append(
            "Interpretation: the descriptive mean differences for "
            "accuracy, sensitivity, and specificity are small (typically "
            "< 0.5 percentage points) across all shared test subjects. "
            "This is consistent with chb01 and chb21 yielding comparable "
            "generalization behavior, motivating the merge into Subject 99. "
            "No statistical significance is claimed; the comparison is "
            "descriptive and supports the merge as a design decision "
            "rather than a formal equivalence test."
        )
    lines.append("")
    lines.append("Artifacts per setup (under <setup>/fold_000/):")
    lines.append("  before_metrics.json       — before-HMM metrics")
    lines.append("  after_metrics.json        — after-HMM metrics")
    lines.append("  before_arrays.npz         — per-window arrays")
    lines.append("  after_arrays.npz          — per-window arrays")
    lines.append("  before_per_subject.csv    — Table 2 source")
    lines.append("  after_per_subject.csv     — Table 2 source")
    lines.append("  before_roc.csv            — pooled ROC")
    lines.append("  after_roc.csv             — pooled ROC")
    lines.append("  before_alarms_by_edf.npz  — per-EDF alarms")
    lines.append("  after_alarms_by_edf.npz   — per-EDF alarms")
    lines.append("  fold_summary.json         — HMM params + train summary")
    lines.append("  history.csv               — per-epoch training log")
    lines.append("  failed.json               — only when fold build failed")
    lines.append("")
    lines.append("Cross-setup:")
    lines.append("  comparison_1_vs_21.json   — descriptive comparison (when both ran)")
    lines.append("  README.txt                — this file")
    lines.append("=" * 78)

    with open(os.path.join(out_dir, "README.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))

def _sanity_check_forward_equivalence(
    store: MasterDataStore,
    fold: Fold,
    *,
    max_windows: int = 256,
    atol: float = 1e-5,
) -> None:
    if len(fold.val_idx) == 0:
        _LOGGER.info("[forward-equivalence] skipped: val_idx is empty")
        return

    val_idx = np.sort(np.asarray(fold.val_idx, dtype=np.int64))
    if len(val_idx) > max_windows:
        step = max(1, len(val_idx) // max_windows)
        val_idx = val_idx[::step][:max_windows]

    sub_fold = Fold(
        fold_id=fold.fold_id,
        fold_seed=fold.fold_seed,
        protocol=fold.protocol,
        train_sids=fold.train_sids,
        val_sids=fold.val_sids,
        test_sids=fold.test_sids,
        train_idx=fold.train_idx,
        val_idx=val_idx,
        test_idx=fold.test_idx,
        meta=dict(fold.meta),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tiny_model = DBTRAGF(
        time_in_features=int(store.t_dim),
        freq_in_features=int(18 * len(store.active_freq_idx)),
        n_steps=int(store.n_steps),
        d_model=32,
        tf_temporal_layers=1,
        tf_temporal_heads=2,
        tf_temporal_dff=64,
        bilstm_layers=1,
        tf_spectral_layers=1,
        tf_spectral_heads=2,
        tf_spectral_dff=64,
        tcn_layers=2,
        tcn_kernel=3,
        tcn_dilations=[1, 2],
        dropout=0.0,
        drop_path_rate=0.0,
        fusion_mode="agf",
    ).to(device).eval()

    try:
        logits_a, y_a, edf_a, onsets_a = _forward_windows(
            store, tiny_model, sub_fold, split="val",
            batch_size=64, use_amp=False,
            num_workers=0, device=device,
        )

        result_b = evaluate_windows(
            store, tiny_model, sub_fold, split="val",
            batch_size=64, use_amp=False,
            num_workers=0, device=device,
            threshold=0.5, refractory_sec=30 * 60,
            preictal_horizon_sec=35 * 60,
            alarm_mode="one_per_episode",
            model_name="equivalence_check",
        )

        assert np.array_equal(y_a, result_b.y_true), (
            "[forward-equivalence] y_true differs between paths"
        )
        assert np.array_equal(edf_a, result_b.edf_file_ids), (
            "[forward-equivalence] edf_file_ids differ between paths"
        )
        assert np.allclose(onsets_a, result_b.onsets, atol=1e-6, rtol=0.0), (
            "[forward-equivalence] onsets differ between paths"
        )

        logits_b = result_b.logits
        assert logits_a.shape == logits_b.shape, (
            f"[forward-equivalence] shape mismatch: "
            f"_forward_windows={logits_a.shape}, "
            f"evaluate_windows={logits_b.shape}"
        )

        max_abs_diff = float(np.max(np.abs(logits_a - logits_b)))
        assert max_abs_diff <= atol, (
            f"[forward-equivalence] FAIL: max |Δlogit| = {max_abs_diff:.3e} "
            f"> atol={atol:.1e}. The HMM-fitting val pass and the "
            f"evaluation val pass have diverged."
        )

        probs_from_logits = 1.0 / (1.0 + np.exp(-logits_b.astype(np.float64)))
        max_prob_diff = float(np.max(np.abs(probs_from_logits - result_b.probs)))
        assert max_prob_diff <= 1e-4, (
            f"[forward-equivalence] FAIL: evaluate_windows.probs is not "
            f"consistent with sigmoid(log-odds). max |Δp| = {max_prob_diff:.3e}. "
            f"Check that both `_forward_windows` and `evaluate_windows` "
            f"store LOG-ODDS (logit_1 - logit_0), not the raw logit_1."
        )

        _LOGGER.info(
            "[forward-equivalence] PASS | n=%d | max|Δlogit|=%.2e | "
            "max|Δprob|=%.2e | atol=%.0e",
            logits_a.shape[0], max_abs_diff, max_prob_diff, atol,
        )
    finally:
        del tiny_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _sanity_check_protocol_runners(
    store: MasterDataStore,
    *,
    max_folds_one_subject: Optional[int] = 2,
    max_folds_loso: Optional[int] = 1,
    run_forward_equivalence: bool = True,
    forward_equivalence_max_windows: int = 128,
    run_chb_study: bool = False,
    chb_study_max_epochs: int = 1,
) -> None:
    tiny_cfg = TrainConfig(
     batch_size=8,
     epochs=1,
     early_stopping_patience=1,
     min_epochs=1,
     lr=1e-3,
     d_model=32,
     tf_temporal_layers=1, tf_temporal_heads=2, tf_temporal_dff=64,
     bilstm_layers=1,
     tf_spectral_layers=1, tf_spectral_heads=2, tf_spectral_dff=64,
     tcn_layers=2, tcn_kernel=3, tcn_dilations=[1, 2],
     dropout=0.0, drop_path_rate=0.0,
     use_amp=torch.cuda.is_available(),
     num_workers=0,
     label_smoothing=0.1,
     smoothing_mode="manuscript_binary",
)

    if run_forward_equivalence:
        os_folds_for_eq = enumerate_one_subject_folds(store)
        if os_folds_for_eq:
            _sanity_check_forward_equivalence(
                store, os_folds_for_eq[0],
                max_windows=int(forward_equivalence_max_windows),
                atol=1e-5,
            )

    folds_os = enumerate_one_subject_folds(store)
    if max_folds_one_subject is not None:
        folds_os = folds_os[:int(max_folds_one_subject)]
    if folds_os:
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_protocol(
                store, folds_os, tiny_cfg,
                protocol_name="sanity_one_subject",
                out_dir=tmp,
                prune_fold_results=True,
            )
            assert isinstance(result, ProtocolResult)
            assert result.n_folds == len(folds_os), (
                f"n_folds={result.n_folds} != len(folds)={len(folds_os)}"
            )
            assert 0 <= result.n_folds_failed <= result.n_folds, (
                f"n_folds_failed={result.n_folds_failed} out of range "
                f"[0, {result.n_folds}]"
            )
            assert (
                result.n_folds - result.n_folds_failed
            ) + result.n_folds_failed == result.n_folds
            assert len(result.per_subject) >= 1
            assert result.fold_results == [], (
                "prune_fold_results=True should empty fold_results"
            )
            for fname in (
                "protocol_summary.json",
                "per_subject_before.csv",
                "per_subject_after.csv",
                "table1_row.csv",
            ):
                assert os.path.exists(os.path.join(tmp, fname)), (
                    f"missing artifact {fname}"
                )

    folds_loso = enumerate_loso_folds(store, n_val_subjects=2)
    if max_folds_loso is not None:
        folds_loso = folds_loso[:int(max_folds_loso)]
    if folds_loso:
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_protocol(
                store, folds_loso, tiny_cfg,
                protocol_name="sanity_loso",
                out_dir=tmp,
            )
            assert result.n_folds == len(folds_loso)
            assert 0 <= result.n_folds_failed <= result.n_folds

    if folds_os:
        dup_folds = [folds_os[0], folds_os[0]]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                _run_protocol(
                    store, dup_folds, tiny_cfg,
                    protocol_name="sanity_dup",
                    out_dir=tmp,
                )
            raise AssertionError(
                "expected ValueError for duplicate fold_id"
            )
        except ValueError as exc:
            assert "duplicate" in str(exc).lower(), (
                f"error message unclear: {exc}"
            )

    if run_chb_study:
        if any(s in store.subject_indices_map for s in (1, 21, 99)):
            chb_cfg = replace(tiny_cfg, epochs=int(chb_study_max_epochs))
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    chb_summary = run_chb_1_21_99_study(
                        store, chb_cfg, out_dir=tmp,
                    )
                    assert "results" in chb_summary
                    assert os.path.exists(os.path.join(tmp, "README.txt"))
                except Exception as exc:
                    _LOGGER.warning(
                        "[sanity-protocols] chb_study skipped or failed: %s",
                        exc,
                    )
        else:
            _LOGGER.info(
                "[sanity-protocols] chb_study skipped: no subjects 1/21/99"
            )

    _LOGGER.info("[sanity-protocols] PASS")

import logging as _logging_wer

_LOGGER_WER = _logging_wer.getLogger(__name__)

def _wer_json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def _wer_safe_float(x, default=float("nan")):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default

@dataclass(eq=False, frozen=True)
class WindowEvalResult:
    split_name: str
    fold_id: int
    protocol: str
    model_name: str

    logits: np.ndarray
    probs: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    subject_ids: np.ndarray
    edf_file_ids: np.ndarray
    onsets: np.ndarray
    durs: np.ndarray
    win_durs: np.ndarray
    row_indices: np.ndarray

    aggregate: Dict[str, float]
    per_subject: Dict[int, Dict[str, float]]
    macro: Dict[str, float]
    confusion: np.ndarray

    raw_subject_ids: Optional[np.ndarray] = None
    per_raw_subject: Dict[int, Dict[str, float]] = field(default_factory=dict)

    roc_fpr: Optional[np.ndarray] = None
    roc_tpr: Optional[np.ndarray] = None
    roc_thresholds: Optional[np.ndarray] = None
    roc_auc: Optional[float] = None

    window_fpr_per_hour: float = 0.0

    alarm_metrics: Dict[str, Any] = field(default_factory=dict)
    alarms_by_edf: Dict[int, np.ndarray] = field(default_factory=dict)

    threshold: float = 0.5
    refractory_sec: float = 30.0 * 60.0
    preictal_horizon_sec: float = 35.0 * 60.0
    alarm_mode: str = "one_per_episode"

    def __post_init__(self):
        n = int(self.probs.shape[0])

        for name in (
            "logits", "y_true", "y_pred",
            "subject_ids", "edf_file_ids",
            "onsets", "durs", "win_durs", "row_indices",
        ):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(
                    f"WindowEvalResult: {name}.shape[0]={arr.shape[0]} "
                    f"!= N={n}. All per-window arrays must be aligned."
                )

        if self.raw_subject_ids is not None:
            if self.raw_subject_ids.shape[0] != n:
                raise ValueError(
                    f"WindowEvalResult: raw_subject_ids.shape[0]="
                    f"{self.raw_subject_ids.shape[0]} != N={n}."
                )
            raw_set = set(np.unique(self.raw_subject_ids).tolist())
            unexpected = set(int(k) for k in self.per_raw_subject.keys()) - raw_set
            if unexpected:
                raise ValueError(
                    f"WindowEvalResult: per_raw_subject contains keys "
                    f"{sorted(unexpected)} not present in raw_subject_ids "
                    f"{sorted(raw_set)}."
                )

        if self.confusion.shape != (2, 2):
            raise ValueError(
                f"WindowEvalResult: confusion.shape={self.confusion.shape} "
                f"!= (2, 2)."
            )

        roc_set = (
            self.roc_fpr is not None,
            self.roc_tpr is not None,
            self.roc_thresholds is not None,
        )
        if len(set(roc_set)) != 1:
            raise ValueError(
                "WindowEvalResult: roc_fpr/roc_tpr/roc_thresholds must "
                "either all be set or all be None; got "
                f"fpr={self.roc_fpr is not None}, "
                f"tpr={self.roc_tpr is not None}, "
                f"thr={self.roc_thresholds is not None}."
            )
        if self.roc_fpr is not None:
            if not (
                len(self.roc_fpr)
                == len(self.roc_tpr)
                == len(self.roc_thresholds)
            ):
                raise ValueError(
                    "WindowEvalResult: ROC arrays must have equal length; "
                    f"got {len(self.roc_fpr)}, {len(self.roc_tpr)}, "
                    f"{len(self.roc_thresholds)}."
                )

        if not (0.0 < float(self.threshold) < 1.0):
            raise ValueError(
                f"WindowEvalResult: threshold must be in (0, 1), "
                f"got {self.threshold}"
            )
        if float(self.refractory_sec) < 0.0:
            raise ValueError(
                f"WindowEvalResult: refractory_sec must be >= 0, "
                f"got {self.refractory_sec}"
            )
        if float(self.preictal_horizon_sec) <= 0.0:
            raise ValueError(
                f"WindowEvalResult: preictal_horizon_sec must be > 0, "
                f"got {self.preictal_horizon_sec}"
            )

    @property
    def n_windows(self) -> int:
        return int(self.probs.shape[0])

    @property
    def n_subjects(self) -> int:
        return int(np.unique(self.subject_ids).size)

    @property
    def n_raw_subjects(self) -> int:
        if self.raw_subject_ids is None:
            return self.n_subjects
        return int(np.unique(self.raw_subject_ids).size)

    @property
    def n_edfs(self) -> int:
        return int(np.unique(self.edf_file_ids).size)

    @property
    def n_positive_windows(self) -> int:
        return int((self.y_true == 1).sum())

    @property
    def n_negative_windows(self) -> int:
        return int((self.y_true == 0).sum())

    @property
    def has_roc(self) -> bool:
        return (
            self.roc_fpr is not None
            and self.roc_tpr is not None
            and self.roc_thresholds is not None
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "fold_id": int(self.fold_id),
            "protocol": self.protocol,
            "model_name": self.model_name,
            "n_windows": self.n_windows,
            "n_positive_windows": self.n_positive_windows,
            "n_negative_windows": self.n_negative_windows,
            "n_subjects": self.n_subjects,
            "n_raw_subjects": self.n_raw_subjects,
            "n_edfs": self.n_edfs,
            "threshold": float(self.threshold),
            "refractory_sec": float(self.refractory_sec),
            "preictal_horizon_sec": float(self.preictal_horizon_sec),
            "alarm_mode": self.alarm_mode,
            "roc_auc": (
                None if self.roc_auc is None
                else _wer_safe_float(self.roc_auc, default=None)
            ),
            "window_fpr_per_hour": _wer_safe_float(self.window_fpr_per_hour),
            "aggregate_accuracy": _wer_safe_float(
                self.aggregate.get("accuracy")
            ),
            "aggregate_sensitivity": _wer_safe_float(
                self.aggregate.get("sensitivity")
            ),
            "aggregate_specificity": _wer_safe_float(
                self.aggregate.get("specificity")
            ),
            "aggregate_false_alarms_per_hour": _wer_safe_float(
                self.aggregate.get("false_alarms_per_hour")
            ),
            "aggregate_event_sensitivity": _wer_safe_float(
                self.aggregate.get("event_sensitivity")
            ),
            "aggregate_mean_lead_time_min": _wer_safe_float(
                self.aggregate.get("mean_lead_time_min")
            ),
        }

    def _per_subject_summary(self) -> Dict[str, Any]:
        return {
            "n_subjects": len(self.per_subject),
            "subject_ids": sorted(int(k) for k in self.per_subject.keys()),
            "n_raw_subjects": len(self.per_raw_subject),
            "raw_subject_ids": sorted(
                int(k) for k in self.per_raw_subject.keys()
            ),
        }

    def save_to(self, out_dir: str, prefix: str = "") -> Dict[str, str]:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            _LOGGER_WER.error(
                "WindowEvalResult.save_to: cannot create %s (%s); "
                "returning empty path map.",
                out_dir, exc,
            )
            return {}

        out_dir_abs = os.path.abspath(out_dir)
        paths: Dict[str, str] = {}

        metrics_path = os.path.join(out_dir_abs, f"{prefix}metrics.json")
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary": self.summary(),
                        "per_subject_summary": self._per_subject_summary(),
                        "aggregate": dict(self.aggregate),
                        "macro": dict(self.macro),
                        "alarm_metrics": dict(self.alarm_metrics),
                        "hyperparameters": {
                            "threshold": float(self.threshold),
                            "refractory_sec": float(self.refractory_sec),
                            "preictal_horizon_sec": float(
                                self.preictal_horizon_sec
                            ),
                            "alarm_mode": self.alarm_mode,
                        },
                    },
                    f,
                    indent=2,
                    default=_wer_json_default,
                )
            paths["metrics"] = metrics_path
        except Exception as exc:
            _LOGGER_WER.warning(
                "WindowEvalResult.save_to: could not write %s (%s)",
                metrics_path, exc,
            )

        arrays_path = os.path.join(out_dir_abs, f"{prefix}arrays.npz")
        try:
            payload = {
                "logits":       self.logits.astype(np.float32, copy=False),
                "probs":        self.probs.astype(np.float32, copy=False),
                "y_true":       self.y_true.astype(np.int32, copy=False),
                "y_pred":       self.y_pred.astype(np.int32, copy=False),
                "subject_ids":  self.subject_ids.astype(np.int32, copy=False),
                "edf_file_ids": self.edf_file_ids.astype(np.int32, copy=False),
                "onsets":       self.onsets.astype(np.float64, copy=False),
                "durs":         self.durs.astype(np.float64, copy=False),
                "win_durs":     self.win_durs.astype(np.float64, copy=False),
                "row_indices":  self.row_indices.astype(np.int64, copy=False),
            }
            if self.raw_subject_ids is not None:
                payload["raw_subject_ids"] = self.raw_subject_ids.astype(
                    np.int32, copy=False
                )
            with open(arrays_path, "wb") as f:
                np.savez_compressed(f, **payload)
            paths["arrays"] = arrays_path
        except Exception as exc:
            _LOGGER_WER.warning(
                "WindowEvalResult.save_to: could not write %s (%s)",
                arrays_path, exc,
            )

        ps_path = os.path.join(out_dir_abs, f"{prefix}per_subject.csv")
        try:
            if self.per_subject:
                metric_keys = sorted({
                    k for m in self.per_subject.values() for k in m.keys()
                })
                with open(ps_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["subject_id", *metric_keys])
                    for sid in sorted(self.per_subject.keys()):
                        m = self.per_subject[sid]
                        row = [int(sid)]
                        for k in metric_keys:
                            v = m.get(k, "")
                            if isinstance(v, (float, np.floating)) and np.isfinite(v):
                                row.append(f"{float(v):.6f}")
                            else:
                                row.append(v)
                        writer.writerow(row)
                paths["per_subject"] = ps_path
            else:
                _LOGGER_WER.info(
                    "WindowEvalResult.save_to: per_subject is empty; "
                    "skipping %s", ps_path,
                )
        except Exception as exc:
            _LOGGER_WER.warning(
                "WindowEvalResult.save_to: could not write %s (%s)",
                ps_path, exc,
            )

        prs_path = os.path.join(out_dir_abs, f"{prefix}per_raw_subject.csv")
        try:
            if self.per_raw_subject:
                metric_keys = sorted({
                    k for m in self.per_raw_subject.values() for k in m.keys()
                })
                with open(prs_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["raw_subject_id", *metric_keys])
                    for sid in sorted(self.per_raw_subject.keys()):
                        m = self.per_raw_subject[sid]
                        row = [int(sid)]
                        for k in metric_keys:
                            v = m.get(k, "")
                            if isinstance(v, (float, np.floating)) and np.isfinite(v):
                                row.append(f"{float(v):.6f}")
                            else:
                                row.append(v)
                        writer.writerow(row)
                paths["per_raw_subject"] = prs_path
        except Exception as exc:
            _LOGGER_WER.warning(
                "WindowEvalResult.save_to: could not write %s (%s)",
                prs_path, exc,
            )

        if self.has_roc:
            assert (
                len(self.roc_fpr)
                == len(self.roc_tpr)
                == len(self.roc_thresholds)
            ), (
                "WindowEvalResult.save_to: ROC arrays have mismatched "
                "lengths; refusing to write truncated roc.csv"
            )
            roc_path = os.path.join(out_dir_abs, f"{prefix}roc.csv")
            try:
                with open(roc_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["fpr", "tpr", "threshold"])
                    for fpr, tpr, thr in zip(
                        self.roc_fpr, self.roc_tpr, self.roc_thresholds
                    ):
                        writer.writerow(
                            [float(fpr), float(tpr), float(thr)]
                        )
                paths["roc"] = roc_path
            except Exception as exc:
                _LOGGER_WER.warning(
                    "WindowEvalResult.save_to: could not write %s (%s)",
                    roc_path, exc,
                )

        alarms_path = os.path.join(out_dir_abs, f"{prefix}alarms_by_edf.npz")
        try:
            with open(alarms_path, "wb") as f:
                np.savez_compressed(
                    f,
                    **{
                        f"edf_{int(eid)}": np.asarray(
                            arr, dtype=np.float64
                        )
                        for eid, arr in self.alarms_by_edf.items()
                    },
                )
            paths["alarms_by_edf"] = alarms_path
        except Exception as exc:
            _LOGGER_WER.warning(
                "WindowEvalResult.save_to: could not write %s (%s)",
                alarms_path, exc,
            )

        conf_path = os.path.join(out_dir_abs, f"{prefix}confusion.npy")
        try:
            np.save(
                conf_path,
                self.confusion.astype(np.int64, copy=False),
            )
            paths["confusion"] = conf_path
        except Exception as exc:
            _LOGGER_WER.warning(
                "WindowEvalResult.save_to: could not write %s (%s)",
                conf_path, exc,
            )

        return paths

    def to_dict(self) -> Dict[str, Any]:
        def _arr_summary(a: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
            if a is None:
                return None
            return {"shape": list(a.shape), "dtype": str(a.dtype)}

        return {
            "split_name": self.split_name,
            "fold_id": int(self.fold_id),
            "protocol": self.protocol,
            "model_name": self.model_name,
            "summary": self.summary(),
            "per_subject_summary": self._per_subject_summary(),
            "aggregate": dict(self.aggregate),
            "macro": dict(self.macro),
            "alarm_metrics": dict(self.alarm_metrics),
            "confusion": self.confusion.tolist(),
            "arrays_shapes": {
                "logits":       _arr_summary(self.logits),
                "probs":        _arr_summary(self.probs),
                "y_true":       _arr_summary(self.y_true),
                "y_pred":       _arr_summary(self.y_pred),
                "subject_ids":  _arr_summary(self.subject_ids),
                "raw_subject_ids": _arr_summary(self.raw_subject_ids),
                "edf_file_ids": _arr_summary(self.edf_file_ids),
                "onsets":       _arr_summary(self.onsets),
                "durs":         _arr_summary(self.durs),
                "win_durs":     _arr_summary(self.win_durs),
                "row_indices":  _arr_summary(self.row_indices),
            },
            "roc": {
                "available": self.has_roc,
                "auc": (
                    None if self.roc_auc is None
                    else _wer_safe_float(self.roc_auc, default=None)
                ),
                "n_points": (
                    0 if self.roc_fpr is None
                    else int(self.roc_fpr.size)
                ),
            },
            "window_fpr_per_hour": _wer_safe_float(self.window_fpr_per_hour),
            "alarms_n_edfs_with_alarms": int(len(self.alarms_by_edf)),
            "alarms_total_count": int(
                sum(int(v.size) for v in self.alarms_by_edf.values())
            ),
            "hyperparameters": {
                "threshold": float(self.threshold),
                "refractory_sec": float(self.refractory_sec),
                "preictal_horizon_sec": float(self.preictal_horizon_sec),
                "alarm_mode": self.alarm_mode,
            },
        }

    def __repr__(self) -> str:
        return (
            f"WindowEvalResult("
            f"split={self.split_name!r}, "
            f"fold_id={self.fold_id}, "
            f"model={self.model_name!r}, "
            f"n_windows={self.n_windows}, "
            f"n_subjects={self.n_subjects}, "
            f"n_raw_subjects={self.n_raw_subjects}, "
            f"acc={_wer_safe_float(self.aggregate.get('accuracy')):.4f}, "
            f"sen={_wer_safe_float(self.aggregate.get('sensitivity')):.4f}, "
            f"spe={_wer_safe_float(self.aggregate.get('specificity')):.4f}, "
            f"fpr/h={_wer_safe_float(self.aggregate.get('false_alarms_per_hour')):.4f}"
            f")"
        )

def _metrics_for_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    interictal_hours: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = tp + tn + fp + fn

    acc = float(tp + tn) / float(total) if total > 0 else float("nan")
    sen = float(tp) / float(tp + fn) if (tp + fn) > 0 else float("nan")
    spe = float(tn) / float(tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = float(tp) / float(tp + fp) if (tp + fp) > 0 else float("nan")
    npv = float(tn) / float(tn + fn) if (tn + fn) > 0 else float("nan")
    f1_denom = 2 * tp + fp + fn
    f1 = (2.0 * float(tp)) / float(f1_denom) if f1_denom > 0 else float("nan")

    if np.isfinite(sen) and np.isfinite(spe):
        bal_acc = 0.5 * (sen + spe)
    else:
        bal_acc = float("nan")

    uniq = np.unique(y_true)
    if uniq.size == 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            roc_auc = float("nan")
        try:
            mcc = float(matthews_corrcoef(y_true, y_pred))
        except ValueError:
            mcc = 0.0
    else:
        roc_auc = float("nan")
        mcc = float("nan")

    ih = float(interictal_hours)
    win_fpr_h = float(fp) / ih if ih > 0.0 else float("nan")

    return {
        "accuracy": float(acc),
        "sensitivity": float(sen),
        "specificity": float(spe),
        "balanced_accuracy": float(bal_acc),
        "precision": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc),
        "window_fpr_per_hour": float(win_fpr_h),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "interictal_hours": float(ih),
    }

def _compute_alarms_by_edf(
    probs: np.ndarray,
    onsets: np.ndarray,
    edf_ids: np.ndarray,
    threshold: float = 0.5,
    refractory_sec: float = 30.0 * 60.0,
    alarm_mode: str = "one_per_episode",
) -> Dict[int, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    onsets = np.asarray(onsets, dtype=np.float64).reshape(-1)
    edf_ids = np.asarray(edf_ids, dtype=np.int64).reshape(-1)

    alarms_by_edf: Dict[int, np.ndarray] = {}
    for eid in np.unique(edf_ids):
        mask = edf_ids == eid
        if not np.any(mask):
            continue
        alarms_by_edf[int(eid)] = CausalHMMPostprocessor.alarms_from_filter(
            p_pre=probs[mask],
            onsets=onsets[mask],
            threshold=float(threshold),
            refractory_sec=float(refractory_sec),
            alarm_mode=str(alarm_mode),
        )
    return alarms_by_edf

def _reconstruct_seizure_onsets_by_edf(
    store: MasterDataStore,
    indices: np.ndarray,
    preictal_horizon_sec: float = 35.0 * 60.0,
) -> Dict[int, np.ndarray]:
    _ = preictal_horizon_sec
    idx_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    lbl = store.labels[idx_arr]
    fids = store.edf_file_ids[idx_arr]
    blks = store.block_ids[idx_arr]
    ons = store.onsets[idx_arr].astype(np.float64)
    wdurs = store.win_durs[idx_arr].astype(np.float64)

    seizures_by_edf: Dict[int, np.ndarray] = {
        int(eid): np.empty(0, dtype=np.float64) for eid in np.unique(fids)
    }

    pre_mask = lbl == 1
    if not np.any(pre_mask):
        return seizures_by_edf

    pre_fids = fids[pre_mask]
    pre_blks = blks[pre_mask]
    pre_ends = ons[pre_mask] + wdurs[pre_mask]

    for eid in np.unique(pre_fids):
        e_mask = pre_fids == eid
        e_blks = pre_blks[e_mask]
        e_ends = pre_ends[e_mask]
        sz_list: List[float] = []
        for b in np.unique(e_blks):
            sz_list.append(float(np.max(e_ends[e_blks == b])))
        seizures_by_edf[int(eid)] = np.sort(
            np.asarray(sz_list, dtype=np.float64)
        )

    return seizures_by_edf

@torch.inference_mode()
def evaluate_windows(
    store: MasterDataStore,
    model: nn.Module,
    fold: Fold,
    *,
    split: str = "test",
    batch_size: int = 256,
    use_amp: bool = True,
    amp_dtype: str = "bfloat16",
    num_workers: int = 0,
    device: Optional[str] = None,
    threshold: float = 0.5,
    refractory_sec: float = 30.0 * 60.0,
    preictal_horizon_sec: float = 35.0 * 60.0,
    alarm_mode: str = "one_per_episode",
    model_name: str = "dbtr_agf",
) -> WindowEvalResult:
    if split not in {"val", "test", "train"}:
        raise ValueError(
            f"evaluate_windows: split must be 'val', 'test', or 'train', "
            f"got {split!r}"
        )
    indices_map = {
        "val": fold.val_idx,
        "test": fold.test_idx,
        "train": fold.train_idx,
    }
    indices = np.asarray(indices_map[split], dtype=np.int64)
    if indices.shape[0] == 0:
        raise ValueError(f"evaluate_windows: split '{split}' is empty (n=0)")

    if device is None:
        device_obj = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device_obj = torch.device(device)
    amp_dtype_obj = (
        torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    )
    _use_amp = bool(use_amp and device_obj.type == "cuda")

    ds = DBTRDataset(store, indices, return_metadata=False)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=_MixupCollateCallable(use_mixup=False, n_classes=2),
        num_workers=int(num_workers),
        pin_memory=bool(device_obj.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    was_training = model.training
    model.eval()
    try:
        all_logits_pos: List[np.ndarray] = []
        all_probs_pos: List[np.ndarray] = []
        for batch in loader:
            time_b = batch["time"].to(device_obj, non_blocking=True)
            freq_b = batch["freq"].to(device_obj, non_blocking=True)
            with torch.amp.autocast(
                device_type=device_obj.type,
                dtype=amp_dtype_obj,
                enabled=_use_amp,
            ):
                logits_2d = model(time_b, freq_b)["logits"]
            logits_fp32 = logits_2d.float()
            probs_pos = torch.softmax(logits_fp32, dim=-1)[:, 1]
            logits_diff = logits_fp32[:, 1] - logits_fp32[:, 0]
            all_logits_pos.append(logits_diff.cpu().numpy())
            all_probs_pos.append(probs_pos.cpu().numpy())
    finally:
        model.train(was_training)

    logits_arr = np.concatenate(all_logits_pos, axis=0).astype(np.float32)
    probs_arr = np.concatenate(all_probs_pos, axis=0).astype(np.float32)

    return _build_result_from_probs(
        store=store,
        fold=fold,
        split=split,
        indices=indices,
        probs=probs_arr,
        model_logits=logits_arr,
        threshold=float(threshold),
        refractory_sec=float(refractory_sec),
        preictal_horizon_sec=float(preictal_horizon_sec),
        alarm_mode=str(alarm_mode),
        model_name=str(model_name),
    )

class _MixupCollateCallable:
    def __init__(
        self,
        alpha: float = 0.2,
        use_mixup: bool = False,
        p_mixup: float = 1.0,
        n_classes: int = 2,
    ) -> None:
        self.alpha = float(alpha)
        self.use_mixup = bool(use_mixup)
        self.p_mixup = float(p_mixup)
        self.n_classes = int(n_classes)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return mixup_collate_fn(
            batch,
            alpha=self.alpha,
            use_mixup=self.use_mixup,
            p_mixup=self.p_mixup,
            n_classes=self.n_classes,
        )

@dataclass
class TrainConfig:
    d_model: int = 128
    tf_temporal_layers: int = 2
    tf_temporal_heads: int = 4
    tf_temporal_dff: int = 512
    bilstm_layers: int = 1
    tf_spectral_layers: int = 2
    tf_spectral_heads: int = 4
    tf_spectral_dff: int = 512
    tcn_layers: int = 3
    tcn_kernel: int = 3
    tcn_dilations: Optional[List[int]] = None
    agf_gate_mode: str = "per_sample"
    agf_bias_init: float = 0.0
    agf_weight_init_std: float = 0.01
    dropout: float = 0.15
    drop_path_rate: float = 0.10
    fusion_mode: str = "agf"
    classifier_hidden: Optional[int] = None
    pooling: str = "mean"
    n_classes: int = 2

    epochs: int = 30
    batch_size: int = 128
    lr: float = 5e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 3
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    smoothing_mode: str = "uniform"
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    p_mixup: float = 0.5
    entropy_weight: float = 1e-2
    gate_l2_weight: float = 1e-4
    grad_clip_norm: float = 1.0

    early_stopping_patience: int = 8
    min_epochs: int = 5
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    num_workers: int = 0
    deterministic: bool = False
    device: Optional[str] = None

def build_model_from_config(
    store: MasterDataStore,
    cfg: TrainConfig,
) -> DBTRAGF:
    f_dim_active = int(len(store.active_freq_idx)) * int(store.n_channels)
    return DBTRAGF(
        time_in_features=int(store.t_dim),
        freq_in_features=f_dim_active,
        n_steps=int(store.n_steps),
        d_model=int(cfg.d_model),
        tf_temporal_layers=int(cfg.tf_temporal_layers),
        tf_temporal_heads=int(cfg.tf_temporal_heads),
        tf_temporal_dff=int(cfg.tf_temporal_dff),
        bilstm_layers=int(cfg.bilstm_layers),
        tf_spectral_layers=int(cfg.tf_spectral_layers),
        tf_spectral_heads=int(cfg.tf_spectral_heads),
        tf_spectral_dff=int(cfg.tf_spectral_dff),
        tcn_layers=int(cfg.tcn_layers),
        tcn_kernel=int(cfg.tcn_kernel),
        tcn_dilations=cfg.tcn_dilations,
        agf_gate_mode=str(cfg.agf_gate_mode),
        agf_bias_init=float(cfg.agf_bias_init),
        agf_weight_init_std=float(cfg.agf_weight_init_std),
        dropout=float(cfg.dropout),
        drop_path_rate=float(cfg.drop_path_rate),
        fusion_mode=str(cfg.fusion_mode),
        classifier_hidden=cfg.classifier_hidden,
        pooling=str(cfg.pooling),
        n_classes=int(cfg.n_classes),
    )

def train_one_fold(
    store: MasterDataStore,
    fold: Fold,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    set_global_seed(int(fold.fold_seed), deterministic=bool(cfg.deterministic))

    if cfg.device is None:
        device_obj = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device_obj = torch.device(cfg.device)

    amp_dtype_obj = (
        torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    )
    use_amp = bool(cfg.use_amp and device_obj.type == "cuda")
    use_scaler = bool(use_amp and amp_dtype_obj == torch.float16)

    train_ds = DBTRDataset(store, fold.train_idx, return_metadata=False)
    val_ds = DBTRDataset(store, fold.val_idx, return_metadata=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"train_one_fold: empty split (n_train={len(train_ds)}, "
            f"n_val={len(val_ds)})"
        )

    g_loader = torch.Generator()
    g_loader.manual_seed(int(fold.fold_seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        collate_fn=_MixupCollateCallable(
            alpha=cfg.mixup_alpha,
            use_mixup=cfg.use_mixup,
            p_mixup=cfg.p_mixup,
            n_classes=cfg.n_classes,
        ),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(device_obj.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g_loader,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(int(cfg.batch_size) * 2, 128),
        shuffle=False,
        collate_fn=_MixupCollateCallable(
            use_mixup=False, n_classes=cfg.n_classes,
        ),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(device_obj.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    model = build_model_from_config(store, cfg).to(device_obj)
    loss_fn = LabelSmoothingCrossEntropy(
        epsilon=float(cfg.label_smoothing),
        reduction="mean",
        smoothing_mode=str(cfg.smoothing_mode),
    )
    val_loss_fn = LabelSmoothingCrossEntropy(epsilon=0.0, reduction="mean")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    total_epochs = max(int(cfg.epochs), 1)
    warmup_ep = max(0, min(int(cfg.warmup_epochs), total_epochs - 1))

    def _lr_lambda(ep_idx: int) -> float:
        if warmup_ep > 0 and ep_idx < warmup_ep:
            return float(ep_idx + 1) / float(warmup_ep)
        if total_epochs <= warmup_ep + 1:
            return 1.0
        progress = float(ep_idx - warmup_ep) / float(
            max(total_epochs - warmup_ep - 1, 1)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        min_ratio = float(cfg.min_lr) / max(float(cfg.lr), 1e-12)
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.amp.GradScaler(
        device=device_obj.type, enabled=use_scaler
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_score = -float("inf")
    best_epoch = 1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, total_epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        tr_loss_sum = 0.0
        tr_correct = 0
        tr_total = 0
        gate_mean_sum = 0.0
        gate_batches = 0

        for batch in train_loader:
            time_b = batch["time"].to(device_obj, non_blocking=True)
            freq_b = batch["freq"].to(device_obj, non_blocking=True)
            hard_y = batch["label"].to(device_obj, non_blocking=True)
            targets = (
                batch["label_soft"].to(device_obj, non_blocking=True)
                if batch["is_mixed"]
                else hard_y
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device_obj.type,
                dtype=amp_dtype_obj,
                enabled=use_amp,
            ):
                out = model(time_b, freq_b)
                logits = out["logits"]
                ce_loss = loss_fn(logits, targets)
                aux_loss = model.auxiliary_loss(
                    out["gate"],
                    entropy_weight=float(cfg.entropy_weight),
                    l2_weight=float(cfg.gate_l2_weight),
                )
                loss = ce_loss + aux_loss

            if use_scaler:
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.grad_clip_norm)
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.grad_clip_norm)
                    )
                optimizer.step()

            bsz = int(hard_y.shape[0])
            tr_loss_sum += float(ce_loss.detach().item()) * bsz
            preds = logits.detach().argmax(dim=-1)
            tr_correct += int((preds == hard_y).sum().item())
            tr_total += bsz

            if out["gate"] is not None:
                gate_mean_sum += float(out["gate"].detach().float().mean().item())
                gate_batches += 1

        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = tr_loss_sum / max(tr_total, 1)
        train_acc = float(tr_correct) / max(tr_total, 1)
        gate_mean = (
            gate_mean_sum / gate_batches if gate_batches > 0 else float("nan")
        )

        model.eval()
        va_loss_sum = 0.0
        va_correct = 0
        va_total = 0
        with torch.inference_mode():
            for batch in val_loader:
                time_b = batch["time"].to(device_obj, non_blocking=True)
                freq_b = batch["freq"].to(device_obj, non_blocking=True)
                hard_y = batch["label"].to(device_obj, non_blocking=True)
                with torch.amp.autocast(
                    device_type=device_obj.type,
                    dtype=amp_dtype_obj,
                    enabled=use_amp,
                ):
                    logits = model(time_b, freq_b)["logits"]
                    v_loss = val_loss_fn(logits, hard_y)
                bsz = int(hard_y.shape[0])
                va_loss_sum += float(v_loss.item()) * bsz
                va_correct += int((logits.argmax(dim=-1) == hard_y).sum().item())
                va_total += bsz

        val_loss = va_loss_sum / max(va_total, 1)
        val_acc = float(va_correct) / max(va_total, 1)
        val_score = val_acc - 0.05 * val_loss

        history.append({
            "epoch": int(epoch),
            "lr": current_lr,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "gate_mean": float(gate_mean),
            "elapsed_sec": float(time.perf_counter() - t_ep),
        })

        if val_score > best_score + 1e-5:
            best_score = val_score
            best_val_loss = float(val_loss)
            best_val_acc = float(val_acc)
            best_epoch = int(epoch)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (
            epoch >= int(cfg.min_epochs)
            and epochs_no_improve >= int(cfg.early_stopping_patience)
        ):
            _LOGGER.info(
                "[fold %d] early stopping at epoch %d (best=%d, val_acc=%.4f)",
                fold.fold_id, epoch, best_epoch, best_val_acc,
            )
            break

    if best_state is not None:
        model.load_state_dict(
            {k: v.to(device_obj) for k, v in best_state.items()}
        )

    return {
        "model": model,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "device": str(device_obj),
    }

def _write_subject_csv(
    per_subject: Dict[int, SubjectSummary],
    mode: str,
    path: str,
) -> None:
    if mode not in {"before", "after"}:
        raise ValueError(f"mode must be 'before' or 'after', got {mode!r}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    all_metrics: List[str] = sorted(
        set().union(
            *[
                (s.before_mean.keys() if mode == "before" else s.after_mean.keys())
                for s in per_subject.values()
            ]
        )
    ) if per_subject else []

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["subject_id", "n_folds"]
        for m in all_metrics:
            header += [f"{m}_mean", f"{m}_std"]
        writer.writerow(header)

        def _fmt(v: Any) -> Any:
            if v is None:
                return ""
            if isinstance(v, float) and not np.isfinite(v):
                return ""
            return v

        for sid in sorted(per_subject.keys()):
            s = per_subject[sid]
            means = s.before_mean if mode == "before" else s.after_mean
            stds = s.before_std if mode == "before" else s.after_std
            row: List[Any] = [int(sid), int(s.n_folds)]
            for m in all_metrics:
                row += [_fmt(means.get(m)), _fmt(stds.get(m))]
            writer.writerow(row)

def run_ablation_protocol(
    store: MasterDataStore,
    base_cfg: TrainConfig,
    out_dir: Optional[str] = None,
    *,
    fusion_modes: Optional[List[str]] = None,
    val_ratio: float = 0.15,
    block_guard: int = 1,
    **pipeline_kwargs: Any,
) -> Dict[str, ProtocolResult]:
    if fusion_modes is None:
        fusion_modes = [
            "agf", "concat", "fixed", "temporal_only", "spectral_only",
        ]
    if out_dir is None:
        out_dir = os.path.join(BIGMODEL_RESULT_DIR, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    folds = enumerate_one_subject_folds(
        store,
        val_ratio=float(val_ratio),
        block_guard=int(block_guard),
    )

    results: Dict[str, ProtocolResult] = {}
    for mode in fusion_modes:
        mode_cfg = replace(base_cfg, fusion_mode=mode)
        mode_out = os.path.join(out_dir, f"mode_{mode}")
        results[mode] = _run_protocol(
            store,
            folds,
            mode_cfg,
            protocol_name=f"ablation_{mode}",
            out_dir=mode_out,
            model_name=f"dbtr_{mode}",
            **pipeline_kwargs,
        )

    comp_csv = os.path.join(out_dir, "ablation_comparison.csv")
    metrics = ["accuracy", "sensitivity", "specificity", "f1", "roc_auc", "false_alarms_per_hour"]
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["fusion_mode", "n_folds_ok"]
        for m in metrics:
            header += [f"{m}_before_mean", f"{m}_before_std", f"{m}_after_mean", f"{m}_after_std"]
        writer.writerow(header)
        for mode, res in results.items():
            row: List[Any] = [mode, res.n_folds - res.n_folds_failed]
            for m in metrics:
                row += [
                    res.before_hmm_mean.get(m, ""),
                    res.before_hmm_std.get(m, ""),
                    res.after_hmm_mean.get(m, ""),
                    res.after_hmm_std.get(m, ""),
                ]
            writer.writerow(row)

    return results

def _disaggregate_subject99_predictions(
    store: MasterDataStore,
    protocol_result: ProtocolResult,
    out_dir: str,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    ok_folds = [
        f for f in protocol_result.fold_results
        if f.error is None and f.after_hmm is not None and f.before_hmm is not None
    ]

    per_raw_after: Dict[int, List[Dict[str, float]]] = {1: [], 21: []}
    per_raw_before: Dict[int, List[Dict[str, float]]] = {1: [], 21: []}

    for fr in ok_folds:
        for phase, ev, bucket in (
            ("before", fr.before_hmm, per_raw_before),
            ("after", fr.after_hmm, per_raw_after),
        ):
            if ev is None:
                continue
            raw_sids_fold = store.raw_sids[ev.row_indices]
            for r_sid in (1, 21):
                mask = raw_sids_fold == r_sid
                if not np.any(mask):
                    continue
                sub_indices = ev.row_indices[mask]
                sub_ih = store.interictal_hours(sub_indices)
                m = _metrics_for_group(
                    y_true=ev.y_true[mask],
                    y_pred=ev.y_pred[mask],
                    y_score=ev.probs[mask],
                    interictal_hours=float(sub_ih),
                )
                bucket[r_sid].append(m)

    summary: Dict[str, Any] = {}
    for r_sid, name in ((1, "chb01"), (21, "chb21")):
        b_mean, b_std = _mean_std_over_dicts(per_raw_before[r_sid])
        a_mean, a_std = _mean_std_over_dicts(per_raw_after[r_sid])
        summary[name] = {
            "raw_subject_id": r_sid,
            "n_folds_evaluated": len(per_raw_after[r_sid]),
            "before_hmm_mean": b_mean,
            "before_hmm_std": b_std,
            "after_hmm_mean": a_mean,
            "after_hmm_std": a_std,
        }
    n_pairs = min(len(per_raw_after[1]), len(per_raw_after[21]))
    descriptive_comparison: Dict[str, Any] = {
        "n_pairs": int(n_pairs),
        "note": (
            "Descriptive comparison only; no statistical hypothesis test "
            "is performed (see Section 3.6 of the manuscript)."
        ),
        "metrics": {},
    }
    if n_pairs >= 2:
        metric_keys = sorted(per_raw_after[1][0].keys())
        for k in metric_keys:
            v1 = np.asarray(
                [d.get(k, np.nan) for d in per_raw_after[1][:n_pairs]],
                dtype=np.float64,
            )
            v21 = np.asarray(
                [d.get(k, np.nan) for d in per_raw_after[21][:n_pairs]],
                dtype=np.float64,
            )
            valid = np.isfinite(v1) & np.isfinite(v21)
            if int(valid.sum()) >= 2:
                diff = v1[valid] - v21[valid]
                descriptive_comparison["metrics"][k] = {
                    "chb01_mean": float(np.mean(v1[valid])),
                    "chb21_mean": float(np.mean(v21[valid])),
                    "mean_diff": float(np.mean(diff)),
                    "std_diff": float(np.std(diff, ddof=1)),
                }

    summary["descriptive_comparison_1_vs_21"] = descriptive_comparison
    out_json = os.path.join(out_dir, "chb01_chb21_99_study.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    return summary

def plot_publication_figures(
    protocol_result: ProtocolResult,
    out_dir: str,
    ablation_results: Optional[Dict[str, ProtocolResult]] = None,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    saved_Charts: Dict[str, str] = {}

    ok_folds = [
        f for f in protocol_result.fold_results
        if f.error is None and f.before_hmm is not None and f.after_hmm is not None
    ]
    if not ok_folds:
        return saved_Charts

    cm_before = sum((f.before_hmm.confusion for f in ok_folds), np.zeros((2, 2), dtype=np.int64))
    cm_after = sum((f.after_hmm.confusion for f in ok_folds), np.zeros((2, 2), dtype=np.int64))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)
    for ax, cm, title in (
        (axes[0], cm_before, "Without HMM (Base DBTR-AGF)"),
        (axes[1], cm_after, "With Causal HMM Post-Processing"),
    ):
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        total = max(tn + fp + fn + tp, 1)
        acc = 100.0 * (tn + tp) / total
        sen = 100.0 * tp / max(tp + fn, 1)
        spe = 100.0 * tn / max(tn + fp, 1)

        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Interictal (Pred)", "Preictal (Pred)"])
        ax.set_yticklabels(["Interictal (True)", "Preictal (True)"])
        ax.set_title(
            f"{title}\nMicro-ACC={acc:.2f}% | SEN={sen:.2f}% | SPE={spe:.2f}%",
            fontsize=10, fontweight="bold",
        )
        for r in range(2):
            for c in range(2):
                val = int(cm[r, c])
                pct = 100.0 * val / max(int(cm[r].sum()), 1)
                ax.text(
                    c, r, f"{val:,}\n({pct:.2f}%)",
                    ha="center", va="center",
                    color="white" if val > cm.max() / 2 else "black",
                    fontsize=10, fontweight="bold",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    cm_path = os.path.join(out_dir, "figure4_confusion_matrices.png")
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_Charts["confusion_matrices"] = cm_path

    mean_fpr = np.linspace(0.0, 1.0, 201)
    tprs_before: List[np.ndarray] = []
    tprs_after: List[np.ndarray] = []

    for f in ok_folds:
        for ev, bucket in ((f.before_hmm, tprs_before), (f.after_hmm, tprs_after)):
            if ev.roc_fpr is not None and ev.roc_tpr is not None:
                interp_tpr = np.interp(mean_fpr, ev.roc_fpr, ev.roc_tpr)
                interp_tpr[0] = 0.0
                bucket.append(interp_tpr)

    if tprs_before and tprs_after:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        mb = np.mean(tprs_before, axis=0); mb[-1] = 1.0
        ma = np.mean(tprs_after, axis=0); ma[-1] = 1.0
        sb = np.std(tprs_before, axis=0)
        sa = np.std(tprs_after, axis=0)

        auc_b = float(auc(mean_fpr, mb))
        auc_a = float(auc(mean_fpr, ma))

        ax.plot(mean_fpr, mb, label=f"DBTR-AGF (HMM OFF) — AUC = {auc_b:.4f}", lw=2.0, color="#1f77b4")
        ax.fill_between(mean_fpr, np.clip(mb - sb, 0, 1), np.clip(mb + sb, 0, 1), color="#1f77b4", alpha=0.18)

        ax.plot(mean_fpr, ma, label=f"DBTR-AGF + Causal HMM — AUC = {auc_a:.4f}", lw=2.2, color="#d62728")
        ax.fill_between(mean_fpr, np.clip(ma - sa, 0, 1), np.clip(ma + sa, 0, 1), color="#d62728", alpha=0.18)

        ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Chance (AUC = 0.50)")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        ax.set_title(f"ROC Comparison ({protocol_result.protocol_name})", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        roc_path = os.path.join(out_dir, "figure_roc_curves.png")
        fig.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_Charts["roc_curves"] = roc_path

    if ablation_results:
        fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)
        modes = list(ablation_results.keys())
        x = np.arange(len(modes))
        width = 0.25
        for idx_m, (m_key, m_label) in enumerate([
            ("accuracy", "Accuracy (%)"),
            ("sensitivity", "Sensitivity (%)"),
            ("specificity", "Specificity (%)"),
        ]):
            vals = [100.0 * ablation_results[m].after_hmm_mean.get(m_key, 0.0) for m in modes]
            ax.bar(x + (idx_m - 1) * width, vals, width, label=m_label)

        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=15)
        ax.set_ylim(50.0, 101.0)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Architectural Ablation Study (With Causal HMM)", fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        abl_path = os.path.join(out_dir, "figure_ablation_study.png")
        fig.savefig(abl_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_Charts["ablation_chart"] = abl_path

    return saved_Charts

def reconcile_macro_vs_micro_report(
    protocol_result: ProtocolResult,
    out_dir: str,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    ok_folds = [
        f for f in protocol_result.fold_results
        if f.error is None and f.before_hmm is not None and f.after_hmm is not None
    ]
    cm_before = sum((f.before_hmm.confusion for f in ok_folds), np.zeros((2, 2), dtype=np.int64))
    cm_after = sum((f.after_hmm.confusion for f in ok_folds), np.zeros((2, 2), dtype=np.int64))

    def _cm_stats(cm: np.ndarray) -> Dict[str, float]:
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        tot = max(tn + fp + fn + tp, 1)
        return {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "micro_accuracy": float(tp + tn) / tot,
            "micro_sensitivity": float(tp) / max(tp + fn, 1),
            "micro_specificity": float(tn) / max(tn + fp, 1),
            "micro_ppv": float(tp) / max(tp + fp, 1),
            "micro_npv": float(tn) / max(tn + fn, 1),
            "micro_f1": (2.0 * tp) / max(2 * tp + fp + fn, 1),
        }

    reconciliation = {
        "protocol": protocol_result.protocol_name,
        "explanation": (
            "Tables 1 and 2 report the Subject-Wise Macro-Average (unweighted "
            "mean across patients), whereas Figure 4 reports the Pooled "
            "Window-Level Micro-Average from the summed confusion matrix."
        ),
        "macro_table1_before_hmm": protocol_result.before_hmm_mean,
        "macro_table1_after_hmm": protocol_result.after_hmm_mean,
        "micro_figure4_before_hmm": _cm_stats(cm_before),
        "micro_figure4_after_hmm": _cm_stats(cm_after),
    }
    path = os.path.join(out_dir, "reviewer2_point4_macro_vs_micro.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reconciliation, f, indent=2, default=_json_default)
    return reconciliation

_TABLE1_DEFAULT_METRICS: Tuple[str, ...] = (
    "accuracy",
    "sensitivity",
    "specificity",
    "false_alarms_per_hour",
)

_TABLE1_METRIC_LABELS: Dict[str, str] = {
    "accuracy": "ACC (%)",
    "sensitivity": "SEN (%)",
    "specificity": "SPE (%)",
    "balanced_accuracy": "Bal-ACC (%)",
    "f1": "F1 (%)",
    "precision": "PPV (%)",
    "npv": "NPV (%)",
    "mcc": "MCC",
    "roc_auc": "AUC",
    "window_fpr_per_hour": "Win-FPR (/h)",
    "false_alarms_per_hour": "FPR/h",
    "event_sensitivity": "Event SEN (%)",
    "mean_lead_time_min": "Lead (min)",
}

_TABLE1_METRIC_SCALE: Dict[str, float] = {
    "accuracy": 100.0,
    "sensitivity": 100.0,
    "specificity": 100.0,
    "balanced_accuracy": 100.0,
    "f1": 100.0,
    "precision": 100.0,
    "npv": 100.0,
    "mcc": 1.0,
    "roc_auc": 1.0,
    "window_fpr_per_hour": 1.0,
    "false_alarms_per_hour": 1.0,
    "event_sensitivity": 100.0,
    "mean_lead_time_min": 1.0,
}

_TABLE1_METRIC_DECIMALS: Dict[str, int] = {
    "accuracy": 2,
    "sensitivity": 2,
    "specificity": 2,
    "balanced_accuracy": 2,
    "f1": 2,
    "precision": 2,
    "npv": 2,
    "mcc": 3,
    "roc_auc": 4,
    "window_fpr_per_hour": 3,
    "false_alarms_per_hour": 3,
    "event_sensitivity": 2,
    "mean_lead_time_min": 1,
}

_PROTOCOL_DISPLAY_NAMES: Dict[str, str] = {
    "one_subject": "Single-subject",
    "holdout_70_30": "Shuffle-based",
    "low_data_50": "Low-data (50%)",
    "low_data_30": "Low-data (30%)",
    "low_data_70": "Low-data (70%)",
    "loso": "LOSO",
}

_PROTOCOL_EVAL_METHODS: Dict[str, str] = {
    "one_subject": "Single-subject training",
    "holdout_70_30": "70/30 shuffle split",
    "low_data_50": "Single-subject with 50% data",
    "low_data_30": "Single-subject with 30% data",
    "low_data_70": "Single-subject with 70% data",
    "loso": "Leave-one-subject-out",
}

def _protocol_display_name(protocol_key: str) -> str:
    return _PROTOCOL_DISPLAY_NAMES.get(protocol_key, protocol_key)

def _protocol_eval_method(protocol_key: str) -> str:
    return _PROTOCOL_EVAL_METHODS.get(protocol_key, protocol_key)

def _format_metric_cell(
    mean_val: Optional[float],
    std_val: Optional[float],
    metric: str,
    *,
    separator: str = " ± ",
    use_std: bool = True,
) -> str:
    if mean_val is None or not np.isfinite(mean_val):
        return "—"

    scale = _TABLE1_METRIC_SCALE.get(metric, 1.0)
    dec = _TABLE1_METRIC_DECIMALS.get(metric, 3)

    mean_scaled = float(mean_val) * scale

    if use_std and std_val is not None and np.isfinite(std_val):
        std_scaled = float(std_val) * scale
        return f"{mean_scaled:.{dec}f}{separator}{std_scaled:.{dec}f}"
    return f"{mean_scaled:.{dec}f}"

def write_manuscript_table1(
    results: Dict[str, ProtocolResult],
    out_dir: str,
    *,
    metric_keys: Optional[Sequence[str]] = None,
    protocol_order: Optional[Sequence[str]] = None,
    include_before_hmm: bool = True,
    include_after_hmm: bool = True,
    write_markdown: bool = True,
    write_latex: bool = True,
) -> Dict[str, str]:
    if not results:
        raise ValueError("write_manuscript_table1: `results` is empty")

    metrics = (
        tuple(metric_keys) if metric_keys is not None
        else _TABLE1_DEFAULT_METRICS
    )
    if not metrics:
        raise ValueError(
            "write_manuscript_table1: `metric_keys` must be non-empty"
        )

    if protocol_order is not None:
        missing_from_results = [
            k for k in protocol_order if k not in results
        ]
        if missing_from_results:
            _LOGGER.warning(
                "write_manuscript_table1: protocol_order contains keys "
                "not present in results: %s (skipped)",
                missing_from_results,
            )
        order = [k for k in protocol_order if k in results]
        order += [k for k in results.keys() if k not in set(order)]
    else:
        order = list(results.keys())

    if not order:
        raise ValueError(
            "write_manuscript_table1: no protocols in `results` after "
            "filtering by `protocol_order`"
        )

    rows: List[Dict[str, Any]] = []
    for proto_key in order:
        res = results[proto_key]
        display_name = _protocol_display_name(proto_key)
        eval_method = _protocol_eval_method(proto_key)

        phases_to_emit: List[Tuple[str, str, str]] = []
        if include_before_hmm and res.before_hmm_mean:
            phases_to_emit.append(
                ("OFF", "before_hmm_mean", "before_hmm_std")
            )
        if include_after_hmm and res.after_hmm_mean:
            phases_to_emit.append(
                ("ON", "after_hmm_mean", "after_hmm_std")
            )

        if not phases_to_emit:
            _LOGGER.warning(
                "write_manuscript_table1: protocol %r has no data for "
                "either phase; skipping.",
                proto_key,
            )
            continue

        for postproc_label, mean_attr, std_attr in phases_to_emit:
            mean_dict: Dict[str, float] = getattr(res, mean_attr)
            std_dict: Dict[str, float] = getattr(res, std_attr)

            raw_row: Dict[str, Any] = {
                "protocol_key": proto_key,
                "protocol": display_name,
                "evaluation": eval_method,
                "postprocessing": postproc_label,
                "n_folds_ok": int(res.n_folds - res.n_folds_failed),
                "n_folds_failed": int(res.n_folds_failed),
            }
            for m in metrics:
                raw_row[f"{m}_mean"] = (
                    float(mean_dict[m]) if m in mean_dict else None
                )
                raw_row[f"{m}_std"] = (
                    float(std_dict[m]) if m in std_dict else None
                )
            rows.append(raw_row)

    if not rows:
        raise RuntimeError(
            "write_manuscript_table1: no rows produced. Check that "
            "`results` contains non-empty `before_hmm_mean` or "
            "`after_hmm_mean`."
        )

    os.makedirs(out_dir, exist_ok=True)
    out_dir_abs = os.path.abspath(out_dir)
    paths: Dict[str, str] = {}

    csv_path = os.path.join(out_dir_abs, "table1_manuscript.csv")
    try:
        header: List[str] = [
            "protocol_key", "protocol", "evaluation", "postprocessing",
            "n_folds_ok", "n_folds_failed",
        ]
        for m in metrics:
            header += [f"{m}_mean", f"{m}_std"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in rows:
                out_row: List[Any] = [
                    r["protocol_key"], r["protocol"], r["evaluation"],
                    r["postprocessing"], r["n_folds_ok"],
                    r["n_folds_failed"],
                ]
                for m in metrics:
                    mv = r.get(f"{m}_mean")
                    sv = r.get(f"{m}_std")
                    out_row.append("" if mv is None else f"{mv:.10g}")
                    out_row.append("" if sv is None else f"{sv:.10g}")
                writer.writerow(out_row)
        paths["csv"] = csv_path
    except Exception as exc:
        _LOGGER.warning(
            "write_manuscript_table1: could not write %s (%s)",
            csv_path, exc,
        )

    json_path = os.path.join(out_dir_abs, "table1_manuscript.json")
    try:
        json_payload = {
            "table": "Table 1 — Multi-protocol summary (mean ± std)",
            "metrics": list(metrics),
            "metric_labels": {
                m: _TABLE1_METRIC_LABELS.get(m, m) for m in metrics
            },
            "metric_scale": {
                m: _TABLE1_METRIC_SCALE.get(m, 1.0) for m in metrics
            },
            "rows": rows,
            "protocol_display_labels": {
                k: _protocol_display_name(k) for k in order
            },
            "protocol_eval_methods": {
                k: _protocol_eval_method(k) for k in order
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, default=_json_default)
        paths["json"] = json_path
    except Exception as exc:
        _LOGGER.warning(
            "write_manuscript_table1: could not write %s (%s)",
            json_path, exc,
        )

    if write_markdown:
        md_path = os.path.join(out_dir_abs, "table1_manuscript.md")
        try:
            md_lines: List[str] = []
            md_lines.append(
                "# Table 1 — Multi-protocol summary (mean ± std)"
            )
            md_lines.append("")
            header_cells = [
                "Protocol", "Evaluation Method", "Postprocessing", "N folds",
            ]
            header_cells += [
                _TABLE1_METRIC_LABELS.get(m, m) for m in metrics
            ]
            md_lines.append("| " + " | ".join(header_cells) + " |")
            md_lines.append(
                "|" + "|".join(["---"] * len(header_cells)) + "|"
            )
            for r in rows:
                cells: List[str] = [
                    r["protocol"],
                    r["evaluation"],
                    r["postprocessing"],
                    str(r["n_folds_ok"]),
                ]
                for m in metrics:
                    cells.append(
                        _format_metric_cell(
                            r.get(f"{m}_mean"),
                            r.get(f"{m}_std"),
                            m,
                        )
                    )
                md_lines.append("| " + " | ".join(cells) + " |")
            md_lines.append("")
            md_lines.append(
                "_All metrics are reported as mean ± standard deviation "
                "across the successful folds of each protocol. "
                "ACC/SEN/SPE are percentages; FPR/h is false alarms per hour._"
            )
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines))
            paths["markdown"] = md_path
        except Exception as exc:
            _LOGGER.warning(
                "write_manuscript_table1: could not write %s (%s)",
                md_path, exc,
            )

    if write_latex:
        tex_path = os.path.join(out_dir_abs, "table1_manuscript.tex")
        try:
            def _tex_escape(s: str) -> str:
                return (
                    s.replace("_", r"\_")
                     .replace("%", r"\%")
                     .replace("&", r"\&")
                )

            n_metric_cols = len(metrics)
            col_spec = "lll" + "c" * (1 + n_metric_cols)

            tex_lines: List[str] = [
                r"\begin{table*}[htbp]",
                r"\centering",
                r"\caption{Multi-protocol summary of DBTR-AGF performance "
                r"on the CHB-MIT dataset. All metrics are reported as "
                r"mean $\pm$ standard deviation across the successful "
                r"folds of each protocol. ACC/SEN/SPE are percentages; "
                r"FPR/h is false alarms per hour.}",
                r"\label{tab:table1}",
                r"\begin{tabular}{" + col_spec + "}",
                r"\toprule",
            ]
            header_cells_tex = [
                "Protocol", "Evaluation", "Post.", "N",
            ]
            header_cells_tex += [
                _tex_escape(_TABLE1_METRIC_LABELS.get(m, m))
                for m in metrics
            ]
            tex_lines.append(" & ".join(header_cells_tex) + r" \\")
            tex_lines.append(r"\midrule")

            last_proto = None
            for r in rows:
                if (
                    last_proto is not None
                    and r["protocol_key"] != last_proto
                ):
                    tex_lines.append(r"\midrule")
                last_proto = r["protocol_key"]

                cells_tex: List[str] = [
                    _tex_escape(r["protocol"]),
                    _tex_escape(r["evaluation"]),
                    r["postprocessing"],
                    str(r["n_folds_ok"]),
                ]
                for m in metrics:
                    cells_tex.append(
                        _format_metric_cell(
                            r.get(f"{m}_mean"),
                            r.get(f"{m}_std"),
                            m,
                            separator=r" $\pm$ ",
                        )
                    )
                tex_lines.append(" & ".join(cells_tex) + r" \\")

            tex_lines += [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
            ]
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write("\n".join(tex_lines))
            paths["latex"] = tex_path
        except Exception as exc:
            _LOGGER.warning(
                "write_manuscript_table1: could not write %s (%s)",
                tex_path, exc,
            )

    _LOGGER.info(
        "write_manuscript_table1: wrote %d rows for %d protocol(s) → %s",
        len(rows), len(order), list(paths.keys()),
    )
    return paths

def write_manuscript_table2(
    protocol_result: ProtocolResult,
    store: MasterDataStore,
    out_dir: str,
    *,
    prefer_after_hmm: bool = True,
    preictal_horizon_sec: float = 35.0 * 60.0,
    refractory_sec: float = 30.0 * 60.0,
    threshold: float = 0.5,
    write_markdown: bool = True,
    write_latex: bool = True,
) -> Dict[str, str]:
    if not protocol_result.fold_results:
        raise ValueError(
            "write_manuscript_table2: `protocol_result.fold_results` is "
            "empty. Table 2 needs per-fold predictions to dis-aggregate "
            "Subject 99 into chb01/chb21. Do NOT call "
            "_run_protocol(..., prune_fold_results=True) before calling "
            "this function."
        )

    ok_folds = [
        f for f in protocol_result.fold_results
        if f.error is None
        and f.before_hmm is not None
        and f.after_hmm is not None
    ]
    if not ok_folds:
        raise RuntimeError(
            "write_manuscript_table2: no successful folds in protocol "
            f"{protocol_result.protocol_name!r}"
        )

    metric_source = "after_hmm" if prefer_after_hmm else "before_hmm"
    _LOGGER.info(
        "write_manuscript_table2: aggregating %d folds using %s",
        len(ok_folds), metric_source,
    )

    per_raw_metrics: Dict[int, List[Dict[str, float]]] = {}
    per_raw_indices: Dict[int, List[np.ndarray]] = {}

    for fr in ok_folds:
        ev = fr.after_hmm if prefer_after_hmm else fr.before_hmm
        if ev is None:
            continue

        if ev.raw_subject_ids is not None:
            raw_ids = ev.raw_subject_ids
        else:
            raw_ids = ev.subject_ids

        for r_sid in np.unique(raw_ids):
            r_sid = int(r_sid)
            mask = raw_ids == r_sid
            if not np.any(mask):
                continue
            idx_r = ev.row_indices[mask]

            window_metrics = _metrics_for_group(
                y_true=ev.y_true[mask],
                y_pred=ev.y_pred[mask],
                y_score=ev.probs[mask],
                interictal_hours=float(store.interictal_hours(idx_r)),
            )

            subject_edfs = np.unique(store.edf_file_ids[idx_r])

            all_seizures = _reconstruct_seizure_onsets_by_edf(
                store, idx_r, preictal_horizon_sec=float(preictal_horizon_sec),
            )
            subj_seizures = {
                int(eid): all_seizures.get(int(eid), np.empty(0))
                for eid in subject_edfs
            }

            subj_alarms = {
                int(eid): ev.alarms_by_edf.get(
                    int(eid), np.empty(0, dtype=np.float64),
                )
                for eid in subject_edfs
            }

            subj_durations: Dict[int, float] = {}
            for eid in subject_edfs:
                edf_mask = store.edf_file_ids[idx_r] == int(eid)
                subj_durations[int(eid)] = (
                    float(store.durs[idx_r][edf_mask].sum()) / 3600.0
                )

            event_metrics = event_level_metrics_per_edf(
                seizures_by_edf=subj_seizures,
                alarms_by_edf=subj_alarms,
                edf_durations_hours=subj_durations,
                preictal_horizon_sec=float(preictal_horizon_sec),
            )

            combined = dict(window_metrics)
            combined.update({
                "event_fpr_per_hour": float(
                    event_metrics["false_alarms_per_hour"]
                ),
                "event_sensitivity": float(
                    event_metrics["event_sensitivity"]
                ),
                "n_seizures_fold": int(event_metrics["n_seizures"]),
                "n_detected_fold": int(event_metrics["n_detected"]),
                "event_hours_fold": float(event_metrics["total_hours"]),
                "mean_lead_time_min_fold": float(
                    event_metrics["mean_lead_time_min"]
                ),
            })

            per_raw_metrics.setdefault(r_sid, []).append(combined)
            per_raw_indices.setdefault(r_sid, []).append(idx_r)

    if not per_raw_metrics:
        raise RuntimeError(
            "write_manuscript_table2: no raw subjects recovered from "
            "the fold results. Check that the fold results contain "
            "non-empty test predictions."
        )

    table_rows: List[Dict[str, Any]] = []
    for r_sid in sorted(per_raw_metrics.keys()):
        fold_metrics = per_raw_metrics[r_sid]
        if not fold_metrics:
            continue

        row: Dict[str, Any] = {
            "subject_id": int(r_sid),
            "subject_label": f"chb{r_sid:02d}",
            "n_folds_in_test": int(len(fold_metrics)),
        }

        for metric_key, output_key in (
            ("accuracy",            "accuracy"),
            ("sensitivity",         "sensitivity"),
            ("specificity",         "specificity"),
            ("event_fpr_per_hour",  "fpr_per_hour"),
        ):
            vals = [
                float(m[metric_key])
                for m in fold_metrics
                if metric_key in m and np.isfinite(m[metric_key])
            ]
            if vals:
                row[f"{output_key}_mean"] = float(np.mean(vals))
                row[f"{output_key}_std"] = (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                )
            else:
                row[f"{output_key}_mean"] = None
                row[f"{output_key}_std"] = None

        n_seiz_vals = [
            int(m.get("n_seizures_fold", 0)) for m in fold_metrics
        ]
        row["n_seizures"] = int(max(n_seiz_vals)) if n_seiz_vals else 0

        hours_per_fold = [
            float(store.durs[idx].sum()) / 3600.0
            for idx in per_raw_indices[r_sid]
        ]
        row["duration_hours"] = (
            float(np.mean(hours_per_fold)) if hours_per_fold else 0.0
        )

        table_rows.append(row)

    src_mean = (
        protocol_result.after_hmm_mean
        if prefer_after_hmm
        else protocol_result.before_hmm_mean
    )
    src_std = (
        protocol_result.after_hmm_std
        if prefer_after_hmm
        else protocol_result.before_hmm_std
    )

    mean_row: Dict[str, Any] = {
        "subject_id": -1,
        "subject_label": "Mean ± SD (unique patients)",
        "n_folds_in_test": None,
    }

    mean_row["accuracy_mean"] = src_mean.get("accuracy")
    mean_row["accuracy_std"] = src_std.get("accuracy")
    mean_row["sensitivity_mean"] = src_mean.get("sensitivity")
    mean_row["sensitivity_std"] = src_std.get("sensitivity")
    mean_row["specificity_mean"] = src_mean.get("specificity")
    mean_row["specificity_std"] = src_std.get("specificity")
    mean_row["fpr_per_hour_mean"] = src_mean.get("false_alarms_per_hour")
    mean_row["fpr_per_hour_std"] = src_std.get("false_alarms_per_hour")

    n_seiz_vals = []
    dur_vals = []
    for s in protocol_result.per_subject.values():
        d = s.after_mean if prefer_after_hmm else s.before_mean
        if "n_seizures" in d and np.isfinite(d["n_seizures"]):
            n_seiz_vals.append(float(d["n_seizures"]))
        if "event_total_hours" in d and np.isfinite(d["event_total_hours"]):
            dur_vals.append(float(d["event_total_hours"]))

    mean_row["n_seizures"] = (
        float(np.mean(n_seiz_vals)) if n_seiz_vals else 0.0
    )
    mean_row["n_seizures_std"] = (
        float(np.std(n_seiz_vals, ddof=1)) if len(n_seiz_vals) > 1 else 0.0
    )
    mean_row["duration_hours"] = (
        float(np.mean(dur_vals)) if dur_vals else 0.0
    )
    mean_row["duration_hours_std"] = (
        float(np.std(dur_vals, ddof=1)) if len(dur_vals) > 1 else 0.0
    )

    table_rows.append(mean_row)

    os.makedirs(out_dir, exist_ok=True)
    out_dir_abs = os.path.abspath(out_dir)
    paths: Dict[str, str] = {}

    def _fmt_pair(
        mean_val: Optional[float],
        std_val: Optional[float],
        *,
        scale: float = 1.0,
        decimals: int = 2,
        separator: str = " ± ",
    ) -> str:
        if mean_val is None or not np.isfinite(mean_val):
            return "—"
        m = float(mean_val) * float(scale)
        if std_val is not None and np.isfinite(std_val):
            s = float(std_val) * float(scale)
            return f"{m:.{decimals}f}{separator}{s:.{decimals}f}"
        return f"{m:.{decimals}f}"

    csv_path = os.path.join(out_dir_abs, "table2_manuscript.csv")
    try:
        header = [
            "subject_id", "subject_label", "n_folds_in_test",
            "accuracy_mean", "accuracy_std",
            "sensitivity_mean", "sensitivity_std",
            "specificity_mean", "specificity_std",
            "fpr_per_hour_mean", "fpr_per_hour_std",
            "n_seizures",
            "duration_hours", "duration_hours_std",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in table_rows:
                def _v(x: Any) -> Any:
                    if x is None:
                        return ""
                    if isinstance(x, float) and not np.isfinite(x):
                        return ""
                    return x

                writer.writerow([
                    r["subject_id"],
                    r["subject_label"],
                    _v(r.get("n_folds_in_test")),
                    _v(r.get("accuracy_mean")),
                    _v(r.get("accuracy_std")),
                    _v(r.get("sensitivity_mean")),
                    _v(r.get("sensitivity_std")),
                    _v(r.get("specificity_mean")),
                    _v(r.get("specificity_std")),
                    _v(r.get("fpr_per_hour_mean")),
                    _v(r.get("fpr_per_hour_std")),
                    _v(r.get("n_seizures")),
                    _v(r.get("duration_hours")),
                    _v(r.get("duration_hours_std")),
                ])
        paths["csv"] = csv_path
    except Exception as exc:
        _LOGGER.warning(
            "write_manuscript_table2: could not write %s (%s)",
            csv_path, exc,
        )

    if write_markdown:
        md_path = os.path.join(out_dir_abs, "table2_manuscript.md")
        try:
            md_lines: List[str] = [
                "# Table 2 — Subject-wise performance analysis "
                "(with HMM)" if prefer_after_hmm
                else "# Table 2 — Subject-wise performance analysis",
                "",
                "_Under the single-subject cross-patient protocol. "
                "Subject 99 (merged chb01+chb21) is reported as two "
                "rows, chb01 and chb21, to satisfy Reviewer 2 Point 1. "
                "The FPR/h column is event-level (Eq. 15)._",
                "",
            ]
            header_cells = [
                "Subject ID", "Accuracy (%)", "Sensitivity (%)",
                "Specificity (%)", "FPR/h", "No. Seizures", "Duration (h)",
            ]
            md_lines.append("| " + " | ".join(header_cells) + " |")
            md_lines.append(
                "|" + "|".join(["---"] * len(header_cells)) + "|"
            )
            for r in table_rows:
                cells = [
                    r["subject_label"],
                    _fmt_pair(
                        r.get("accuracy_mean"),
                        r.get("accuracy_std"), scale=100.0,
                    ),
                    _fmt_pair(
                        r.get("sensitivity_mean"),
                        r.get("sensitivity_std"), scale=100.0,
                    ),
                    _fmt_pair(
                        r.get("specificity_mean"),
                        r.get("specificity_std"), scale=100.0,
                    ),
                    _fmt_pair(
                        r.get("fpr_per_hour_mean"),
                        r.get("fpr_per_hour_std"), decimals=2,
                    ),
                    (
                        "—" if r.get("n_seizures") is None
                        else f"{int(r['n_seizures'])}"
                        if isinstance(r["n_seizures"], int)
                        else _fmt_pair(
                            r.get("n_seizures"),
                            r.get("n_seizures_std"), decimals=2,
                        )
                    ),
                    (
                        _fmt_pair(
                            r.get("duration_hours"),
                            r.get("duration_hours_std"), decimals=1,
                        )
                    ),
                ]
                md_lines.append("| " + " | ".join(cells) + " |")
            md_lines.append("")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines))
            paths["markdown"] = md_path
        except Exception as exc:
            _LOGGER.warning(
                "write_manuscript_table2: could not write %s (%s)",
                md_path, exc,
            )

    if write_latex:
        tex_path = os.path.join(out_dir_abs, "table2_manuscript.tex")
        try:
            def _tex_escape(s: str) -> str:
                return (
                    s.replace("_", r"\_")
                     .replace("%", r"\%")
                     .replace("&", r"\&")
                )

            tex_lines: List[str] = [
                r"\begin{table*}[htbp]",
                r"\centering",
                r"\caption{Subject-wise performance of DBTR-AGF with "
                r"HMM post-processing under the single-subject "
                r"cross-patient protocol on the CHB-MIT dataset. "
                r"Subject 99 (merged chb01+chb21) is reported as two "
                r"rows. FPR/h is event-level (Eq. 15).}",
                r"\label{tab:table2}",
                r"\begin{tabular}{lcccccc}",
                r"\toprule",
                r"Subject ID & Accuracy (\%) & Sensitivity (\%) & "
                r"Specificity (\%) & FPR/h & No. Seizures & Duration (h) \\",
                r"\midrule",
            ]
            last_was_mean = False
            for r in table_rows:
                if r["subject_id"] == -1 and not last_was_mean:
                    tex_lines.append(r"\midrule")
                    last_was_mean = True
                cells_tex: List[str] = [
                    _tex_escape(r["subject_label"]),
                    _fmt_pair(
                        r.get("accuracy_mean"),
                        r.get("accuracy_std"),
                        scale=100.0,
                        separator=r" $\pm$ ",
                    ),
                    _fmt_pair(
                        r.get("sensitivity_mean"),
                        r.get("sensitivity_std"),
                        scale=100.0,
                        separator=r" $\pm$ ",
                    ),
                    _fmt_pair(
                        r.get("specificity_mean"),
                        r.get("specificity_std"),
                        scale=100.0,
                        separator=r" $\pm$ ",
                    ),
                    _fmt_pair(
                        r.get("fpr_per_hour_mean"),
                        r.get("fpr_per_hour_std"),
                        decimals=2,
                        separator=r" $\pm$ ",
                    ),
                    (
                        "—" if r.get("n_seizures") is None
                        else f"{int(r['n_seizures'])}"
                        if isinstance(r["n_seizures"], int)
                        else _fmt_pair(
                            r.get("n_seizures"),
                            r.get("n_seizures_std"),
                            decimals=2,
                            separator=r" $\pm$ ",
                        )
                    ),
                    _fmt_pair(
                        r.get("duration_hours"),
                        r.get("duration_hours_std"),
                        decimals=1,
                        separator=r" $\pm$ ",
                    ),
                ]
                tex_lines.append(" & ".join(cells_tex) + r" \\")
            tex_lines += [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
            ]
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write("\n".join(tex_lines))
            paths["latex"] = tex_path
        except Exception as exc:
            _LOGGER.warning(
                "write_manuscript_table2: could not write %s (%s)",
                tex_path, exc,
            )

    _LOGGER.info(
        "write_manuscript_table2: wrote %d subject rows "
        "(including the Mean ± SD row) → %s",
        len(table_rows), list(paths.keys()),
    )
    return paths

def run_all_component_sanity_checks() -> None:
    _sanity_check_positional_encoding()
    _sanity_check_channel_encoder()
    _sanity_check_transformer()
    _sanity_check_bilstm()
    _sanity_check_tcn()
    _sanity_check_agf()
    _sanity_check_dbtr_agf()
    _sanity_check_hmm()

def plot_gate_evolution(
    protocol_result: ProtocolResult,
    out_dir: str,
) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    ok_folds = [
        f for f in protocol_result.fold_results
        if f.error is None and f.history
    ]
    if not ok_folds:
        return None

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)
    for fr in ok_folds:
        epochs = [int(h["epoch"]) for h in fr.history]
        gmean = [
            float(h.get("gate_mean", np.nan)) for h in fr.history
        ]
        ax.plot(
            epochs, gmean, lw=1.0, alpha=0.55,
            label=f"train subject {fr.train_sids[0]}",
        )

    ax.axhline(0.5, color="red", linestyle="--", lw=1.4,
               label="Balanced gate (g = 0.5)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean gate value g (temporal↔spectral)", fontsize=11)
    ax.set_title(
        f"AGF Gate Evolution ({protocol_result.protocol_name})",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 12:
        step = max(1, len(handles) // 10)
        handles = handles[::step] + [handles[-1]]
        labels = labels[::step] + [labels[-1]]
    ax.legend(handles, labels, loc="lower right",
              fontsize=7, ncol=2, frameon=True)

    out_path = os.path.join(
        out_dir, f"figure_agf_gate_evolution_{protocol_result.protocol_name}.png"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
