import os
import re
import sys
import gc
import time
import random
import logging
import difflib
import h5py
import numpy as np
import mne
import pywt
import scipy
from scipy.signal import windows
from scipy.fft import rfft as scipy_rfft, rfftfreq as scipy_rfftfreq

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

BASE_DIR = r"G:\Python_codes\MIT_PRE_TRAIN\MIT_dataset\NEW_EX"
SAVE_DIR = r"E:\Neuroscience_Lab\Article reviews final codes and check\result of preprocessing"
H5_FILE = os.path.join(SAVE_DIR, "spectral_data_production_ready.h5")

KNOWN_MMAP_DIRS = [
    SAVE_DIR,
    r"E:\Neuroscience_Lab\Article reviews final codes and check\Final_Evaluation_Master_v103",
    r"E:\Neuroscience_Lab\Article reviews final codes and check\Final_Evaluation_Master_v104",
]
CLEAN_KNOWN_MMAP_CACHES = True

STRICT_SAFE_HALT = True
INCLUDE_SEPARATE_1_AND_21 = True
INCLUDE_COMBINED_1_21 = True
COMBINED_SUBJECT_ID = 99
INTERICTAL_SAMPLING_MODE = "fixed_5s_adaptive_stride"

TARGET_STFT_FREQ_BINS = 641
FIXED_N_TIME = 14
N_CWT_SCALES = 32

Nominal_FS = 256.0
EXCLUDED_SUBJECTS = [4, 12, 13, 24]
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 100.0
NOTCH_FREQS = [60.0, 120.0]
NOTCH_WIDTH = 10.0

SELECTED_CHANNELS = [
    'P3-O1', 'FP1-F3', 'FP2-F4', 'F3-C3', 'F4-C4', 'F7-T7',
    'T8-P8', 'F8-T8', 'FP2-F8', 'T7-P7',
    'C3-P3', 'P4-O2', 'FP1-F7', 'P7-O1', 'P8-O2',
    'C4-P4', 'CZ-PZ', 'FZ-CZ',
]
N_CHANNELS = len(SELECTED_CHANNELS)

PREICTAL_HORIZON_SEC = 35 * 60
POSTICTAL_BUFFER_SEC = 10 * 60
WINDOW_SEC = 5.0
PRE_STEP_SEC = 2.5

MIN_INTER_STEP_SEC = 2.5
MAX_INTER_STEP_SEC = 600.0
DEFAULT_INTER_STEP_SEC = 37.5
STRIDE_SEARCH_LOW = 0.5
STRIDE_SEARCH_HIGH = 2.0

WIN_COUNT_EPS = 1e-9
WINDOW_FIT_TOL_SEC = 1e-5

STFT_SUBWIN_SEC = 2.0
STFT_TAPER_START_HZ = 95.0
CWT_MIN_FREQ_HZ = 1.0
CWT_MAX_FREQ_HZ = 100.0
WAVELET_NAME = "cmor1.5-1.0"

NORM_PERCENTILE_LOW = 0.10
NORM_PERCENTILE_HIGH = 99.90

CHUNK_BATCH_SIZE = 96
SEED = 42

NORM_SUBSAMPLE_MAX = 2000
H5_WRITE_CHUNK = 128
MAX_INRAM_EPOCHS = 8500
INITIAL_BUFFER_FLOOR = 256
FS_TOLERANCE_HZ = 1e-3
MNE_FILTER_JOBS = min(8, max(1, (os.cpu_count() or 4) // 2))
FFT_WORKERS = min(8, max(1, (os.cpu_count() or 4) // 2))

MIN_STAT_SAMPLES = int(5.0 * Nominal_FS)
MAX_STAT_ACCUMULATION_SAMPLES = int(15 * 3600 * Nominal_FS)

MEMORY_SAFETY_FRACTION = 0.65
IN_RAM_OVERHEAD_GB = 2.0
MEMORY_FALLBACK_GB = 8.0

ROW_BYTES_STFT = N_CHANNELS * TARGET_STFT_FREQ_BINS * FIXED_N_TIME * 4
ROW_BYTES_CWT = N_CHANNELS * N_CWT_SCALES * FIXED_N_TIME * 4
ROW_GB_ESTIMATE = (ROW_BYTES_STFT + ROW_BYTES_CWT) / (1024 ** 3)

H5_DATASET_SPECS = [
    ("time_matrix", "float32"),
    ("freq_matrix", "float32"),
    ("labels", "int64"),
    ("subject_ids", "int32"),
    ("raw_subject_ids", "int32"),
    ("block_ids", "int32"),
    ("window_duration_seconds", "float32"),
    ("effective_step_seconds", "float32"),
    ("interictal_window_seconds", "float32"),
    ("window_onset_seconds", "float32"),
    ("edf_file_ids", "int32"),
]

SEIZURE_START_RE = re.compile(r'^Seizure(?:\s+\d+)?\s+Start\s+Time\s*[:=]\s*([\d:\.]+)', re.IGNORECASE)
SEIZURE_END_RE = re.compile(r'^Seizure(?:\s+\d+)?\s+End\s+Time\s*[:=]\s*([\d:\.]+)', re.IGNORECASE)
NUM_SEIZURES_RE = re.compile(r'^Number\s+of\s+Seizures\s+in\s+File\s*[:=]\s*(\d+)', re.IGNORECASE)


def verify_runtime_environment(logger_ref):
    try:
        pywt.cwt(
            np.zeros((4, 64), dtype=np.float32),
            [2.0, 4.0],
            WAVELET_NAME,
            sampling_period=1.0 / Nominal_FS,
            axis=-1,
        )
    except (TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError(f"PyWavelets {pywt.__version__} lacks vectorized axis=-1 support.") from exc
    logger_ref.info("Runtime environment check passed: Vectorized PyWavelets CWT verified.")


def _get_available_gb():
    if PSUTIL_AVAILABLE:
        return psutil.virtual_memory().available / (1024 ** 3)
    return MEMORY_FALLBACK_GB


def _route_decision(n_rows, sid, logger_ref):
    est_peak_gb = n_rows * ROW_GB_ESTIMATE + IN_RAM_OVERHEAD_GB
    available_gb = _get_available_gb()
    safe_gb = available_gb * MEMORY_SAFETY_FRACTION
    logger_ref.info(
        f"  [Memory Router] Subject {sid:02d}: Planned Rows={n_rows:,} | "
        f"Est.Peak={est_peak_gb:.2f} GB | Avail={available_gb:.2f} GB | SafeBudget={safe_gb:.2f} GB"
    )
    if est_peak_gb < safe_gb and n_rows <= MAX_INRAM_EPOCHS * 2:
        logger_ref.info(f"  [Route] Subject {sid:02d} -> IN-RAM (Fast Vectorized Path)")
        return "in-ram"
    logger_ref.info(f"  [Route] Subject {sid:02d} -> DISK-STREAMING (OOM-Proof Safe Path)")
    return "disk-streaming"


def _norm_ch(name):
    if not isinstance(name, str):
        return ""
    name = name.replace('\u200b', '').replace('\ufeff', '').replace('\u00a0', ' ')
    name = name.lower().strip()
    name = re.sub(r'[\s_]+', '-', name)
    name = re.sub(r'-+', '-', name)
    return name.strip('-')


def match_18_channels_strict(available_channels, target_channels, edf_name, logger_ref):
    picks = []
    used_indices = set()
    missing_targets = []
    norm_available = [_norm_ch(ch) for ch in available_channels]

    for target in target_channels:
        t_norm = _norm_ch(target)
        matched_idx = None
        for idx, ch_norm in enumerate(norm_available):
            if idx not in used_indices and ch_norm == t_norm:
                matched_idx = idx
                break
        if matched_idx is None:
            for idx, ch_norm in enumerate(norm_available):
                if idx not in used_indices and (ch_norm.startswith(t_norm + '-') or t_norm.startswith(ch_norm + '-')):
                    matched_idx = idx
                    break
        if matched_idx is None:
            unclaimed = [(i, ch_norm) for i, ch_norm in enumerate(norm_available) if i not in used_indices]
            if unclaimed:
                candidates = [c[1] for c in unclaimed]
                close = difflib.get_close_matches(t_norm, candidates, n=1, cutoff=0.85)
                if close:
                    for i, ch_norm in unclaimed:
                        if ch_norm == close[0]:
                            matched_idx = i
                            logger_ref.info(f"  [Fuzzy Channel Match] {edf_name}: '{target}' -> '{available_channels[i]}'")
                            break
        if matched_idx is not None:
            picks.append(available_channels[matched_idx])
            used_indices.add(matched_idx)
        else:
            missing_targets.append(target)

    if len(picks) != len(target_channels) or missing_targets:
        err_msg = (
            f"Channel mismatch in {edf_name}: matched {len(picks)}/{len(target_channels)}. "
            f"Missing channels: {missing_targets}"
        )
        if STRICT_SAFE_HALT:
            logger_ref.error(f"  [CRITICAL CHANNEL ERROR] {err_msg}")
            raise RuntimeError(err_msg)
        logger_ref.warning(f"  [Channel Skip] {err_msg}")
        return []

    return picks


def find_summary_file(patient_dir):
    try:
        candidates = [
            f for f in os.listdir(patient_dir)
            if 'summary' in f.lower() and not f.lower().endswith(('.edf', '.seizures', '.edf.seizures'))
        ]
        if candidates:
            candidates.sort(key=lambda f: (not f.lower().endswith('.txt'), f.lower()))
            return os.path.join(patient_dir, candidates[0])
    except OSError:
        pass
    return None


def parse_time_val(val_str):
    val_str = val_str.replace("seconds", "").strip()
    if ":" in val_str:
        p = val_str.split(":")
        if len(p) == 3:
            return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
        if len(p) == 2:
            return int(p[0]) * 60 + float(p[1])
    return float(val_str)


def parse_summary_file(summary_path, logger_ref):
    events = {}
    expected_counts = {}
    try:
        with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        err_msg = f"Failed reading summary file {summary_path}: {e}"
        logger_ref.error(err_msg)
        raise RuntimeError(err_msg) from e

    current_file = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("File Name:"):
            current_file = s.split(":", 1)[1].strip()
            events[current_file] = []
            continue
        if current_file is None:
            continue

        m_num = NUM_SEIZURES_RE.match(s)
        if m_num:
            try:
                expected_counts[current_file] = int(m_num.group(1))
            except ValueError as ex:
                if STRICT_SAFE_HALT:
                    raise RuntimeError(f"Invalid seizure count line in {summary_path}: {s}") from ex
            continue

        m_st = SEIZURE_START_RE.match(s)
        if m_st:
            try:
                events[current_file].append([parse_time_val(m_st.group(1)), -1.0])
            except Exception as ex:
                err_msg = f"Failed parsing seizure start time in {summary_path} ({current_file}): {s}"
                logger_ref.error(err_msg)
                if STRICT_SAFE_HALT:
                    raise RuntimeError(err_msg) from ex
            continue

        m_en = SEIZURE_END_RE.match(s)
        if m_en:
            try:
                val = parse_time_val(m_en.group(1))
                if events[current_file] and events[current_file][-1][1] < 0:
                    events[current_file][-1][1] = val
                else:
                    err_msg = f"Orphan seizure end time in {summary_path} ({current_file}): {s}"
                    logger_ref.error(err_msg)
                    if STRICT_SAFE_HALT:
                        raise RuntimeError(err_msg)
            except Exception as ex:
                err_msg = f"Failed parsing seizure end time in {summary_path} ({current_file}): {s}"
                logger_ref.error(err_msg)
                if STRICT_SAFE_HALT:
                    raise RuntimeError(err_msg) from ex
            continue

    for fn in list(events.keys()):
        unclosed = [pair for pair in events[fn] if pair[1] <= pair[0] or pair[0] < 0]
        if unclosed:
            err_msg = f"Invalid or unclosed seizure intervals in {summary_path} ({fn}): {unclosed}"
            logger_ref.error(err_msg)
            if STRICT_SAFE_HALT:
                raise RuntimeError(err_msg)
        valid = [(st, en) for st, en in events[fn] if en > st >= 0.0]
        events[fn] = valid
        exp = expected_counts.get(fn)
        if exp is not None and exp != len(valid):
            err_msg = f"Seizure count mismatch in {summary_path} ({fn}): declared={exp}, parsed={len(valid)}"
            logger_ref.error(err_msg)
            if STRICT_SAFE_HALT:
                raise RuntimeError(err_msg)

    return events


def build_clinical_intervals(seizure_events, total_sec, preictal_sec=PREICTAL_HORIZON_SEC, postictal_sec=POSTICTAL_BUFFER_SEC):
    if not seizure_events:
        return [(0.0, float(total_sec), "interictal")]
    sorted_sz = sorted(seizure_events, key=lambda x: x[0])
    breakpoints = {0.0, float(total_sec)}
    for s, e in sorted_sz:
        if s < 0.0 or e > total_sec + 1.0 or e <= s:
            if STRICT_SAFE_HALT:
                raise RuntimeError(f"Seizure timestamp ({s}, {e}) out of bounds for EDF duration {total_sec:.2f}s.")
        breakpoints.update([
            max(0.0, float(s) - preictal_sec),
            max(0.0, float(s)),
            min(float(total_sec), float(e)),
            min(float(total_sec), float(e) + postictal_sec),
        ])
    pts = sorted(breakpoints)
    raw_intervals = []
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a <= 1e-4:
            continue
        mid = 0.5 * (a + b)
        lbl = "interictal"
        for s, e in sorted_sz:
            if s <= mid <= e:
                lbl = "seizure"
                break
        if lbl == "interictal" and postictal_sec > 0:
            for s, e in sorted_sz:
                if e < mid <= min(total_sec, e + postictal_sec):
                    lbl = "postictal"
                    break
        if lbl == "interictal":
            for s, e in sorted_sz:
                if max(0.0, s - preictal_sec) <= mid < s:
                    lbl = "preictal"
                    break
        raw_intervals.append((a, b, lbl))
    merged = []
    for a, b, lbl in raw_intervals:
        if merged and merged[-1][2] == lbl and abs(merged[-1][1] - a) < 1e-4:
            merged[-1] = (merged[-1][0], b, lbl)
        else:
            merged.append((a, b, lbl))
    return merged


def count_windows_in_intervals(intervals, target_label, win_len, step_sec):
    total = 0
    if step_sec <= 0 or win_len <= 0:
        return 0
    for s, e, lbl in intervals:
        d = e - s
        if lbl == target_label and d >= win_len:
            total += int(np.floor((d - win_len) / step_sec + WIN_COUNT_EPS)) + 1
    return total


def accumulate_interictal_stats(data, fs, intervals, sum_x, sum_x2, count_x):
    n_samples = data.shape[1]
    for s, e, lbl in intervals:
        if lbl == "interictal":
            i0 = max(0, int(round(s * fs)))
            i1 = min(n_samples, int(round(e * fs)))
            if i1 > i0:
                seg = data[:, i0:i1].astype(np.float64)
                sum_x += np.sum(seg, axis=1)
                sum_x2 += np.sum(seg ** 2, axis=1)
                count_x += (i1 - i0)
    return sum_x, sum_x2, count_x


def compute_balanced_stride(intervals, target_pre_count, win_len, logger_ref, sid):
    if target_pre_count <= 0:
        err_msg = f"Subject {sid:02d} has 0 preictal windows."
        if STRICT_SAFE_HALT:
            raise RuntimeError(err_msg)
        return DEFAULT_INTER_STEP_SEC, 0, 0

    total_inter = sum(e - s for s, e, lbl in intervals if lbl == "interictal" and (e - s) >= win_len)
    if total_inter <= 0:
        err_msg = f"Subject {sid:02d} has 0s of clean interictal data."
        if STRICT_SAFE_HALT:
            raise RuntimeError(err_msg)
        return DEFAULT_INTER_STEP_SEC, 0, 0

    analytical = total_inter / float(target_pre_count)
    search_lo = max(MIN_INTER_STEP_SEC, analytical * STRIDE_SEARCH_LOW)
    search_hi = min(MAX_INTER_STEP_SEC, analytical * STRIDE_SEARCH_HIGH)

    def count_at(st):
        return count_windows_in_intervals(intervals, "interictal", win_len, st)

    if search_lo >= search_hi:
        stride = float(np.clip(analytical, MIN_INTER_STEP_SEC, MAX_INTER_STEP_SEC))
        return stride, count_at(stride), target_pre_count

    n_lo = count_at(search_lo)
    n_hi = count_at(search_hi)

    if n_hi >= target_pre_count:
        return search_hi, n_hi, target_pre_count
    if n_lo <= target_pre_count:
        return search_lo, n_lo, target_pre_count

    lo, hi = search_lo, search_hi
    best_stride = lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        n = count_at(mid)
        if n == target_pre_count:
            best_stride = mid
            break
        if n > target_pre_count:
            lo = mid
            best_stride = lo
        else:
            hi = mid

    final_count = count_at(best_stride)
    if final_count < target_pre_count and count_at(lo) >= target_pre_count:
        best_stride = lo
        final_count = count_at(best_stride)

    logger_ref.info(
        f"  [Balance Stride] Subject {sid:02d}: Stride={best_stride:.4f}s "
        f"(Analytical={analytical:.2f}s) -> Planned Interictal={final_count:,} vs Preictal={target_pre_count:,}"
    )
    return best_stride, final_count, target_pre_count


def extract_chronological_windows_from_edf(data, fs, intervals, pre_win_sec, pre_step_sec, inter_win_sec, inter_step_sec, start_block_id):
    segs, labels, block_ids, onsets, win_durs, step_durs = [], [], [], [], [], []
    current_block = int(start_block_id)
    total_samples = data.shape[1]
    for s, e, lbl in intervals:
        if lbl == "preictal":
            win_sec, step_sec, y_val = float(pre_win_sec), float(pre_step_sec), 1
        elif lbl == "interictal":
            win_sec, step_sec, y_val = float(inter_win_sec), float(inter_step_sec), 0
        else:
            continue
        d = e - s
        if d < win_sec:
            continue
        n_samples_win = int(round(win_sec * fs))
        n_win = int(np.floor((d - win_sec) / step_sec + WIN_COUNT_EPS)) + 1
        added_in_block = False
        for i in range(n_win):
            w_start = s + i * step_sec
            w_end = w_start + win_sec
            if w_end > e + WINDOW_FIT_TOL_SEC:
                continue
            i0 = int(round(w_start * fs))
            i1 = i0 + n_samples_win
            if i1 > total_samples:
                continue
            segs.append(data[:, i0:i1].copy())
            labels.append(y_val)
            block_ids.append(current_block)
            onsets.append(float(w_start))
            win_durs.append(float(win_sec))
            step_durs.append(float(step_sec))
            added_in_block = True
        if added_in_block:
            current_block += 1
    return segs, labels, block_ids, onsets, win_durs, step_durs, current_block


def _verify_extracted_subject_metadata(sid, labels_arr, blocks_arr, win_durs_arr, step_durs_arr, onsets_arr, fids_arr, logger_ref):
    n = len(labels_arr)
    if n == 0:
        raise RuntimeError(f"Subject {sid:02d} metadata audit failed: 0 windows present.")
    if not (len(blocks_arr) == len(win_durs_arr) == len(step_durs_arr) == len(onsets_arr) == len(fids_arr) == n):
        raise RuntimeError(f"Subject {sid:02d} metadata length mismatch across arrays.")
    unique_lbls = np.unique(labels_arr)
    if not np.array_equal(np.sort(unique_lbls), np.array([0, 1], dtype='int64')):
        raise RuntimeError(f"Subject {sid:02d} label audit failed: expected binary classes [0, 1], got {unique_lbls.tolist()}")
    n_pre = int((labels_arr == 1).sum())
    n_int = int((labels_arr == 0).sum())
    if n_pre != n_int:
        raise RuntimeError(f"Subject {sid:02d} class balance audit failed: Preictal={n_pre:,} != Interictal={n_int:,}")
    if not np.allclose(win_durs_arr, WINDOW_SEC, atol=1e-4):
        raise RuntimeError(f"Subject {sid:02d} window duration audit failed: non-5.0s window detected.")
    if np.any(step_durs_arr <= 0.0) or not np.all(np.isfinite(step_durs_arr)):
        raise RuntimeError(f"Subject {sid:02d} effective step audit failed: invalid step values.")
    if np.any(np.diff(blocks_arr) < 0):
        raise RuntimeError(f"Subject {sid:02d} block monotonicity audit failed: block_ids decreased.")
    for blk in np.unique(blocks_arr):
        blk_lbls = labels_arr[blocks_arr == blk]
        if len(np.unique(blk_lbls)) != 1:
            raise RuntimeError(f"Subject {sid:02d} block purity audit failed: block_id={blk} contains mixed classes.")
    for fid in np.unique(fids_arr):
        f_onsets = onsets_arr[fids_arr == fid]
        if len(f_onsets) > 1 and np.any(np.diff(f_onsets) <= 0.0):
            raise RuntimeError(f"Subject {sid:02d} chronological audit failed: non-increasing window onsets in edf_file_id={fid}.")
    logger_ref.info(
        f"  [Audit PASS] Subject {sid:02d}: {n:,} windows verified "
        f"(1:1 balanced, 5.0s fixed duration, {len(np.unique(blocks_arr))} pure blocks, strict chronological order)."
    )


def _verify_normalized_chunks(cwt_2d, stft_2d, passband_mask, sid):
    if not np.all(np.isfinite(cwt_2d)) or not np.all(np.isfinite(stft_2d)):
        raise RuntimeError(f"Subject {sid:02d} tensor audit failed: NaN or Inf in normalized features.")
    if float(cwt_2d.min()) < -1e-6 or float(cwt_2d.max()) > 1.0 + 1e-6:
        raise RuntimeError(f"Subject {sid:02d} CWT range audit failed: [{cwt_2d.min():.6f}, {cwt_2d.max():.6f}] outside [0, 1].")
    if float(stft_2d.min()) < -1e-6 or float(stft_2d.max()) > 1.0 + 1e-6:
        raise RuntimeError(f"Subject {sid:02d} STFT range audit failed: [{stft_2d.min():.6f}, {stft_2d.max():.6f}] outside [0, 1].")
    cwt_win_std = cwt_2d.reshape(cwt_2d.shape[0], -1).std(axis=1)
    stft_win_std = stft_2d.reshape(stft_2d.shape[0], -1).std(axis=1)
    if np.any(cwt_win_std <= 1e-12) or np.any(stft_win_std <= 1e-12):
        raise RuntimeError(f"Subject {sid:02d} tensor audit failed: dead/flatline window (std == 0) detected.")
    stopband_mask = ~passband_mask
    if np.any(stopband_mask) and float(np.max(np.abs(stft_2d[:, stopband_mask, :]))) > 1e-7:
        raise RuntimeError(f"Subject {sid:02d} STFT stopband audit failed: non-zero energy outside [0.5, 100.0] Hz.")


def apply_class_balance_blockaware(labels_arr, block_ids_arr, rng, logger_ref, sid):
    unique_labels = np.unique(labels_arr)
    if not np.all(np.isin(unique_labels, [0, 1])):
        err_msg = f"Subject {sid:02d}: Corrupted labels detected: {unique_labels}"
        logger_ref.error(f"  [Label Integrity] {err_msg}")
        raise RuntimeError(err_msg)

    n_pre = int((labels_arr == 1).sum())
    n_inter = int((labels_arr == 0).sum())
    if n_pre == 0 or n_inter == 0:
        err_msg = f"Subject {sid:02d} has empty class (Preictal={n_pre}, Interictal={n_inter})."
        logger_ref.error(f"  [Class Error] {err_msg}")
        raise RuntimeError(err_msg)

    target = min(n_pre, n_inter)
    if n_pre == n_inter:
        return np.ones(len(labels_arr), dtype=bool)

    keep_mask = np.zeros(len(labels_arr), dtype=bool)
    for class_val in (0, 1):
        class_idx = np.where(labels_arr == class_val)[0]
        if len(class_idx) == target:
            keep_mask[class_idx] = True
            continue
        blocks_in_class = block_ids_arr[class_idx]
        unique_blocks = np.unique(blocks_in_class)
        rng.shuffle(unique_blocks)
        keep_idx_parts = []
        accum = 0
        for blk in unique_blocks:
            if accum >= target:
                break
            blk_idx = class_idx[blocks_in_class == blk]
            need = target - accum
            if len(blk_idx) <= need:
                keep_idx_parts.append(blk_idx)
                accum += len(blk_idx)
            else:
                keep_idx_parts.append(blk_idx[:need])
                accum += need
        if keep_idx_parts:
            keep_mask[np.concatenate(keep_idx_parts)] = True

    kept_pre = int((labels_arr[keep_mask] == 1).sum())
    kept_int = int((labels_arr[keep_mask] == 0).sum())
    if kept_pre != target or kept_int != target:
        raise RuntimeError(f"Subject {sid:02d} failed reaching exact target balance: Pre={kept_pre}, Inter={kept_int}, Target={target}")

    logger_ref.info(
        f"  [Block-Aware Balance] Subject {sid:02d}: Raw(Pre={n_pre:,}, Inter={n_inter:,}) -> "
        f"Balanced(Pre={kept_pre:,}, Inter={kept_int:,}, Ratio={kept_int/max(kept_pre,1):.3f})"
    )
    return keep_mask


def get_coregistered_bins_and_centers(n_samples_total, n_steps=FIXED_N_TIME):
    edges = np.linspace(0, n_samples_total, n_steps + 1).round().astype(int)
    bins = [(int(edges[k]), int(edges[k + 1])) for k in range(n_steps)]
    centers = [int(round(0.5 * (edges[k] + edges[k + 1]))) for k in range(n_steps)]
    return bins, centers


def get_cwt_scales(fs):
    target_freqs = np.logspace(np.log10(CWT_MIN_FREQ_HZ), np.log10(CWT_MAX_FREQ_HZ), N_CWT_SCALES)
    return pywt.frequency2scale(WAVELET_NAME, target_freqs / float(fs))


def build_stft_spectral_taper(freqs):
    passband_mask = (freqs >= BANDPASS_LOW) & (freqs <= BANDPASS_HIGH)
    taper = np.zeros_like(freqs, dtype=np.float32)
    taper[passband_mask] = 1.0
    roll_mask = (freqs > STFT_TAPER_START_HZ) & (freqs <= BANDPASS_HIGH)
    if np.any(roll_mask):
        phase = (freqs[roll_mask] - STFT_TAPER_START_HZ) / (BANDPASS_HIGH - STFT_TAPER_START_HZ)
        taper[roll_mask] = (0.5 * (1.0 + np.cos(np.pi * phase))).astype(np.float32)
    return passband_mask, taper


def compute_raw_log_stft_batch(epochs_batch, fs):
    B, C, T = epochs_batch.shape
    _, centers = get_coregistered_bins_and_centers(T, FIXED_N_TIME)
    subwin = int(round(STFT_SUBWIN_SEC * fs))
    if subwin % 2 != 0:
        subwin += 1
    half_w = subwin // 2
    padded = np.pad(epochs_batch, ((0, 0), (0, 0), (half_w, half_w)), mode='reflect')
    slices = np.stack([padded[:, :, c:c + subwin] for c in centers], axis=2)
    win = windows.cosine(subwin, sym=False).astype(np.float32)
    win = win / np.sqrt(np.sum(win ** 2) + 1e-12)
    windowed = slices * win[None, None, None, :]
    n_fft = (TARGET_STFT_FREQ_BINS - 1) * 2
    Z = scipy_rfft(windowed, n=n_fft, axis=-1, workers=FFT_WORKERS)
    Z = np.transpose(Z, (0, 1, 3, 2))
    power = np.abs(Z).astype(np.float32) ** 2
    return np.log10(power + 1e-10).astype(np.float32)


def compute_raw_log_cwt_batch(epochs_batch, fs, scales):
    B, C, T = epochs_batch.shape
    bins, _ = get_coregistered_bins_and_centers(T, FIXED_N_TIME)
    flat_signals = epochs_batch.reshape(B * C, T)
    coeffs, _ = pywt.cwt(flat_signals, scales, WAVELET_NAME, sampling_period=1.0 / fs, axis=-1)
    mag_sq = (np.abs(coeffs).astype(np.float32)) ** 2
    mag_sq = mag_sq.reshape(len(scales), B, C, T).transpose(1, 2, 0, 3)
    pooled = np.stack([np.sqrt(np.mean(mag_sq[:, :, :, b0:b1], axis=-1) + 1e-12) for b0, b1 in bins], axis=-1)
    return np.log10(pooled + 1e-10).astype(np.float32)


def _compute_percentiles_from_subsample(raw_cwt_4d, raw_stft_4d, fs, rng, sid):
    N = raw_cwt_4d.shape[0]
    if N > NORM_SUBSAMPLE_MAX:
        idx = rng.choice(N, NORM_SUBSAMPLE_MAX, replace=False)
        cwt_s = raw_cwt_4d[idx]
        stft_s = raw_stft_4d[idx]
    else:
        cwt_s = raw_cwt_4d
        stft_s = raw_stft_4d

    c_pcts = np.percentile(cwt_s, [NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH], axis=(0, 1, 3), keepdims=True)
    c_low, c_high = c_pcts[0].astype(np.float32), c_pcts[1].astype(np.float32)
    del c_pcts
    if cwt_s is not raw_cwt_4d:
        del cwt_s

    if np.any((c_high - c_low) <= 1e-12):
        raise RuntimeError(f"Subject {sid:02d}: degenerate CWT scale percentile range detected.")

    n_fft = (TARGET_STFT_FREQ_BINS - 1) * 2
    freqs = scipy_rfftfreq(n_fft, d=1.0 / fs)
    passband_mask, spectral_taper = build_stft_spectral_taper(freqs)
    pass_idx = np.where(passband_mask)[0]

    stft_pass_s = stft_s[:, :, pass_idx, :]
    s_pcts = np.percentile(stft_pass_s, [NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH], axis=(0, 1, 3), keepdims=True)
    s_low, s_high = s_pcts[0].astype(np.float32), s_pcts[1].astype(np.float32)
    del stft_pass_s, s_pcts
    if stft_s is not raw_stft_4d:
        del stft_s

    if np.any((s_high - s_low) <= 1e-12):
        raise RuntimeError(f"Subject {sid:02d}: degenerate STFT passband percentile range detected.")

    return c_low, c_high, s_low, s_high, passband_mask, spectral_taper, pass_idx


def _normalize_and_write_chunked(hf, cur_row, raw_cwt_4d, raw_stft_4d, fs, logger_ref, sid):
    N = raw_cwt_4d.shape[0]
    rng = np.random.default_rng(SEED)
    c_low, c_high, s_low, s_high, passband_mask, spectral_taper, pass_idx = _compute_percentiles_from_subsample(
        raw_cwt_4d, raw_stft_4d, fs, rng, sid
    )
    c_denom = np.maximum(c_high - c_low, 1e-8)
    s_denom = np.maximum(s_high - s_low, 1e-8)

    for start in range(0, N, H5_WRITE_CHUNK):
        end = min(start + H5_WRITE_CHUNK, N)
        B = end - start

        cwt_chunk = raw_cwt_4d[start:end]
        cwt_norm = np.clip((cwt_chunk - c_low) / c_denom, 0.0, 1.0).astype(np.float32)
        cwt_out = cwt_norm.transpose(0, 3, 1, 2).reshape(B, FIXED_N_TIME, N_CHANNELS * N_CWT_SCALES)
        del cwt_chunk, cwt_norm

        stft_chunk = raw_stft_4d[start:end]
        stft_pass_norm = np.clip((stft_chunk[:, :, pass_idx, :] - s_low) / s_denom, 0.0, 1.0).astype(np.float32)
        stft_pass_norm *= spectral_taper[None, None, pass_idx, None]
        stft_full = np.zeros((B, N_CHANNELS, TARGET_STFT_FREQ_BINS, FIXED_N_TIME), dtype=np.float32)
        stft_full[:, :, passband_mask, :] = stft_pass_norm
        stft_out = stft_full.transpose(0, 2, 1, 3).reshape(B, TARGET_STFT_FREQ_BINS, N_CHANNELS * FIXED_N_TIME)
        del stft_chunk, stft_pass_norm, stft_full

        _verify_normalized_chunks(cwt_out, stft_out, passband_mask, sid)
        hf["time_matrix"][cur_row + start:cur_row + end] = cwt_out
        hf["freq_matrix"][cur_row + start:cur_row + end] = stft_out
        del cwt_out, stft_out

    hf.flush()
    logger_ref.info(f"    [In-RAM Write] {N:,} normalized rows verified and committed to HDF5.")


def _create_h5_datasets(hf):
    hf.create_dataset(
        "time_matrix",
        shape=(0, FIXED_N_TIME, N_CHANNELS * N_CWT_SCALES),
        maxshape=(None, FIXED_N_TIME, N_CHANNELS * N_CWT_SCALES),
        dtype='float32',
        chunks=(H5_WRITE_CHUNK, FIXED_N_TIME, N_CHANNELS * N_CWT_SCALES),
        compression="gzip",
        shuffle=True,
    )
    hf.create_dataset(
        "freq_matrix",
        shape=(0, TARGET_STFT_FREQ_BINS, N_CHANNELS * FIXED_N_TIME),
        maxshape=(None, TARGET_STFT_FREQ_BINS, N_CHANNELS * FIXED_N_TIME),
        dtype='float32',
        chunks=(16, 128, N_CHANNELS * FIXED_N_TIME),
        compression="gzip",
        shuffle=True,
    )
    for name, dtype in H5_DATASET_SPECS[2:]:
        hf.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(65536,))


def _ensure_h5_datasets(hf, n_new_rows, logger_ref):
    if "time_matrix" not in hf:
        _create_h5_datasets(hf)
    cur = int(hf["time_matrix"].shape[0])
    resized = []
    try:
        for name, _ in H5_DATASET_SPECS:
            hf[name].resize(cur + n_new_rows, axis=0)
            resized.append(name)
    except Exception as e:
        logger_ref.error(f"HDF5 resize failed: {e}. Rolling back {len(resized)} datasets to row {cur}.")
        for name in resized:
            try:
                hf[name].resize(cur, axis=0)
            except Exception:
                pass
        raise
    return cur


def _write_metadata(hf, cur_row, n_total, sid, labels_arr, all_blocks_arr, all_win_durs_arr, all_step_durs_arr, all_onsets_arr, all_fids_arr):
    hf["labels"][cur_row:cur_row + n_total] = labels_arr
    hf["subject_ids"][cur_row:cur_row + n_total] = np.full(n_total, int(sid), dtype='int32')
    hf["raw_subject_ids"][cur_row:cur_row + n_total] = np.full(n_total, int(sid), dtype='int32')
    hf["block_ids"][cur_row:cur_row + n_total] = all_blocks_arr
    hf["window_duration_seconds"][cur_row:cur_row + n_total] = all_win_durs_arr
    hf["effective_step_seconds"][cur_row:cur_row + n_total] = all_step_durs_arr
    weight_arr = np.where(labels_arr == 0, all_step_durs_arr, float(PRE_STEP_SEC)).astype('float32')
    hf["interictal_window_seconds"][cur_row:cur_row + n_total] = weight_arr
    hf["window_onset_seconds"][cur_row:cur_row + n_total] = all_onsets_arr
    hf["edf_file_ids"][cur_row:cur_row + n_total] = all_fids_arr


def _ensure_buffer_capacity(cwt_buf, stft_buf, required_rows, logger_ref):
    cur_cap = cwt_buf.shape[0]
    if required_rows <= cur_cap:
        return cwt_buf, stft_buf
    growth = max(required_rows - cur_cap, max(INITIAL_BUFFER_FLOOR, cur_cap // 4))
    new_cap = cur_cap + growth
    logger_ref.info(f"    [Buffer Expand] Expanding in-RAM buffer: {cur_cap:,} -> {new_cap:,} rows")
    new_cwt = np.empty((new_cap, N_CHANNELS, N_CWT_SCALES, FIXED_N_TIME), dtype=np.float32)
    new_cwt[:cur_cap] = cwt_buf
    del cwt_buf
    new_stft = np.empty((new_cap, N_CHANNELS, TARGET_STFT_FREQ_BINS, FIXED_N_TIME), dtype=np.float32)
    new_stft[:cur_cap] = stft_buf
    del stft_buf
    gc.collect()
    return new_cwt, new_stft


def _accumulate_features_from_edfs(valid_edf_records, subj_mean, subj_std, subj_inter_stride, initial_block_id, initial_edf_fid, write_callback, logger_ref):
    all_labels, all_blocks, all_win_durs, all_step_durs, all_onsets, all_fids = [], [], [], [], [], []
    cur_block_id = int(initial_block_id)
    cur_edf_fid = int(initial_edf_fid)
    scales = get_cwt_scales(Nominal_FS)
    write_ptr = 0

    for edf_file, edf_path, picks, intervals, sfreq_hdr, n_times_hdr in valid_edf_records:
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
            raw.pick_channels(picks, ordered=True)
            raw.notch_filter(
                freqs=NOTCH_FREQS,
                notch_widths=NOTCH_WIDTH,
                method='fir',
                filter_length='auto',
                n_jobs=MNE_FILTER_JOBS,
                verbose='ERROR',
            )
            raw.filter(
                l_freq=BANDPASS_LOW,
                h_freq=BANDPASS_HIGH,
                n_jobs=MNE_FILTER_JOBS,
                verbose='ERROR',
            )
            fs = float(raw.info['sfreq'])
            n_times_actual = int(raw.n_times)
            if abs(fs - Nominal_FS) > FS_TOLERANCE_HZ or abs(fs - sfreq_hdr) > 1e-4 or n_times_actual != n_times_hdr:
                del raw
                err_msg = (
                    f"EDF integrity mismatch in {edf_file}: header=({sfreq_hdr}Hz, {n_times_hdr}) "
                    f"vs loaded=({fs}Hz, {n_times_actual})"
                )
                logger_ref.error(f"  [EDF Integrity Error] {err_msg}")
                raise RuntimeError(err_msg)

            raw_data = raw.get_data().astype(np.float32)
            del raw

            if not np.all(np.isfinite(raw_data)):
                del raw_data
                err_msg = f"Non-finite (NaN/Inf) raw signal values detected in {edf_file}."
                logger_ref.error(f"  [Signal Error] {err_msg}")
                raise RuntimeError(err_msg)

            ch_stds = raw_data.std(axis=1)
            if np.any(ch_stds <= 1e-12):
                dead_chs = [SELECTED_CHANNELS[idx] for idx, val in enumerate(ch_stds) if val <= 1e-12]
                del raw_data
                err_msg = f"Dead/flatline channel(s) detected in {edf_file}: {dead_chs}"
                logger_ref.error(f"  [Channel Signal Error] {err_msg}")
                raise RuntimeError(err_msg)

            if subj_mean is not None and subj_std is not None:
                data = np.clip((raw_data - subj_mean) / (subj_std + 1e-8), -15.0, 15.0).astype(np.float32)
            else:
                m_f = raw_data.mean(axis=1, keepdims=True)
                s_f = raw_data.std(axis=1, keepdims=True) + 1e-8
                data = np.clip((raw_data - m_f) / s_f, -15.0, 15.0).astype(np.float32)
            del raw_data

            segs, lbls, blks, ons, w_durs, s_durs, cur_block_id = extract_chronological_windows_from_edf(
                data, fs, intervals, WINDOW_SEC, PRE_STEP_SEC, WINDOW_SEC, subj_inter_stride, cur_block_id
            )
            del data

            if not segs:
                cur_edf_fid += 1
                continue

            arr_3d = np.stack(segs, axis=0)
            del segs

            for start_idx in range(0, len(arr_3d), CHUNK_BATCH_SIZE):
                chunk = arr_3d[start_idx:start_idx + CHUNK_BATCH_SIZE]
                b_len = len(chunk)
                cwt_b = compute_raw_log_cwt_batch(chunk, fs, scales)
                stft_b = compute_raw_log_stft_batch(chunk, fs)
                if not np.all(np.isfinite(cwt_b)) or not np.all(np.isfinite(stft_b)):
                    del cwt_b, stft_b, arr_3d
                    raise RuntimeError(f"Non-finite CWT/STFT coefficients generated in {edf_file}.")
                write_callback(write_ptr, b_len, cwt_b, stft_b)
                write_ptr += b_len
                del cwt_b, stft_b
            del arr_3d

            all_labels.extend(lbls)
            all_blocks.extend(blks)
            all_win_durs.extend(w_durs)
            all_step_durs.extend(s_durs)
            all_onsets.extend(ons)
            all_fids.extend([cur_edf_fid] * len(lbls))
            cur_edf_fid += 1

        except MemoryError:
            raise
        except Exception as e:
            logger_ref.error(f"  [EDF Processing Error] {edf_file}: {type(e).__name__}: {e}")
            raise

    return all_labels, all_blocks, all_win_durs, all_step_durs, all_onsets, all_fids, cur_block_id, cur_edf_fid, write_ptr


def _finalize_write(hf, sid, labels_arr, all_blocks_arr, all_win_durs_arr, all_step_durs_arr, all_onsets_arr, all_fids_arr, write_callback, logger_ref):
    _verify_extracted_subject_metadata(
        sid, labels_arr, all_blocks_arr, all_win_durs_arr,
        all_step_durs_arr, all_onsets_arr, all_fids_arr, logger_ref
    )
    n_total = len(labels_arr)
    cur_row = None
    try:
        cur_row = _ensure_h5_datasets(hf, n_total, logger_ref)
        _write_metadata(
            hf, cur_row, n_total, sid, labels_arr, all_blocks_arr,
            all_win_durs_arr, all_step_durs_arr, all_onsets_arr, all_fids_arr
        )
        write_callback(hf, cur_row, n_total)
        hf.flush()
        n_pre = int((labels_arr == 1).sum())
        n_int = int((labels_arr == 0).sum())
        logger_ref.info(
            f"  [HDF5 Commit] Subject {sid:02d}: Total={n_total:,} "
            f"(Preictal={n_pre:,}, Interictal={n_int:,}, Ratio={n_int/max(n_pre,1):.3f})"
        )
        return True
    except Exception as e:
        logger_ref.error(f"  [Commit Failed] Subject {sid:02d}: {type(e).__name__}: {e}")
        if cur_row is not None:
            for name, _ in H5_DATASET_SPECS:
                if name in hf and hf[name].shape[0] > cur_row:
                    try:
                        hf[name].resize(cur_row, axis=0)
                    except Exception:
                        pass
            try:
                hf.flush()
            except Exception:
                pass
        raise


def _process_in_ram(sid, valid_edf_records, subj_mean, subj_std, subj_inter_stride, global_block_id, global_edf_fid, hf, logger_ref, total_pre, planned_inter):
    n_planned = max(INITIAL_BUFFER_FLOOR, total_pre + planned_inter)
    raw_cwt_4d = np.empty((n_planned, N_CHANNELS, N_CWT_SCALES, FIXED_N_TIME), dtype=np.float32)
    raw_stft_4d = np.empty((n_planned, N_CHANNELS, TARGET_STFT_FREQ_BINS, FIXED_N_TIME), dtype=np.float32)

    def write_cb(ptr, b_len, cwt_b, stft_b):
        nonlocal raw_cwt_4d, raw_stft_4d
        if ptr + b_len > raw_cwt_4d.shape[0]:
            raw_cwt_4d, raw_stft_4d = _ensure_buffer_capacity(raw_cwt_4d, raw_stft_4d, ptr + b_len, logger_ref)
        raw_cwt_4d[ptr:ptr + b_len] = cwt_b
        raw_stft_4d[ptr:ptr + b_len] = stft_b

    try:
        (all_labels, all_blocks, all_win_durs, all_step_durs, all_onsets, all_fids,
         cur_block_id, cur_edf_fid, write_ptr) = _accumulate_features_from_edfs(
            valid_edf_records, subj_mean, subj_std, subj_inter_stride,
            global_block_id, global_edf_fid, write_cb, logger_ref
        )
    except Exception:
        del raw_cwt_4d, raw_stft_4d
        gc.collect()
        raise

    if write_ptr == 0 or not all_labels or len(all_labels) != write_ptr:
        del raw_cwt_4d, raw_stft_4d
        gc.collect()
        err_msg = f"Subject {sid:02d} in-RAM extraction mismatch: write_ptr={write_ptr}, labels={len(all_labels)}"
        logger_ref.error(f"  [Extraction Error] {err_msg}")
        raise RuntimeError(err_msg)

    if write_ptr < raw_cwt_4d.shape[0]:
        raw_cwt_4d = raw_cwt_4d[:write_ptr]
        raw_stft_4d = raw_stft_4d[:write_ptr]

    labels_arr = np.asarray(all_labels, dtype='int64')
    all_blocks_arr = np.asarray(all_blocks, dtype='int32')
    all_win_durs_arr = np.asarray(all_win_durs, dtype='float32')
    all_step_durs_arr = np.asarray(all_step_durs, dtype='float32')
    all_onsets_arr = np.asarray(all_onsets, dtype='float32')
    all_fids_arr = np.asarray(all_fids, dtype='int32')

    rng = np.random.default_rng(SEED + sid)
    keep_mask = apply_class_balance_blockaware(labels_arr, all_blocks_arr, rng, logger_ref, sid)
    keep_idx = np.where(keep_mask)[0]

    for i, src in enumerate(keep_idx):
        if i != src:
            raw_cwt_4d[i] = raw_cwt_4d[src]
            raw_stft_4d[i] = raw_stft_4d[src]
    n_keep = len(keep_idx)
    raw_cwt_4d = raw_cwt_4d[:n_keep]
    raw_stft_4d = raw_stft_4d[:n_keep]

    labels_arr = labels_arr[keep_mask]
    all_blocks_arr = all_blocks_arr[keep_mask]
    all_win_durs_arr = all_win_durs_arr[keep_mask]
    all_step_durs_arr = all_step_durs_arr[keep_mask]
    all_onsets_arr = all_onsets_arr[keep_mask]
    all_fids_arr = all_fids_arr[keep_mask]

    def final_cb(hf_final, cur_row, n_total):
        _normalize_and_write_chunked(hf_final, cur_row, raw_cwt_4d, raw_stft_4d, Nominal_FS, logger_ref, sid)

    try:
        _finalize_write(
            hf, sid, labels_arr, all_blocks_arr, all_win_durs_arr,
            all_step_durs_arr, all_onsets_arr, all_fids_arr, final_cb, logger_ref
        )
    finally:
        del raw_cwt_4d, raw_stft_4d
        gc.collect()

    return None, cur_block_id, cur_edf_fid, float(subj_inter_stride)


def _process_disk_streaming(sid, valid_edf_records, subj_mean, subj_std, subj_inter_stride, global_block_id, global_edf_fid, hf, logger_ref, total_pre, planned_inter):
    n_planned = total_pre + planned_inter
    if n_planned == 0:
        raise RuntimeError(f"Subject {sid:02d} has 0 planned windows.")

    temp_h5_path = os.path.join(SAVE_DIR, f"temp_feat_{sid:02d}.h5")
    if os.path.exists(temp_h5_path):
        try:
            os.remove(temp_h5_path)
        except Exception:
            pass

    logger_ref.info(f"  [Disk Streaming] Initializing scratch HDF5: {temp_h5_path}")
    cur_block_id = int(global_block_id)
    cur_edf_fid = int(global_edf_fid)

    try:
        with h5py.File(temp_h5_path, 'w') as temp_hf:
            temp_hf.create_dataset(
                "raw_cwt",
                shape=(0, N_CHANNELS, N_CWT_SCALES, FIXED_N_TIME),
                maxshape=(None, N_CHANNELS, N_CWT_SCALES, FIXED_N_TIME),
                dtype='float32',
                chunks=(H5_WRITE_CHUNK, N_CHANNELS, N_CWT_SCALES, FIXED_N_TIME),
            )
            temp_hf.create_dataset(
                "raw_stft",
                shape=(0, N_CHANNELS, TARGET_STFT_FREQ_BINS, FIXED_N_TIME),
                maxshape=(None, N_CHANNELS, TARGET_STFT_FREQ_BINS, FIXED_N_TIME),
                dtype='float32',
                chunks=(16, N_CHANNELS, TARGET_STFT_FREQ_BINS, FIXED_N_TIME),
            )

            def temp_write_cb(ptr, b_len, cwt_b, stft_b):
                temp_hf["raw_cwt"].resize(ptr + b_len, axis=0)
                temp_hf["raw_stft"].resize(ptr + b_len, axis=0)
                temp_hf["raw_cwt"][ptr:ptr + b_len] = cwt_b
                temp_hf["raw_stft"][ptr:ptr + b_len] = stft_b

            (all_labels, all_blocks, all_win_durs, all_step_durs, all_onsets, all_fids,
             cur_block_id, cur_edf_fid, write_ptr) = _accumulate_features_from_edfs(
                valid_edf_records, subj_mean, subj_std, subj_inter_stride,
                global_block_id, global_edf_fid, temp_write_cb, logger_ref
            )

            if write_ptr == 0 or not all_labels or len(all_labels) != write_ptr:
                err_msg = f"Subject {sid:02d} disk-streaming extraction mismatch: write_ptr={write_ptr}, labels={len(all_labels)}"
                logger_ref.error(f"  [Extraction Error] {err_msg}")
                raise RuntimeError(err_msg)

            labels_arr = np.asarray(all_labels, dtype='int64')
            all_blocks_arr = np.asarray(all_blocks, dtype='int32')
            all_win_durs_arr = np.asarray(all_win_durs, dtype='float32')
            all_step_durs_arr = np.asarray(all_step_durs, dtype='float32')
            all_onsets_arr = np.asarray(all_onsets, dtype='float32')
            all_fids_arr = np.asarray(all_fids, dtype='int32')

            rng = np.random.default_rng(SEED + sid)
            keep_mask = apply_class_balance_blockaware(labels_arr, all_blocks_arr, rng, logger_ref, sid)

            labels_arr = labels_arr[keep_mask]
            all_blocks_arr = all_blocks_arr[keep_mask]
            all_win_durs_arr = all_win_durs_arr[keep_mask]
            all_step_durs_arr = all_step_durs_arr[keep_mask]
            all_onsets_arr = all_onsets_arr[keep_mask]
            all_fids_arr = all_fids_arr[keep_mask]
            n_total = len(labels_arr)

            valid_idx = np.where(keep_mask)[0]
            sample_idx = rng.choice(valid_idx, min(len(valid_idx), NORM_SUBSAMPLE_MAX), replace=False)
            sample_idx.sort()

            cwt_s = temp_hf['raw_cwt'][sample_idx]
            c_pcts = np.percentile(cwt_s, [NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH], axis=(0, 1, 3), keepdims=True, overwrite_input=True)
            c_low, c_high = c_pcts[0].astype(np.float32), c_pcts[1].astype(np.float32)
            del cwt_s, c_pcts

            if np.any((c_high - c_low) <= 1e-12):
                raise RuntimeError(f"Subject {sid:02d}: degenerate CWT scale percentile range in disk-streaming.")

            n_fft = (TARGET_STFT_FREQ_BINS - 1) * 2
            freqs = scipy_rfftfreq(n_fft, d=1.0 / Nominal_FS)
            passband_mask, spectral_taper = build_stft_spectral_taper(freqs)
            pass_idx = np.where(passband_mask)[0]

            stft_s_pass = temp_hf['raw_stft'][sample_idx][:, :, pass_idx, :]
            s_pcts = np.percentile(stft_s_pass, [NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH], axis=(0, 1, 3), keepdims=True, overwrite_input=True)
            s_low, s_high = s_pcts[0].astype(np.float32), s_pcts[1].astype(np.float32)
            del stft_s_pass, s_pcts

            if np.any((s_high - s_low) <= 1e-12):
                raise RuntimeError(f"Subject {sid:02d}: degenerate STFT passband percentile range in disk-streaming.")

            c_denom = np.maximum(c_high - c_low, 1e-8)
            s_denom = np.maximum(s_high - s_low, 1e-8)

            def final_cb(hf_main, cur_row, n_write):
                write_offset = 0
                for chunk_start in range(0, write_ptr, H5_WRITE_CHUNK):
                    chunk_end = min(chunk_start + H5_WRITE_CHUNK, write_ptr)
                    mask_chunk = keep_mask[chunk_start:chunk_end]
                    if not np.any(mask_chunk):
                        continue
                    B = int(mask_chunk.sum())

                    cwt_chunk = temp_hf['raw_cwt'][chunk_start:chunk_end][mask_chunk]
                    cwt_norm = np.clip((cwt_chunk - c_low) / c_denom, 0.0, 1.0).astype(np.float32)
                    cwt_out = cwt_norm.transpose(0, 3, 1, 2).reshape(B, FIXED_N_TIME, N_CHANNELS * N_CWT_SCALES)
                    del cwt_chunk, cwt_norm

                    stft_chunk = temp_hf['raw_stft'][chunk_start:chunk_end][mask_chunk]
                    stft_pass_norm = np.clip((stft_chunk[:, :, pass_idx, :] - s_low) / s_denom, 0.0, 1.0).astype(np.float32)
                    stft_pass_norm *= spectral_taper[None, None, pass_idx, None]
                    stft_full = np.zeros((B, N_CHANNELS, TARGET_STFT_FREQ_BINS, FIXED_N_TIME), dtype=np.float32)
                    stft_full[:, :, passband_mask, :] = stft_pass_norm
                    stft_out = stft_full.transpose(0, 2, 1, 3).reshape(B, TARGET_STFT_FREQ_BINS, N_CHANNELS * FIXED_N_TIME)
                    del stft_chunk, stft_pass_norm, stft_full

                    _verify_normalized_chunks(cwt_out, stft_out, passband_mask, sid)
                    hf_main["time_matrix"][cur_row + write_offset:cur_row + write_offset + B] = cwt_out
                    hf_main["freq_matrix"][cur_row + write_offset:cur_row + write_offset + B] = stft_out
                    del cwt_out, stft_out
                    write_offset += B
                if write_offset != n_write:
                    raise RuntimeError(f"Subject {sid:02d} disk-streaming write count mismatch: {write_offset} != {n_write}")
                logger_ref.info(f"    [Disk Streaming Write] {n_total:,} normalized rows verified and committed to HDF5.")

            _finalize_write(
                hf, sid, labels_arr, all_blocks_arr, all_win_durs_arr,
                all_step_durs_arr, all_onsets_arr, all_fids_arr, final_cb, logger_ref
            )
    finally:
        if os.path.exists(temp_h5_path):
            try:
                os.remove(temp_h5_path)
            except Exception:
                pass
        gc.collect()

    return None, cur_block_id, cur_edf_fid, float(subj_inter_stride)


def process_single_patient(patient_dir, sid, global_block_id, global_edf_fid, hf, logger_ref):
    summary_path = find_summary_file(patient_dir)
    if not summary_path:
        err_msg = f"No summary file found in {patient_dir} for Subject {sid:02d}."
        logger_ref.error(err_msg)
        raise RuntimeError(err_msg)

    logger_ref.info(f"  [Summary] Reading {os.path.basename(summary_path)}")
    seizure_dict = parse_summary_file(summary_path, logger_ref)
    edf_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('.edf') and not f.endswith('.seizures')])
    if not edf_files:
        err_msg = f"No EDF files found in {patient_dir} for Subject {sid:02d}."
        logger_ref.error(err_msg)
        raise RuntimeError(err_msg)

    edf_set = set(edf_files)
    missing_seizure_edfs = [fn for fn, sz_list in seizure_dict.items() if sz_list and fn not in edf_set]
    if missing_seizure_edfs:
        err_msg = f"Subject {sid:02d} summary lists seizures in missing EDF file(s): {missing_seizure_edfs}"
        logger_ref.error(f"  [Missing EDF Error] {err_msg}")
        raise RuntimeError(err_msg)

    valid_edf_records, total_pre, total_inter_sec = [], 0, 0.0
    sum_x = np.zeros(N_CHANNELS, dtype=np.float64)
    sum_x2 = np.zeros(N_CHANNELS, dtype=np.float64)
    count_x = 0

    for edf_file in edf_files:
        edf_path = os.path.join(patient_dir, edf_file)
        try:
            raw_hdr = mne.io.read_raw_edf(edf_path, preload=False, verbose='ERROR')
        except Exception as ex:
            err_msg = f"Corrupted EDF header in {edf_file}: {ex}"
            logger_ref.error(f"  [Header Error] {err_msg}")
            raise RuntimeError(err_msg) from ex

        picks = match_18_channels_strict(raw_hdr.ch_names, SELECTED_CHANNELS, edf_file, logger_ref)
        if not picks:
            continue

        sfreq_hdr = float(raw_hdr.info['sfreq'])
        if abs(sfreq_hdr - Nominal_FS) > FS_TOLERANCE_HZ:
            err_msg = f"Sampling rate mismatch in {edf_file}: {sfreq_hdr} Hz (expected {Nominal_FS} Hz)"
            logger_ref.error(f"  [Sampling Rate Error] {err_msg}")
            raise RuntimeError(err_msg)

        n_times_hdr = int(raw_hdr.n_times)
        total_sec = float(n_times_hdr) / sfreq_hdr
        if total_sec < WINDOW_SEC:
            err_msg = f"EDF file {edf_file} is shorter ({total_sec:.2f}s) than 5.0s window."
            logger_ref.error(f"  [Duration Error] {err_msg}")
            raise RuntimeError(err_msg)

        intervals = build_clinical_intervals(seizure_dict.get(edf_file, []), total_sec)
        n_pre_file = count_windows_in_intervals(intervals, "preictal", WINDOW_SEC, PRE_STEP_SEC)
        inter_sec_file = sum(e - s for s, e, lbl in intervals if lbl == "interictal" and (e - s) >= WINDOW_SEC)
        total_pre += n_pre_file
        total_inter_sec += inter_sec_file
        valid_edf_records.append((edf_file, edf_path, picks, intervals, sfreq_hdr, n_times_hdr))

        if inter_sec_file >= 300.0 and count_x < MAX_STAT_ACCUMULATION_SAMPLES:
            try:
                raw_cal = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
                raw_cal.pick_channels(picks, ordered=True)
                raw_cal.notch_filter(
                    freqs=NOTCH_FREQS,
                    notch_widths=NOTCH_WIDTH,
                    method='fir',
                    filter_length='auto',
                    n_jobs=MNE_FILTER_JOBS,
                    verbose='ERROR',
                )
                raw_cal.filter(
                    l_freq=BANDPASS_LOW,
                    h_freq=BANDPASS_HIGH,
                    n_jobs=MNE_FILTER_JOBS,
                    verbose='ERROR',
                )
                sum_x, sum_x2, count_x = accumulate_interictal_stats(
                    raw_cal.get_data(), sfreq_hdr, intervals, sum_x, sum_x2, count_x
                )
                del raw_cal
            except Exception as ex:
                logger_ref.error(f"  [Pass 1 Baseline Error] {edf_file}: {ex}")
                raise RuntimeError(f"Failed baseline calibration on {edf_file}: {ex}") from ex

    if len(valid_edf_records) != len(edf_files):
        raise RuntimeError(
            f"Subject {sid:02d}: only {len(valid_edf_records)}/{len(edf_files)} EDF files passed validation."
        )

    if count_x >= MIN_STAT_SAMPLES:
        subj_mean = (sum_x / count_x).reshape(N_CHANNELS, 1).astype(np.float32)
        subj_std = np.sqrt(np.maximum((sum_x2 / count_x) - (sum_x / count_x) ** 2, 1e-12)).reshape(N_CHANNELS, 1).astype(np.float32)
    else:
        subj_mean, subj_std = None, None

    all_intervals = []
    for _, _, _, iv, _, _ in valid_edf_records:
        all_intervals.extend(iv)
    subj_inter_stride, planned_inter, _ = compute_balanced_stride(all_intervals, total_pre, WINDOW_SEC, logger_ref, sid)

    route = _route_decision(total_pre + max(planned_inter, total_pre), sid, logger_ref)
    if route == "in-ram":
        try:
            return _process_in_ram(
                sid, valid_edf_records, subj_mean, subj_std, subj_inter_stride,
                global_block_id, global_edf_fid, hf, logger_ref, total_pre, planned_inter
            )
        except MemoryError:
            gc.collect()
            logger_ref.warning(
                f"  [OOM Guard] MemoryError caught during In-RAM processing of Subject {sid:02d}. "
                f"Automatically falling back to Disk-Streaming!"
            )
            return _process_disk_streaming(
                sid, valid_edf_records, subj_mean, subj_std, subj_inter_stride,
                global_block_id, global_edf_fid, hf, logger_ref, total_pre, planned_inter
            )
    return _process_disk_streaming(
        sid, valid_edf_records, subj_mean, subj_std, subj_inter_stride,
        global_block_id, global_edf_fid, hf, logger_ref, total_pre, planned_inter
    )


def _stream_combine_subjects_in_h5(hf, source_sids, target_sid, logger_ref):
    ranges = []
    for src_sid in source_sids:
        mask = hf["subject_ids"][:] == src_sid
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise RuntimeError(f"Source Subject {src_sid} has 0 rows in HDF5; cannot build Combined Subject {target_sid}.")
        ranges.append((int(idx[0]), int(idx[-1]) + 1) if int(idx[-1]) - int(idx[0]) + 1 == len(idx) else idx.tolist())

    n_total = sum((r[1] - r[0]) if isinstance(r, tuple) else len(r) for r in ranges)
    dst_pos = _ensure_h5_datasets(hf, n_total, logger_ref)
    start_pos = dst_pos

    try:
        for r in ranges:
            if isinstance(r, tuple):
                src_start, src_end = r
                for off in range(src_start, src_end, H5_WRITE_CHUNK):
                    B = min(off + H5_WRITE_CHUNK, src_end) - off
                    for name, _ in H5_DATASET_SPECS:
                        if name == "subject_ids":
                            hf[name][dst_pos:dst_pos + B] = np.full(B, int(target_sid), dtype='int32')
                        else:
                            block = hf[name][off:off + B]
                            hf[name][dst_pos:dst_pos + B] = block
                    dst_pos += B
            else:
                idx_arr = np.array(r, dtype=np.int64)
                for off in range(0, len(idx_arr), H5_WRITE_CHUNK):
                    chunk_idx = idx_arr[off:off + H5_WRITE_CHUNK]
                    B = len(chunk_idx)
                    for name, _ in H5_DATASET_SPECS:
                        if name == "subject_ids":
                            hf[name][dst_pos:dst_pos + B] = np.full(B, int(target_sid), dtype='int32')
                        else:
                            block = hf[name][chunk_idx.tolist()]
                            hf[name][dst_pos:dst_pos + B] = block
                    dst_pos += B
        hf.flush()
        logger_ref.info(f"  [Combined Subject] Subject {target_sid} built from {source_sids} ({n_total:,} rows).")
        return True
    except Exception as e:
        logger_ref.error(f"  [Combine Failed] Error combining subjects {source_sids}: {e}. Rolling back to row {start_pos}.")
        for name, _ in H5_DATASET_SPECS:
            try:
                hf[name].resize(start_pos, axis=0)
            except Exception:
                pass
        hf.flush()
        return False


def _remove_subjects_from_h5(hf, target_sids, logger_ref):
    sids_all = hf["subject_ids"][:]
    keep_mask = ~np.isin(sids_all, target_sids)
    if np.all(keep_mask):
        return
    keep_idx = np.where(keep_mask)[0]
    n_keep = len(keep_idx)
    logger_ref.info(f"  [Compaction] Removing separate subjects {target_sids}; retaining {n_keep:,} rows.")
    write_pos = 0
    n_chunks = (n_keep + H5_WRITE_CHUNK - 1) // H5_WRITE_CHUNK
    for c in range(n_chunks):
        chunk_idx = keep_idx[c * H5_WRITE_CHUNK:min((c + 1) * H5_WRITE_CHUNK, n_keep)]
        B = len(chunk_idx)
        is_contig = (int(chunk_idx[-1]) - int(chunk_idx[0]) + 1 == B)
        if is_contig and int(chunk_idx[0]) == write_pos:
            write_pos += B
            continue
        sel = slice(int(chunk_idx[0]), int(chunk_idx[-1]) + 1) if is_contig else chunk_idx.tolist()
        for name, _ in H5_DATASET_SPECS:
            block = hf[name][sel]
            hf[name][write_pos:write_pos + B] = block
        write_pos += B
    for name, _ in H5_DATASET_SPECS:
        hf[name].resize(n_keep, axis=0)
    hf.flush()


def _verify_final_h5_integrity(hf, logger_ref):
    expected_rows = int(hf["time_matrix"].shape[0])
    if expected_rows == 0:
        raise RuntimeError("Final HDF5 verification failed: 0 total rows in file.")
    for name, _ in H5_DATASET_SPECS:
        if name not in hf:
            raise RuntimeError(f"Final HDF5 verification failed: missing dataset '{name}'.")
        if int(hf[name].shape[0]) != expected_rows:
            raise RuntimeError(
                f"Final HDF5 row alignment mismatch: '{name}' has {hf[name].shape[0]} rows vs {expected_rows}."
            )
    sids = hf["subject_ids"][:]
    raw_sids = hf["raw_subject_ids"][:]
    labels = hf["labels"][:]
    for sid in np.unique(sids):
        mask = sids == sid
        n_pre = int((labels[mask] == 1).sum())
        n_int = int((labels[mask] == 0).sum())
        if n_pre == 0 or n_int == 0 or n_pre != n_int:
            raise RuntimeError(
                f"Final HDF5 class balance verification failed for Subject {sid}: Pre={n_pre}, Inter={n_int}"
            )
    if INCLUDE_COMBINED_1_21 and COMBINED_SUBJECT_ID in sids and INCLUDE_SEPARATE_1_AND_21:
        n1 = int((sids == 1).sum())
        n21 = int((sids == 21).sum())
        n99 = int((sids == COMBINED_SUBJECT_ID).sum())
        n99_1 = int(((sids == COMBINED_SUBJECT_ID) & (raw_sids == 1)).sum())
        n99_21 = int(((sids == COMBINED_SUBJECT_ID) & (raw_sids == 21)).sum())
        if n99 != n1 + n21 or n99_1 != n1 or n99_21 != n21:
            raise RuntimeError(
                f"Subject 99 provenance mismatch: N99={n99} (from_1={n99_1}, from_21={n99_21}) vs N1={n1}, N21={n21}."
            )
    logger_ref.info(
        f"  [Final HDF5 Audit PASS] Total Rows={expected_rows:,} | "
        f"Subjects={sorted(np.unique(sids).tolist())} | All 11 datasets aligned and 1:1 balanced."
    )


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    mne.set_log_level('ERROR')
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("DBTR-Preprocessor-v13")

    logger.info("=" * 78)
    logger.info(" DBTR-AGF PRODUCTION PREPROCESSING PIPELINE (v13.0.0 - ULTRA-SAFE AUTO-ROUTER) ")
    logger.info("=" * 78)

    try:
        verify_runtime_environment(logger)
        logger.info(f"psutil available: {PSUTIL_AVAILABLE} | Available RAM: {_get_available_gb():.2f} GB")

        if not os.path.isdir(BASE_DIR):
            raise FileNotFoundError(f"BASE_DIR not found: {BASE_DIR}")
        os.makedirs(SAVE_DIR, exist_ok=True)

        if CLEAN_KNOWN_MMAP_CACHES:
            for m_dir in KNOWN_MMAP_DIRS:
                if os.path.isdir(m_dir):
                    for m_name in ("_fast_t_fp16.mmap", "_fast_f_fp16.mmap"):
                        m_path = os.path.join(m_dir, m_name)
                        if os.path.exists(m_path):
                            try:
                                os.remove(m_path)
                            except Exception:
                                pass

        patient_dirs = sorted([
            d for d in os.listdir(BASE_DIR)
            if os.path.isdir(os.path.join(BASE_DIR, d)) and re.match(r'^chb\d{1,2}$', d.lower())
        ])
        if not patient_dirs:
            raise RuntimeError(f"No chbXX folders found inside {BASE_DIR}")

        global_block_id = 0
        global_edf_fid = 1
        processed_subjects = set()
        empirical_strides = []
        open_mode = "a" if os.path.exists(H5_FILE) else "w"

        with h5py.File(H5_FILE, open_mode, libver="latest") as hf:
            hf.attrs["pipeline_version"] = "13.0.0"
            hf.attrs["interictal_sampling_mode"] = INTERICTAL_SAMPLING_MODE
            hf.attrs["interictal_window_seconds_semantics"] = "per_epoch_effective_step_weight_seconds"
            hf.attrs["window_sec"] = float(WINDOW_SEC)
            hf.attrs["pre_step_sec"] = float(PRE_STEP_SEC)
            hf.attrs["inter_win_sec"] = float(WINDOW_SEC)
            hf.attrs["inter_stride_range_sec"] = np.array([MIN_INTER_STEP_SEC, MAX_INTER_STEP_SEC], dtype='float32')
            hf.attrs["stft_subwin_sec"] = float(STFT_SUBWIN_SEC)
            hf.attrs["stft_n_fft"] = int((TARGET_STFT_FREQ_BINS - 1) * 2)
            hf.attrs["stft_bin_spacing_hz"] = float(Nominal_FS / ((TARGET_STFT_FREQ_BINS - 1) * 2))
            hf.attrs["stft_resolution_hz"] = float(Nominal_FS / int(round(STFT_SUBWIN_SEC * Nominal_FS)))
            hf.attrs["preictal_horizon_sec"] = float(PREICTAL_HORIZON_SEC)
            hf.attrs["postictal_buffer_sec"] = float(POSTICTAL_BUFFER_SEC)
            hf.attrs["cwt_freq_range_hz"] = np.array([CWT_MIN_FREQ_HZ, CWT_MAX_FREQ_HZ], dtype='float32')
            hf.attrs["cwt_wavelet"] = WAVELET_NAME
            hf.attrs["combined_99_sources"] = np.array([1, 21], dtype='int32')
            hf.attrs["combined_99_norm_mode"] = "concatenated_per_subject_normalized_tensors"
            hf.attrs["boundary_resolution_order"] = "seizure > postictal > preictal > interictal"
            hf.attrs["normalization_scheme"] = "subject_interictal_zscore + per_band_cross_epoch_minmax"
            hf.attrs["class_balance_scheme"] = "binary_search_stride + block_aware_subsampling"
            hf.attrs["routing_mode"] = "hybrid_dynamic_by_n_epochs"
            hf.attrs["max_inram_epochs"] = int(MAX_INRAM_EPOCHS)
            hf.attrs["norm_subsample_max"] = int(NORM_SUBSAMPLE_MAX)
            hf.attrs["h5_write_chunk"] = int(H5_WRITE_CHUNK)
            hf.attrs["selected_channels"] = np.array(SELECTED_CHANNELS, dtype=h5py.string_dtype())
            hf.attrs["time_matrix_layout"] = "N, time, channel_x_cwt_scale"
            hf.attrs["freq_matrix_layout"] = "N, freq_bin, channel_x_time"
            n_fft_attr = (TARGET_STFT_FREQ_BINS - 1) * 2
            hf.attrs["stft_freqs_hz"] = scipy_rfftfreq(n_fft_attr, d=1.0 / Nominal_FS).astype(np.float32)
            hf.attrs["cwt_scales"] = get_cwt_scales(Nominal_FS).astype(np.float32)

            if "subject_ids" in hf and hf["subject_ids"].shape[0] > 0:
                processed_subjects = set(np.unique(hf["subject_ids"][:]).tolist())
                logger.info(f"Resuming execution. Subjects already in HDF5: {sorted(list(processed_subjects))}")
                if "block_ids" in hf and hf["block_ids"].shape[0] > 0:
                    global_block_id = int(hf["block_ids"][:].max()) + 1
                if "edf_file_ids" in hf and hf["edf_file_ids"].shape[0] > 0:
                    global_edf_fid = int(hf["edf_file_ids"][:].max()) + 1

            for d in patient_dirs:
                m = re.match(r'^chb(\d{1,2})$', d.lower())
                if not m:
                    continue
                sid = int(m.group(1))
                if sid in EXCLUDED_SUBJECTS:
                    continue
                if sid in processed_subjects:
                    continue
                if sid in (1, 21) and not INCLUDE_SEPARATE_1_AND_21 and COMBINED_SUBJECT_ID in processed_subjects:
                    continue

                logger.info(f">>> Processing Subject {sid:02d} (Folder: {d}) <<<")
                t0 = time.perf_counter()
                _, global_block_id, global_edf_fid, used_stride = process_single_patient(
                    os.path.join(BASE_DIR, d), sid, global_block_id, global_edf_fid, hf, logger
                )
                empirical_strides.append(used_stride)
                processed_subjects.add(sid)
                logger.info(f"  Subject {sid:02d} completed in {time.perf_counter() - t0:.1f}s.")

            if empirical_strides:
                hf.attrs["empirical_median_interictal_stride_sec"] = float(np.median(empirical_strides))

            if INCLUDE_COMBINED_1_21 and "subject_ids" in hf:
                current_sids = set(np.unique(hf["subject_ids"][:]).tolist())
                if COMBINED_SUBJECT_ID not in current_sids and 1 in current_sids and 21 in current_sids:
                    logger.info(f">>> Building Combined Subject {COMBINED_SUBJECT_ID} from Subjects 1 + 21 <<<")
                    combine_success = _stream_combine_subjects_in_h5(hf, [1, 21], COMBINED_SUBJECT_ID, logger)
                    if combine_success:
                        current_sids.add(COMBINED_SUBJECT_ID)
                    else:
                        raise RuntimeError(f"Failed to build combined Subject {COMBINED_SUBJECT_ID}.")
                if not INCLUDE_SEPARATE_1_AND_21 and COMBINED_SUBJECT_ID in current_sids:
                    _remove_subjects_from_h5(hf, [1, 21], logger)

            _verify_final_h5_integrity(hf, logger)
            hf.flush()

        logger.info("=" * 78)
        logger.info(f" PREPROCESSING v13.0.0 COMPLETED SUCCESSFULLY! File: {H5_FILE}")
        logger.info("=" * 78)

    except Exception as fatal_err:
        logger.error(f"CRITICAL HALT: {fatal_err}. HDF5 file flushed and safely closed.")
        sys.exit(1)
