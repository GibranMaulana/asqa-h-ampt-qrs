"""
Exact evaluation metrics, dataset loaders, RR/HR MAE, and Wilcoxon statistical tests.

Guarantees 100% mathematical parity with acy_ampt/mit_processing/mit_analysis.py
and result_replication/REPLICATION_REPORT.md (Neri et al., Sensors 2023).
"""

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import wfdb
from scipy.stats import ttest_rel, wilcoxon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACY_DIR = os.path.join(PROJECT_ROOT, "acy_ampt")
REPLICATION_DIR = os.path.join(PROJECT_ROOT, "result_replication")

if ACY_DIR not in sys.path:
    sys.path.insert(0, ACY_DIR)

from mit_processing.load_annotation import (  # noqa: E402
    loadAnnotationSampleFromPath,
    loadAnnotationSampleFromPathD,
    loadAnnotationSampleFromPathSinus,
)

TIME_WINDOW_MS = 150  # ANSI/AAMI EC57 tolerance window (ms)

B2_PACED_EXCLUDED = {"102.dat", "104.dat", "107.dat", "217.dat"}

DATASETS: List[Dict] = [
    {
        "code": "A1",
        "name": "High Quality (PhysioNet 2014 SET-P)",
        "rel_path": "ECG DBs/A - Physionet Challange 2014/A1 - High Quality - SET-P",
        "fs": 250,
        "record_len_min": 10.0,
        "expected_records": 100,
        "expected_beats": 72415,
    },
    {
        "code": "A2",
        "name": "Low Quality (PhysioNet 2014 Training)",
        "rel_path": "ECG DBs/A - Physionet Challange 2014/A2 - Low Quality - Training",
        "fs": 360,
        "record_len_min": 10.0,
        "expected_records": 100,
        "expected_beats": 78618,
    },
    {
        "code": "B1",
        "name": "Normal Sinus Rhythm (MIT-BIH NSR DB)",
        "rel_path": "ECG DBs/B - MIT-BIH NSR & ARRHYTHMIA/B1 - NSR DB 1.0.0",
        "fs": 128,
        "record_len_min": 30.0,
        "expected_records": 18,
        "expected_beats": 48494,
    },
    {
        "code": "B2",
        "name": "Arrhythmias (MIT-BIH Arrhythmia DB, 44 non-paced)",
        "rel_path": "ECG DBs/B - MIT-BIH NSR & ARRHYTHMIA/B2 - ARRHYTHMIA DB 1.0.0",
        "fs": 360,
        "record_len_min": 30.0,
        "expected_records": 44,
        "expected_beats": 103724,
    },
    {
        "code": "C",
        "name": "Paced Rhythm (MIT-BIH Pacemaker Subset)",
        "rel_path": "ECG DBs/C - MIT-BIH Pacemaker Rhythm - part of ARRHYT DB",
        "fs": 360,
        "record_len_min": 30.0,
        "expected_records": 4,
        "expected_beats": 8923,
    },
    {
        "code": "D",
        "name": "TeleHealth (Harvard Dataverse Dry Electrode, 134 valid)",
        "rel_path": "ECG DBs/D - Harvard Telehealth DB",
        "fs": 500,
        "record_len_min": 0.5,
        "expected_records": 134,
        "expected_beats": 3726,
    },
]

DATASET_MAP: Dict[str, Dict] = {d["code"]: d for d in DATASETS}


def get_valid_d_filenames() -> List[str]:
    """Read the exact 134 valid Dataset D filenames from Week 1 replication."""
    csv_path = os.path.join(REPLICATION_DIR, "D", "YouCare_not_resampled_134records.csv")
    df = pd.read_csv(csv_path)
    return sorted(df["file"].tolist())


def load_dataset_records(ds_code: str) -> Tuple[str, List[str], int]:
    """
    Return (full_dir_path, filtered_sorted_filenames, nominal_fs) for dataset code.
    Applies exact Week 1 filtering rules:
      - B2 excludes 4 pacemaker records (102, 104, 107, 217) -> 44 records
      - D selects exact 134 valid records from Week 1 replication
    """
    info = DATASET_MAP[ds_code]
    full_path = os.path.join(ACY_DIR, info["rel_path"])
    if ds_code == "D":
        files = get_valid_d_filenames()
    else:
        files = sorted([f for f in os.listdir(full_path) if f.endswith(".dat")])
        if ds_code == "B2":
            files = [f for f in files if f not in B2_PACED_EXCLUDED]
    return full_path, files, int(info["fs"])


def load_single_record(ds_code: str, full_path: str, filename: str, nominal_fs: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load raw ECG signal (1D float64 array), ground-truth annotation sample indices,
    and sampling frequency matching run_replication.py identically.
    """
    if ds_code == "D":
        rec = np.genfromtxt(
            os.path.join(full_path, filename[:-12] + "_samples.dat"),
            delimiter="",
        )
        ecg = np.asarray(np.ravel(rec), dtype=np.float64)
        fs = 500
        ann = loadAnnotationSampleFromPathD(full_path + "/", filename[:-12])
        ann = np.asarray(np.multiply(np.divide(ann, 1000.0), 500.0), dtype=np.float64)
    else:
        rec_path = os.path.join(full_path, filename[:-4])
        if ds_code == "B1":
            record = wfdb.rdrecord(rec_path, sampfrom=0, sampto=230400, channels=[0])
            ann = loadAnnotationSampleFromPathSinus(
                full_path + "/", filename[:-4], counter=0, sampfrom=0, sampTo=230400
            )
        else:
            record = wfdb.rdrecord(rec_path, channels=[0])
            ann = loadAnnotationSampleFromPath(full_path + "/", filename[:-4])
        ecg = np.asarray(np.ravel(record.p_signal), dtype=np.float64)
        # Replace any NaN/inf samples defensively (e.g. dropout in raw WFDB)
        if not np.all(np.isfinite(ecg)):
            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
        fs = int(nominal_fs)
        ann = np.asarray(ann, dtype=np.float64)
    return ecg, ann, fs


def fast_check_positive(
    ann: Sequence[float],
    peaks: Sequence[float],
    fs: float,
    time_window: float = TIME_WINDOW_MS,
) -> List[float]:
    """
    O((M + N) log M) exact equivalent of acy_ampt.mit_processing.mit_analysis.checkPositive.

    A detected peak `i` is a False Positive iff there is NO annotation `n` in `ann`
    satisfying the strict inequality:
        i + w > n > i - w,   where w = time_window * fs / 1000.
    """
    peaks_arr = np.asarray(peaks, dtype=np.float64)
    if peaks_arr.size == 0:
        return []
    ann_arr = np.asarray(ann, dtype=np.float64)
    if ann_arr.size == 0:
        return list(peaks)

    ann_sorted = np.sort(ann_arr)
    w = time_window * fs / 1000.0

    low = peaks_arr - w
    high = peaks_arr + w
    # First element in ann_sorted strictly greater than (peak - w)
    idx = np.searchsorted(ann_sorted, low, side="right")
    valid = idx < ann_sorted.size
    matched = np.zeros(peaks_arr.size, dtype=bool)
    if np.any(valid):
        matched[valid] = ann_sorted[idx[valid]] < high[valid]

    return [peaks[k] for k in np.flatnonzero(~matched)]


def fast_check_negative(
    ann: Sequence[float],
    peaks: Sequence[float],
    fs: float,
    time_window: float = TIME_WINDOW_MS,
) -> List[float]:
    """
    O((M + N) log N) exact equivalent of acy_ampt.mit_processing.mit_analysis.checkNegative.

    An annotation `i` is a False Negative iff there is NO detected peak `n` in `peaks`
    satisfying the strict inequality:
        i + w > n > i - w,   where w = time_window * fs / 1000.
    """
    ann_arr = np.asarray(ann, dtype=np.float64)
    if ann_arr.size == 0:
        return []
    peaks_arr = np.asarray(peaks, dtype=np.float64)
    if peaks_arr.size == 0:
        return list(ann)

    peaks_sorted = np.sort(peaks_arr)
    w = time_window * fs / 1000.0

    low = ann_arr - w
    high = ann_arr + w
    idx = np.searchsorted(peaks_sorted, low, side="right")
    valid = idx < peaks_sorted.size
    matched = np.zeros(ann_arr.size, dtype=bool)
    if np.any(valid):
        matched[valid] = peaks_sorted[idx[valid]] < high[valid]

    return [ann[k] for k in np.flatnonzero(~matched)]


def compute_rr_hr_mae(
    ann: Sequence[float],
    peaks: Sequence[float],
    fs: float,
    time_window: float = TIME_WINDOW_MS,
) -> Tuple[float, float, int]:
    """
    Compute Mean Absolute Error (MAE) of RR intervals (ms) and instantaneous Heart Rate (bpm)
    over matched consecutive ground-truth annotations.

    Each annotation is matched to its closest detected peak strictly inside +/- time_window ms
    (unique greedy match). For consecutive annotations (k, k+1) both matched to distinct
    chronological peaks with physiologically valid RR >= 200 ms (<= 300 bpm) and <= 3000 ms:
      - RR MAE = mean(|RR_est_ms - RR_true_ms|)
      - HR MAE = mean(|60000/RR_est_ms - 60000/RR_true_ms|)

    Returns (rr_mae_ms, hr_mae_bpm, n_matched_intervals).
    """
    ann_arr = np.sort(np.asarray(ann, dtype=np.float64))
    peaks_arr = np.sort(np.asarray(peaks, dtype=np.float64))
    if ann_arr.size < 2 or peaks_arr.size < 2:
        return np.nan, np.nan, 0

    w = time_window * fs / 1000.0
    matched_peak = np.full(ann_arr.size, np.nan, dtype=np.float64)

    # Nearest peak lookup for each annotation via searchsorted
    pos = np.searchsorted(peaks_arr, ann_arr)
    for k in range(ann_arr.size):
        best_dist = w
        best_val = np.nan
        p = pos[k]
        if p > 0:
            d = abs(peaks_arr[p - 1] - ann_arr[k])
            if d < best_dist:
                best_dist = d
                best_val = peaks_arr[p - 1]
        if p < peaks_arr.size:
            d = abs(peaks_arr[p] - ann_arr[k])
            if d < best_dist:
                best_dist = d
                best_val = peaks_arr[p]
        matched_peak[k] = best_val

    valid_pairs = np.isfinite(matched_peak[:-1]) & np.isfinite(matched_peak[1:])
    if not np.any(valid_pairs):
        return np.nan, np.nan, 0

    rr_true_ms = (ann_arr[1:][valid_pairs] - ann_arr[:-1][valid_pairs]) * (1000.0 / fs)
    rr_est_ms = (matched_peak[1:][valid_pairs] - matched_peak[:-1][valid_pairs]) * (1000.0 / fs)

    # Filter out non-beat annotation artifacts (e.g. <200 ms or >3000 ms) and duplicate matches
    phys = (rr_true_ms >= 200.0) & (rr_true_ms <= 3000.0) & (rr_est_ms >= 200.0) & (rr_est_ms <= 3000.0)
    if not np.any(phys):
        return np.nan, np.nan, 0

    rr_true_ms = rr_true_ms[phys]
    rr_est_ms = rr_est_ms[phys]

    rr_mae = float(np.mean(np.abs(rr_est_ms - rr_true_ms)))
    hr_true = 60000.0 / rr_true_ms
    hr_est = 60000.0 / rr_est_ms
    hr_mae = float(np.mean(np.abs(hr_est - hr_true)))
    return rr_mae, hr_mae, int(rr_true_ms.size)


def compute_record_metrics(
    ann: Sequence[float],
    peaks: Sequence[float],
    fs: float,
    elapsed_sec: float,
    duration_sec: float,
    filename: str,
    time_window: float = TIME_WINDOW_MS,
) -> Dict:
    """Compute full per-record detection metrics, RR/HR MAE, and normalized timing."""
    fp_list = fast_check_positive(ann, peaks, fs, time_window=time_window)
    fn_list = fast_check_negative(ann, peaks, fs, time_window=time_window)

    our_peaks = int(len(peaks))
    ann_peaks = int(len(ann))
    fp = int(len(fp_list))
    fn = int(len(fn_list))
    tp = int(our_peaks - fp)

    sens = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    ppv = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    acc = (tp / (tp + fp + fn) * 100.0) if (tp + fp + fn) > 0 else 0.0
    f1 = (2.0 * tp / (2.0 * tp + fp + fn) * 100.0) if (2 * tp + fp + fn) > 0 else 0.0
    err_rate = ((fp + fn) / ann_peaks * 100.0) if ann_peaks > 0 else 0.0
    time_per_10s_ms = (elapsed_sec * 1000.0) / max(duration_sec / 10.0, 1e-6)

    rr_mae_ms, hr_mae_bpm, n_rr = compute_rr_hr_mae(ann, peaks, fs, time_window=time_window)

    return {
        "file": filename,
        "timing": float(elapsed_sec),
        "duration_s": float(duration_sec),
        "time_per_10s_ms": float(time_per_10s_ms),
        "Our Peaks": our_peaks,
        "Annotation Peaks": ann_peaks,
        "TP": tp,
        "False Positive": fp,
        "False Negative": fn,
        "Sensitivity_Pct": float(sens),
        "PPV_Pct": float(ppv),
        "Accuracy_Pct": float(acc),
        "F1_Pct": float(f1),
        "Error_Rate_Pct": float(err_rate),
        "RR_MAE_ms": float(rr_mae_ms) if np.isfinite(rr_mae_ms) else np.nan,
        "HR_MAE_bpm": float(hr_mae_bpm) if np.isfinite(hr_mae_bpm) else np.nan,
        "Matched_RR_Pairs": int(n_rr),
    }


def add_per_record_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure per-record DataFrame (including Week 1 replication CSVs) has standardized
    TP, Sensitivity_Pct, PPV_Pct, Accuracy_Pct, F1_Pct, Error_Rate_Pct columns.
    """
    out = df.copy()
    if "TP" not in out.columns:
        if "tp" in out.columns:
            out["TP"] = out["tp"].astype(int)
        else:
            out["TP"] = (out["Our Peaks"] - out["False Positive"]).astype(int)

    tp = out["TP"].astype(float)
    fp = out["False Positive"].astype(float)
    fn = out["False Negative"].astype(float)
    ann = out["Annotation Peaks"].astype(float)

    out["Sensitivity_Pct"] = np.where((tp + fn) > 0, tp / (tp + fn) * 100.0, 0.0)
    out["PPV_Pct"] = np.where((tp + fp) > 0, tp / (tp + fp) * 100.0, 0.0)
    out["Accuracy_Pct"] = np.where((tp + fp + fn) > 0, tp / (tp + fp + fn) * 100.0, 0.0)
    out["F1_Pct"] = np.where((2 * tp + fp + fn) > 0, 2.0 * tp / (2.0 * tp + fp + fn) * 100.0, 0.0)
    out["Error_Rate_Pct"] = np.where(ann > 0, (fp + fn) / ann * 100.0, 0.0)
    return out


def aggregate_dataset_metrics(
    df: pd.DataFrame,
    ds_code: str,
    algorithm: str,
    use_nominal_chunks_for_time: bool = False,
) -> Dict:
    """
    Aggregate per-record CSV exactly matching REPLICATION_REPORT.md:
      - Error Rate (%) = (sum(FP) + sum(FN)) / sum(Annotation Peaks) * 100
      - Sensitivity, PPV, Accuracy, F1 = macro-averaged across records
      - Time_per_10s_ms = mean per-record ms per 10s (or nominal chunks for Week 1 CSVs)
    """
    df_std = add_per_record_derived_columns(df)
    tp_sum = int(df_std["TP"].sum())
    fp_sum = int(df_std["False Positive"].sum())
    fn_sum = int(df_std["False Negative"].sum())
    ann_sum = int(df_std["Annotation Peaks"].sum())
    failed_sum = fp_sum + fn_sum
    err_rate = (failed_sum / ann_sum * 100.0) if ann_sum > 0 else 0.0

    sens_mean = float(df_std["Sensitivity_Pct"].mean())
    ppv_mean = float(df_std["PPV_Pct"].mean())
    acc_mean = float(df_std["Accuracy_Pct"].mean())
    f1_mean = float(df_std["F1_Pct"].mean())

    if "time_per_10s_ms" in df_std.columns and not use_nominal_chunks_for_time:
        time_per_10s = float(df_std["time_per_10s_ms"].mean())
    else:
        chunks = DATASET_MAP[ds_code]["record_len_min"] * 6.0  # 60s / 10s = 6 chunks/min
        time_per_10s = float((df_std["timing"].mean() * 1000.0) / chunks)

    row = {
        "Dataset": ds_code,
        "Algorithm": algorithm,
        "Records": int(len(df_std)),
        "Annotated_Peaks": ann_sum,
        "TP": tp_sum,
        "FP": fp_sum,
        "FN": fn_sum,
        "Failed_Detection": failed_sum,
        "Error_Rate_Pct": round(err_rate, 2),
        "Sensitivity_Pct": round(sens_mean, 2),
        "PPV_Pct": round(ppv_mean, 2),
        "Accuracy_Pct": round(acc_mean, 2),
        "F1_Pct": round(f1_mean, 2),
        "Time_per_10s_ms": round(time_per_10s, 2),
    }

    if "RR_MAE_ms" in df_std.columns:
        valid_rr = df_std["RR_MAE_ms"].dropna()
        valid_hr = df_std["HR_MAE_bpm"].dropna()
        row["RR_MAE_ms"] = round(float(valid_rr.mean()), 2) if len(valid_rr) > 0 else 0.0
        row["HR_MAE_bpm"] = round(float(valid_hr.mean()), 2) if len(valid_hr) > 0 else 0.0

    return row


def run_wilcoxon_tests(
    df_proposed: pd.DataFrame,
    df_baseline: pd.DataFrame,
    ds_code: str,
    proposed_name: str,
    baseline_name: str,
) -> List[Dict]:
    """
    Run paired two-sided Wilcoxon signed-rank tests AND paired Student's t-tests
    (`ttest_rel`) on per-record F1_Pct and Error_Rate_Pct aligned by filename.
    """
    p_df = add_per_record_derived_columns(df_proposed).set_index("file")
    b_df = add_per_record_derived_columns(df_baseline).set_index("file")
    common = sorted(p_df.index.intersection(b_df.index))
    p_df = p_df.loc[common]
    b_df = b_df.loc[common]

    results = []
    for metric in ["F1_Pct", "Error_Rate_Pct"]:
        x = p_df[metric].to_numpy(dtype=np.float64)
        y = b_df[metric].to_numpy(dtype=np.float64)
        diff = x - y
        nonzero = diff[np.abs(diff) > 1e-12]

        # 1. Paired Wilcoxon signed-rank test (non-parametric)
        if len(nonzero) == 0:
            w_stat, w_pval = 0.0, 1.0
        else:
            res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            w_stat = float(res.statistic)
            w_pval = float(res.pvalue)

        # 2. Paired t-test (parametric)
        if len(common) < 2 or float(np.std(diff, ddof=1)) < 1e-12:
            t_stat, t_pval = 0.0, 1.0
        else:
            tres = ttest_rel(x, y, alternative="two-sided")
            t_stat = float(tres.statistic) if np.isfinite(tres.statistic) else 0.0
            t_pval = float(tres.pvalue) if np.isfinite(tres.pvalue) else 1.0

        # Positive improvement direction
        if metric == "F1_Pct":
            improved = int(np.sum(diff > 1e-12))
            worsened = int(np.sum(diff < -1e-12))
        else:
            improved = int(np.sum(diff < -1e-12))
            worsened = int(np.sum(diff > 1e-12))
        tied = int(len(diff) - improved - worsened)

        results.append(
            {
                "Dataset": ds_code,
                "Comparison": f"{proposed_name} vs {baseline_name}",
                "Metric": metric,
                "N_Records": int(len(common)),
                "Proposed_Mean": round(float(np.mean(x)), 4),
                "Baseline_Mean": round(float(np.mean(y)), 4),
                "Mean_Delta": round(float(np.mean(diff)), 4),
                "Median_Delta": round(float(np.median(diff)), 4),
                "wilcoxon_stat": round(w_stat, 4),
                "wilcoxon_pvalue": w_pval,
                "t_stat": round(t_stat, 4),
                "t_pvalue": t_pval,
                "Wilcoxon_Sig_0.05": bool(w_pval < 0.05),
                "Wilcoxon_Sig_0.01": bool(w_pval < 0.01),
                "TTest_Sig_0.05": bool(t_pval < 0.05),
                "TTest_Sig_0.01": bool(t_pval < 0.01),
                "Records_Improved": improved,
                "Records_Worsened": worsened,
                "Records_Tied": tied,
            }
        )
    return results
