import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample

# Add acy_ampt to sys.path so we import the exact reference modules
ACY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acy_ampt")
sys.path.insert(0, ACY_DIR)

from ecg_processing.signal_filtering import (
    bandpassFilt,
    derivateStep,
    movingAverageMeanPanTompkins,
)
from ecg_processing.signal_peak_detector import AMPT
from mit_processing.load_annotation import (
    loadAnnotationSampleFromPath,
    loadAnnotationSampleFromPathSinus,
    loadAnnotationSampleFromPathD,
)
from mit_processing.mit_analysis import checkNegative, checkPositive
from mit_processing.pan import panTompkins

TIME_WINDOW = 150


def process_single_record(task):
    """Run exact acy_ampt pipeline on a single ECG record."""
    ds_name, path, filename, signal_sampling, pan, resample_ecg = task

    if ds_name == "D":
        record = np.genfromtxt(
            os.path.join(path, filename[:-12] + "_samples.dat"), delimiter=""
        )
        not_resampled_signal = [i for i in np.ravel(record)]
        if resample_ecg:
            ecg_sig = resample(
                not_resampled_signal,
                int(len(not_resampled_signal) * 200 / signal_sampling),
            )
            fs = 200
        else:
            ecg_sig = not_resampled_signal
            fs = 500
    else:
        rec_path = os.path.join(path, filename[:-4])
        if ds_name == "B1":
            record = wfdb.rdrecord(rec_path, sampfrom=0, sampto=230400, channels=[0])
        else:
            record = wfdb.rdrecord(rec_path, channels=[0])
        not_resampled_signal = [i for i in np.ravel(record.p_signal)]
        if resample_ecg:
            ecg_sig = resample(
                not_resampled_signal,
                int(len(not_resampled_signal) * 200 / signal_sampling),
            )
            fs = 200
        else:
            ecg_sig = not_resampled_signal
            fs = signal_sampling

    if not pan:
        new_ecg_filt = bandpassFilt(ecg_sig, 4, fs, 15, 5)
        derivate_signal = derivateStep(new_ecg_filt)
        squared_ecg = np.power(np.abs(derivate_signal), 2)
        integrated_ecg = movingAverageMeanPanTompkins(squared_ecg, fs)
        start_time = time.time()
        peaks, _ = AMPT(integrated_ecg, fs)
        elapsed = time.time() - start_time
    else:
        peaks, elapsed = panTompkins(ecg_sig, fs)

    # Load annotations
    if ds_name == "D":
        ann = loadAnnotationSampleFromPathD(path + "/", filename[:-12])
        ann = np.divide(ann, 1000)
        ann = np.multiply(ann, 500)
    elif ds_name == "B1":
        ann = loadAnnotationSampleFromPathSinus(
            path + "/", filename[:-4], counter=0, sampfrom=0, sampTo=230400
        )
    else:
        ann = loadAnnotationSampleFromPath(path + "/", filename[:-4])

    if resample_ecg:
        ann = [int(i * 200 / signal_sampling) for i in ann]

    # Suppress stdout spam from checkPositive / checkNegative
    fp = checkPositive(ann, peaks, fs, TIME_WINDOW)
    fn = checkNegative(ann, peaks, fs, TIME_WINDOW)

    return {
        "timing": elapsed,
        "file": filename,
        "Our Peaks": len(peaks),
        "Annotation Peaks": len(ann),
        "False Positive": len(fp),
        "False Negative": len(fn),
    }


def run_dataset(ds_name, rel_path, fs, out_root, resample_ecg=False, workers=10):
    full_path = os.path.join(ACY_DIR, rel_path)
    out_dir = os.path.join(out_root, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    suffix = "_samples.dat" if ds_name == "D" else ".dat"
    files = sorted([f for f in os.listdir(full_path) if f.endswith(suffix)])

    for pan, out_name in [
        (False, "YouCare_resampled.csv" if resample_ecg else "YouCare_not_resampled.csv"),
        (True, "pan_resampled_1.csv" if resample_ecg else "pan_not_resampled_1.csv"),
    ]:
        out_csv = os.path.join(out_dir, out_name)
        if os.path.exists(out_csv):
            df_existing = pd.read_csv(out_csv)
            if len(df_existing) == len(files):
                print(f"[SKIP] {ds_name} {out_name} already exists ({len(files)} rows)")
                continue

        algo_label = "Pan-Tompkins" if pan else "AMPT"
        print(f"[RUN] {ds_name} ({len(files)} records) - {algo_label} (resample={resample_ecg})...", flush=True)
        t0 = time.time()
        tasks = [(ds_name, full_path, f, fs, pan, resample_ecg) for f in files]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(process_single_record, tasks))
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False, header=True)
        print(f"[DONE] {ds_name} - {algo_label} saved to {out_csv} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_replication")
    datasets = [
        ("A1", "ECG DBs/A - Physionet Challange 2014/A1 - High Quality - SET-P", 250),
        ("A2", "ECG DBs/A - Physionet Challange 2014/A2 - Low Quality - Training", 360),
        ("B1", "ECG DBs/B - MIT-BIH NSR & ARRHYTHMIA/B1 - NSR DB 1.0.0", 128),
        ("B2", "ECG DBs/B - MIT-BIH NSR & ARRHYTHMIA/B2 - ARRHYTHMIA DB 1.0.0", 360),
        ("C", "ECG DBs/C - MIT-BIH Pacemaker Rhythm - part of ARRHYT DB", 360),
        ("D", "ECG DBs/D - Harvard Telehealth DB", 500),
    ]
    target = sys.argv[1:] if len(sys.argv) > 1 else [d[0] for d in datasets]
    for ds_name, rel_path, fs in datasets:
        if ds_name in target:
            run_dataset(ds_name, rel_path, fs, out_root, resample_ecg=False, workers=10)
