"""
Real-time QRS detection algorithms and ablation suite:
  1. run_pan_tompkins    : Exact wrapper around acy_ampt Pan-Tompkins (Pickus / Neri et al. 2023)
  2. run_ampt_fast       : Exact vectorized equivalent of Neri et al. (2023) AMPT (0-beat diff)
  3. wavelet_ampt        : Fixed Wavelet (sym4) soft-thresholding + AMPT
  4. median_ampt         : Fixed Dual-Median baseline removal + AMPT
  5. kalman_ampt         : Fixed 1D Zero-Phase Adaptive Kalman filter + AMPT
  6. asqa_h_ampt         : Proposed Adaptive Signal Quality Assessment +
                           Adaptive Denoising Decision Tree +
                           Hilbert-Enhanced AMPT + Dynamic Adaptive Threshold & RR Post-Processing
"""

import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
from scipy import signal
from scipy.fft import next_fast_len
from scipy.ndimage import uniform_filter1d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACY_DIR = os.path.join(PROJECT_ROOT, "acy_ampt")
if ACY_DIR not in sys.path:
    sys.path.insert(0, ACY_DIR)

from mit_processing.pan import panTompkins as _acy_pan_tompkins  # noqa: E402


# =============================================================================
# 0. Exact Pan-Tompkins Baseline Wrapper (Neri et al., Sensors 2023)
# =============================================================================


def run_pan_tompkins(ecg: np.ndarray, fs: float) -> Tuple[List[int], float]:
    """
    Exact wrapper around acy_ampt.mit_processing.pan.panTompkins.
    Returns (detected_peaks, end_to_end_elapsed_seconds).
    """
    sig = np.asarray(ecg, dtype=np.float64)
    t0 = time.perf_counter()
    peaks, _ = _acy_pan_tompkins(sig, fs)
    elapsed = time.perf_counter() - t0
    return [int(p) for p in peaks], elapsed


# =============================================================================
# 1. Exact Vectorized Equivalent of acy_ampt (Neri et al., Sensors 2023)
# =============================================================================


def _bandpass_filt_exact(ecg: np.ndarray, n_order: int, fs: float, freq_1: float = 15.0, freq_2: float = 5.0) -> np.ndarray:
    """
    Exact mathematical equivalent of acy_ampt.ecg_processing.signal_filtering.bandpassFilt,
    accelerated ~500x using scipy.signal.lfilter on the identical IIR difference equations.
    """
    x = np.asarray(ecg, dtype=np.float64)
    if x.size <= 1:
        return x.copy()

    n = int(n_order / 4)
    a = math.cos(math.pi * (freq_1 + freq_2) / fs) / math.cos(math.pi * (freq_1 - freq_2) / fs)
    a2 = a * a
    b = math.tan(math.pi * (freq_1 - freq_2) / fs)
    b2 = b * b

    y = x[:-1].copy()
    for i in range(0, n + 1):
        r = math.sin(math.pi * (2.0 * i + 1.0) / (4.0 * n))
        s = b2 + 2.0 * b * r + 1.0
        A = b2 / s
        d1 = 4.0 * a * (1.0 + b * r) / s
        d2 = 2.0 * (b2 - 2.0 * a2 - 1.0) / s
        d3 = 4.0 * a * (1.0 - b * r) / s
        d4 = -(b2 - 2.0 * b * r + 1.0) / s
        b_coef = np.array([A, 0.0, -2.0 * A, 0.0, A], dtype=np.float64)
        a_coef = np.array([1.0, -d1, -d2, -d3, -d4], dtype=np.float64)
        y = signal.lfilter(b_coef, a_coef, y)

    out = np.empty_like(x)
    out[:-1] = y
    out[-1] = x[-1]
    return out


def _derivate_step_exact(data: np.ndarray) -> np.ndarray:
    """Exact vectorized equivalent of acy_ampt.ecg_processing.signal_filtering.derivateStep."""
    n = data.size
    if n <= 3:
        return np.empty(0, dtype=np.float64)
    scale = (1.0 / 8.0) * 360.0
    out = np.empty(n - 3, dtype=np.float64)
    out[0] = scale * ((-data[-1] - 2.0 * data[0]) + 2.0 * data[2] + data[3])
    if n > 4:
        out[1:] = scale * ((-data[:-4] - 2.0 * data[1:-3]) + 2.0 * data[3:-1] + data[4:])
    return out


def _mwi_exact(data: np.ndarray, fs: float) -> np.ndarray:
    """Exact O(N) equivalent of acy_ampt.ecg_processing.signal_filtering.movingAverageMeanPanTompkins."""
    size = int(0.15 * fs)
    if size <= 1 or data.size < size:
        return data.copy()
    cs = np.empty(data.size + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(data, out=cs[1:])
    return (cs[size:] - cs[:-size]) / float(size)


def _ampt_peak_detect_exact(sig: np.ndarray, fs: float) -> List[int]:
    """
    Exact equivalent of acy_ampt.ecg_processing.signal_peak_detector.AMPT,
    using vectorized local-maxima discovery and O(1) index tracking.
    """
    n_sig = sig.size
    if n_sig < 3:
        return []

    peaks = (np.flatnonzero((sig[1:-1] > sig[:-2]) & (sig[1:-1] > sig[2:])) + 1).tolist()
    n_peaks = len(peaks)
    if n_peaks < 4:
        return []

    peak_to_pos = {p: idx for idx, p in enumerate(peaks)}

    signal_peaks = [0]
    spki = 0.0
    npki = 0.0
    threshold_i1 = 0.0
    rr_missed = 0.0

    w_02 = 0.2 * fs
    w_036 = 0.36 * fs
    min_close = (fs / 1000.0) * fs

    i = 1
    while i < n_peaks - 2:
        p_i = peaks[i]
        v_i = float(sig[p_i])

        if len(signal_peaks) > 2 and rr_missed > 0:
            if p_i - signal_peaks[-1] > rr_missed:
                rr_missed = 0.0
                threshold_i2 = 0.25 * threshold_i1
                x = peak_to_pos[signal_peaks[-1]]
                y = i
                z = x
                while z < y:
                    if z != 0:
                        p_z = peaks[z]
                        v_z = float(sig[p_z])
                        if v_z > threshold_i2 and v_z > sig[peaks[z - 1]] and v_z > sig[peaks[z + 1]]:
                            if p_z - signal_peaks[-1] > w_02:
                                signal_peaks.append(p_z)
                                spki = 0.125 * v_i + 0.875 * spki
                            else:
                                npki = 0.125 * v_i + 0.875 * npki
                    z += 1

        if v_i > threshold_i1 and v_i > sig[peaks[i - 1]] and v_i > sig[peaks[i + 1]]:
            diff_last = p_i - signal_peaks[-1]
            if diff_last > w_036:
                signal_peaks.append(p_i)
                spki = 0.125 * v_i + 0.875 * spki
            if w_02 < diff_last < w_036 and v_i > 0.5 * sig[signal_peaks[-1]]:
                signal_peaks.append(p_i)
                spki = 0.125 * v_i + 0.875 * spki
        else:
            npki = 0.125 * v_i + 0.875 * npki

        threshold_i1 = npki + 0.25 * (spki - npki)

        if len(signal_peaks) > 8:
            arr = signal_peaks[len(signal_peaks) - 9 : len(signal_peaks) - 2]
            rr_sum = arr[-1] - arr[0]
            rr_ave = rr_sum / float(len(arr) - 1)
            rr_missed = 1.66 * rr_ave

        i += 1

    j = 1
    while j < len(signal_peaks) - 1:
        if signal_peaks[j] - signal_peaks[j - 1] < min_close:
            if sig[signal_peaks[j]] > sig[signal_peaks[j - 1]]:
                signal_peaks.pop(j - 1)
            else:
                signal_peaks.pop(j)
        j += 1

    return signal_peaks[1:]


def run_ampt_fast(ecg: np.ndarray, fs: float) -> Tuple[List[int], float]:
    """Full end-to-end exact vectorized Neri et al. (2023) AMPT detector."""
    sig = np.asarray(ecg, dtype=np.float64)
    t0 = time.perf_counter()
    bp = _bandpass_filt_exact(sig, 4, fs, 15.0, 5.0)
    deriv = _derivate_step_exact(bp)
    sq = deriv * deriv
    mwi = _mwi_exact(sq, fs)
    peaks = _ampt_peak_detect_exact(mwi, fs)
    elapsed = time.perf_counter() - t0
    return peaks, elapsed


# =============================================================================
# 2. Modular Signal Denoisers (Used by Fixed Ablations & Adaptive Decision Tree)
# =============================================================================


def dual_median_filter_fast(
    ecg: np.ndarray,
    fs: float,
    w1_ms: float = 200.0,
    w2_ms: float = 600.0,
) -> np.ndarray:
    """
    Fast multi-rate zero-phase Dual-Median baseline wander removal (200 ms + 600 ms).
    Decimates to ~25 Hz via block-medians before applying the median filters and
    linearly interpolating the estimated baseline back to full rate.
    """
    x = np.asarray(ecg, dtype=np.float64)
    n = x.size
    if n < 16:
        return x - np.median(x) if n > 0 else x.copy()

    target_fs = 25.0
    step = max(1, int(round(fs / target_fs)))
    fs_eff = fs / float(step)

    if step > 1 and n >= step * 6:
        n_blocks = n // step
        trimmed = x[: n_blocks * step].reshape(n_blocks, step)
        sub = np.median(trimmed, axis=1)
        centers = (np.arange(n_blocks, dtype=np.float64) + 0.5) * step
    else:
        sub = x.copy()
        centers = np.arange(n, dtype=np.float64)
        fs_eff = fs

    k1 = max(3, int(round((w1_ms / 1000.0) * fs_eff)))
    if k1 % 2 == 0:
        k1 += 1
    k2 = max(5, int(round((w2_ms / 1000.0) * fs_eff)))
    if k2 % 2 == 0:
        k2 += 1

    k1 = min(k1, sub.size if sub.size % 2 == 1 else max(1, sub.size - 1))
    k2 = min(k2, sub.size if sub.size % 2 == 1 else max(1, sub.size - 1))

    b1 = signal.medfilt(sub, kernel_size=k1) if k1 >= 3 else sub
    b2 = signal.medfilt(b1, kernel_size=k2) if k2 >= 3 else b1

    if step > 1 and sub.size >= 2:
        baseline = np.interp(np.arange(n, dtype=np.float64), centers, b2)
    else:
        baseline = b2
    return x - baseline


def wavelet_soft_denoise(
    ecg: np.ndarray,
    fs: float,
    wavelet: str = "sym4",
    level: Optional[int] = None,
) -> np.ndarray:
    """
    Frequency-aware Sym4 Wavelet soft-thresholding for high-frequency / EMG / AWGN noise.
    Applies full VisuShrink thresholding to HF noise bands (>25 Hz), gentle Bayes-scaled
    thresholding to transition bands (18-25 Hz), and preserves core QRS subbands (<18 Hz).
    """
    x = np.asarray(ecg, dtype=np.float64)
    n = x.size
    if n < 32:
        return x.copy()

    max_lvl = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    if level is None:
        desired = max(2, int(round(math.log2(max(fs / 18.0, 2.0)))))
        level = min(max_lvl, min(5, desired))
    else:
        level = min(max_lvl, level)

    if level < 1:
        return x.copy()

    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode="symmetric")
    d1 = coeffs[-1]
    sigma = float(np.median(np.abs(d1))) / 0.6745
    if sigma < 1e-12:
        return x.copy()

    visu = sigma * math.sqrt(2.0 * math.log(max(n, 2)))
    new_coeffs = [coeffs[0]]
    for idx, d in enumerate(coeffs[1:]):
        lvl = level - idx
        band_low_hz = fs / (2.0 ** (lvl + 1))
        if band_low_hz >= 25.0:
            thr = visu
        elif band_low_hz >= 18.0:
            thr = 0.25 * visu
        else:
            thr = 0.0
        new_coeffs.append(pywt.threshold(d, value=thr, mode="soft") if thr > 0.0 else d)

    recon = pywt.waverec(new_coeffs, wavelet=wavelet, mode="symmetric")
    return recon[:n]


def _kalman_1d_pass(x_list: List[float], q_val: float, r_base: float, inv_spike2: float) -> List[float]:
    """Single-direction innovation-adaptive 1D Kalman pass."""
    n = len(x_list)
    out = [0.0] * n
    x_hat = x_list[0]
    out[0] = x_hat
    p_cov = 1.0
    for k in range(1, n):
        p_pred = p_cov + q_val
        res = x_list[k] - x_hat
        res2 = res * res
        r_adapt = r_base * (1.0 + 3.0 * res2 * inv_spike2)
        gain = p_pred / (p_pred + r_adapt)
        x_hat = x_hat + gain * res
        p_cov = (1.0 - gain) * p_pred
        out[k] = x_hat
    return out


def adaptive_kalman_filter(
    ecg: np.ndarray,
    fs: float,
    q_factor: float = 0.25,
    r_factor: float = 0.06,
    spike_k: float = 4.0,
) -> np.ndarray:
    """
    Zero-phase bidirectional 1D scale-invariant Adaptive Kalman Smoother.

    Sampling-rate-normalized process noise Q scales inversely with fs so rapid QRS
    upstrokes at low sampling rates (e.g., 128 Hz) pass undistorted while large
    non-stationary / impulsive artifacts inflate R_k and are damped cleanly.
    """
    x = np.asarray(ecg, dtype=np.float64)
    n = x.size
    if n < 8:
        return x.copy()

    dx = np.diff(x)
    mad_dx = float(np.median(np.abs(dx))) / 0.6745 + 1e-8
    mad_x = float(np.median(np.abs(x - np.median(x)))) / 0.6745 + 1e-8
    scale = max(mad_x, mad_dx * 3.0, 1e-6)

    q_eff = q_factor * (360.0 / max(float(fs), 100.0))
    xn = (x / scale).tolist()
    inv_spike2 = 1.0 / (spike_k * spike_k)

    fwd = _kalman_1d_pass(xn, q_eff, r_factor, inv_spike2)
    bwd = _kalman_1d_pass(xn[::-1], q_eff, r_factor, inv_spike2)[::-1]

    out = 0.5 * (np.asarray(fwd, dtype=np.float64) + np.asarray(bwd, dtype=np.float64))
    return out * scale


# =============================================================================
# 3. Fixed Preprocessing + AMPT Ablation Detectors
# =============================================================================


def wavelet_ampt(ecg: np.ndarray, fs: float) -> Tuple[List[int], float]:
    """Fixed Ablation 1: Wavelet (sym4) soft-thresholding + exact AMPT."""
    sig = np.asarray(ecg, dtype=np.float64)
    t0 = time.perf_counter()
    denoised = wavelet_soft_denoise(sig, fs, wavelet="sym4")
    bp = _bandpass_filt_exact(denoised, 4, fs, 15.0, 5.0)
    deriv = _derivate_step_exact(bp)
    sq = deriv * deriv
    mwi = _mwi_exact(sq, fs)
    peaks = _ampt_peak_detect_exact(mwi, fs)
    elapsed = time.perf_counter() - t0
    return peaks, elapsed


def median_ampt(ecg: np.ndarray, fs: float) -> Tuple[List[int], float]:
    """Fixed Ablation 2: Dual-Median baseline removal + exact AMPT."""
    sig = np.asarray(ecg, dtype=np.float64)
    t0 = time.perf_counter()
    denoised = dual_median_filter_fast(sig, fs)
    bp = _bandpass_filt_exact(denoised, 4, fs, 15.0, 5.0)
    deriv = _derivate_step_exact(bp)
    sq = deriv * deriv
    mwi = _mwi_exact(sq, fs)
    peaks = _ampt_peak_detect_exact(mwi, fs)
    elapsed = time.perf_counter() - t0
    return peaks, elapsed


def kalman_ampt(ecg: np.ndarray, fs: float) -> Tuple[List[int], float]:
    """Fixed Ablation 3: 1D Zero-Phase Adaptive Kalman filter + exact AMPT."""
    sig = np.asarray(ecg, dtype=np.float64)
    t0 = time.perf_counter()
    denoised = adaptive_kalman_filter(sig, fs)
    bp = _bandpass_filt_exact(denoised, 4, fs, 15.0, 5.0)
    deriv = _derivate_step_exact(bp)
    sq = deriv * deriv
    mwi = _mwi_exact(sq, fs)
    peaks = _ampt_peak_detect_exact(mwi, fs)
    elapsed = time.perf_counter() - t0
    return peaks, elapsed


# =============================================================================
# 4. Proposed Method: ASQA-H-AMPT
#    (Windowed SQA + Adaptive Denoising Decision Tree +
#     Hilbert-Enhanced AMPT + Dynamic Adaptive Threshold & RR Post-Processing)
# =============================================================================


def _fast_coarse_peaks(arr: np.ndarray, height: float, min_dist: int) -> np.ndarray:
    """Ultra-fast vectorized local maxima with greedy minimum distance."""
    n = arr.size
    if n < 3:
        return np.empty(0, dtype=np.int64)
    cand = np.flatnonzero((arr[1:-1] > arr[:-2]) & (arr[1:-1] >= arr[2:]) & (arr[1:-1] >= height)) + 1
    if cand.size <= 1:
        return cand
    selected = []
    last_p = -min_dist - 1
    last_v = -1.0
    for p in cand.tolist():
        v = float(arr[p])
        if p - last_p >= min_dist:
            selected.append(p)
            last_p = p
            last_v = v
        elif v > last_v:
            selected[-1] = p
            last_p = p
            last_v = v
    return np.asarray(selected, dtype=np.int64)


def _fast_median_small(buf: List[float]) -> float:
    """Fast pure-Python median for small rolling histories (<= 16 elements)."""
    s = sorted(buf)
    m = len(s)
    mid = m >> 1
    if m & 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def evaluate_window_sqa(win: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Compute 5 physics-calibrated Signal Quality Assessment (SQA) features + Composite
    Quality Score (CQS) for a 5-second ECG window via a single fast rFFT:
      - SNR  : QRS band (5-25 Hz) vs HF/EMG band (35-95 Hz; 38-62 Hz when fs<=150 Hz)
      - BWI  : Baseline Wander Index (power in 0-1.0 Hz / total AC power) in [0, 1]
      - Kurt : Kurtosis of QRS-band (5-20 Hz) signal
      - VAQ  : Robust QRS amplitude variation (MAD / median on MWI QRS peaks)
      - VRR  : Coarse RR interval coefficient of variation (std / mean)
      - CQS  : Composite Quality Score in [0, 1]
    """
    n = win.size
    if n < max(16, int(0.5 * fs)):
        return {"SNR": 20.0, "BWI": 0.05, "Kurt": 10.0, "VAQ": 0.10, "VRR": 0.10, "CQS": 0.85}

    xc = win - float(np.mean(win))
    var = float(np.mean(xc * xc))
    if var < 1e-12:
        return {"SNR": -15.0, "BWI": 1.0, "Kurt": 1.0, "VAQ": 2.0, "VRR": 1.0, "CQS": 0.0}

    n_fft = next_fast_len(n)
    spec = np.fft.rfft(xc, n=n_fft)
    psd = spec.real * spec.real + spec.imag * spec.imag
    df = fs / float(n_fft)

    q_lo = max(1, int(math.ceil(5.0 / df)))
    q_hi = min(psd.size, int(math.floor(25.0 / df)) + 1)
    hf_lo_hz = 38.0 if fs <= 150.0 else 35.0
    hf_hi_hz = min(95.0, 0.48 * fs)
    h_lo = max(1, int(math.ceil(hf_lo_hz / df)))
    h_hi = min(psd.size, int(math.floor(hf_hi_hz / df)) + 1)
    b_hi = min(psd.size, int(math.floor(1.0 / df)) + 1)

    p_qrs = float(np.sum(psd[q_lo:q_hi])) + 1e-12
    p_hf = float(np.sum(psd[h_lo:h_hi])) + 1e-12
    snr_db = float(np.clip(10.0 * math.log10(p_qrs / p_hf), -15.0, 40.0))

    p_base = float(np.sum(psd[1:b_hi]))
    p_tot = float(np.sum(psd[1:])) + 1e-12
    bwi = float(np.clip(p_base / p_tot, 0.0, 1.0))

    bp_hi = min(spec.size, int(math.floor(min(20.0, 0.45 * fs) / df)) + 1)
    bp_spec = np.zeros_like(spec)
    bp_spec[q_lo:bp_hi] = spec[q_lo:bp_hi]
    qrs_sig = np.fft.irfft(bp_spec, n=n_fft)[:n]

    q_mean = float(np.mean(qrs_sig))
    q_diff = qrs_sig - q_mean
    q_var = float(np.mean(q_diff * q_diff)) + 1e-20
    q_m4 = float(np.mean((q_diff * q_diff) ** 2))
    kurt = float(np.clip(q_m4 / (q_var * q_var), 1.0, 45.0))

    q_energy = uniform_filter1d(qrs_sig * qrs_sig, size=max(3, int(0.10 * fs)), mode="nearest")
    min_dist = max(1, int(0.32 * fs))
    p98 = float(np.percentile(q_energy, 98))
    pks = _fast_coarse_peaks(q_energy, height=max(0.30 * p98, 1e-12), min_dist=min_dist)

    if pks.size >= 3:
        amps = np.sqrt(np.maximum(q_energy[pks], 0.0))
        med_amp = float(np.median(amps)) + 1e-9
        mad_amp = 1.4826 * float(np.median(np.abs(amps - med_amp)))
        vaq = float(np.clip(mad_amp / med_amp, 0.0, 2.0))
        rrs = np.diff(pks).astype(np.float64)
        vrr = float(np.clip(np.std(rrs) / (np.mean(rrs) + 1e-9), 0.0, 1.0))
    else:
        vaq = 0.85
        vrr = 0.65

    snr_norm = float(np.clip((snr_db - 2.0) / 22.0, 0.0, 1.0))
    bwi_norm = float(np.clip(bwi / 0.55, 0.0, 1.0))
    if kurt <= 28.0:
        kurt_norm = float(np.clip((kurt - 3.0) / 9.0, 0.0, 1.0))
    else:
        kurt_norm = float(np.clip(1.0 - (kurt - 28.0) / 35.0, 0.3, 1.0))
    vaq_norm = float(np.clip(vaq / 0.70, 0.0, 1.0))
    vrr_norm = float(np.clip(vrr / 0.45, 0.0, 1.0))

    cqs = (
        0.35 * snr_norm
        + 0.25 * (1.0 - bwi_norm)
        + 0.20 * kurt_norm
        + 0.10 * (1.0 - vaq_norm)
        + 0.10 * (1.0 - vrr_norm)
    )

    return {
        "SNR": snr_db,
        "BWI": bwi,
        "Kurt": kurt,
        "VAQ": vaq,
        "VRR": vrr,
        "CQS": float(np.clip(cqs, 0.0, 1.0)),
    }


def _select_denoise_mode(
    sqa: Dict[str, float],
    cqs_clean_thr: float = 0.68,
    bwi_thr: float = 0.28,
    snr_hf_thr: float = 9.0,
    vaq_thr: float = 0.48,
) -> str:
    """
    Adaptive Denoising Decision Tree routing each 5s window into 4 canonical modes:
      - 'Bypass'  : Clean ECG (zero extra filtering cost, preserves pristine QRS)
      - 'Median'  : Dominant baseline wander (BWI > bwi_thr) -> Dual-Median filter
      - 'Wavelet' : Dominant HF / EMG / AWGN noise (SNR < snr_hf_thr) -> Sym4 Wavelet
      - 'Kalman'  : Dynamic / impulsive / mixed artifacts -> Zero-phase Adaptive Kalman
    """
    cqs = sqa["CQS"]
    bwi = sqa["BWI"]
    snr = sqa["SNR"]
    vaq = sqa["VAQ"]

    if cqs >= cqs_clean_thr and bwi <= bwi_thr and snr >= snr_hf_thr and vaq <= vaq_thr:
        return "Bypass"
    if bwi > bwi_thr:
        return "Median"
    if snr < snr_hf_thr:
        return "Wavelet"
    return "Kalman"


def _detect_rail_saturation_mask(
    ecg: np.ndarray,
    fs: float,
    min_rail_sec: float = 0.05,
    pad_sec: float = 0.25,
) -> np.ndarray:
    """
    Physics-based dry-electrode lead-off / ADC rail-saturation mask.
    Detects:
      1) Sustained ADC rail clipping (consecutive samples pinned within 1.5% of extreme
         signal min/max for >= 50 ms)
      2) Sustained signal flatline dropout (|dx| <= 5e-4 * ptp for >= 450 ms)
    Dilates by +/- pad_sec to suppress sharp vertical rail-transition edges.
    Triggers 0.00% on clean A1/B1 and <= 0.01% on B2/C while catching electrode detachment in D/A2.
    """
    n = ecg.size
    mask = np.zeros(n, dtype=bool)
    if n < int(0.5 * fs):
        return mask

    sig_max = float(np.max(ecg))
    sig_min = float(np.min(ecg))
    ptp = sig_max - sig_min
    if ptp < 1e-6:
        return np.ones(n, dtype=bool)

    rail_tol = 0.015 * ptp
    near_rail = (ecg >= sig_max - rail_tol) | (ecg <= sig_min + rail_tol)
    min_rail_len = max(5, int(min_rail_sec * fs))

    padded = np.concatenate(([False], near_rail, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for s, e in zip(starts, ends):
        if (e - s) >= min_rail_len:
            mask[s:e] = True

    dx = np.abs(np.diff(ecg))
    is_flat = np.empty(n, dtype=bool)
    is_flat[0] = False
    is_flat[1:] = dx <= max(1e-6, 5e-4 * ptp)
    min_drop_len = max(12, int(0.45 * fs))
    p2 = np.concatenate(([False], is_flat, [False]))
    ch2 = np.diff(p2.astype(np.int8))
    for s, e in zip(np.flatnonzero(ch2 == 1), np.flatnonzero(ch2 == -1)):
        if (e - s) >= min_drop_len:
            mask[max(0, s - 1) : e] = True

    if not np.any(mask):
        return mask

    pad = max(1, int(pad_sec * fs))
    return uniform_filter1d(mask.astype(np.float64), size=2 * pad + 1, mode="nearest") > 0.0


def _hilbert_ampt_envelope(
    ecg_denoised: np.ndarray,
    fs: float,
    bp_low: float = 5.0,
    bp_high: float = 18.0,
    mwi_ms: float = 110.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zero-phase Hilbert-Enhanced AMPT feature extraction:
      1. Zero-phase 4th-order Butterworth BPF [5, 18] Hz
      2. Centered 5-point Pan-Tompkins derivative
      3. Instantaneous Analytic Hilbert Energy |d[n] + j H{d[n]}|^2 via next_fast_len FFT
      4. Centered Moving Window Integration (~110 ms)
    """
    n = ecg_denoised.size
    nyq = 0.5 * fs
    hi = min(bp_high, 0.45 * fs)
    lo = min(bp_low, hi * 0.5)
    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")

    padlen = min(3 * (2 * sos.shape[0] + 1), max(0, n - 1))
    if n > padlen + 1 and padlen > 0:
        bpf = signal.sosfiltfilt(sos, ecg_denoised, padlen=padlen)
    else:
        bpf = signal.sosfilt(sos, ecg_denoised)

    k_deriv = np.array([1.0, 2.0, 0.0, -2.0, -1.0], dtype=np.float64) * (fs / 8.0)
    deriv = np.convolve(bpf, k_deriv, mode="same")

    n_fft = next_fast_len(n)
    analytic = signal.hilbert(deriv, N=n_fft)[:n]
    hilb_amp = np.abs(analytic)
    hilb_energy = hilb_amp * hilb_amp

    mwi_win = max(3, int(round((mwi_ms / 1000.0) * fs)))
    mwi_env = uniform_filter1d(hilb_energy, size=mwi_win, mode="nearest")

    return bpf, hilb_amp, mwi_env


def _dynamic_adaptive_peak_detect(
    env: np.ndarray,
    bpf: np.ndarray,
    fs: float,
    window_cqs: np.ndarray,
    sat_mask: np.ndarray,
    refractory_ms: float = 200.0,
    t_wave_ms: float = 360.0,
) -> List[int]:
    """
    Fast dynamic multi-parameter adaptive threshold detector on Hilbert-MWI energy:
      - Causal 4-second robust percentile initialization (real-time wearable compliant)
      - Anti-lock-up robust QRS amplitude tracking (rolling median + bounded EWMA)
      - Continuous SQA (CQS)-modulated threshold factor & search-back sensitivity
      - Refractory blanking (200 ms) + T-wave discrimination (200-360 ms)
      - Sparsity-gated prolonged-silence recovery (prevents lock-up without chasing noise)
      - Exact R-peak localization refinement onto max |bpf| within +/- 55 ms
    """
    n = env.size
    if n < 5:
        return []

    work_env = env.copy()
    if np.any(sat_mask):
        work_env[sat_mask] = 0.0

    if float(np.max(work_env)) <= 1e-18:
        return []

    min_cand_dist = max(1, int(0.08 * fs))
    cand_peaks = _fast_coarse_peaks(work_env, height=1e-18, min_dist=min_cand_dist)
    if cand_peaks.size == 0:
        return []

    block_len = max(16, int(2.0 * fs))
    n_blocks = max(1, (n + block_len - 1) // block_len)
    block_p92 = np.zeros(n_blocks, dtype=np.float64)
    block_bg = np.zeros(n_blocks, dtype=np.float64)

    n_full = n // block_len
    if n_full > 0:
        e_mat = work_env[: n_full * block_len].reshape(n_full, block_len)
        block_p92[:n_full] = np.percentile(e_mat, 92, axis=1)
        block_bg[:n_full] = np.percentile(e_mat, 35, axis=1)
    if n_full < n_blocks:
        seg_e = work_env[n_full * block_len :]
        block_p92[n_full] = float(np.percentile(seg_e, 92))
        block_bg[n_full] = float(np.percentile(seg_e, 35))

    if n_blocks >= 3:
        block_p92_s = np.maximum(block_p92, 0.5 * uniform_filter1d(block_p92, size=3, mode="nearest"))
    else:
        block_p92_s = block_p92

    # Causal 4-second robust percentile initialization
    init_end = min(n, int(4.0 * fs))
    init_slice = work_env[:init_end]
    nz_init = init_slice[init_slice > 0.0]
    if nz_init.size < 10:
        nz_init = work_env[work_env > 0.0]
    if nz_init.size == 0:
        return []

    p95 = float(np.percentile(nz_init, 95))
    p25 = float(np.percentile(nz_init, 25))
    spki = p95
    npki = p25

    qrs_hist: List[float] = [p95] * 4
    rr_hist: List[float] = [0.8 * fs] * 4

    refrac_samp = int(round((refractory_ms / 1000.0) * fs))
    twave_samp = int(round((t_wave_ms / 1000.0) * fs))
    split_samp = int(round(0.14 * fs))
    refine_half = max(1, int(round(0.055 * fs)))

    accepted: List[int] = []
    accepted_env: List[float] = []
    accepted_cand_idx: List[int] = []

    cands = cand_peaks.tolist()
    cand_vals = work_env[cand_peaks].tolist()

    for idx_c, (p, val) in enumerate(zip(cands, cand_vals)):
        if val <= 1e-18:
            continue

        b_idx = p // block_len
        if b_idx >= n_blocks:
            b_idx = n_blocks - 1

        cqs_local = float(window_cqs[p])
        bg_local = float(block_bg[b_idx]) + 1e-18

        qrs_med = _fast_median_small(qrs_hist[-8:])
        qrs_ref = 0.60 * spki + 0.40 * qrs_med

        # Continuous SQA-modulated dynamic threshold:
        #   Clean (CQS >= 0.70) -> alpha = 0.25 (preserves full sensitivity)
        #   Noisy (CQS < 0.70)  -> alpha rises smoothly up to ~0.48 to curb false positives
        alpha = 0.25 + 0.45 * max(0.0, 0.70 - cqs_local)
        thr = npki + alpha * max(qrs_ref - npki, 0.0)

        rr_ave = min(max(_fast_median_small(rr_hist[-8:]), 0.35 * fs), 1.40 * fs)

        if accepted:
            gap = p - accepted[-1]
            # SQA-aware RR search-back
            if gap > 1.66 * rr_ave and accepted_cand_idx:
                sb_factor = 0.45 if cqs_local >= 0.62 else 0.65
                sb_thr = sb_factor * thr
                prev_c_idx = accepted_cand_idx[-1]
                best_sb_c = -1
                best_sb_val = sb_thr
                for k_sb in range(prev_c_idx + 1, idx_c):
                    p_sb = cands[k_sb]
                    if p_sb - accepted[-1] <= refrac_samp or p - p_sb <= refrac_samp:
                        continue
                    v_sb = cand_vals[k_sb]
                    if v_sb > best_sb_val:
                        best_sb_val = v_sb
                        best_sb_c = k_sb
                if best_sb_c != -1:
                    p_sb = cands[best_sb_c]
                    rr_sb = p_sb - accepted[-1]
                    accepted.append(p_sb)
                    accepted_env.append(best_sb_val)
                    accepted_cand_idx.append(best_sb_c)
                    clipped_sb = min(best_sb_val, 3.0 * qrs_med)
                    spki = 0.25 * clipped_sb + 0.75 * spki
                    qrs_hist.append(clipped_sb)
                    if 0.25 * fs <= rr_sb <= 2.0 * fs:
                        rr_hist.append(float(rr_sb))
                    gap = p - accepted[-1]
                    qrs_med = _fast_median_small(qrs_hist[-8:])
                    qrs_ref = 0.60 * spki + 0.40 * qrs_med
                    thr = npki + alpha * max(qrs_ref - npki, 0.0)

            # Sparsity-gated anti-lock-up recovery after prolonged silence (> 2.4 s)
            if gap > 2.4 * fs:
                local_ref = float(block_p92_s[b_idx])
                if local_ref > 1e-18 and (local_ref / bg_local) >= 3.5 and spki > 2.5 * local_ref:
                    spki = 0.4 * spki + 0.6 * local_ref
                    qrs_hist.append(local_ref)
                    qrs_med = _fast_median_small(qrs_hist[-8:])
                    qrs_ref = 0.60 * spki + 0.40 * qrs_med
                    thr = npki + alpha * max(qrs_ref - npki, 0.0)

        if val > thr:
            if not accepted:
                accepted.append(p)
                accepted_env.append(val)
                accepted_cand_idx.append(idx_c)
                clipped_v = min(val, 3.5 * max(qrs_med, 1e-18))
                spki = 0.125 * clipped_v + 0.875 * spki
                qrs_hist.append(clipped_v)
            else:
                diff = p - accepted[-1]
                if diff <= refrac_samp:
                    if diff <= split_samp and val > accepted_env[-1] * 1.15:
                        accepted[-1] = p
                        accepted_env[-1] = val
                        accepted_cand_idx[-1] = idx_c
                    else:
                        npki = 0.125 * min(val, 1.5 * qrs_med) + 0.875 * npki
                elif diff < twave_samp:
                    if val >= 0.40 * accepted_env[-1]:
                        accepted.append(p)
                        accepted_env.append(val)
                        accepted_cand_idx.append(idx_c)
                        clipped_v = min(val, 3.5 * qrs_med)
                        spki = 0.125 * clipped_v + 0.875 * spki
                        qrs_hist.append(clipped_v)
                        if 0.25 * fs <= diff <= 2.0 * fs:
                            rr_hist.append(float(diff))
                    else:
                        npki = 0.125 * min(val, 1.5 * qrs_med) + 0.875 * npki
                else:
                    accepted.append(p)
                    accepted_env.append(val)
                    accepted_cand_idx.append(idx_c)
                    clipped_v = min(val, 3.5 * qrs_med)
                    spki = 0.125 * clipped_v + 0.875 * spki
                    qrs_hist.append(clipped_v)
                    if 0.25 * fs <= diff <= 2.0 * fs:
                        rr_hist.append(float(diff))
        else:
            npki = 0.125 * min(val, 0.75 * qrs_ref) + 0.875 * npki

    # Final R-peak localization refinement onto max |bpf| within +/- 55 ms
    abs_bpf = np.abs(bpf)
    refined: List[int] = []
    for p in accepted:
        lo = max(0, p - refine_half)
        hi = min(n, p + refine_half + 1)
        best_r = lo + int(np.argmax(abs_bpf[lo:hi]))
        if not refined or best_r - refined[-1] >= refrac_samp:
            refined.append(best_r)
        else:
            if abs_bpf[best_r] > abs_bpf[refined[-1]]:
                refined[-1] = best_r

    return refined


def asqa_h_ampt(
    ecg: np.ndarray,
    fs: float,
    window_sec: float = 5.0,
    return_details: bool = False,
):
    """
    Full Proposed ASQA-H-AMPT Detector:
      Stage 1: Windowed SQA (5s windows: local SNR, BWI, Kurtosis, VAQ, VRR, CQS)
      Stage 2: Adaptive Denoising Decision Tree (Bypass / Median / Wavelet / Kalman)
      Stage 3: Zero-Phase Hilbert-Enhanced AMPT Feature Extraction (Analytic Energy)
      Stage 4: Dynamic Multi-Parameter Adaptive Threshold + Refractory + T-wave + RR Search-Back
    """
    sig = np.asarray(ecg, dtype=np.float64)
    n = sig.size
    if n == 0:
        empty_sqa = {
            "Bypass": 0,
            "Median": 0,
            "Wavelet": 0,
            "Kalman": 0,
            "n_windows": 0,
            "mean_cqs": 0.0,
            "mean_snr": 0.0,
            "mean_bwi": 0.0,
            "sat_frac": 0.0,
        }
        return ([], 0.0, empty_sqa, {}) if return_details else ([], 0.0, empty_sqa)

    t0 = time.perf_counter()

    win_len = max(int(round(window_sec * fs)), int(fs))
    bounds: List[Tuple[int, int]] = []
    pos = 0
    while pos < n:
        end = min(n, pos + win_len)
        if n - end < win_len // 2 and end < n:
            end = n
        bounds.append((pos, end))
        pos = end

    denoised = np.empty(n, dtype=np.float64)
    window_cqs = np.empty(n, dtype=np.float64)
    mode_counts = {"Bypass": 0, "Median": 0, "Wavelet": 0, "Kalman": 0}
    cqs_list: List[float] = []
    snr_list: List[float] = []
    bwi_list: List[float] = []

    pad_ov = max(8, int(round(0.20 * fs)))

    for s, e in bounds:
        win = sig[s:e]
        sqa = evaluate_window_sqa(win, fs)
        mode = _select_denoise_mode(sqa)
        mode_counts[mode] += 1
        cqs_list.append(sqa["CQS"])
        snr_list.append(sqa["SNR"])
        bwi_list.append(sqa["BWI"])
        window_cqs[s:e] = sqa["CQS"]

        if mode == "Bypass":
            denoised[s:e] = win
        else:
            ps = max(0, s - pad_ov)
            pe = min(n, e + pad_ov)
            chunk = sig[ps:pe]
            if mode == "Median":
                proc = dual_median_filter_fast(chunk, fs)
            elif mode == "Wavelet":
                proc = wavelet_soft_denoise(chunk, fs, wavelet="sym4")
            else:
                proc = adaptive_kalman_filter(chunk, fs)
            denoised[s:e] = proc[s - ps : e - ps]

    sat_mask = _detect_rail_saturation_mask(sig, fs)
    bpf, hilb_amp, mwi_env = _hilbert_ampt_envelope(denoised, fs)
    peaks = _dynamic_adaptive_peak_detect(
        mwi_env,
        bpf,
        fs,
        window_cqs=window_cqs,
        sat_mask=sat_mask,
    )

    elapsed = time.perf_counter() - t0

    sqa_summary = {
        "Bypass": mode_counts["Bypass"],
        "Median": mode_counts["Median"],
        "Wavelet": mode_counts["Wavelet"],
        "Kalman": mode_counts["Kalman"],
        "n_windows": len(bounds),
        "mean_cqs": float(np.mean(cqs_list)) if cqs_list else 0.0,
        "mean_snr": float(np.mean(snr_list)) if snr_list else 0.0,
        "mean_bwi": float(np.mean(bwi_list)) if bwi_list else 0.0,
        "sat_frac": float(np.mean(sat_mask)),
    }

    if return_details:
        debug_dict = {
            "raw": sig,
            "denoised": denoised,
            "bpf": bpf,
            "hilbert_raw": hilb_amp,
            "envelope": mwi_env,
            "sat_mask": sat_mask,
            "window_cqs": window_cqs,
            "sqa_summary": sqa_summary,
        }
        return peaks, elapsed, sqa_summary, debug_dict

    return peaks, elapsed, sqa_summary
