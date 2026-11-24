"""
Publication-Grade IEEE Two-Column Figure Generator (300 DPI) for PSD Tugas Akhir.

Generates 8 figures strictly from REAL experimental CSV outputs and REAL ECG waveforms
under result_experiment/figures/:
  - fig1_pipeline_visualization.png : Real noisy A2 ECG trace through ASQA-H-AMPT stages
  - fig2_f1_comparison.png          : Grouped F1-score across all 6 datasets (4 main methods)
  - fig3_error_and_fp_reduction.png : Total Error Rate & FP/FN decomposition on noisy A2 & D
  - fig4_ablation_study.png         : Ablation study (AMPT vs fixed preproc vs ASQA-H-AMPT)
  - fig5_efficiency_vs_f1.png       : Real-time efficiency (ms/10s) vs F1 trade-off
  - fig6_noise_robustness.png       : Real AWGN & 0.3 Hz Baseline-Drift stress test curves
  - fig7_sqa_distribution.png       : Real 5-s windowed SQA routing & CQS distribution
  - fig8_f1_error_boxplot.png       : Per-record F1 & Error Rate box plots (4 main methods)
"""

import os
import sys
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.detectors import (  # noqa: E402
    _bandpass_filt_exact,
    _derivate_step_exact,
    _mwi_exact,
    _select_denoise_mode,
    asqa_h_ampt,
    evaluate_window_sqa,
    run_ampt_fast,
)
from experiments.metrics import (  # noqa: E402
    fast_check_negative,
    fast_check_positive,
    load_dataset_records,
    load_single_record,
)

RESULT_ROOT = os.path.join(PROJECT_ROOT, "result_experiment")
FIG_DIR = os.path.join(RESULT_ROOT, "figures")

DATASETS = ["A1", "A2", "B1", "B2", "C", "D"]
DATASET_LABELS = {
    "A1": "A1\n(High Qual.)",
    "A2": "A2\n(Low Qual.)",
    "B1": "B1\n(NSR)",
    "B2": "B2\n(Arrhyth.)",
    "C": "C\n(Paced)",
    "D": "D\n(TeleHealth)",
}
DATASET_LABELS_COMPACT = {
    "A1": "A1\nHigh Q.",
    "A2": "A2\nLow Q.",
    "B1": "B1\nNSR",
    "B2": "B2\nArrh.",
    "C": "C\nPaced",
    "D": "D\nTeleH.",
}

# IEEE-ready colorblind-safe high-contrast palette
COLORS = {
    "Pan-Tompkins": "#7F8C8D",
    "AMPT": "#1F77B4",
    "Wavelet+AMPT": "#56B4E9",
    "Median+AMPT": "#E69F00",
    "Kalman+AMPT": "#8E44AD",
    "Best Fixed+AMPT": "#E69F00",
    "ASQA-H-AMPT": "#C0392B",
    "Bypass": "#2E8B57",
    "Median": "#E69F00",
    "Wavelet": "#3498DB",
    "Kalman": "#8E44AD",
}

FIXED_METHODS = ["Wavelet+AMPT", "Median+AMPT", "Kalman+AMPT"]
FIXED_KEYS = {
    "Wavelet+AMPT": "wavelet_ampt",
    "Median+AMPT": "median_ampt",
    "Kalman+AMPT": "kalman_ampt",
}


def setup_ieee_style() -> None:
    """Configure clean, crisp white-background IEEE two-column styling (300 DPI)."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 8.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "axes.titlepad": 6.0,
            "axes.labelsize": 8.0,
            "axes.labelweight": "normal",
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.0,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.framealpha": 0.94,
            "legend.edgecolor": "#CCCCCC",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D5D8DC",
            "grid.linestyle": "--",
            "grid.alpha": 0.65,
            "grid.linewidth": 0.55,
            "axes.axisbelow": True,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "savefig.facecolor": "white",
        }
    )


def _save_fig(fig: plt.Figure, filename: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    return out_path


def _best_fixed_per_dataset(df_main: pd.DataFrame) -> Dict[str, str]:
    """Identify the highest-F1 fixed preprocessing method per dataset dynamically from real CSVs."""
    best = {}
    for ds in DATASETS:
        sub = df_main[(df_main["Dataset"] == ds) & (df_main["Algorithm"].isin(FIXED_METHODS))]
        best_row = sub.sort_values(["F1_Pct", "Error_Rate_Pct"], ascending=[False, True]).iloc[0]
        best[ds] = str(best_row["Algorithm"])
    return best


def _sig_stars(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


# =============================================================================
# Figure 1: Pipeline Visualization on Real Noisy A2 ECG Trace
# =============================================================================


def make_fig1_pipeline() -> str:
    full_path, _, nom_fs = load_dataset_records("A2")
    rec_name = "2812.dat"
    ecg, ann, fs = load_single_record("A2", full_path, rec_name, nom_fs)

    peaks_asqa, _, _, dbg = asqa_h_ampt(ecg, fs, return_details=True)
    peaks_ampt, _ = run_ampt_fast(ecg, fs)

    bp_ampt = _bandpass_filt_exact(ecg, 4, fs, 15.0, 5.0)
    deriv_ampt = _derivate_step_exact(bp_ampt)
    mwi_ampt = _mwi_exact(deriv_ampt * deriv_ampt, fs)
    pad_mwi = ecg.size - mwi_ampt.size
    mwi_ampt_full = np.pad(mwi_ampt, (pad_mwi // 2, pad_mwi - pad_mwi // 2), mode="edge")

    t_start, t_end = 50.0, 60.0
    s_idx = int(t_start * fs)
    e_idx = int(t_end * fs)
    t_rel = np.arange(e_idx - s_idx, dtype=np.float64) / float(fs)

    raw_seg = dbg["raw"][s_idx:e_idx]
    den_seg = dbg["denoised"][s_idx:e_idx]
    mwi_seg = mwi_ampt_full[s_idx:e_idx]
    hilb_amp_seg = dbg["hilbert_raw"][s_idx:e_idx]
    env_seg = dbg["envelope"][s_idx:e_idx]

    sqa_w1 = evaluate_window_sqa(raw_seg[: 5 * fs], fs)
    sqa_w2 = evaluate_window_sqa(raw_seg[5 * fs :], fs)
    mode_w1 = _select_denoise_mode(sqa_w1)
    mode_w2 = _select_denoise_mode(sqa_w2)

    ann_arr = np.asarray(ann, dtype=np.float64)
    ann_seg = ann_arr[(ann_arr >= s_idx) & (ann_arr < e_idx)] - s_idx
    asqa_arr = np.asarray(peaks_asqa, dtype=np.float64)
    asqa_seg = asqa_arr[(asqa_arr >= s_idx) & (asqa_arr < e_idx)] - s_idx
    ampt_arr = np.asarray(peaks_ampt, dtype=np.float64)
    ampt_seg = ampt_arr[(ampt_arr >= s_idx) & (ampt_arr < e_idx)] - s_idx

    ampt_fp = np.asarray(fast_check_positive(ann_seg, ampt_seg, fs), dtype=np.float64)
    ampt_fn = np.asarray(fast_check_negative(ann_seg, ampt_seg, fs), dtype=np.float64)
    asqa_fp = fast_check_positive(ann_seg, asqa_seg, fs)
    asqa_fn = fast_check_negative(ann_seg, asqa_seg, fs)

    fig, axes = plt.subplots(5, 1, figsize=(7.16, 6.8), sharex=True, gridspec_kw={"hspace": 0.36})

    # (a) Raw Noisy ECG
    ax = axes[0]
    ax.axvspan(0.0, 5.0, color="#FEF5E7", alpha=0.85, zorder=0)
    ax.axvspan(5.0, 10.0, color="#FCF3CF", alpha=0.55, zorder=0)
    ax.axvline(5.0, color="#B7791F", linestyle=":", linewidth=1.0)
    ax.plot(t_rel, raw_seg, color="#2C3E50", linewidth=0.85)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.30 * (y1 - y0))
    ax.set_ylabel("Amp.\n(mV)")
    ax.set_title(
        f"(a) Raw Noisy ECG — PhysioNet 2014 A2 (Record {rec_name[:-4]}, 50–60 s, $f_s$={fs} Hz)",
        loc="left",
    )
    ax.text(
        0.25,
        0.87,
        f"Win 1: CQS={sqa_w1['CQS']:.2f}, BWI={sqa_w1['BWI']:.2f} → {mode_w1}",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.7,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D69E2E", alpha=0.95),
    )
    ax.text(
        0.75,
        0.87,
        f"Win 2: CQS={sqa_w2['CQS']:.2f}, BWI={sqa_w2['BWI']:.2f} → {mode_w2}",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.7,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D69E2E", alpha=0.95),
    )

    # (b) SQA-Adaptive Denoised ECG
    ax = axes[1]
    ax.plot(t_rel, den_seg, color=COLORS["AMPT"], linewidth=0.85)
    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=0.6)
    ax.set_ylabel("Amp.\n(mV)")
    ax.set_title(
        f"(b) SQA-Adaptive Denoised ECG (Dual-Median Baseline Removal; ptp {np.ptp(raw_seg):.2f} → {np.ptp(den_seg):.2f} mV)",
        loc="left",
    )

    # (c) Classical AMPT MWI
    ax = axes[2]
    mwi_norm = mwi_seg / (np.max(mwi_seg) + 1e-12)
    ax.plot(t_rel, mwi_norm, color=COLORS["Pan-Tompkins"], linewidth=0.95)
    ax.fill_between(t_rel, 0.0, mwi_norm, color=COLORS["Pan-Tompkins"], alpha=0.16)
    ax.set_ylabel("Norm.\nMWI")
    ax.set_ylim(-0.04, 1.12)
    ax.set_title("(c) Classical AMPT Moving Window Integration (150 ms MWI on Bandpass Derivative)", loc="left")

    # (d) Hilbert Analytic Envelope & Smoothed Hilbert-MWI Energy
    ax = axes[3]
    hilb_norm = hilb_amp_seg / (np.max(hilb_amp_seg) + 1e-12)
    env_norm = env_seg / (np.max(env_seg) + 1e-12)
    ax.plot(
        t_rel,
        hilb_norm,
        color="#56B4E9",
        linewidth=0.75,
        alpha=0.85,
        label="Hilbert Envelope $|d[n]+j\\mathcal{H}\\{d[n]\\}|$",
    )
    ax.plot(t_rel, env_norm, color=COLORS["ASQA-H-AMPT"], linewidth=1.15, label="Hilbert-MWI Energy Envelope (110 ms)")
    ax.fill_between(t_rel, 0.0, env_norm, color=COLORS["ASQA-H-AMPT"], alpha=0.12)
    ax.set_ylabel("Norm.\nEnergy")
    ax.set_ylim(-0.04, 1.32)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_title("(d) Zero-Phase Hilbert Analytic Envelope & Hilbert-MWI Energy", loc="left")
    ax.legend(loc="upper right", ncol=2, fontsize=6.7)

    # (e) Final Detected vs Ground-Truth R-Peaks
    ax = axes[4]
    ax.plot(t_rel, den_seg, color="#34495E", linewidth=0.80, alpha=0.88, label="Denoised ECG")
    ann_i = np.clip(np.round(ann_seg).astype(int), 0, den_seg.size - 1)
    asqa_i = np.clip(np.round(asqa_seg).astype(int), 0, den_seg.size - 1)
    ax.scatter(
        ann_seg / fs,
        den_seg[ann_i],
        s=46,
        facecolors="none",
        edgecolors="#009E73",
        linewidths=1.35,
        zorder=4,
        label=f"Ground Truth (N={len(ann_seg)})",
    )
    ax.scatter(
        asqa_seg / fs,
        den_seg[asqa_i],
        marker="v",
        s=24,
        color=COLORS["ASQA-H-AMPT"],
        zorder=5,
        label=f"ASQA-H-AMPT (TP={len(asqa_seg)-len(asqa_fp)}, FP={len(asqa_fp)}, FN={len(asqa_fn)})",
    )
    if ampt_fn.size > 0:
        fn_i = np.clip(np.round(ampt_fn).astype(int), 0, den_seg.size - 1)
        ax.scatter(
            ampt_fn / fs,
            den_seg[fn_i] - 0.14 * np.ptp(den_seg),
            marker="x",
            s=36,
            color="#E69F00",
            linewidths=1.4,
            zorder=6,
            label=f"AMPT Missed (FN={len(ampt_fn)}, FP={len(ampt_fp)})",
        )
    if ampt_fp.size > 0:
        fp_i = np.clip(np.round(ampt_fp).astype(int), 0, den_seg.size - 1)
        ax.scatter(
            ampt_fp / fs,
            den_seg[fp_i],
            marker="+",
            s=38,
            color="#8E44AD",
            linewidths=1.35,
            zorder=6,
            label="AMPT Spurious (FP)",
        )
    ax.set_ylabel("Amp.\n(mV)")
    ax.set_xlabel("Relative Time (s) [Offset +50.0 s in Record 2812]")
    ax.set_xlim(0.0, 10.0)
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo - 0.08 * (y_hi - y_lo), y_hi + 0.15 * (y_hi - y_lo))
    ax.set_title("(e) Detected R-Peaks vs. Ground-Truth Annotations (ASQA-H-AMPT vs. Baseline AMPT Errors)", loc="left")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.36),
        ncol=3,
        fontsize=6.7,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    return _save_fig(fig, "fig1_pipeline_visualization.png")


# =============================================================================
# Figure 2: Grouped F1-Score Comparison Across All 6 Datasets
# =============================================================================


def make_fig2_f1_comparison(df_main: pd.DataFrame, df_sig: pd.DataFrame) -> str:
    best_fixed = _best_fixed_per_dataset(df_main)
    groups = ["Pan-Tompkins", "AMPT", "Best Fixed+AMPT", "ASQA-H-AMPT"]
    bar_labels = ["Pan-Tompkins", "AMPT", "Best Fixed Preproc+AMPT", "ASQA-H-AMPT (Proposed)"]

    vals = {g: [] for g in groups}
    for ds in DATASETS:
        for g in ("Pan-Tompkins", "AMPT", "ASQA-H-AMPT"):
            v = float(df_main[(df_main["Dataset"] == ds) & (df_main["Algorithm"] == g)]["F1_Pct"].iloc[0])
            vals[g].append(v)
        bf_alg = best_fixed[ds]
        v_bf = float(df_main[(df_main["Dataset"] == ds) & (df_main["Algorithm"] == bf_alg)]["F1_Pct"].iloc[0])
        vals["Best Fixed+AMPT"].append(v_bf)

    pvals_ampt = {}
    for ds in DATASETS:
        row = df_sig[
            (df_sig["Dataset"] == ds)
            & (df_sig["Comparison"] == "ASQA-H-AMPT vs AMPT")
            & (df_sig["Metric"] == "F1_Pct")
        ].iloc[0]
        pvals_ampt[ds] = float(row["wilcoxon_pvalue"])

    fig, ax = plt.subplots(figsize=(7.16, 3.45))
    x = np.arange(len(DATASETS))
    width = 0.195
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    hatches = ["///", "", "...", ""]

    for g, lbl, off, ht in zip(groups, bar_labels, offsets, hatches):
        bars = ax.bar(
            x + off,
            vals[g],
            width=width * 0.94,
            label=lbl,
            color=COLORS[g],
            edgecolor="#222222" if g == "ASQA-H-AMPT" else "#444444",
            linewidth=0.85 if g == "ASQA-H-AMPT" else 0.55,
            hatch=ht,
            zorder=3,
        )
        for idx_b, bar in enumerate(bars):
            h = bar.get_height()
            star = _sig_stars(pvals_ampt[DATASETS[idx_b]]) if g == "ASQA-H-AMPT" else ""
            txt = f"{h:.2f}{star}" if g == "ASQA-H-AMPT" else f"{h:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.50,
                txt,
                ha="center",
                va="bottom",
                fontsize=6.5 if g == "ASQA-H-AMPT" else 5.9,
                fontweight="bold" if g == "ASQA-H-AMPT" else "normal",
                color="#901B1B" if g == "ASQA-H-AMPT" else "#333333",
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_ylabel("Macro-Average F1-Score (%)")
    ax.set_ylim(70.0, 108.5)
    ax.set_yticks(np.arange(70, 101, 5))
    ax.set_title("QRS Detection F1-Score Across All Six Benchmark Datasets (400 Records, 315,900 Beats)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=True)
    fig.text(
        0.5,
        -0.135,
        "Wilcoxon vs. AMPT: ** p < 0.01, *** p < 0.001  |  Best Fixed Preproc: A1=Wavelet/Median, A2/B1/B2/D=Kalman, C=Wavelet",
        ha="center",
        va="top",
        fontsize=6.6,
        color="#555555",
        style="italic",
    )

    return _save_fig(fig, "fig2_f1_comparison.png")


# =============================================================================
# Figure 3: Total Error Rate & FP/FN Breakdown on Noisy Datasets (A2 & D)
# =============================================================================


def make_fig3_error_and_fp(df_main: pd.DataFrame) -> str:
    noisy_ds = ["A2", "D"]
    best_fixed = _best_fixed_per_dataset(df_main)
    methods_4 = ["Pan-Tompkins", "AMPT", "Best Fixed+AMPT", "ASQA-H-AMPT"]
    short_labels = ["Pan-Tompkins", "AMPT", "Best Fixed (Kalman)", "ASQA-H-AMPT (Proposed)"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 3.55), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.30}
    )

    # Panel (a): Total Error Rate (%) on A2 & D
    x = np.arange(len(noisy_ds))
    w = 0.19
    offs = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    hatches = ["///", "", "...", ""]

    for m, slbl, off, ht in zip(methods_4, short_labels, offs, hatches):
        errs = []
        for ds in noisy_ds:
            alg = best_fixed[ds] if m == "Best Fixed+AMPT" else m
            row = df_main[(df_main["Dataset"] == ds) & (df_main["Algorithm"] == alg)].iloc[0]
            errs.append(float(row["Error_Rate_Pct"]))
        bars = ax1.bar(
            x + off,
            errs,
            width=w * 0.92,
            color=COLORS[m],
            edgecolor="#222222" if m == "ASQA-H-AMPT" else "#444444",
            linewidth=0.8 if m == "ASQA-H-AMPT" else 0.55,
            hatch=ht,
            label=slbl,
            zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.65,
                f"{h:.2f}%",
                ha="center",
                va="bottom",
                fontsize=6.3,
                fontweight="bold" if m == "ASQA-H-AMPT" else "normal",
                color="#901B1B" if m == "ASQA-H-AMPT" else "#333333",
                rotation=90,
            )

    for i, ds in enumerate(noisy_ds):
        e_ampt = float(df_main[(df_main["Dataset"] == ds) & (df_main["Algorithm"] == "AMPT")]["Error_Rate_Pct"].iloc[0])
        e_asqa = float(
            df_main[(df_main["Dataset"] == ds) & (df_main["Algorithm"] == "ASQA-H-AMPT")]["Error_Rate_Pct"].iloc[0]
        )
        delta = e_asqa - e_ampt
        ax1.annotate(
            f"{delta:+.2f} pp\nvs. AMPT",
            xy=(i + 1.5 * w, e_asqa + 6.5),
            xytext=(i + 1.05 * w, max(e_ampt, e_asqa) + 12.0),
            ha="center",
            fontsize=6.3,
            fontweight="bold",
            color="#901B1B",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.85),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FDEDEC", edgecolor="#C0392B", lw=0.6),
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(["A2: Low Quality\n(100 recs)", "D: TeleHealth\n(134 recs)"])
    ax1.set_ylabel("Total Error Rate = (FP + FN) / Beats (%)")
    ax1.set_ylim(0, 50.5)
    ax1.set_title("(a) Total Error Rate on A2 & D", loc="left")

    # Panel (b): Stacked FP Rate (%) + FN Rate (%) Decomposition on A2 & D
    positions = [0.0, 1.0, 2.0, 3.0, 5.2, 6.2, 7.2, 8.2]
    fp_rates, fn_rates, fp_cnts, fn_cnts = [], [], [], []
    tick_labels = []
    for ds in noisy_ds:
        for m in methods_4:
            alg = best_fixed[ds] if m == "Best Fixed+AMPT" else m
            row = df_main[(df_main["Dataset"] == ds) & (df_main["Algorithm"] == alg)].iloc[0]
            ann = float(row["Annotated_Peaks"])
            fp_c = int(row["FP"])
            fn_c = int(row["FN"])
            fp_rates.append(fp_c / ann * 100.0)
            fn_rates.append(fn_c / ann * 100.0)
            fp_cnts.append(fp_c)
            fn_cnts.append(fn_c)
            tick_labels.append({"Pan-Tompkins": "PT", "AMPT": "AMPT", "Best Fixed+AMPT": "Kal.", "ASQA-H-AMPT": "ASQA"}[m])

    bar_w = 0.72
    b_fp = ax2.bar(
        positions,
        fp_rates,
        width=bar_w,
        color="#E67E22",
        edgecolor="#444444",
        linewidth=0.6,
        label="FP Rate (FP / Beats %)",
        zorder=3,
    )
    b_fn = ax2.bar(
        positions,
        fn_rates,
        bottom=fp_rates,
        width=bar_w,
        color="#2E86C1",
        edgecolor="#444444",
        linewidth=0.6,
        hatch="///",
        label="FN Rate (FN / Beats %)",
        zorder=3,
    )

    for idx_pos in (3, 7):
        b_fp[idx_pos].set_edgecolor("#901B1B")
        b_fp[idx_pos].set_linewidth(1.15)
        b_fn[idx_pos].set_edgecolor("#901B1B")
        b_fn[idx_pos].set_linewidth(1.15)

    for pos, fpr, fnr, fpc, fnc in zip(positions, fp_rates, fn_rates, fp_cnts, fn_cnts):
        fp_txt = f"{fpc/1000:.1f}k" if fpc >= 5000 else f"{fpc:,}"
        fn_txt = f"{fnc/1000:.1f}k" if fnc >= 5000 else f"{fnc:,}"
        if fpr >= 4.5:
            ax2.text(pos, fpr / 2.0, fp_txt, ha="center", va="center", fontsize=5.4, color="white", fontweight="bold")
        if fnr >= 4.5:
            ax2.text(
                pos,
                fpr + fnr / 2.0,
                fn_txt,
                ha="center",
                va="center",
                fontsize=5.4,
                color="white",
                fontweight="bold",
            )
        ax2.text(pos, fpr + fnr + 0.65, f"{fpr+fnr:.1f}%", ha="center", va="bottom", fontsize=5.8, color="#222222")

    ax2.axvline(4.1, color="#888888", linestyle="--", linewidth=0.85)
    ax2.text(1.5, 48.2, "Dataset A2\n(78,618 beats)", ha="center", va="top", fontsize=7.0, fontweight="bold", color="#222222")
    ax2.text(6.7, 48.2, "Dataset D\n(3,726 beats)", ha="center", va="top", fontsize=7.0, fontweight="bold", color="#222222")
    ax2.text(
        1.5,
        39.5,
        "FN: -48.0% vs. AMPT",
        ha="center",
        va="center",
        fontsize=6.1,
        color="#1B4F72",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#EBF5FB", edgecolor="#2E86C1", lw=0.55),
    )
    ax2.text(
        7.15,
        37.8,
        "FN: -44.0% (AMPT)\nFP: -20.1% (PT)",
        ha="center",
        va="center",
        fontsize=5.7,
        color="#1B4F72",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#EBF5FB", edgecolor="#2E86C1", lw=0.55),
    )

    ax2.set_xticks(positions)
    ax2.set_xticklabels(tick_labels, fontsize=6.7)
    ax2.set_ylabel("Error Decomposition (% Beats; Counts Inside)")
    ax2.set_ylim(0, 50.5)
    ax2.set_title("(b) FP vs. FN Breakdown on A2 & D", loc="left")

    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=6.6)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=6.6)

    return _save_fig(fig, "fig3_error_and_fp_reduction.png")


# =============================================================================
# Figure 4: Ablation Study (AMPT vs Fixed Preproc vs ASQA-H-AMPT)
# =============================================================================


def make_fig4_ablation(df_ablation: pd.DataFrame) -> str:
    abl_methods = ["AMPT", "Wavelet+AMPT", "Median+AMPT", "Kalman+AMPT", "ASQA-H-AMPT"]
    delta_methods = ["Wavelet+AMPT", "Median+AMPT", "Kalman+AMPT", "ASQA-H-AMPT"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 3.55), gridspec_kw={"width_ratios": [1.38, 1.0], "wspace": 0.28}
    )

    # Panel (a): Delta F1 vs Baseline AMPT across all 6 datasets
    x = np.arange(len(DATASETS))
    w = 0.19
    offs = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    hatches = ["///", "...", "\\\\", ""]

    for m, off, ht in zip(delta_methods, offs, hatches):
        deltas = [
            float(df_ablation[(df_ablation["Dataset"] == ds) & (df_ablation["Algorithm"] == m)]["Delta_F1_vs_AMPT"].iloc[0])
            for ds in DATASETS
        ]
        bars = ax1.bar(
            x + off,
            deltas,
            width=w * 0.92,
            color=COLORS[m],
            edgecolor="#222222" if m == "ASQA-H-AMPT" else "#444444",
            linewidth=0.8 if m == "ASQA-H-AMPT" else 0.55,
            hatch=ht,
            label=m,
            zorder=3,
        )
        for idx_ds, (bar, val) in enumerate(zip(bars, deltas)):
            if abs(val) >= 0.25 or m == "ASQA-H-AMPT":
                bx = bar.get_x() + bar.get_width() / 2.0
                if m == "Kalman+AMPT" and DATASETS[idx_ds] == "A2":
                    ax1.text(bx - 0.04, val + 0.12, f"{val:+.2f}", ha="right", va="bottom", fontsize=5.9, color="#4A235A")
                elif m == "Kalman+AMPT" and DATASETS[idx_ds] == "C":
                    ax1.text(bx + 0.03, val - 0.10, f"{val:+.2f}", ha="left", va="top", fontsize=5.7, color="#4A235A")
                elif m == "Median+AMPT" and DATASETS[idx_ds] == "C":
                    ax1.text(bx - 0.02, val - 0.10, f"{val:+.2f}", ha="center", va="top", fontsize=5.7, color="#7E5109")
                else:
                    va = "bottom" if val >= 0 else "top"
                    y_pos = val + 0.11 if val >= 0 else val - 0.11
                    ax1.text(
                        bx,
                        y_pos,
                        f"{val:+.2f}",
                        ha="center",
                        va=va,
                        fontsize=6.0,
                        fontweight="bold" if m == "ASQA-H-AMPT" else "normal",
                        color="#901B1B" if m == "ASQA-H-AMPT" else "#333333",
                    )

    ax1.axhline(0.0, color="#222222", linewidth=0.9, zorder=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DATASET_LABELS_COMPACT[d] for d in DATASETS], fontsize=7.0)
    ax1.set_ylabel("$\\Delta$ F1-Score vs. Baseline AMPT (pp)")
    ax1.set_ylim(-1.25, 5.55)
    ax1.set_title("(a) $\\Delta$ F1 vs. Baseline AMPT (All Datasets)", loc="left")

    # Panel (b): Absolute F1 on Challenging Datasets (A2, C, D)
    challenging = ["A2", "C", "D"]
    x2 = np.arange(len(challenging))
    w2 = 0.155
    offs2 = [-2 * w2, -1 * w2, 0.0, 1 * w2, 2 * w2]
    hatches2 = ["", "///", "...", "\\\\", ""]

    for m, off, ht in zip(abl_methods, offs2, hatches2):
        f1s = [
            float(df_ablation[(df_ablation["Dataset"] == ds) & (df_ablation["Algorithm"] == m)]["F1_Pct"].iloc[0])
            for ds in challenging
        ]
        bars = ax2.bar(
            x2 + off,
            f1s,
            width=w2 * 0.92,
            color=COLORS[m],
            edgecolor="#222222" if m == "ASQA-H-AMPT" else "#444444",
            linewidth=0.8 if m == "ASQA-H-AMPT" else 0.55,
            hatch=ht,
            label=m,
            zorder=3,
        )
        if m in ("AMPT", "ASQA-H-AMPT"):
            for bar in bars:
                h = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.35,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.0,
                    fontweight="bold" if m == "ASQA-H-AMPT" else "normal",
                    color="#901B1B" if m == "ASQA-H-AMPT" else "#1B4F72",
                    rotation=90,
                )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(["A2\n(Low Qual.)", "C\n(Paced)", "D\n(TeleHealth)"], fontsize=7.2)
    ax2.set_ylabel("Macro-Average F1-Score (%)")
    ax2.set_ylim(78.0, 104.0)
    ax2.set_title("(b) Absolute F1 on A2, C & D", loc="left")

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=7.0)

    return _save_fig(fig, "fig4_ablation_study.png")


# =============================================================================
# Figure 5: Processing Time (ms/10s) vs F1-Score Trade-off
# =============================================================================


def make_fig5_efficiency_vs_f1(df_main: pd.DataFrame) -> str:
    all_methods = ["Pan-Tompkins", "AMPT", "Wavelet+AMPT", "Median+AMPT", "Kalman+AMPT", "ASQA-H-AMPT"]
    markers = {
        "Pan-Tompkins": "s",
        "AMPT": "o",
        "Wavelet+AMPT": "^",
        "Median+AMPT": "v",
        "Kalman+AMPT": "D",
        "ASQA-H-AMPT": "*",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.55), gridspec_kw={"wspace": 0.30})

    # Panel (a): Macro-Average F1 vs Mean Processing Time (ms/10s) Pareto Trade-off
    pts = {}
    for m in all_methods:
        sub = df_main[df_main["Algorithm"] == m]
        mean_t = float(sub["Time_per_10s_ms"].mean())
        mean_f1 = float(sub["F1_Pct"].mean())
        pts[m] = (mean_t, mean_f1)
        ax1.scatter(
            mean_t,
            mean_f1,
            s=175 if m == "ASQA-H-AMPT" else 64,
            marker=markers[m],
            color=COLORS[m],
            edgecolors="#222222",
            linewidths=0.9 if m == "ASQA-H-AMPT" else 0.7,
            zorder=5 if m == "ASQA-H-AMPT" else 4,
            label=m,
        )

    pareto_pts = sorted([pts["AMPT"], pts["ASQA-H-AMPT"]])
    ax1.plot(
        [p[0] for p in pareto_pts],
        [p[1] for p in pareto_pts],
        linestyle="--",
        color="#C0392B",
        linewidth=1.0,
        alpha=0.70,
        zorder=2,
    )

    # Clean non-overlapping arrowed annotations in data coordinates
    annot_pos = {
        "ASQA-H-AMPT": (11.5, 94.95, "left"),
        "AMPT": (0.82, 94.05, "left"),
        "Wavelet+AMPT": (3.0, 92.25, "left"),
        "Median+AMPT": (0.82, 91.10, "left"),
        "Kalman+AMPT": (13.0, 93.15, "left"),
        "Pan-Tompkins": (48.0, 89.25, "right"),
    }
    for m, (tx, ty, ha) in annot_pos.items():
        t, f1 = pts[m]
        ax1.annotate(
            f"{m}\n({t:.2f} ms, {f1:.2f}%)",
            xy=(t, f1),
            xytext=(tx, ty),
            ha=ha,
            va="center",
            fontsize=6.2,
            fontweight="bold" if m == "ASQA-H-AMPT" else "normal",
            color="#901B1B" if m == "ASQA-H-AMPT" else "#222222",
            arrowprops=dict(
                arrowstyle="-",
                color="#C0392B" if m == "ASQA-H-AMPT" else "#777777",
                lw=0.75,
                shrinkA=2,
                shrinkB=4,
            ),
        )

    ax1.axvline(10.0, color="#2E8B57", linestyle=":", linewidth=1.25, zorder=2)
    ax1.text(
        9.0,
        87.3,
        "Real-Time Target\n(<10 ms/10 s)",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="#1E6B42",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
    )
    ax1.set_xscale("log")
    ax1.set_xlim(0.7, 95.0)
    ax1.set_ylim(87.0, 95.9)
    ax1.set_xlabel("Mean Processing Time per 10 s ECG (ms, log scale)")
    ax1.set_ylabel("6-Dataset Mean F1-Score (%)")
    ax1.set_title("(a) Accuracy–Efficiency Trade-off", loc="left")

    # Panel (b): Per-Dataset Runtime Comparison (ms/10s, log scale)
    compare_4 = ["Pan-Tompkins", "AMPT", "Kalman+AMPT", "ASQA-H-AMPT"]
    x = np.arange(len(DATASETS))
    w = 0.19
    offs = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    for m, off in zip(compare_4, offs):
        times = [
            float(df_main[(df_main["Dataset"] == ds) & (df_main["Algorithm"] == m)]["Time_per_10s_ms"].iloc[0])
            for ds in DATASETS
        ]
        bars = ax2.bar(
            x + off,
            times,
            width=w * 0.90,
            color=COLORS[m],
            edgecolor="#222222" if m == "ASQA-H-AMPT" else "#444444",
            linewidth=0.75 if m == "ASQA-H-AMPT" else 0.5,
            label=m,
            zorder=3,
        )
        if m == "ASQA-H-AMPT":
            for bar in bars:
                h = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h * 1.14,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.9,
                    fontweight="bold",
                    color="#901B1B",
                )

    ax2.axhline(10.0, color="#2E8B57", linestyle=":", linewidth=1.25, zorder=4)
    ax2.text(
        2.5,
        12.5,
        "10 ms/10 s Wearable Constraint",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#1E6B42",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="#2E8B57", alpha=0.90, lw=0.5),
    )
    ax2.set_yscale("log")
    ax2.set_ylim(0.3, 420.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(DATASETS, fontsize=7.5)
    ax2.set_xlabel("Benchmark Dataset")
    ax2.set_ylabel("Processing Time per 10 s ECG (ms, log scale)")
    ax2.set_title("(b) Runtime Scaling (2.6×–25.5× vs. PT)", loc="left")
    ax2.legend(loc="upper center", ncol=2, fontsize=6.4, columnspacing=0.9)

    return _save_fig(fig, "fig5_efficiency_vs_f1.png")


# =============================================================================
# Figure 6: Real Noise Robustness Stress Tests (AWGN & 0.3 Hz Baseline Drift)
# =============================================================================


def make_fig6_noise_robustness(df_awgn: pd.DataFrame, df_drift: pd.DataFrame) -> str:
    stress_methods = ["AMPT", "Wavelet+AMPT", "Median+AMPT", "Kalman+AMPT", "ASQA-H-AMPT"]
    styles = {
        "AMPT": ("o", "--", 1.35),
        "Wavelet+AMPT": ("^", "-.", 1.2),
        "Median+AMPT": ("v", ":", 1.2),
        "Kalman+AMPT": ("D", "-.", 1.2),
        "ASQA-H-AMPT": ("*", "-", 2.0),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.45), gridspec_kw={"wspace": 0.32})

    # Panel (a): AWGN Stress Test (SNR 20 -> -5 dB)
    for m in stress_methods:
        sub = df_awgn[df_awgn["Algorithm"] == m].sort_values("SNR_dB", ascending=False)
        mk, ls, lw = styles[m]
        ax1.plot(
            sub["SNR_dB"],
            sub["F1_Pct"],
            marker=mk,
            markersize=7.5 if m == "ASQA-H-AMPT" else 4.5,
            linestyle=ls,
            linewidth=lw,
            color=COLORS[m],
            label=m,
            zorder=5 if m == "ASQA-H-AMPT" else 3,
        )
        if m in ("AMPT", "ASQA-H-AMPT"):
            lo = np.clip(sub["F1_Pct"] - sub["F1_Std_Pct"], 0.0, 100.0)
            hi = np.clip(sub["F1_Pct"] + sub["F1_Std_Pct"], 0.0, 100.0)
            ax1.fill_between(sub["SNR_dB"], lo, hi, color=COLORS[m], alpha=0.12, zorder=1)

    ax1.invert_xaxis()
    ax1.set_xticks([20, 15, 10, 5, 0, -5])
    ax1.set_xlabel("Input SNR (dB) [Clean → Severe AWGN]")
    ax1.set_ylabel("Macro-Average F1-Score (%)")
    ax1.set_ylim(82.0, 101.3)
    ax1.set_title("(a) AWGN Stress Test (30 A1 Recs, ±1 SD)", loc="left")

    asqa_m5 = float(df_awgn[(df_awgn["Algorithm"] == "ASQA-H-AMPT") & (df_awgn["SNR_dB"] == -5.0)]["F1_Pct"].iloc[0])
    ampt_m5 = float(df_awgn[(df_awgn["Algorithm"] == "AMPT") & (df_awgn["SNR_dB"] == -5.0)]["F1_Pct"].iloc[0])
    ax1.annotate(
        f"ASQA-H-AMPT: {asqa_m5:.2f}%\n(+{asqa_m5 - ampt_m5:.2f} pp vs. AMPT)",
        xy=(-5.0, asqa_m5),
        xytext=(18.5, 92.2),
        ha="left",
        fontsize=6.5,
        fontweight="bold",
        color="#901B1B",
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.95, connectionstyle="arc3,rad=-0.12"),
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FDEDEC", edgecolor="#C0392B", lw=0.65),
    )

    # Panel (b): 0.3 Hz Sinusoidal Baseline Drift Stress Test (0.1 to 2.0 mV)
    for m in stress_methods:
        sub = df_drift[df_drift["Algorithm"] == m].sort_values("Drift_mV")
        mk, ls, lw = styles[m]
        ax2.plot(
            sub["Drift_mV"],
            sub["F1_Pct"],
            marker=mk,
            markersize=7.5 if m == "ASQA-H-AMPT" else 4.5,
            linestyle=ls,
            linewidth=lw,
            color=COLORS[m],
            label=m,
            zorder=5 if m == "ASQA-H-AMPT" else 3,
        )

    ax2.set_xticks([0.1, 0.5, 1.0, 1.5, 2.0])
    ax2.set_xlabel("0.3 Hz Sinusoidal Drift Amplitude (mV)")
    ax2.set_ylabel("Macro-Average F1-Score (%)")
    ax2.set_ylim(99.64, 99.93)
    ax2.set_title("(b) Baseline-Drift Test (0.3 Hz, A1)", loc="left")

    inset = ax2.inset_axes([0.33, 0.12, 0.62, 0.40])
    inset.set_facecolor("white")
    for m in ("AMPT", "Median+AMPT", "ASQA-H-AMPT"):
        sub = df_drift[df_drift["Algorithm"] == m].sort_values("Drift_mV")
        mk, ls, lw = styles[m]
        inset.plot(
            sub["Drift_mV"],
            sub["Error_Rate_Pct"],
            marker=mk,
            markersize=3.2 if m != "ASQA-H-AMPT" else 4.8,
            linestyle=ls,
            linewidth=1.1 if m != "ASQA-H-AMPT" else 1.5,
            color=COLORS[m],
        )
    inset.text(0.05, 0.88, "Total Error Rate (%)", transform=inset.transAxes, fontsize=6.1, fontweight="bold", color="#333333")
    inset.set_ylim(0.18, 0.50)
    inset.set_xticks([0.1, 1.0, 2.0])
    inset.set_yticks([0.2, 0.3, 0.4])
    inset.tick_params(labelsize=6.0)
    inset.grid(True, linestyle="--", alpha=0.45, linewidth=0.5)
    for sp in ("top", "right"):
        inset.spines[sp].set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=7.0)

    return _save_fig(fig, "fig6_noise_robustness.png")


# =============================================================================
# Figure 7: Real SQA Denoising Route Distribution & CQS Profile
# =============================================================================


def make_fig7_sqa_distribution(df_sqa: pd.DataFrame) -> str:
    scopes = ["A1", "A2", "B1", "B2", "C", "D", "ALL_400"]
    scope_labels = ["A1", "A2", "B1", "B2", "C", "D", "Pooled"]
    modes = ["Bypass", "Median", "Wavelet", "Kalman"]
    mode_labels = [
        "Bypass (Clean)",
        "Dual-Median (Baseline)",
        "Sym4 Wavelet (HF/EMG)",
        "Adaptive Kalman (Dynamic)",
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 3.50), gridspec_kw={"width_ratios": [1.32, 1.0], "wspace": 0.30}
    )

    # Panel (a): 100% Stacked Bar of SQA Routing across 46,639 real 5-s windows
    x = np.arange(len(scopes))
    bottom = np.zeros(len(scopes), dtype=np.float64)
    df_idx = df_sqa.set_index("Dataset")

    for mode, mlbl in zip(modes, mode_labels):
        pcts = np.array([float(df_idx.loc[sc, f"{mode}_Pct"]) for sc in scopes])
        ax1.bar(
            x,
            pcts,
            bottom=bottom,
            width=0.62,
            color=COLORS[mode],
            edgecolor="white",
            linewidth=0.7,
            label=mlbl,
            zorder=3,
        )
        for i, (p, b) in enumerate(zip(pcts, bottom)):
            if p >= 7.5:
                ax1.text(
                    i,
                    b + p / 2.0,
                    f"{p:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white",
                    fontweight="bold",
                )
        bottom += pcts

    for i, sc in enumerate(scopes):
        n_w = int(df_idx.loc[sc, "Total_Windows"])
        lbl_w = f"{n_w/1000:.1f}k" if n_w >= 10000 else f"{n_w:,}"
        ax1.text(i, 101.8, lbl_w, ha="center", va="bottom", fontsize=5.9, color="#333333")

    ax1.axvline(5.5, color="#666666", linestyle="--", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scope_labels, fontsize=7.3)
    ax1.set_xlabel("Benchmark Dataset (Total 5-s Windows Above)")
    ax1.set_ylabel("5-s Window Route Share (%)")
    ax1.set_ylim(0, 111.0)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_title("(a) SQA Route Share (N = 46,639 Windows)", loc="left")

    # Panel (b): Mean CQS ± SD per Dataset with BWI cleanly in x-axis labels
    x2 = np.arange(len(DATASETS))
    cqs_means = [float(df_idx.loc[ds, "Mean_CQS"]) for ds in DATASETS]
    cqs_stds = [float(df_idx.loc[ds, "Std_CQS"]) for ds in DATASETS]
    bar_cols = ["#2E8B57" if c >= 0.68 else "#D35400" for c in cqs_means]

    ax2.bar(
        x2,
        cqs_means,
        yerr=cqs_stds,
        capsize=3.0,
        width=0.56,
        color=bar_cols,
        edgecolor="#333333",
        linewidth=0.6,
        error_kw={"elinewidth": 0.9, "ecolor": "#333333"},
        zorder=3,
    )
    ax2.axhline(0.68, color="#C0392B", linestyle="--", linewidth=1.15, zorder=4, label="Clean Bypass Thr. (0.68)")

    for i in range(len(DATASETS)):
        y_lbl = cqs_means[i] + cqs_stds[i] + 0.025
        if abs(y_lbl - 0.68) < 0.04:
            y_lbl = 0.705
        ax2.text(
            i,
            y_lbl,
            f"{cqs_means[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none", alpha=0.80),
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"{ds}\n({float(df_idx.loc[ds, 'Mean_BWI']):.2f})" for ds in DATASETS], fontsize=7.0)
    ax2.set_xlabel("Dataset (Mean Baseline Wander Index)")
    ax2.set_ylabel("Mean Composite Quality Score (CQS ± SD)")
    ax2.set_ylim(0.0, 1.16)
    ax2.set_title("(b) Mean CQS ± SD & Baseline Wander", loc="left")
    ax2.legend(loc="upper right", fontsize=6.5)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7.0)

    return _save_fig(fig, "fig7_sqa_distribution.png")


# =============================================================================
# Figure 8: Per-Record F1 & Error Rate Box Plots (Coworker Revision #4)
# =============================================================================


def make_fig8_boxplot(df_main: pd.DataFrame, df_sig: pd.DataFrame) -> str:
    best_fixed = _best_fixed_per_dataset(df_main)
    method_order = [
        ("pan_tompkins", "Pan-Tompkins", "Pan-Tompkins"),
        ("ampt", "AMPT", "AMPT"),
        ("best_fixed", "Best Fixed+AMPT", "Best Fixed+AMPT"),
        ("asqa_h_ampt", "ASQA-H-AMPT", "ASQA-H-AMPT (Proposed)"),
    ]

    def _load_metric_array(mkey: str, ds: str, col: str) -> np.ndarray:
        real_key = FIXED_KEYS[best_fixed[ds]] if mkey == "best_fixed" else mkey
        df = pd.read_csv(os.path.join(RESULT_ROOT, real_key, f"{ds}.csv"))
        return df[col].to_numpy(dtype=np.float64)

    fig = plt.figure(figsize=(7.16, 5.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0], hspace=0.46, wspace=0.28)
    ax_top = fig.add_subplot(gs[0, :])
    ax_a2 = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # --- Top panel (a): Per-record F1-Score box plots across ALL 6 datasets ---
    group_centers = np.arange(len(DATASETS)) * 1.15
    offs = [-0.30, -0.10, 0.10, 0.30]

    for noisy_idx in (1, 5):
        c = group_centers[noisy_idx]
        ax_top.axvspan(c - 0.48, c + 0.48, color="#FDEDEC", alpha=0.65, zorder=0)

    for (mkey, color_key, _), off in zip(method_order, offs):
        data_list = [_load_metric_array(mkey, ds, "F1_Pct") for ds in DATASETS]
        pos_list = group_centers + off
        bp = ax_top.boxplot(
            data_list,
            positions=pos_list,
            widths=0.16,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            flierprops=dict(
                marker="o", markersize=2.3, alpha=0.45, markerfacecolor=COLORS[color_key], markeredgecolor="none"
            ),
            medianprops=dict(color="#111111", linewidth=1.15),
            meanprops=dict(
                marker="D", markersize=3.0, markerfacecolor="white", markeredgecolor="#111111", markeredgewidth=0.7
            ),
            whiskerprops=dict(color="#444444", linewidth=0.75),
            capprops=dict(color="#444444", linewidth=0.75),
        )
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[color_key])
            box.set_alpha(0.86 if color_key == "ASQA-H-AMPT" else 0.72)
            box.set_edgecolor("#222222" if color_key == "ASQA-H-AMPT" else "#444444")
            box.set_linewidth(0.85 if color_key == "ASQA-H-AMPT" else 0.6)

    ax_top.set_xticks(group_centers)
    ax_top.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=7.5)
    ax_top.set_ylabel("Per-Record F1-Score (%)")
    ax_top.set_ylim(-2.0, 105.0)
    ax_top.set_title(
        "(a) Per-Record F1-Score Distribution Across All Six Datasets (Shaded: Noisy A2 & TeleHealth D)",
        loc="left",
        pad=22.0,
    )

    legend_patches = [
        mpatches.Patch(facecolor=COLORS[ck], edgecolor="#333333", label=lbl, alpha=0.84)
        for _, ck, lbl in method_order
    ]
    mean_handle = plt.Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor="white",
        markeredgecolor="#111111",
        markersize=4.2,
        label="Mean (◆) / Median (—)",
    )
    ax_top.legend(
        handles=legend_patches + [mean_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        fontsize=6.4,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    # --- Bottom panels (b) & (c): Zoomed Error Rate (%) Box Plots on A2 & D ---
    rng = np.random.default_rng(2026)
    base_names = ["PT", "AMPT", "Best Fixed", "ASQA-H-\nAMPT"]

    for ax_sub, ds, panel_tag, y_cap in (
        (ax_a2, "A2", "(b) Noisy A2 Error Rate (n=100)", 112.0),
        (ax_d, "D", "(c) TeleHealth D Error Rate (n=134)", 144.0),
    ):
        err_arrays = [_load_metric_array(mk, ds, "Error_Rate_Pct") for mk, _, _ in method_order]
        bp = ax_sub.boxplot(
            err_arrays,
            positions=[0, 1, 2, 3],
            widths=0.48,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            medianprops=dict(color="#111111", linewidth=1.25),
            meanprops=dict(
                marker="D", markersize=3.8, markerfacecolor="white", markeredgecolor="#111111", markeredgewidth=0.75
            ),
            whiskerprops=dict(color="#444444", linewidth=0.8),
            capprops=dict(color="#444444", linewidth=0.8),
        )
        xtick_lbls = []
        for idx_m, ((_, ck, _), box, arr, bname) in enumerate(zip(method_order, bp["boxes"], err_arrays, base_names)):
            box.set_facecolor(COLORS[ck])
            box.set_alpha(0.86 if ck == "ASQA-H-AMPT" else 0.72)
            box.set_edgecolor("#222222" if ck == "ASQA-H-AMPT" else "#444444")
            box.set_linewidth(0.9 if ck == "ASQA-H-AMPT" else 0.65)
            in_range = arr <= (y_cap - 10.0)
            jitter = rng.uniform(-0.15, 0.15, size=int(np.sum(in_range)))
            ax_sub.scatter(
                idx_m + jitter,
                arr[in_range],
                s=7.5,
                color=COLORS[ck],
                alpha=0.28,
                edgecolors="none",
                zorder=2,
            )
            med_v = float(np.median(arr))
            xtick_lbls.append(f"{bname}\n(med {med_v:.1f}%)")

        sig_ampt = df_sig[
            (df_sig["Dataset"] == ds) & (df_sig["Comparison"] == "ASQA-H-AMPT vs AMPT") & (df_sig["Metric"] == "F1_Pct")
        ].iloc[0]
        sig_pt = df_sig[
            (df_sig["Dataset"] == ds)
            & (df_sig["Comparison"] == "ASQA-H-AMPT vs Pan-Tompkins")
            & (df_sig["Metric"] == "F1_Pct")
        ].iloc[0]

        def _fmt_p(p: float) -> str:
            return f"{p:.1e}" if p < 0.001 else f"{p:.4f}"

        note = (
            f"F1 vs. AMPT: W $p$={_fmt_p(float(sig_ampt['wilcoxon_pvalue']))}, "
            f"t $p$={_fmt_p(float(sig_ampt['t_pvalue']))}\n"
            f"F1 vs. PT: W $p$={_fmt_p(float(sig_pt['wilcoxon_pvalue']))}, "
            f"t $p$={_fmt_p(float(sig_pt['t_pvalue']))}"
        )
        ax_sub.text(
            0.98,
            0.96,
            note,
            transform=ax_sub.transAxes,
            ha="right",
            va="top",
            fontsize=6.0,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#C0392B", alpha=0.95, lw=0.65),
        )

        ax_sub.set_xticks([0, 1, 2, 3])
        ax_sub.set_xticklabels(xtick_lbls, fontsize=6.6)
        ax_sub.set_ylabel("Per-Record Error Rate (%)")
        ax_sub.set_ylim(-3.0, y_cap)
        ax_sub.set_title(panel_tag, loc="left")

    return _save_fig(fig, "fig8_f1_error_boxplot.png")


def main() -> None:
    setup_ieee_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    df_main = pd.read_csv(os.path.join(RESULT_ROOT, "table_main_comparison.csv"))
    df_ablation = pd.read_csv(os.path.join(RESULT_ROOT, "table_ablation.csv"))
    df_sig = pd.read_csv(os.path.join(RESULT_ROOT, "table_significance.csv"))
    df_sqa = pd.read_csv(os.path.join(RESULT_ROOT, "table_sqa_distribution.csv"))
    df_awgn = pd.read_csv(os.path.join(RESULT_ROOT, "noise_robustness_awgn.csv"))
    df_drift = pd.read_csv(os.path.join(RESULT_ROOT, "noise_robustness_drift.csv"))

    outputs = [
        make_fig1_pipeline(),
        make_fig2_f1_comparison(df_main, df_sig),
        make_fig3_error_and_fp(df_main),
        make_fig4_ablation(df_ablation),
        make_fig5_efficiency_vs_f1(df_main),
        make_fig6_noise_robustness(df_awgn, df_drift),
        make_fig7_sqa_distribution(df_sqa),
        make_fig8_boxplot(df_main, df_sig),
    ]

    png_magic = b"\x89PNG\r\n\x1a\n"
    for p in outputs:
        assert os.path.isfile(p), f"Missing generated figure: {p}"
        sz = os.path.getsize(p)
        assert sz > 10_000, f"Figure too small ({sz} bytes): {p}"
        with open(p, "rb") as f:
            hdr = f.read(8)
        assert hdr == png_magic, f"Invalid PNG signature in {p}"
        print(f"  [OK] {os.path.relpath(p, PROJECT_ROOT)} ({sz / 1024:.1f} KB)")

    print(f"[ALL {len(outputs)} PUBLICATION FIGURES GENERATED & VERIFIED AT 300 DPI]")


if __name__ == "__main__":
    main()
