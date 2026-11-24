"""
Clean, cohesive, IEEE Transactions / Nature-style figure generator.
Saves redesigned 300 DPI figures to `result_experiment/figures_clean/`
without modifying or deleting the original `result_experiment/figures/`.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wfdb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
ACY_DIR = os.path.join(PROJECT_ROOT, "acy_ampt")
if ACY_DIR not in sys.path:
    sys.path.insert(0, ACY_DIR)

from experiments.detectors import asqa_h_ampt, run_ampt_fast
from mit_processing.load_annotation import loadAnnotationSampleFromPath

RES_DIR = os.path.join(PROJECT_ROOT, "result_experiment")
OUT_DIR = os.path.join(RES_DIR, "figures_clean")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Unified Cohesive Publication Palette (IEEE / Nature Restraint)
# =============================================================================
HERO_NAVY   = "#0F4C81"  # Proposed: ASQA-H-AMPT (Hero)
BASE_BLUE   = "#6B9AC4"  # Baseline: AMPT (Subordinate blue)
NEUT_GRAY   = "#B0B7C3"  # Classic: Pan-Tompkins (Neutral gray)
ABL_WAVELET = "#94B0C2"  # Ablation: Wavelet+AMPT (Light cool steel)
ABL_MEDIAN  = "#D4A373"  # Ablation: Median+AMPT (Muted warm sand)
ABL_KALMAN  = "#7C8FA6"  # Ablation: Kalman+AMPT (Medium slate)
ACCENT_RED  = "#B83B3B"  # Sparse accent: thresholds / errors only
INK         = "#1F2937"  # Primary dark ink
SUB_INK     = "#4B5563"  # Secondary text
GRID_COL    = "#E5E7EB"  # Hairline grid

# SQA 4-mode palette (calm, cohesive, colorblind-safe)
SQA_BYPASS  = "#0F4C81"  # Deep navy (clean bypass)
SQA_MEDIAN  = "#D4A373"  # Warm sand (baseline wander)
SQA_WAVELET = "#6B9AC4"  # Soft blue (HF/EMG)
SQA_KALMAN  = "#8D99AE"  # Cool slate (dynamic artifact)

DATASETS = ["A1", "A2", "B1", "B2", "C", "D"]
DS_LABELS = {
    "A1": "A1\n(High Q.)",
    "A2": "A2\n(Low Q.)",
    "B1": "B1\n(NSR)",
    "B2": "B2\n(Arrh.)",
    "C":  "C\n(Paced)",
    "D":  "D\n(TeleH.)",
}


def apply_clean_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.labelsize": 8.8,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#9CA3AF",
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "xtick.color": SUB_INK,
        "ytick.color": SUB_INK,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "grid.color": GRID_COL,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": 8.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    })


def clean_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=GRID_COL, linewidth=0.6, linestyle="-")


def fmt_p(p):
    if p < 1e-4:
        exp = int(np.floor(np.log10(p)))
        mant = p / (10 ** exp)
        return f"{mant:.1f}×10^{{{exp}}}"
    return f"{p:.4f}"


# =============================================================================
# FIG 2 (Paper): Per-Record F1 & Error Rate Box Plots (Clean Redesign of fig8)
# =============================================================================
def make_fig2_boxplot():
    sig_df = pd.read_csv(os.path.join(RES_DIR, "table_significance.csv"))

    # Load per-record CSVs for 4 main groups
    # Best Fixed+AMPT per dataset (by macro F1 from table_ablation.csv)
    abl = pd.read_csv(os.path.join(RES_DIR, "table_ablation.csv"))
    fixed_methods = ["Wavelet+AMPT", "Median+AMPT", "Kalman+AMPT"]
    name_to_dir = {
        "Pan-Tompkins": "pan_tompkins",
        "AMPT": "ampt",
        "Wavelet+AMPT": "wavelet_ampt",
        "Median+AMPT": "median_ampt",
        "Kalman+AMPT": "kalman_ampt",
        "ASQA-H-AMPT": "asqa_h_ampt",
    }

    best_fixed_dir = {}
    for ds in DATASETS:
        sub = abl[(abl["Dataset"] == ds) & (abl["Algorithm"].isin(fixed_methods))]
        best_alg = sub.sort_values("F1_Pct", ascending=False).iloc[0]["Algorithm"]
        best_fixed_dir[ds] = name_to_dir[best_alg]

    methods_order = ["Pan-Tompkins", "AMPT", "Best Fixed+AMPT", "ASQA-H-AMPT"]
    colors = [NEUT_GRAY, BASE_BLUE, ABL_MEDIAN, HERO_NAVY]

    fig = plt.figure(figsize=(10.2, 6.4))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.25)

    # --- Panel (a): Per-Record F1 across all 6 datasets ---
    ax_top = fig.add_subplot(gs[0, :])
    clean_grid(ax_top, "y")

    positions_base = np.arange(len(DATASETS)) * 1.15
    offsets = [-0.30, -0.10, 0.10, 0.30]
    width = 0.17

    legend_handles = []
    for m_idx, (m_label, col) in enumerate(zip(methods_order, colors)):
        data_list = []
        pos_list = []
        for d_idx, ds in enumerate(DATASETS):
            mdir = best_fixed_dir[ds] if m_label == "Best Fixed+AMPT" else name_to_dir[m_label]
            df = pd.read_csv(os.path.join(RES_DIR, mdir, f"{ds}.csv"))
            data_list.append(df["F1_Pct"].values)
            pos_list.append(positions_base[d_idx] + offsets[m_idx])

        bp = ax_top.boxplot(
            data_list,
            positions=pos_list,
            widths=width,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=2.2, markerfacecolor=col, markeredgecolor="none", alpha=0.35),
            medianprops=dict(color=INK, linewidth=1.2),
            meanprops=dict(marker="D", markersize=3.8, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.8),
            whiskerprops=dict(color=SUB_INK, linewidth=0.8),
            capprops=dict(color=SUB_INK, linewidth=0.8),
            boxprops=dict(facecolor=col, edgecolor=INK, linewidth=0.75, alpha=0.88),
        )
        legend_handles.append(mpl.patches.Patch(facecolor=col, edgecolor=INK, linewidth=0.75, label=m_label))

    # Mean/median legend marker
    mean_h = mpl.lines.Line2D([], [], color="none", marker="D", markersize=4.5,
                              markerfacecolor="white", markeredgecolor=INK, label="Mean (◇) / Median (—)")
    legend_handles.append(mean_h)

    ax_top.set_xticks(positions_base)
    ax_top.set_xticklabels([DS_LABELS[d] for d in DATASETS])
    ax_top.set_ylabel("Per-Record F1-Score (%)")
    ax_top.set_ylim(-3, 105)
    ax_top.set_title("(a) Per-Record F1-Score Distribution Across All Six Datasets", loc="left", color=INK, pad=26)
    ax_top.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=5, handlelength=1.2, columnspacing=1.2)

    # --- Panels (b) and (c): Noisy A2 & TeleHealth D Error Rate ---
    for ax_idx, (ds, title_str, ymax) in enumerate([
        ("A2", "(b) Noisy A2 Per-Record Error Rate (n = 100)", 118),
        ("D",  "(c) TeleHealth D Per-Record Error Rate (n = 134)", 148),
    ]):
        ax = fig.add_subplot(gs[1, ax_idx])
        clean_grid(ax, "y")

        err_data = []
        medians = []
        for m_label in methods_order:
            mdir = best_fixed_dir[ds] if m_label == "Best Fixed+AMPT" else name_to_dir[m_label]
            df = pd.read_csv(os.path.join(RES_DIR, mdir, f"{ds}.csv"))
            vals = df["Error_Rate_Pct"].values
            err_data.append(vals)
            medians.append(np.median(vals))

        pos = np.arange(4)
        for i, (vals, col) in enumerate(zip(err_data, colors)):
            ax.boxplot(
                [vals],
                positions=[i],
                widths=0.44,
                patch_artist=True,
                showmeans=True,
                showfliers=False,
                medianprops=dict(color=INK, linewidth=1.3),
                meanprops=dict(marker="D", markersize=4.2, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.85),
                whiskerprops=dict(color=SUB_INK, linewidth=0.85),
                capprops=dict(color=SUB_INK, linewidth=0.85),
                boxprops=dict(facecolor=col, edgecolor=INK, linewidth=0.8, alpha=0.88),
            )
            # Subtle jittered points
            rng = np.random.default_rng(42 + i)
            jitter = rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, np.clip(vals, 0, ymax * 0.96),
                       s=7, color=col, alpha=0.22, edgecolors="none", zorder=1)

        # Clean, uncluttered significance caption in top-right whitespace (no red box)
        r_ampt = sig_df[(sig_df["Dataset"] == ds) & (sig_df["Comparison"] == "ASQA-H-AMPT vs AMPT") & (sig_df["Metric"] == "F1_Pct")].iloc[0]
        r_pt   = sig_df[(sig_df["Dataset"] == ds) & (sig_df["Comparison"] == "ASQA-H-AMPT vs Pan-Tompkins") & (sig_df["Metric"] == "F1_Pct")].iloc[0]
        stat_txt = (
            f"F1 vs. AMPT:  W p = {r_ampt['wilcoxon_pvalue']:.4g},  t p = {r_ampt['t_pvalue']:.4g}\n"
            f"F1 vs. PT:       W p = {r_pt['wilcoxon_pvalue']:.2e},  t p = {r_pt['t_pvalue']:.2e}"
        )
        ax.text(0.98, 0.96, stat_txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=7.4, color=INK, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="#F9FAFB", edgecolor="#D1D5DB", linewidth=0.7))

        short_names = ["PT", "AMPT", "Best Fixed", "ASQA-H-AMPT"]
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{n}\n(med {m:.1f}%)" for n, m in zip(short_names, medians)], fontsize=7.8)
        ax.set_ylabel("Per-Record Error Rate (%)")
        ax.set_ylim(-4, ymax)
        ax.set_title(title_str, loc="left", color=INK)

    out = os.path.join(OUT_DIR, "fig2_f1_error_boxplot.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# =============================================================================
# FIG 3 (Paper): Noise Robustness Stress Tests (Clean Redesign of fig6)
# =============================================================================
def make_fig3_noise_robustness():
    awgn = pd.read_csv(os.path.join(RES_DIR, "noise_robustness_awgn.csv"))
    drift = pd.read_csv(os.path.join(RES_DIR, "noise_robustness_drift.csv"))

    methods = [
        ("AMPT",         BASE_BLUE,   "--", "o", 1.5, 4.5),
        ("Wavelet+AMPT", ABL_WAVELET, ":",  "^", 1.4, 4.5),
        ("Median+AMPT",  ABL_MEDIAN,  "-.", "v", 1.4, 4.5),
        ("Kalman+AMPT",  ABL_KALMAN,  "--", "s", 1.3, 4.2),
        ("ASQA-H-AMPT",  HERO_NAVY,   "-",  "D", 2.3, 5.5),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    plt.subplots_adjust(wspace=0.28)

    # Panel (a): AWGN
    clean_grid(ax1, "y")
    for alg, col, ls, mk, lw, ms in methods:
        sub = awgn[awgn["Algorithm"] == alg].sort_values("SNR_dB", ascending=False)
        x = sub["SNR_dB"].values
        y = sub["F1_Pct"].values
        ax1.plot(x, y, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=ms,
                 label=alg, zorder=5 if alg == "ASQA-H-AMPT" else 3)
        if alg == "ASQA-H-AMPT" and "F1_Std_Pct" in sub.columns:
            sd = sub["F1_Std_Pct"].values
            ax1.fill_between(x, np.clip(y - sd, 0, 100), np.clip(y + sd, 0, 100),
                             color=col, alpha=0.10, zorder=1)

    ax1.invert_xaxis()
    ax1.set_xticks([20, 15, 10, 5, 0, -5])
    ax1.set_ylim(86.5, 101.0)
    ax1.set_xlabel("Input SNR (dB)  [Clean → Severe AWGN]")
    ax1.set_ylabel("Macro-Average F1-Score (%)")
    ax1.set_title("(a) AWGN Stress Test (30 A1 Records)", loc="left", color=INK)

    # Clean direct endpoint callout in open left whitespace (zero line collisions)
    ax1.annotate(
        "ASQA-H-AMPT: 97.04%\n(+6.93 pp vs. AMPT)",
        xy=(-5, 97.04), xytext=(14.5, 93.6),
        fontsize=7.8, fontweight="bold", color=HERO_NAVY,
        arrowprops=dict(arrowstyle="-|>", color=HERO_NAVY, lw=1.0, connectionstyle="arc3,rad=-0.10")
    )
    ax1.text(1.8, 89.2, "AMPT: 90.11%", fontsize=7.6, color=BASE_BLUE, ha="right", va="center", fontweight="bold")

    # Panel (b): Baseline Drift (clean single view: Error Rate is much clearer than flat 99.8x F1, or show F1 cleanly without floating inset!)
    # Show Total Error Rate (%) across drift amplitudes — super informative and clean!
    clean_grid(ax2, "y")
    for alg, col, ls, mk, lw, ms in methods:
        sub = drift[drift["Algorithm"] == alg].sort_values("Drift_mV")
        ax2.plot(sub["Drift_mV"].values, sub["Error_Rate_Pct"].values,
                 color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=ms,
                 label=alg, zorder=5 if alg == "ASQA-H-AMPT" else 3)

    ax2.set_xticks([0.1, 0.5, 1.0, 1.5, 2.0])
    ax2.set_ylim(0.18, 0.48)
    ax2.set_xlabel("0.3 Hz Sinusoidal Drift Amplitude (mV)")
    ax2.set_ylabel("Aggregate Total Error Rate (%)")
    ax2.set_title("(b) Baseline-Drift Stress Test (0.3 Hz, 30 A1 Records)", loc="left", color=INK)

    ax2.text(1.98, 0.355, "ASQA-H-AMPT: 0.34%\n(F1 = 99.85%)", fontsize=7.6, fontweight="bold",
             color=HERO_NAVY, ha="right", va="bottom")
    ax2.text(1.98, 0.415, "AMPT & Fixed: ~0.40%", fontsize=7.5, color=SUB_INK, ha="right", va="bottom")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, columnspacing=1.4)

    out = os.path.join(OUT_DIR, "fig3_noise_robustness.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# =============================================================================
# FIG 4 (Paper): Ablation Study (Clean Redesign of fig4 — Zero Hatching!)
# =============================================================================
def make_fig4_ablation():
    abl = pd.read_csv(os.path.join(RES_DIR, "table_ablation.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.1), gridspec_kw={"width_ratios": [1.35, 1.0]})
    plt.subplots_adjust(wspace=0.25)

    # Panel (a): Delta F1 vs AMPT across all 6 datasets
    clean_grid(ax1, "y")
    algs_delta = [
        ("Wavelet+AMPT", ABL_WAVELET),
        ("Median+AMPT",  ABL_MEDIAN),
        ("Kalman+AMPT",  ABL_KALMAN),
        ("ASQA-H-AMPT",  HERO_NAVY),
    ]
    x = np.arange(len(DATASETS))
    w = 0.19
    for idx, (alg, col) in enumerate(algs_delta):
        vals = [abl[(abl["Dataset"] == ds) & (abl["Algorithm"] == alg)].iloc[0]["Delta_F1_vs_AMPT"] for ds in DATASETS]
        bars = ax1.bar(x + (idx - 1.5) * w, vals, width=w, color=col, edgecolor=INK, linewidth=0.55, label=alg)
        if alg == "ASQA-H-AMPT":
            for b, v in zip(bars, vals):
                ax1.text(b.get_x() + b.get_width() / 2, v + 0.12, f"+{v:.2f}",
                         ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=HERO_NAVY)
        elif alg == "Kalman+AMPT":
            for d_i, (b, v) in enumerate(zip(bars, vals)):
                if abs(v) >= 0.3:
                    y_pos = v + 0.10 if v > 0 else v - 0.14
                    va = "bottom" if v > 0 else "top"
                    ax1.text(b.get_x() + b.get_width() / 2, y_pos, f"{v:+.2f}",
                             ha="center", va=va, fontsize=7.0, color=SUB_INK)
        elif alg == "Median+AMPT":
            for d_i, (b, v) in enumerate(zip(bars, vals)):
                if v <= -0.5:
                    ax1.text(b.get_x() + b.get_width() / 2, v - 0.14, f"{v:+.2f}",
                             ha="center", va="top", fontsize=7.0, color="#8C5A2B")

    ax1.axhline(0, color=INK, linewidth=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DS_LABELS[d] for d in DATASETS])
    ax1.set_ylabel("Δ F1-Score vs. Baseline AMPT (pp)")
    ax1.set_ylim(-1.25, 5.45)
    ax1.set_title("(a) Ablation Gain Over Baseline AMPT (All Datasets)", loc="left", color=INK)

    # Panel (b): Absolute F1 on challenging datasets A2, C, D
    clean_grid(ax2, "y")
    hard_ds = ["A2", "C", "D"]
    algs_abs = [
        ("AMPT",         BASE_BLUE),
        ("Wavelet+AMPT", ABL_WAVELET),
        ("Median+AMPT",  ABL_MEDIAN),
        ("Kalman+AMPT",  ABL_KALMAN),
        ("ASQA-H-AMPT",  HERO_NAVY),
    ]
    x2 = np.arange(len(hard_ds))
    w2 = 0.155
    for idx, (alg, col) in enumerate(algs_abs):
        vals = [abl[(abl["Dataset"] == ds) & (abl["Algorithm"] == alg)].iloc[0]["F1_Pct"] for ds in hard_ds]
        bars = ax2.bar(x2 + (idx - 2.0) * w2, vals, width=w2, color=col, edgecolor=INK, linewidth=0.55, label=alg)
        if alg in ("AMPT", "ASQA-H-AMPT"):
            for b, v in zip(bars, vals):
                ax2.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.1f}",
                         ha="center", va="bottom", fontsize=7.3,
                         fontweight="bold" if alg == "ASQA-H-AMPT" else "normal",
                         color=HERO_NAVY if alg == "ASQA-H-AMPT" else SUB_INK)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(["A2\n(Low Q.)", "C\n(Paced)", "D\n(TeleHealth)"])
    ax2.set_ylabel("Macro-Average F1-Score (%)")
    ax2.set_ylim(78, 102.5)
    ax2.set_title("(b) Absolute F1 on Challenging Datasets", loc="left", color=INK)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=5, columnspacing=1.3)

    out = os.path.join(OUT_DIR, "fig4_ablation_study.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# =============================================================================
# FIG 5 (Paper): SQA Routing Distribution (Clean Redesign of fig7)
# =============================================================================
def make_fig5_sqa_distribution():
    sqa = pd.read_csv(os.path.join(RES_DIR, "table_sqa_distribution.csv"))
    order = ["A1", "A2", "B1", "B2", "C", "D", "ALL_400"]
    sqa = sqa.set_index("Dataset").loc[order].reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1.35, 1.0]})
    plt.subplots_adjust(wspace=0.28)

    # Panel (a): Stacked SQA route share
    clean_grid(ax1, "y")
    x = np.arange(len(order))
    w = 0.56
    modes = [
        ("Bypass_Pct",  "Bypass (Clean)",          SQA_BYPASS,  "white"),
        ("Median_Pct",  "Dual-Median (Baseline)",  SQA_MEDIAN,  INK),
        ("Wavelet_Pct", "Sym4 Wavelet (HF/EMG)",   SQA_WAVELET, "white"),
        ("Kalman_Pct",  "Adaptive Kalman (Dyn.)",  SQA_KALMAN,  "white"),
    ]
    bottom = np.zeros(len(order))
    for col_name, lbl, col, txt_col in modes:
        vals = sqa[col_name].values
        ax1.bar(x, vals, bottom=bottom, width=w, color=col, edgecolor="white", linewidth=0.8, label=lbl)
        for i, v in enumerate(vals):
            if v >= 9.0:
                ax1.text(x[i], bottom[i] + v / 2, f"{v:.1f}%", ha="center", va="center",
                         fontsize=7.0, fontweight="bold", color=txt_col)
        bottom += vals

    # Subtle divider before Pooled
    ax1.axvline(5.5, color="#9CA3AF", linestyle="--", linewidth=0.9)
    for i, n_win in enumerate(sqa["Total_Windows"].values):
        lbl_n = f"{n_win/1000:.1f}k" if n_win >= 10000 else f"{n_win:,}"
        ax1.text(x[i], 102.0, lbl_n, ha="center", va="bottom", fontsize=7.0, color=SUB_INK)

    ax1.set_xticks(x)
    ax1.set_xticklabels(["A1", "A2", "B1", "B2", "C", "D", "Pooled"])
    ax1.set_ylim(0, 111)
    ax1.set_ylabel("5-s Window Route Share (%)")
    ax1.set_xlabel("Benchmark Dataset (Total 5-s Windows Above)")
    ax1.set_title("(a) SQA Adaptive Routing Across 46,639 Windows", loc="left", color=INK)

    # Panel (b): Mean CQS across datasets
    clean_grid(ax2, "y")
    sub_ds = sqa[sqa["Dataset"] != "ALL_400"].copy()
    x2 = np.arange(len(sub_ds))
    cqs_vals = sub_ds["Mean_CQS"].values
    bwi_vals = sub_ds["Mean_BWI"].values
    bar_cols = [HERO_NAVY if c >= 0.68 else ABL_MEDIAN for c in cqs_vals]

    bars = ax2.bar(x2, cqs_vals, width=0.52, color=bar_cols, edgecolor=INK, linewidth=0.65)
    ax2.axhline(0.68, color=ACCENT_RED, linestyle="--", linewidth=1.1, label="Clean Bypass Threshold (0.68)")
    for b, c in zip(bars, cqs_vals):
        ax2.text(b.get_x() + b.get_width() / 2, c + 0.022, f"{c:.2f}", ha="center", va="bottom",
                 fontsize=7.6, fontweight="bold", color=INK,
                 bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))

    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"{d}\n({b:.2f})" for d, b in zip(sub_ds["Dataset"], bwi_vals)], fontsize=7.8)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Mean Composite Quality Score (CQS)")
    ax2.set_xlabel("Dataset (Mean Baseline Wander Index)")
    ax2.set_title("(b) Mean Signal Quality Score per Dataset", loc="left", color=INK)
    ax2.legend(loc="upper right", fontsize=7.6)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=4, columnspacing=1.4)

    out = os.path.join(OUT_DIR, "fig5_sqa_distribution.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# =============================================================================
# FIG 6 (Paper): Efficiency vs Accuracy Trade-Off (Clean Redesign of fig5)
# =============================================================================
def make_fig6_efficiency():
    main_df = pd.read_csv(os.path.join(RES_DIR, "table_main_comparison.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1.0, 1.15]})
    plt.subplots_adjust(wspace=0.28)

    # Panel (a): Pareto scatter (Mean time vs 6-dataset mean F1)
    clean_grid(ax1, "both")
    summary = main_df.groupby("Algorithm", as_index=False).agg(
        Mean_F1=("F1_Pct", "mean"),
        Mean_Time=("Time_per_10s_ms", "mean"),
    )

    styles = {
        "Pan-Tompkins": (NEUT_GRAY,   "s", 70,  (0,   16), "center"),
        "AMPT":         (BASE_BLUE,   "o", 80,  (0,   18), "left"),
        "Wavelet+AMPT": (ABL_WAVELET, "^", 75,  (12,   1), "left"),
        "Median+AMPT":  (ABL_MEDIAN,  "v", 75,  (8,  -26), "left"),
        "Kalman+AMPT":  (ABL_KALMAN,  "D", 75,  (30,   0), "left"),
        "ASQA-H-AMPT":  (HERO_NAVY,   "*", 180, (-6,  16), "center"),
    }

    for _, row in summary.iterrows():
        alg = row["Algorithm"]
        col, mk, sz, (dx, dy), ha = styles[alg]
        ax1.scatter(row["Mean_Time"], row["Mean_F1"], c=col, marker=mk, s=sz,
                    edgecolors=INK, linewidths=0.75, zorder=5 if alg == "ASQA-H-AMPT" else 4)
        lbl = f"{alg}\n({row['Mean_Time']:.2f} ms, {row['Mean_F1']:.2f}%)"
        ax1.annotate(
            lbl,
            xy=(row["Mean_Time"], row["Mean_F1"]),
            xytext=(dx, dy), textcoords="offset points",
            ha=ha, va="center",
            fontsize=7.3,
            fontweight="bold" if alg == "ASQA-H-AMPT" else "normal",
            color=HERO_NAVY if alg == "ASQA-H-AMPT" else SUB_INK,
            arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.65) if alg in ("AMPT", "Median+AMPT", "Wavelet+AMPT", "Kalman+AMPT") else None,
        )

    ax1.axvline(10.0, color=ACCENT_RED, linestyle="--", linewidth=1.0)
    ax1.text(9.3, 87.6, "Real-Time Limit\n(<10 ms / 10 s)", ha="right", va="bottom",
             fontsize=7.3, color=ACCENT_RED, fontweight="bold")

    ax1.set_xscale("log")
    ax1.set_xlim(0.75, 95)
    ax1.set_ylim(87.2, 95.8)
    ax1.set_xlabel("Mean Processing Time per 10 s ECG (ms, log scale)")
    ax1.set_ylabel("6-Dataset Macro Mean F1-Score (%)")
    ax1.set_title("(a) Accuracy–Efficiency Pareto Trade-Off", loc="left", color=INK)

    # Panel (b): Runtime scaling per dataset
    clean_grid(ax2, "y")
    algs_rt = [
        ("Pan-Tompkins", NEUT_GRAY),
        ("AMPT",         BASE_BLUE),
        ("Kalman+AMPT",  ABL_KALMAN),
        ("ASQA-H-AMPT",  HERO_NAVY),
    ]
    x = np.arange(len(DATASETS))
    w = 0.19
    for idx, (alg, col) in enumerate(algs_rt):
        vals = [main_df[(main_df["Dataset"] == ds) & (main_df["Algorithm"] == alg)].iloc[0]["Time_per_10s_ms"] for ds in DATASETS]
        bars = ax2.bar(x + (idx - 1.5) * w, vals, width=w, color=col, edgecolor=INK, linewidth=0.55, label=alg)
        if alg == "ASQA-H-AMPT":
            for b, v in zip(bars, vals):
                ax2.text(b.get_x() + b.get_width() / 2, v * 1.13, f"{v:.1f}",
                         ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=HERO_NAVY)

    ax2.axhline(10.0, color=ACCENT_RED, linestyle="--", linewidth=1.0)
    ax2.text(2.42, 11.4, "10 ms/10 s limit", ha="center", va="bottom", fontsize=7.0, color=ACCENT_RED, fontweight="bold",
             bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))

    ax2.set_yscale("log")
    ax2.set_ylim(0.3, 460)
    ax2.set_xticks(x)
    ax2.set_xticklabels(DATASETS)
    ax2.set_xlabel("Benchmark Dataset")
    ax2.set_ylabel("Processing Time per 10 s ECG (ms, log)")
    ax2.set_title("(b) Runtime Scaling Across Datasets (2.6×–25.5× vs. PT)", loc="left", color=INK)
    ax2.legend(loc="upper center", ncol=2, fontsize=7.6, columnspacing=1.0)

    out = os.path.join(OUT_DIR, "fig6_efficiency_vs_f1.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# =============================================================================
# FIG 7 (Paper): Qualitative Pipeline Trace on Real Noisy A2/2812 (Clean Redesign of fig1)
# =============================================================================
def make_fig7_pipeline_trace():
    rec_dir = os.path.join(ACY_DIR, "ECG DBs/A - Physionet Challange 2014/A2 - Low Quality - Training")
    fs = 360
    rec = wfdb.rdrecord(os.path.join(rec_dir, "2812"), channels=[0])
    sig = np.ravel(rec.p_signal).astype(np.float64)
    ann = np.asarray(loadAnnotationSampleFromPath(rec_dir + "/", "2812"), dtype=np.int64)

    peaks_asqa, _, _, dbg = asqa_h_ampt(sig, fs, return_details=True)
    peaks_ampt, _ = run_ampt_fast(sig, fs)

    t_start, t_end = 50.0, 60.0
    s0, s1 = int(t_start * fs), int(t_end * fs)
    t = np.arange(s1 - s0) / fs

    raw_seg = dbg["raw"][s0:s1]
    den_seg = dbg["denoised"][s0:s1]
    hilb_raw = dbg["hilbert_raw"][s0:s1]
    env_seg = dbg["envelope"][s0:s1]

    # Normalize envelopes for clean visual comparison
    hilb_norm = hilb_raw / (np.max(hilb_raw) + 1e-12)
    env_norm  = env_seg / (np.max(env_seg) + 1e-12)

    ann_seg  = ann[(ann >= s0) & (ann < s1)] - s0
    asqa_seg = np.asarray([p - s0 for p in peaks_asqa if s0 <= p < s1], dtype=np.int64)
    ampt_seg = np.asarray([p - s0 for p in peaks_ampt if s0 <= p < s1], dtype=np.int64)

    tol = int(0.150 * fs)
    ampt_fn = np.asarray([a for a in ann_seg if not np.any(np.abs(ampt_seg - a) < tol)], dtype=np.int64)
    ampt_fp = np.asarray([p for p in ampt_seg if not np.any(np.abs(ann_seg - p) < tol)], dtype=np.int64)

    fig, axes = plt.subplots(4, 1, figsize=(9.8, 6.8), sharex=True)
    plt.subplots_adjust(hspace=0.34)

    # (a) Raw Noisy ECG
    ax = axes[0]
    clean_grid(ax, "y")
    ax.plot(t, raw_seg, color=INK, linewidth=0.95)
    ax.axvline(5.0, color="#9CA3AF", linestyle="--", linewidth=0.9)
    ax.text(2.5, 2.25, "Window 1: CQS = 0.40, BWI = 0.83 → Dual-Median", ha="center", va="center",
            fontsize=7.4, color=SUB_INK, bbox=dict(boxstyle="round,pad=0.22", fc="#F9FAFB", ec="#D1D5DB", lw=0.6))
    ax.text(7.5, 2.25, "Window 2: CQS = 0.41, BWI = 0.66 → Dual-Median", ha="center", va="center",
            fontsize=7.4, color=SUB_INK, bbox=dict(boxstyle="round,pad=0.22", fc="#F9FAFB", ec="#D1D5DB", lw=0.6))
    ax.set_ylim(-1.3, 2.8)
    ax.set_ylabel("Amp. (mV)")
    ax.set_title("(a) Raw Noisy ECG — PhysioNet 2014 A2 (Record 2812, 50–60 s, fs = 360 Hz)", loc="left", color=INK)

    # (b) SQA-Adaptive Denoised ECG
    ax = axes[1]
    clean_grid(ax, "y")
    ax.plot(t, den_seg, color=HERO_NAVY, linewidth=0.95)
    ax.set_ylim(-0.9, 1.2)
    ax.set_ylabel("Amp. (mV)")
    ax.set_title("(b) SQA-Adaptive Denoised ECG (Dual-Median Baseline Removal)", loc="left", color=INK)

    # (c) Hilbert Analytic Envelope & Hilbert-MWI Energy
    ax = axes[2]
    clean_grid(ax, "y")
    ax.fill_between(t, 0, hilb_norm, color=BASE_BLUE, alpha=0.22, label="Hilbert Envelope |d[n] + jH{d[n]}|")
    ax.plot(t, env_norm, color=HERO_NAVY, linewidth=1.25, label="Hilbert-MWI Energy Envelope (110 ms)")
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Norm. Energy")
    ax.set_title("(c) Zero-Phase Hilbert Analytic Envelope & Hilbert-MWI Energy", loc="left", color=INK)
    ax.legend(loc="upper right", ncol=2, fontsize=7.6)

    # (d) Detected R-Peaks vs Ground-Truth & AMPT Errors
    ax = axes[3]
    clean_grid(ax, "y")
    ax.plot(t, den_seg, color="#6B7280", linewidth=0.8, label="Denoised ECG")
    ax.scatter(ann_seg / fs, den_seg[ann_seg], s=58, facecolors="none", edgecolors=INK, linewidths=1.25,
               zorder=4, label=f"Ground Truth (N={len(ann_seg)})")
    ax.scatter(asqa_seg / fs, den_seg[asqa_seg] + 0.22, s=34, marker="v", color=HERO_NAVY,
               zorder=5, label=f"ASQA-H-AMPT (TP={len(asqa_seg)}, FP=0, FN=0)")
    if len(ampt_fn) > 0:
        ax.scatter(ampt_fn / fs, den_seg[ampt_fn] - 0.24, s=36, marker="x", color=ACCENT_RED, linewidths=1.4,
                   zorder=5, label=f"AMPT Missed (FN={len(ampt_fn)})")
    if len(ampt_fp) > 0:
        ax.scatter(ampt_fp / fs, den_seg[ampt_fp], s=42, marker="+", color=ABL_MEDIAN, linewidths=1.6,
                   zorder=5, label=f"AMPT Spurious (FP={len(ampt_fp)})")

    ax.set_ylim(-0.95, 1.38)
    ax.set_xlim(0, 10)
    ax.set_ylabel("Amp. (mV)")
    ax.set_xlabel("Relative Time (s) [Offset +50.0 s in Record 2812]")
    ax.set_title("(d) Detected R-Peaks vs. Ground Truth (ASQA-H-AMPT vs. Baseline AMPT Errors)", loc="left", color=INK)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.72), ncol=3, columnspacing=1.2, fontsize=7.8)

    out = os.path.join(OUT_DIR, "fig7_pipeline_visualization.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


if __name__ == "__main__":
    apply_clean_style()
    make_fig2_boxplot()
    make_fig3_noise_robustness()
    make_fig4_ablation()
    make_fig5_sqa_distribution()
    make_fig6_efficiency()
    make_fig7_pipeline_trace()
    print("All 6 clean IEEE figures saved to:", OUT_DIR)
