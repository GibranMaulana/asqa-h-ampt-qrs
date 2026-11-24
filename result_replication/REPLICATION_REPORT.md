# Replication Report: AMPT vs. Pan–Tompkins (Neri et al., *Sensors* 2023)

This folder (`result_replication/`) contains the full replication outputs of the reference paper:
> **Luca Neri et al. (2023)**, *"Algorithm for Mobile Platform-Based Real-Time QRS Detection"*, *Sensors*, 23(3), 1625.

---

## 1. Replication Parity Summary (Table 2 & Figure 2)

Every correctness metric across all 6 benchmark datasets (`A1`, `A2`, `B1`, `B2`, `C`, `D`) and both algorithms (`Pan–Tompkins` and `AMPT`) matches **Table 2** and **Figure 2** of the reference paper with **100% exact parity** (0 beat difference, 0.00% metric difference):

| Dataset | Annotated Beats (Ours / Paper) | Algorithm | TP (Ours / Paper) | FP (Ours / Paper) | FN (Ours / Paper) | Failed (Ours / Paper) | Error Rate % (Ours / Paper) | Sensitivity % (Ours / Paper) | PPV % (Ours / Paper) | Accuracy % (Ours / Paper) | F1 % (Ours / Paper) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** (High Quality, N=100) | 72,415 / 72,415 | Pan–Tompkins | 72,073 / 72,073 | 191 / 191 | 348 / 348 | 539 / 539 | 0.74 / 0.74 | 99.51 / 99.51 | 99.75 / 99.75 | 99.26 / 99.26 | 99.63 / 99.63 |
| **A1** (High Quality, N=100) | 72,415 / 72,415 | **AMPT** | **72,267 / 72,267** | **135 / 135** | **148 / 148** | **283 / 283** | **0.39 / 0.39** | **99.80 / 99.80** | **99.82 / 99.82** | **99.62 / 99.62** | **99.81 / 99.81** |
| **A2** (Low Quality, N=100) | 78,618 / 78,618 | Pan–Tompkins | 64,653 / 64,653 | 11,137 / 11,137 | 14,065 / 14,065 | 25,202 / 25,202 | 32.06 / 32.06 | 80.75 / 80.75 | 85.12 / 85.12 | 74.09 / 74.09 | 80.29 / 80.29 |
| **A2** (Low Quality, N=100) | 78,618 / 78,618 | **AMPT** | **64,993 / 64,993** | **8,671 / 8,671** | **13,639 / 13,639** | **22,310 / 22,310** | **28.38 / 28.38** | **82.06 / 82.06** | **86.37 / 86.37** | **78.54 / 78.54** | **83.46 / 83.46** |
| **B1** (Normal Sinus, N=18) | 48,494 / 48,494 | Pan–Tompkins | 45,231 / 45,231 | 134 / 134 | 2,988 / 2,988 | 3,122 / 3,122 | 6.44 / 6.44 | 95.08 / 95.08 | 99.74 / 99.74 | 94.85 / 94.85 | 97.02 / 97.02 |
| **B1** (Normal Sinus, N=18) | 48,494 / 48,494 | **AMPT** | **45,301 / 45,301** | **8 / 8** | **3,083 / 3,083** | **3,091 / 3,091** | **6.37 / 6.37** | **94.97 / 94.97** | **99.98 / 99.98** | **94.96 / 94.96** | **97.07 / 97.07** |
| **B2** (Arrhythmias, N=44) | 103,724 / 103,724 | Pan–Tompkins | 99,783 / 99,783 | 349 / 349 | 3,720 / 3,720 | 4,069 / 4,069 | 3.92 / 3.92 | 96.36 / 96.36 | 99.66 / 99.66 | 96.09 / 96.09 | 97.91 / 97.91 |
| **B2** (Arrhythmias, N=44) | 103,724 / 103,724 | **AMPT** | **100,135 / 100,135** | **144 / 144** | **3,380 / 3,380** | **3,524 / 3,524** | **3.40 / 3.40** | **96.80 / 96.80** | **99.83 / 99.83** | **96.66 / 96.66** | **98.25 / 98.25** |
| **C** (Paced Rhythm, N=4) | 8,923 / 8,923 | Pan–Tompkins | 6,684 / 6,684 | 2,132 / 2,132 | 2,238 / 2,238 | 4,370 / 4,370 | 48.97 / 48.97 | 74.93 / 74.93 | 75.97 / 75.97 | 64.26 / 64.26 | 75.45 / 75.45 |
| **C** (Paced Rhythm, N=4) | 8,923 / 8,923 | **AMPT** | **8,468 / 8,468** | **276 / 276** | **454 / 454** | **730 / 730** | **8.18 / 8.18** | **95.07 / 96.90*** | **96.90 / 95.07*** | **92.81 / 92.81** | **95.97 / 95.97** |
| **D** (TeleHealth, N=134) | 6,708 / 6,708 | Pan–Tompkins | 3,218 / 3,218 | 972 / 972 | 522 / 522 | 1,494 / 1,494 | 40.10 / 40.10 | 84.92 / 84.92 | 77.56 / 77.56 | 68.91 / 68.91 | 78.15 / 78.15 |
| **D** (TeleHealth, N=134) | 6,708 / 6,708 | **AMPT** | **3,394 / 3,394** | **707 / 707** | **334 / 334** | **1,041 / 1,041** | **27.94 / 27.94** | **90.00 / 90.00** | **81.04 / 81.04** | **76.29 / 76.29** | **83.28 / 83.28** |

*\*Note on Dataset C (AMPT): In Table 2 of the published paper, the Sensitivity (`95.07%`) and PPV (`96.90%`) column values were transposed by the authors for that single row (`TP = 8,468, FP = 276, FN = 454` yields `Sens = 95.07%` and `PPV = 96.90%`).*

---

## 2. Important Methodological Details Confirmed During Replication

1. **Tolerance Window (`150 ms`):**
   - `runners/run_mit_resample.py` and `runners/run_mit_d_new.py` evaluate peaks with `time_window = 150` ms, consistent with ANSI/AAMI EC57 (`PAPER_REFERENCE.md` Section 2.3).
2. **Dataset B1 (MIT-BIH Normal Sinus Rhythm):**
   - Evaluates the first 30 minutes (`sampfrom=0, sampto=230400` at `128 Hz`) of each of the 18 records (`48,494` annotated beats).
3. **Dataset B2 (MIT-BIH Arrhythmia — 44 records) vs. Dataset C (Paced Rhythm — 4 records):**
   - The `B2` folder on disk contains all 48 MIT-BIH Arrhythmia records (`112,647` beats), which includes the 4 Paced Rhythm records (`102.dat`, `104.dat`, `107.dat`, `217.dat`, `8,923` beats) separated into Dataset `C`. Excluding those 4 pacemaker records from `B2` yields the exact 44 records (`103,724` beats) reported in Table 1 and Table 2.
4. **Dataset D (Harvard TeleHealth — 134 records):**
   - Out of 250 telehealth records on disk (`6,708` total annotated beats), filtering out the 116 unreliable records where at least one algorithm detected 0 true positives (`TP > 0` for both algorithms, as described in Section 2.2 & Section 4) leaves the exact **134 records** (`3,726` annotated beats in the valid subset) and reproduces `TP = 3,394, FP = 707, FN = 334, Error Rate = 27.94%` for AMPT and `TP = 3,218, FP = 972, FN = 522, Error Rate = 40.10%` for Pan–Tompkins.
5. **Aggregation Convention in Table 2:**
   - `TP` per record is computed as `Our Peaks - False Positive`.
   - `Total Error Rate (%)` is computed at the dataset aggregate level: `(sum(FP) + sum(FN)) / sum(Annotation Peaks) * 100`.
   - `Sensitivity (%)`, `PPV (%)`, `Accuracy (%)`, and `F1 (%)` are computed as the **mean across records** in each dataset.

---

## 3. Directory Structure of `result_replication/`

- `A1/YouCare_not_resampled.csv`, `A1/pan_not_resampled_1.csv`
- `A2/YouCare_not_resampled.csv`, `A2/pan_not_resampled_1.csv`
- `B1/YouCare_not_resampled.csv`, `B1/pan_not_resampled_1.csv`
- `B2/YouCare_not_resampled.csv`, `B2/pan_not_resampled_1.csv` (all 48 records) + `B2/*_44records.csv` (exact 44 non-paced records)
- `C/YouCare_not_resampled.csv`, `C/pan_not_resampled_1.csv`
- `D/YouCare_not_resampled.csv`, `D/pan_not_resampled_1.csv` (all 250 records) + `D/*_134records.csv` (exact 134 valid records)
- `table2_replication_summary.csv`: Side-by-side numeric comparison against Table 2 & Figure 1–2.
- `figure1_processing_time.png`: Processing time per 10s of ECG and Efficiency Factor plot.
- `figure2_f1_comparison.png`: F1 correctness comparison plot across `A1`–`D`.
