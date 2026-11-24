# ASQA-H-AMPT: Adaptive Signal-Quality Assessment and Hilbert-Enhanced AMPT for Real-Time QRS Detection

Official code and reproduction repository for **ASQA-H-AMPT**, developed collaboratively by our research team for the Digital Signal Processing (PSD) Final Project. The framework combines 5-second windowed Signal Quality Assessment (ASQA), a 4-mode Adaptive Denoising Decision Tree (`Bypass`, `Dual-Median`, `Sym4 Wavelet`, `Adaptive Kalman 1D`), dry-electrode rail-saturation masking, zero-phase Hilbert-enhanced AMPT feature extraction, and CQS-modulated dynamic adaptive thresholding with RR-based post-processing.

**Team Members / Authors (Collaborative Work):**
- **Lutfiyah Istiana**
- **Gibran Maulana Azmi**
- **Radit Alfa Anugerah Bombing**
- **Hasan Dwi Cahyono**

**Affiliation:** Informatics Study Program, Faculty of Information Technology and Data Science, Universitas Sebelas Maret (UNS), Surakarta, Indonesia  
**Upstream Reference Baseline:** Neri et al. (2023), *Algorithm for Mobile Platform-Based Real-Time QRS Detection*, Sensors 23(3), 1625 (`Accyourate-Group-S-p-A/acy_ampt`).

---

## Repository Structure

```text
.
├── experiments/
│   ├── metrics.py                # Fast O(N log N) ANSI/AAMI 150 ms matcher, HR/RR MAE, Wilcoxon & paired t-test
│   ├── detectors.py              # Pan-Tompkins, AMPT, Wavelet/Median/Kalman+AMPT ablations, and ASQA-H-AMPT
│   ├── run_all.py                # Full 6-dataset multiprocessing runner + AWGN & Baseline-Drift stress tests
│   ├── test_verification.py      # Independent verification & zero-hardcoding QA suite (4/4 PASS)
│   └── generate_figures_clean.py # Publication-grade 300 DPI IEEE figure generator
├── run_replication.py            # Exact Week 1 baseline replication runner (100% parity with Neri et al. 2023)
├── 01_replication.ipynb          # Interactive baseline replication notebook
├── result_replication/           # Verified baseline CSVs & REPLICATION_REPORT.md
├── result_experiment/            # 36 per-record CSVs, 7 summary tables, EXPERIMENT_SUMMARY.md & figures_clean/
├── requirements.txt
└── .gitignore
```

---

## Key Verified Results (400 Valid Records, 315,900 Annotated Beats)

Evaluated under the exact ANSI/AAMI EC57 ($\pm 150\text{ ms}$) protocol of Neri et al. (2023):

| Dataset | Records / Beats | Pan–Tompkins F1 | AMPT F1 | **ASQA-H-AMPT F1** | Err. Rate (%) | Time (ms/10s) | vs. AMPT Wilcoxon $p$ | vs. AMPT t-test $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** (PhysioNet 2014 High Q.) | 100 / 72,415 | 99.63% | 99.81% | **99.87%** | **0.27** | 3.71 | $3.76\times 10^{-7}$ | $5.75\times 10^{-5}$ |
| **A2** (PhysioNet 2014 Low Q.) | 100 / 78,618 | 80.29% | 83.46% | **88.17%** (+4.71) | **23.67** | 6.47 | 0.0047 | 0.0062 |
| **B1** (MIT-BIH NSR, 30 min) | 18 / 48,494 | 97.02% | 97.07% | **97.08%** | 6.37 | 2.68 | 0.1094 | 0.6569 |
| **B2** (MIT-BIH Arrhythmia) | 44 / 103,724 | 97.91% | 98.25% | **98.27%** | **3.26** | 4.82 | 0.6205 | 0.8438 |
| **C** (MIT-BIH Paced Rhythm) | 4 / 8,923 | 75.45% | 95.97% | **98.98%** (+3.01) | **2.06** | 3.18 | 0.5000 | 0.3879 |
| **D** (Harvard TeleHealth) | 134 / 3,726 | 78.15% | 83.28% | **85.05%** (+1.77) | **25.87** | 8.70 | $8.40\times 10^{-4}$ | 0.0325 |
| **Pooled `ALL_400`** | 400 / 315,900 | 90.06% | 92.86% | **94.68%** (+1.82) | — | **4.93** avg | **$1.72\times 10^{-8}$** | **$4.06\times 10^{-4}$** |

- **Controlled Stress Tests (30 `A1` records):** At **−5 dB AWGN**, ASQA-H-AMPT maintains **97.04% F1** vs. AMPT **90.11%** (+6.93 pp); under **2.0 mV 0.3 Hz baseline drift**, **99.85% F1** (0.34% error rate) vs. AMPT **99.81%** (0.40%).
- **Real-Time Efficiency:** 2.68–8.70 ms per 10 s of ECG (**2.6×–25.5× faster than Pan–Tompkins**) with **98.35%** of clean `A1` windows bypassing heavy denoising.

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Clone Reference Baseline (`acy_ampt`) & Download Benchmark Datasets
The upstream reference baseline (`acy_ampt`) and raw ECG databases (~1.9 GB extracted) are excluded from this repository. Clone the official AccYouRate repository and extract `ECG DBs.zip` (download link in `acy_ampt/README.md`) inside it:
```bash
git clone https://github.com/Accyourate-Group-S-p-A/acy_ampt.git
# Download ECG DBs.zip from the link in acy_ampt/README.md, then extract to:
# acy_ampt/ECG DBs/
```

### 3. Run Baseline Replication, Full Experiments, Verification & Figures
```bash
# 1) Replicate Pan-Tompkins & AMPT baselines (Neri et al., 2023)
python run_replication.py

# 2) Run all 6 methods, ablations, Wilcoxon/t-test, HR/RR MAE, and AWGN/Drift stress tests
python experiments/run_all.py

# 3) Run independent QA & numerical parity verification suite
python experiments/test_verification.py

# 4) Regenerate publication-grade 300 DPI figures into result_experiment/figures_clean/
python experiments/generate_figures_clean.py
```

---

## Acknowledgments & Third-Party Attribution
- **Baseline Implementation:** Original Pan–Tompkins and AMPT implementations are authored by **AccYouRate Group S.p.A.** ([`Accyourate-Group-S-p-A/acy_ampt`](https://github.com/Accyourate-Group-S-p-A/acy_ampt), Neri et al., *Sensors* 2023) and are excluded from this repository; please clone them separately as shown above.
- **Benchmark Datasets:** PhysioNet/CinC Challenge 2014, MIT-BIH Normal Sinus Rhythm Database, MIT-BIH Arrhythmia Database (PhysioNet), and the Harvard Dataverse TELE ECG Database.

---

## License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
