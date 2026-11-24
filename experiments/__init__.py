"""Experimental evaluation suite for ASQA-H-AMPT vs QRS detection baselines."""

from .metrics import (
    DATASETS,
    B2_PACED_EXCLUDED,
    TIME_WINDOW_MS,
    fast_check_positive,
    fast_check_negative,
    load_dataset_records,
    load_single_record,
    compute_record_metrics,
    aggregate_dataset_metrics,
    compute_rr_hr_mae,
    run_wilcoxon_tests,
)

__all__ = [
    "DATASETS",
    "B2_PACED_EXCLUDED",
    "TIME_WINDOW_MS",
    "fast_check_positive",
    "fast_check_negative",
    "load_dataset_records",
    "load_single_record",
    "compute_record_metrics",
    "aggregate_dataset_metrics",
    "compute_rr_hr_mae",
    "run_wilcoxon_tests",
]
