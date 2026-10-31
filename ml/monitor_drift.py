"""
MLOps Data & Concept Drift Monitoring Module.
Detects statistical distribution shifts between baseline training data and production batches.
Calculates Kolmogorov-Smirnov test for numerical features and Population Stability Index (PSI).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import ks_2samp

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "ml"
MODEL_DIR = BASE_DIR / "models"


def calculate_psi(baseline, target, num_bins=10):
    """Calculates Population Stability Index (PSI) for continuous or discrete features."""
    try:
        baseline = pd.to_numeric(baseline, errors="coerce").dropna()
        target = pd.to_numeric(target, errors="coerce").dropna()
        if len(baseline) == 0 or len(target) == 0:
            return 0.0

        quantiles = np.linspace(0, 1, num_bins + 1)
        bins = np.percentile(baseline, quantiles * 100)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        bins[0] = -np.inf
        bins[-1] = np.inf

        base_counts, _ = np.histogram(baseline, bins=bins)
        target_counts, _ = np.histogram(target, bins=bins)

        base_pct = np.where(base_counts == 0, 1e-4, base_counts) / len(baseline)
        target_pct = np.where(target_counts == 0, 1e-4, target_counts) / len(target)

        psi_val = np.sum((target_pct - base_pct) * np.log(target_pct / base_pct))
        return float(psi_val)
    except Exception:
        return 0.0


def run_drift_analysis():
    print("=" * 60)
    print("MLOPS DATA DRIFT & INTEGRITY MONITORING")
    print("=" * 60)

    train_path = DATA_DIR / "X_train.csv"
    test_path = DATA_DIR / "X_test.csv"

    if not train_path.exists() or not test_path.exists():
        print("[-] Training or evaluation batches missing.")
        return

    df_base = pd.read_csv(train_path, low_memory=False)
    df_curr = pd.read_csv(test_path, low_memory=False)

    numeric_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "prior_admission_count",
    ]

    drift_results = []
    drift_detected_count = 0

    for col in numeric_cols:
        if col in df_base.columns and col in df_curr.columns:
            s1 = pd.to_numeric(df_base[col], errors="coerce").dropna()
            s2 = pd.to_numeric(df_curr[col], errors="coerce").dropna()

            ks_stat, p_val = ks_2samp(s1, s2)
            psi = calculate_psi(s1, s2)

            # PSI standard interpretation:
            # PSI < 0.1: No significant change
            # 0.1 <= PSI < 0.25: Moderate change, investigate
            # PSI >= 0.25: Significant drift, retrain recommended
            is_drift = bool(psi >= 0.25 or (p_val < 0.001 and ks_stat > 0.05))
            if is_drift:
                drift_detected_count += 1

            drift_results.append({
                "feature": col,
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 4),
                "psi": round(float(psi), 4),
                "baseline_mean": round(float(s1.mean()), 2),
                "current_mean": round(float(s2.mean()), 2),
                "drift_status": "DRIFT_DETECTED" if is_drift else "STABLE"
            })

    overall_status = "WARNING_DRIFT_DETECTED" if drift_detected_count > 0 else "SYSTEM_HEALTHY"
    recommendation = "Automated retraining trigger recommended." if drift_detected_count > 0 else "Model continues in production serving."

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_sample_size": len(df_base),
        "evaluated_sample_size": len(df_curr),
        "overall_status": overall_status,
        "features_monitored": len(drift_results),
        "features_with_drift": drift_detected_count,
        "recommendation": recommendation,
        "metrics": drift_results
    }

    report_path = MODEL_DIR / "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nMonitored {len(drift_results)} core clinical features.")
    print(f"Overall Status: {overall_status}")
    print(f"Features with drift: {drift_detected_count} / {len(drift_results)}")
    print(f"Drift report exported to: {report_path}")
    return report


if __name__ == "__main__":
    run_drift_analysis()
