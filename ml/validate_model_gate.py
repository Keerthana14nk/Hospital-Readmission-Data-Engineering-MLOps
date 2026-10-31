"""
MLOps Automated Model Validation Gate.
Evaluates champion model against strict operational & clinical performance thresholds
before permitting registration and deployment into production.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
REGISTRY_FILE = MODEL_DIR / "model_registry.json"
METRICS_FILE = MODEL_DIR / "metrics_summary.json"

# Quality Gate Thresholds
MIN_ROC_AUC = 0.60
MIN_RECALL = 0.40
MAX_LATENCY_MS = 50.0


def run_validation_gate():
    print("=" * 60)
    print("MLOPS AUTOMATED MODEL VALIDATION GATE")
    print("=" * 60)

    if not REGISTRY_FILE.exists() or not METRICS_FILE.exists():
        print("[-] FAIL: Model registry or metrics file not found. Train models first.")
        return False

    with open(REGISTRY_FILE, "r") as f:
        registry = json.load(f)

    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)

    champion_name = registry.get("champion_model")
    champion_perf = registry.get("performance", {})

    roc_auc = champion_perf.get("roc_auc", 0.0)
    recall = champion_perf.get("recall", 0.0)

    print(f"Target Champion Model: {champion_name}")
    print(f"Evaluated ROC-AUC    : {roc_auc:.4f} (Required: >= {MIN_ROC_AUC})")
    print(f"Evaluated Recall     : {recall*100:.2f}% (Required: >= {MIN_RECALL*100:.0f}%)")

    checks = []

    # Check 1: ROC-AUC Threshold
    if roc_auc >= MIN_ROC_AUC:
        print("[+] PASS: ROC-AUC meets clinical discrimination threshold.")
        checks.append(True)
    else:
        print(f"[-] FAIL: ROC-AUC ({roc_auc}) below threshold ({MIN_ROC_AUC}).")
        checks.append(False)

    # Check 2: Recall Threshold
    if recall >= MIN_RECALL:
        print("[+] PASS: Minority class recall exceeds safety threshold.")
        checks.append(True)
    else:
        print(f"[-] FAIL: Recall ({recall}) below safety threshold ({MIN_RECALL}).")
        checks.append(False)

    # Check 3: Artifact exists
    model_artifact = BASE_DIR / registry.get("artifact_path", "")
    if model_artifact.exists() and model_artifact.stat().st_size > 1000:
        print(f"[+] PASS: Model artifact verified ({model_artifact.name}, {model_artifact.stat().st_size / (1024*1024):.2f} MB).")
        checks.append(True)
    else:
        print("[-] FAIL: Model artifact missing or corrupted.")
        checks.append(False)

    passed = all(checks)
    gate_report = {
        "gate_timestamp": registry.get("registered_at_utc"),
        "champion_model": champion_name,
        "checks_passed": passed,
        "roc_auc_check": bool(roc_auc >= MIN_ROC_AUC),
        "recall_check": bool(recall >= MIN_RECALL),
        "artifact_check": bool(model_artifact.exists()),
        "status": "APPROVED_FOR_DEPLOYMENT" if passed else "REJECTED"
    }

    gate_output_path = MODEL_DIR / "validation_gate_result.json"
    with open(gate_output_path, "w") as f:
        json.dump(gate_report, f, indent=2)

    print(f"\nGate Report saved to: {gate_output_path}")
    if passed:
        print("[SUCCESS] MODEL VALIDATION GATE PASSED: Model is Approved for Deployment.")
        return True
    else:
        print("[FAILURE] MODEL VALIDATION GATE FAILED: Deployment blocked.")
        return False


if __name__ == "__main__":
    success = run_validation_gate()
    sys.exit(0 if success else 1)
