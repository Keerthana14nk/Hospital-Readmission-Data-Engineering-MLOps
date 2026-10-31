"""
Unified Pipeline Orchestrator for Hospital Readmission Analytics & MLOps.
Executes both Phase 1 (Data Engineering) and Phase 2 (MLOps Pipeline) sequentially or modularly.
"""

import sys
import time
import argparse
from datetime import datetime, timezone


def run_data_pipeline():
    print("\n" + "=" * 65)
    print("PHASE 1: HEALTHCARE DATA ENGINEERING & ETL PIPELINE")
    print("=" * 65)

    start_time = time.time()

    print("\n>>> Task 1: Ingesting Raw Data from Public UCI Source...")
    from src.ingestion.ingest import main as ingest_main
    ingest_main()

    print("\n>>> Task 2: Cleaning, Normalizing & Feature Engineering...")
    from src.transform.clean import main as clean_main
    clean_main()

    print("\n>>> Task 3: Running Automated Data Quality Validations...")
    from src.quality.validate import main as validate_main
    validate_main()

    print("\n>>> Task 4: Loading Curated Data into PostgreSQL Mart...")
    try:
        from src.load.postgres import main as load_main
        load_main()
    except Exception as e:
        print(f"[!] Warning: PostgreSQL load encountered issue ({e}). Local CSV mart preserved.")

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Phase 1 Data Engineering Pipeline Completed in {elapsed:.2f}s")


def run_mlops_pipeline():
    print("\n" + "=" * 65)
    print("PHASE 2: PRODUCTION MLOPS & MODELING PIPELINE")
    print("=" * 65)

    start_time = time.time()

    print("\n>>> Task 5: Splitting Feature Matrix into Stratified Train/Test Sets...")
    from ml.split_data import main as split_main
    split_main()

    print("\n>>> Task 6: Training & Benchmarking Models with Clinical Threshold Analysis...")
    from ml.train_and_evaluate import main as train_main
    train_main()

    print("\n>>> Task 7: Evaluating Automated Operational Quality Gate...")
    from ml.validate_model_gate import run_validation_gate
    passed = run_validation_gate()
    if not passed:
        print("[FAILURE] Model failed operational quality gate. Pipeline halted.")
        sys.exit(1)

    print("\n>>> Task 8: Running Statistical Data & Concept Drift Monitoring...")
    from ml.monitor_drift import run_drift_analysis
    run_drift_analysis()

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Phase 2 MLOps Pipeline Completed in {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Hospital Readmission Integrated Data & MLOps Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "data", "mlops"],
        default="all",
        help="Specify which pipeline stage to execute (default: all)"
    )
    args = parser.parse_args()

    pipeline_start = time.time()
    print("*" * 65)
    print("HOSPITAL READMISSION INTEGRATED ETL + MLOPS PIPELINE")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("*" * 65)

    if args.stage in ["all", "data"]:
        run_data_pipeline()

    if args.stage in ["all", "mlops"]:
        run_mlops_pipeline()

    total_elapsed = time.time() - pipeline_start
    print("\n" + "*" * 65)
    print(f"[SUCCESS] Integrated Pipeline Finished in {total_elapsed:.2f}s")
    print("*" * 65)


if __name__ == "__main__":
    main()
