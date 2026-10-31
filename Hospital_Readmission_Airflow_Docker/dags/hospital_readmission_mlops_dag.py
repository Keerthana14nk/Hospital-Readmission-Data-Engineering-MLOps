"""
Airflow MLOps Orchestration DAG for Hospital Readmission Analytics & Prediction — Part 2.
Orchestrates the entire lifecycle: Ingestion -> ETL -> Data Quality -> DW Mart Load ->
Feature Engineering -> Model Training & Benchmarking -> Quality Gate -> Drift Monitoring.
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


def task_ingest():
    from src.ingestion.ingest import main
    main()


def task_clean():
    from src.transform.clean import main
    main()


def task_validate():
    from src.quality.validate import main
    main()


def task_load():
    from src.load.postgres import main
    main()


def task_split_ml():
    from ml.split_data import main as split_main
    # If split_data doesn't define main, execute module logic
    import subprocess, sys
    res = subprocess.run([sys.executable, "-m", "ml.split_data"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Split data failed: {res.stderr}")


def task_train_and_evaluate():
    from ml.train_and_evaluate import main as train_main
    train_main()


def task_validation_gate():
    from ml.validate_model_gate import run_validation_gate
    passed = run_validation_gate()
    if not passed:
        raise ValueError("Model failed operational validation gate. Promotion aborted.")


def task_drift_monitor():
    from ml.monitor_drift import run_drift_analysis
    run_drift_analysis()


with DAG(
    dag_id="hospital_readmission_mlops_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["hospital-readmission", "mlops", "data-engineering", "part2"],
    description="End-to-end Healthcare Data & MLOps Pipeline: Ingestion, ETL, DW Mart, Training, Gate & Monitoring.",
) as dag:

    # Phase 1: Data Engineering Pipeline
    ingest_raw_to_staging = PythonOperator(
        task_id="ingest_raw_to_staging",
        python_callable=task_ingest,
    )

    clean_and_engineer = PythonOperator(
        task_id="clean_and_engineer",
        python_callable=task_clean,
    )

    validate_data_quality = PythonOperator(
        task_id="validate_data_quality",
        python_callable=task_validate,
    )

    load_postgresql_mart = PythonOperator(
        task_id="load_postgresql_mart",
        python_callable=task_load,
    )

    # Phase 2: MLOps Pipeline
    prepare_and_split_ml = PythonOperator(
        task_id="prepare_and_split_ml",
        python_callable=task_split_ml,
    )

    train_and_benchmark_models = PythonOperator(
        task_id="train_and_benchmark_models",
        python_callable=task_train_and_evaluate,
    )

    model_validation_gate = PythonOperator(
        task_id="model_validation_gate",
        python_callable=task_validation_gate,
    )

    monitor_drift_and_health = PythonOperator(
        task_id="monitor_drift_and_health",
        python_callable=task_drift_monitor,
    )

    # End-to-end dependency chain
    (
        ingest_raw_to_staging
        >> clean_and_engineer
        >> validate_data_quality
        >> load_postgresql_mart
        >> prepare_and_split_ml
        >> train_and_benchmark_models
        >> model_validation_gate
        >> monitor_drift_and_health
    )
