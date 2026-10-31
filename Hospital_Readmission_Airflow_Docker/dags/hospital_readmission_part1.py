"""
Airflow orchestration for Hospital Readmission Analytics & Prediction.
Fully integrated pipeline: Data Engineering (Tasks 1-4) + MLOps (Tasks 5-8).
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


# Phase 1: Data Engineering Callables
def ingest():
    from src.ingestion.ingest import main
    main()


def clean():
    from src.transform.clean import main
    main()


def validate():
    from src.quality.validate import main
    main()


def load():
    from src.load.postgres import main
    main()


# Phase 2: MLOps Callables
def split_ml():
    from ml.split_data import main
    main()


def train_and_evaluate():
    from ml.train_and_evaluate import main
    main()


def validate_gate():
    from ml.validate_model_gate import run_validation_gate
    passed = run_validation_gate()
    if not passed:
        raise ValueError("Model failed operational validation gate. Promotion aborted.")


def monitor_drift():
    from ml.monitor_drift import run_drift_analysis
    run_drift_analysis()


with DAG(
    dag_id="hospital_readmission_integrated_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["hospital-readmission", "data-engineering", "mlops", "integrated"],
    description="Integrated Healthcare Pipeline: Ingestion -> ETL -> Data Mart -> Feature Splitting -> ML Training -> Quality Gate -> Drift Monitoring.",
) as dag:

    # 1. Ingestion
    ingest_raw_to_staging = PythonOperator(
        task_id="ingest_raw_to_staging",
        python_callable=ingest,
    )

    # 2. Transformation
    clean_and_engineer = PythonOperator(
        task_id="clean_and_engineer",
        python_callable=clean,
    )

    # 3. Validation
    validate_data_quality = PythonOperator(
        task_id="validate_data_quality",
        python_callable=validate,
    )

    # 4. Warehouse Load
    load_postgresql_mart = PythonOperator(
        task_id="load_postgresql_mart",
        python_callable=load,
    )

    # 5. ML Feature Preparation
    prepare_and_split_ml = PythonOperator(
        task_id="prepare_and_split_ml",
        python_callable=split_ml,
    )

    # 6. Model Training & Benchmarking
    train_and_benchmark_models = PythonOperator(
        task_id="train_and_benchmark_models",
        python_callable=train_and_evaluate,
    )

    # 7. Model Validation Gate
    model_validation_gate = PythonOperator(
        task_id="model_validation_gate",
        python_callable=validate_gate,
    )

    # 8. Post-Deployment Drift Monitoring
    monitor_drift_and_health = PythonOperator(
        task_id="monitor_drift_and_health",
        python_callable=monitor_drift,
    )

    # Fully integrated dependency chain
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
