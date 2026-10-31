import pandas as pd
from sqlalchemy import create_engine, text

from src.config import POSTGRES_URL, STAGING_DIR, CLEANED_DIR, ERROR_DIR


def main():
    engine = create_engine(POSTGRES_URL)

    # Create schemas
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS staging"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS mart"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS ops"))

    # ---------------------------------------------------------
    # 1. Load cleaned/staged encounter data
    # ---------------------------------------------------------
    cleaned_file = CLEANED_DIR / "encounters_cleaned.csv"

    df = pd.read_csv(cleaned_file, low_memory=False)

    print(f"Loading {len(df):,} cleaned records...")

    # Use an SQLAlchemy Connection, not the Engine directly
    with engine.begin() as conn:
        df.to_sql(
            "encounters_cleaned",
            con=conn,
            schema="staging",
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi",
        )

    print("Loaded staging.encounters_cleaned")

    # ---------------------------------------------------------
    # 2. Admission fact table
    # ---------------------------------------------------------
    admission_columns = [
        "encounter_id",
        "patient_key",
        "gender",
        "race",
        "age",
        "age_years_midpoint",
        "diagnosis_group",
        "admission_type_id",
        "admission_source_id",
        "discharge_disposition_id",
        "time_in_hospital",
        "length_of_stay",
        "number_diagnoses",
        "diagnosis_count",
        "medication_count",
        "prior_admission_count",
        "readmission_class",
        "readmitted_within_30_days",
    ]

    available_columns = [
        col for col in admission_columns
        if col in df.columns
    ]

    admission_fact = df[available_columns].copy()

    with engine.begin() as conn:
        admission_fact.to_sql(
            "admission_fact",
            con=conn,
            schema="mart",
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi",
        )

    print("Created mart.admission_fact")

    # ---------------------------------------------------------
    # 3. Diagnosis summary
    # ---------------------------------------------------------
    diagnosis_columns = [
        "encounter_id",
        "patient_key",
        "diag_1",
        "diag_2",
        "diag_3",
        "diagnosis_count",
    ]

    available_diagnosis_columns = [
        col for col in diagnosis_columns
        if col in df.columns
    ]

    diagnosis_summary = df[available_diagnosis_columns].copy()

    with engine.begin() as conn:
        diagnosis_summary.to_sql(
            "diagnosis_summary",
            con=conn,
            schema="mart",
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi",
        )

    print("Created mart.diagnosis_summary")

    # ---------------------------------------------------------
    # 4. Patient summary
    # ---------------------------------------------------------
    patient_columns = [
        "patient_key",
        "age_midpoint",
        "gender",
        "race",
        "prior_admission_count",
    ]

    available_patient_columns = [
        col for col in patient_columns
        if col in df.columns
    ]

    patient_summary = (
        df[available_patient_columns]
        .drop_duplicates(subset=["patient_key"])
        .copy()
    )

    with engine.begin() as conn:
        patient_summary.to_sql(
            "patient_summary",
            con=conn,
            schema="mart",
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi",
        )

    print("Created mart.patient_summary")

    # ---------------------------------------------------------
    # 5. Readmission summary
    # ---------------------------------------------------------
    readmission_summary = (
        df.groupby("readmission_class", dropna=False)
        .size()
        .reset_index(name="encounter_count")
    )

    with engine.begin() as conn:
        readmission_summary.to_sql(
            "readmission_summary",
            con=conn,
            schema="mart",
            if_exists="replace",
            index=False,
            method="multi",
        )

    print("Created mart.readmission_summary")

    # ---------------------------------------------------------
    # 6. Rejected records
    # ---------------------------------------------------------
    rejected_file = ERROR_DIR / "rejected_records.csv"

    if rejected_file.exists():
        rejected = pd.read_csv(rejected_file, low_memory=False)

        with engine.begin() as conn:
            rejected.to_sql(
                "rejected_records",
                con=conn,
                schema="ops",
                if_exists="replace",
                index=False,
                chunksize=5000,
                method="multi",
            )

        print("Loaded ops.rejected_records")

    print()
    print("====================================")
    print("POSTGRESQL LOAD COMPLETED")
    print("====================================")
    print(f"Records loaded: {len(df):,}")
    print("Schemas: staging, mart, ops")
    print("Status: SUCCESS")


if __name__ == "__main__":
    main()