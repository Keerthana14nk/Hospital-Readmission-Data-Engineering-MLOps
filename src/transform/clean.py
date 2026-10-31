import re
import hashlib
import pandas as pd

from src.config import STAGING_DIR, CLEANED_DIR

INPUT = STAGING_DIR / "encounters_staging.csv"
OUTPUT = CLEANED_DIR / "encounters_cleaned.csv"

MISSING_TOKENS = {"?", "Unknown/Invalid", "Unknown/Invalid ", "Not Available"}

def age_midpoint(age):
    if pd.isna(age):
        return pd.NA
    m = re.match(r"\[(\d+),\s*(\d+)\)", str(age))
    if not m:
        return pd.NA
    lo, hi = int(m.group(1)), int(m.group(2))
    return (lo + hi) // 2

def diagnosis_group(row):
    # Use primary diagnosis where available; map broad ICD-9 ranges.
    val = str(row.get("diag_1", ""))
    if val in {"nan", "<NA>", ""}:
        return "Unknown"
    if val.startswith(("250",)):
        return "Diabetes"
    try:
        code = float(val)
        if 390 <= code < 460:
            return "Circulatory"
        if 460 <= code < 520:
            return "Respiratory"
        if 520 <= code < 580:
            return "Digestive"
        if 580 <= code < 630:
            return "Genitourinary"
        if 140 <= code < 240:
            return "Neoplasms"
        if 800 <= code < 1000:
            return "Injury"
    except ValueError:
        pass
    return "Other"

def surrogate_patient_key(value):
    # Deterministic, non-reversible analytical key.
    return hashlib.sha256(f"patient:{value}".encode()).hexdigest()[:16]

def main():
    if not INPUT.exists():
        raise FileNotFoundError("Run ingestion first.")

    df = pd.read_csv(INPUT, low_memory=False)
    df = df.replace(list(MISSING_TOKENS), pd.NA)

    # Remove exact duplicate encounters.
    df = df.drop_duplicates(subset=["encounter_id"], keep="first")

    # Exclude invalid encounter identifiers.
    df["encounter_id"] = pd.to_numeric(df["encounter_id"], errors="coerce")
    df["patient_nbr"] = pd.to_numeric(df["patient_nbr"], errors="coerce")
    df = df.dropna(subset=["encounter_id", "patient_nbr"])

    # Create privacy-safe surrogate key and never expose patient_nbr downstream.
    df["patient_key"] = df["patient_nbr"].astype("int64").map(surrogate_patient_key)

    df["age_years_midpoint"] = df["age"].map(age_midpoint)
    df["diagnosis_group"] = df.apply(diagnosis_group, axis=1)

    numeric_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["length_of_stay"] = df["time_in_hospital"]

    # Dataset-defined readmission classes: <30, >30, NO.
    df["readmission_class"] = df["readmitted"].astype("string")
    df["readmitted_within_30_days"] = (
        df["readmission_class"].eq("<30").astype("int8")
    )

    # Useful engineered variables required by the assignment.
    df["prior_admission_count"] = (
        df[["number_inpatient", "number_outpatient", "number_emergency"]]
        .fillna(0).sum(axis=1)
    )
    df["diagnosis_count"] = df["number_diagnoses"].fillna(0)
    df["medication_count"] = df["num_medications"].fillna(0)

    # Keep patient-safe analytical columns. patient_nbr is deliberately removed.
    drop_cols = ["patient_nbr"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df.to_csv(OUTPUT, index=False)
    print(f"Cleaned rows: {len(df):,}")
    print(f"Output: {OUTPUT}")

if __name__ == "__main__":
    main()
