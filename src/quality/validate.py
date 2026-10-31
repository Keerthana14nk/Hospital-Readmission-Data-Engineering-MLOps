from datetime import datetime, timezone
import pandas as pd

from src.config import CLEANED_DIR, ERROR_DIR

INPUT = CLEANED_DIR / "encounters_cleaned.csv"
ERROR_FILE = ERROR_DIR / "rejected_records.csv"

def main():
    df = pd.read_csv(INPUT, low_memory=False)
    rejected = []

    def reject(mask, rule):
        bad = df.loc[mask].copy()
        if not bad.empty:
            bad["validation_rule"] = rule
            rejected.append(bad)

    reject(df["encounter_id"].isna(), "encounter_id_not_null")
    reject(df["patient_key"].isna(), "patient_key_not_null")
    reject(
        ~df["time_in_hospital"].between(1, 14, inclusive="both"),
        "length_of_stay_between_1_and_14"
    )
    reject(
        ~df["readmitted"].isin(["<30", ">30", "NO"]),
        "readmission_class_valid"
    )

    if rejected:
        rejected_df = pd.concat(rejected, ignore_index=True).drop_duplicates(
            subset=["encounter_id"]
        )
    else:
        rejected_df = pd.DataFrame(columns=list(df.columns) + ["validation_rule"])

    rejected_df.to_csv(ERROR_FILE, index=False)

    valid = df[~df["encounter_id"].isin(rejected_df["encounter_id"])]
    valid.to_csv(INPUT, index=False)

    summary = {
        "validation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(df),
        "rejected_rows": len(rejected_df),
        "valid_rows": len(valid),
        "status": "success",
    }
    print(summary)

if __name__ == "__main__":
    main()
