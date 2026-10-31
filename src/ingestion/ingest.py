from pathlib import Path
import json
from datetime import datetime, timezone

import pandas as pd

from src.config import RAW_DIR, STAGING_DIR


RAW_FILE = RAW_DIR / "diabetic_data.csv"
STAGING_FILE = STAGING_DIR / "encounters_staging.csv"
METADATA_FILE = STAGING_DIR / "ingestion_metadata.json"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names without removing any source fields."""
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def main():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    extraction_timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # Read the raw dataset.
        # low_memory=False avoids the mixed-type warning seen in the UCI file.
        df = pd.read_csv(RAW_FILE, low_memory=False)

        rows_extracted = len(df)

        # Normalize column names but KEEP ALL source columns.
        df = normalize_columns(df)

        # Convert special missing-value marker to proper null values.
        df = df.replace("?", pd.NA)

        # Save staging data.
        df.to_csv(STAGING_FILE, index=False)

        metadata = {
            "extraction_timestamp_utc": extraction_timestamp,
            "source": "UCI Machine Learning Repository — Dataset 296",
            "file": str(RAW_FILE.resolve()),
            "rows_extracted": rows_extracted,
            "rows_staged": len(df),
            "status": "success",
        }

        METADATA_FILE.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8"
        )

        print(json.dumps(metadata, indent=2))

    except Exception as exc:
        metadata = {
            "extraction_timestamp_utc": extraction_timestamp,
            "source": "UCI Machine Learning Repository — Dataset 296",
            "file": str(RAW_FILE.resolve()),
            "rows_extracted": 0,
            "rows_staged": 0,
            "status": "failed",
            "error": str(exc),
        }

        METADATA_FILE.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8"
        )

        raise


if __name__ == "__main__":
    main()
