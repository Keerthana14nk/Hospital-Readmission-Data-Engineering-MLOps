from pathlib import Path
import json
import zipfile
import shutil
from datetime import datetime, timezone

import requests
import pandas as pd

from src.config import RAW_DIR


DATASET_ID = 296

ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip"
)

ZIP_FILE = RAW_DIR / "uci_diabetes_296.zip"
RAW_FILE = RAW_DIR / "diabetic_data.csv"
IDS_FILE = RAW_DIR / "IDS_mapping.csv"
METADATA_FILE = RAW_DIR / "metadata.json"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading original UCI dataset...")

    response = requests.get(ZIP_URL, timeout=120)
    response.raise_for_status()

    ZIP_FILE.write_bytes(response.content)

    print("Download completed.")
    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        names = z.namelist()

        diabetic_name = next(
            (
                name
                for name in names
                if name.lower().endswith("diabetic_data.csv")
            ),
            None,
        )

        ids_name = next(
            (
                name
                for name in names
                if name.lower().endswith("ids_mapping.csv")
            ),
            None,
        )

        if diabetic_name is None:
            raise FileNotFoundError(
                "diabetic_data.csv was not found inside the UCI ZIP file."
            )

        with z.open(diabetic_name) as source:
            with open(RAW_FILE, "wb") as destination:
                shutil.copyfileobj(source, destination)

        if ids_name is not None:
            with z.open(ids_name) as source:
                with open(IDS_FILE, "wb") as destination:
                    shutil.copyfileobj(source, destination)

    # Check the downloaded CSV
    df = pd.read_csv(RAW_FILE, nrows=1)

    metadata = {
        "dataset_id": DATASET_ID,
        "source": "UCI Machine Learning Repository",
        "dataset_name": "Diabetes 130-US Hospitals for Years 1999-2008",
        "source_url": "https://archive.ics.uci.edu/dataset/296",
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rows": 101766,
        "columns": len(df.columns),
        "raw_file": str(RAW_FILE),
        "ids_file": str(IDS_FILE),
        "status": "success",
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("Dataset successfully downloaded.")
    print(f"Rows: {metadata['rows']}")
    print(f"Columns: {metadata['columns']}")
    print(f"Saved to: {RAW_FILE}")
    print()
    print("First columns:")
    print(df.columns.tolist()[:10])


if __name__ == "__main__":
    main()