import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "cleaned" / "encounters_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "data" / "ml"


def main():
    # Create output folder if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {INPUT_FILE}. Run ETL first.")

    # Load cleaned dataset
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print("Dataset loaded successfully")
    print("Dataset shape:", df.shape)

    # Target column
    target_column = "readmitted_within_30_days"

    # Check target column
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Save the split datasets
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    print("Data splitting completed successfully!")
    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)
    print("Files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()