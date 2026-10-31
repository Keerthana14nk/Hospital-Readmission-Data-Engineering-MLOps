import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "ml_dataset.csv"

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
for column in df.columns:
    print(column)

print("\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

print("\nTarget distribution:")
print(df["readmitted_within_30_days"].value_counts())