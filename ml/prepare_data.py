import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "ml_dataset.csv"

# Load original dataset
df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# Create binary target
df["readmitted_within_30_days"] = (
    df["readmitted"].astype(str).str.strip() == "<30"
).astype(int)

# Check target distribution
print("\nTarget distribution:")
print(df["readmitted_within_30_days"].value_counts())

# Save ML dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nML dataset created successfully!")
print("Saved to:", OUTPUT_FILE)
print("Final shape:", df.shape)