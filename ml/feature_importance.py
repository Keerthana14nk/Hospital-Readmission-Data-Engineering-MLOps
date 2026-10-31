import pandas as pd
from pathlib import Path
import joblib

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "ml"
MODEL_DIR = BASE_DIR / "models"

# Load trained model
model_path = MODEL_DIR / "random_forest_model.pkl"
model = joblib.load(model_path)

# Load training data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")

# Remove leakage columns
leakage_columns = [
    "readmitted",
    "readmission_class",
    "readmitted_within_30_days",
    "age_years_midpoint"
]

X_train = X_train.drop(
    columns=[c for c in leakage_columns if c in X_train.columns],
    errors="ignore"
)

# Get preprocessing and classifier
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get importance values
importance_values = classifier.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

# Save results
output_file = MODEL_DIR / "feature_importance.csv"
importance_df.to_csv(output_file, index=False)

# Display top 20
print("\n========== TOP 20 IMPORTANT FEATURES ==========")

print(
    importance_df.head(20).to_string(index=False)
)

print("\nFeature importance saved successfully!")
print("Saved at:", output_file)