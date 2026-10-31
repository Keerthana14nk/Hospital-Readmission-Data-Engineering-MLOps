import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# ==============================
# 1. LOAD DATA
# ==============================

X_train = pd.read_csv("data/ml/X_train.csv", low_memory=False)
X_test = pd.read_csv("data/ml/X_test.csv", low_memory=False)
y_train = pd.read_csv("data/ml/y_train.csv").squeeze()
y_test = pd.read_csv("data/ml/y_test.csv").squeeze()

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==============================
# 2. REMOVE LEAKAGE AND ID COLUMNS
# ==============================

columns_to_remove = [
    "readmitted",
    "readmission_class",
    "readmitted_within_30_days",
    "age_years_midpoint",
    "encounter_id",
    "patient_key"
]

X_train = X_train.drop(
    columns=[c for c in columns_to_remove if c in X_train.columns]
)

X_test = X_test.drop(
    columns=[c for c in columns_to_remove if c in X_test.columns]
)

print("Features after cleaning:", X_train.shape[1])


# ==============================
# 3. IDENTIFY COLUMN TYPES
# ==============================

numerical_columns = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("Numerical columns:", len(numerical_columns))
print("Categorical columns:", len(categorical_columns))


# ==============================
# 4. PREPROCESSING
# ==============================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


# ==============================
# 5. RANDOM FOREST MODEL
# ==============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])


# ==============================
# 6. TRAIN
# ==============================

print("\nTraining Random Forest...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# ==============================
# 7. PREDICTION
# ==============================

y_pred = pipeline.predict(X_test)


# ==============================
# 8. EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== FINAL MODEL RESULTS ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred, zero_division=0))


# ==============================
# 9. SAVE MODEL
# ==============================

os.makedirs("models", exist_ok=True)

import joblib

joblib.dump(
    pipeline,
    "models/random_forest_model.pkl"
)

print("\nModel saved successfully!")
print("Saved at: models/random_forest_model.pkl")