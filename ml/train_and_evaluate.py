"""
Hospital Readmission Prediction — Part 2 MLOps Workflow & 3-Model Comparison.
Integrated with MLflow Experiment Tracking, Model Registry, and Evaluation Curves.
Trains and compares:
  1. Logistic Regression (Balanced)
  2. Random Forest (Balanced)
  3. XGBoost Classifier (Scale Pos Weight)
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "ml"
MODEL_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = MODEL_DIR / "artifacts"
MLRUNS_DIR = BASE_DIR / "mlruns"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv", low_memory=False)
    X_test = pd.read_csv(DATA_DIR / "X_test.csv", low_memory=False)
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    columns_to_remove = [
        "encounter_id",
        "patient_key",
        "readmitted",
        "readmission_class",
        "readmitted_within_30_days",
        "age_years_midpoint",
    ]

    X_train = X_train.drop(
        columns=[c for c in columns_to_remove if c in X_train.columns],
        errors="ignore",
    )
    X_test = X_test.drop(
        columns=[c for c in columns_to_remove if c in X_test.columns],
        errors="ignore",
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    numerical_columns = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_columns),
            ("cat", cat_pipeline, categorical_columns),
        ]
    )

    return preprocessor, numerical_columns, categorical_columns


def evaluate_predictions(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ROC & PR curves
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)

    # Downsample curve points for clean JSON storage and Plotly rendering
    step_roc = max(1, len(fpr) // 60)
    step_pr = max(1, len(prec_curve) // 60)

    roc_points = [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
        for i in range(0, len(fpr), step_roc)
    ]
    # Ensure end point (1, 1) is present
    if roc_points[-1]["fpr"] < 1.0:
        roc_points.append({"fpr": 1.0, "tpr": 1.0})

    pr_points = [
        {"recall": round(float(rec_curve[i]), 4), "precision": round(float(prec_curve[i]), 4)}
        for i in range(0, len(prec_curve), step_pr)
    ]

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "roc_curve": roc_points,
        "pr_curve": pr_points,
    }


def analyze_thresholds(y_true, y_proba):
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        results.append({
            "threshold": t,
            "recall": round(float(rec), 4),
            "precision": round(float(prec), 4),
            "f1": round(float(f1), 4),
        })
    return results


def save_confusion_matrix_plot(cm, model_name, filepath):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        cmap="Blues", interpolation="nearest"
    )
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0 (No)", "Pred 1 (<30d)"])
    ax.set_yticklabels(["Actual 0 (No)", "Actual 1 (<30d)"])
    ax.set_title(f"Confusion Matrix: {model_name}")

    vals = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{vals[i][j]:,}", ha="center", va="center", color="black" if vals[i][j] < cm["tn"]/2 else "white")
    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    plt.close(fig)


def save_roc_plot(fpr, tpr, roc_auc, model_name, filepath):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve: {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    plt.close(fig)


def main():
    print("=" * 70)
    print("HOSPITAL READMISSION MLOPS WORKFLOW & 3-MODEL BENCHMARKING (MLflow)")
    print("=" * 70)

    # 1. Load data
    print("\n[Step 1/7] Loading train and test datasets...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    scale_pos_weight = round(num_neg / num_pos, 2)
    print(f"Target Distribution: Negative = {num_neg:,}, Positive = {num_pos:,} (Ratio 1:{scale_pos_weight:.2f})")

    # 2. Build Preprocessor
    print("\n[Step 2/7] Constructing feature preprocessing pipeline...")
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    print(f"Features: {len(num_cols)} numerical, {len(cat_cols)} categorical")

    # 3. Setup MLflow Experiment
    print("\n[Step 3/7] Initializing MLflow tracking workflow...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    db_path = (BASE_DIR / "mlflow.db").as_posix()
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "hospital_readmission_prediction"
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"MLflow Experiment: {experiment_name}")

    # 4. Define the 3 Models
    models_to_benchmark = {
        "Logistic Regression (Balanced)": {
            "clf": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs"
            ),
            "type": "linear",
            "params": {"solver": "lbfgs", "class_weight": "balanced", "max_iter": 1000}
        },
        "Random Forest (Balanced)": {
            "clf": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                max_depth=12,
                min_samples_leaf=5
            ),
            "type": "ensemble_bagging",
            "params": {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 5, "class_weight": "balanced"}
        },
        "XGBoost (Scale Pos Weight)": {
            "clf": XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ),
            "type": "gradient_boosting",
            "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.08, "scale_pos_weight": scale_pos_weight}
        }
    }

    results = {}
    fitted_pipelines = {}

    # 5. Train, Log to MLflow, and Benchmark
    print("\n[Step 4/7] Training, Logging & Benchmarking the 3 Models...")

    for model_name, info in models_to_benchmark.items():
        print(f"\n" + "-" * 50)
        print(f"  Training Model: {model_name}")
        print("-" * 50)

        t0 = time.time()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", info["clf"])
        ])

        with mlflow.start_run(run_name=model_name):
            # Log params
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", info["type"])
            for p_key, p_val in info["params"].items():
                mlflow.log_param(p_key, p_val)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            # Train
            pipeline.fit(X_train, y_train)
            train_duration = round(time.time() - t0, 2)
            mlflow.log_metric("training_duration_seconds", train_duration)

            # Predict
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            # Evaluate
            eval_metrics = evaluate_predictions(y_test, y_pred, y_proba)
            eval_metrics["training_duration_seconds"] = train_duration
            eval_metrics["threshold_analysis"] = analyze_thresholds(y_test, y_proba)

            # Log metrics to MLflow
            mlflow.log_metric("accuracy", eval_metrics["accuracy"])
            mlflow.log_metric("precision_class_1", eval_metrics["precision"])
            mlflow.log_metric("recall_class_1", eval_metrics["recall"])
            mlflow.log_metric("f1_score_class_1", eval_metrics["f1_score"])
            mlflow.log_metric("roc_auc", eval_metrics["roc_auc"])
            mlflow.log_metric("pr_auc", eval_metrics["pr_auc"])
            mlflow.log_metric("specificity", eval_metrics["specificity"])

            # Generate Artifacts
            safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            cm_plot_path = ARTIFACTS_DIR / f"{safe_name}_confusion_matrix.png"
            roc_plot_path = ARTIFACTS_DIR / f"{safe_name}_roc_curve.png"

            save_confusion_matrix_plot(eval_metrics["confusion_matrix"], model_name, cm_plot_path)
            fpr_raw, tpr_raw, _ = roc_curve(y_test, y_proba)
            save_roc_plot(fpr_raw, tpr_raw, eval_metrics["roc_auc"], model_name, roc_plot_path)

            mlflow.log_artifact(str(cm_plot_path), artifact_path="plots")
            mlflow.log_artifact(str(roc_plot_path), artifact_path="plots")

            # Log tags
            mlflow.set_tag("developer", "Keerthana")
            mlflow.set_tag("dataset", "UCI Diabetes 130-US Hospitals")
            mlflow.set_tag("task", "30-Day Readmission Risk")

            fitted_pipelines[model_name] = pipeline
            results[model_name] = eval_metrics

            print(f"  [+] Accuracy : {eval_metrics['accuracy']*100:.2f}%")
            print(f"  [+] Recall   : {eval_metrics['recall']*100:.2f}% (Catches readmitted patients)")
            print(f"  [+] Precision: {eval_metrics['precision']*100:.2f}%")
            print(f"  [+] F1-Score : {eval_metrics['f1_score']*100:.2f}%")
            print(f"  [+] ROC-AUC  : {eval_metrics['roc_auc']:.4f}")
            print(f"  [+] PR-AUC   : {eval_metrics['pr_auc']:.4f}")
            print(f"  [+] MLflow Run Logged in {train_duration:.2f}s")

    # 6. Champion Selection
    print("\n[Step 5/7] Selecting Champion Model based on ROC-AUC & Recall...")
    # Select highest ROC-AUC amongst models with recall >= 0.40
    champion_name = max(
        results.keys(),
        key=lambda k: (results[k]["roc_auc"] if results[k]["recall"] >= 0.40 else 0)
    )
    print(f"  => Champion Model: {champion_name}")
    print(f"     ROC-AUC: {results[champion_name]['roc_auc']:.4f} | Recall: {results[champion_name]['recall']*100:.2f}%")

    champion_pipeline = fitted_pipelines[champion_name]

    # 7. Extract Feature Importances
    print("\n[Step 6/7] Extracting global clinical feature importances...")
    preprocessor_step = champion_pipeline.named_steps["preprocessor"]
    transformed_feature_names = preprocessor_step.get_feature_names_out()
    classifier_step = champion_pipeline.named_steps["classifier"]

    if hasattr(classifier_step, "feature_importances_"):
        importances = classifier_step.feature_importances_
    elif hasattr(classifier_step, "coef_"):
        importances = np.abs(classifier_step.coef_[0])
    else:
        importances = np.zeros(len(transformed_feature_names))

    feat_df = pd.DataFrame({
        "feature": transformed_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    feat_csv_path = MODEL_DIR / "feature_importance.csv"
    feat_df.to_csv(feat_csv_path, index=False)
    print(f"Saved feature importances to: {feat_csv_path}")

    # Save models to disk
    joblib.dump(champion_pipeline, MODEL_DIR / "champion_model.pkl")
    joblib.dump(fitted_pipelines["Logistic Regression (Balanced)"], MODEL_DIR / "logistic_regression_model.pkl")
    joblib.dump(fitted_pipelines["Random Forest (Balanced)"], MODEL_DIR / "random_forest_model.pkl")
    joblib.dump(fitted_pipelines["XGBoost (Scale Pos Weight)"], MODEL_DIR / "xgboost_model.pkl")

    # 8. Save Metrics Summary & Model Registry
    print("\n[Step 7/7] Exporting MLOps Registry & Metrics Summary...")
    metrics_summary_path = MODEL_DIR / "metrics_summary.json"
    with open(metrics_summary_path, "w") as f:
        json.dump(results, f, indent=2)

    registry = {
        "model_name": "Hospital_Readmission_Classifier",
        "champion_model": champion_name,
        "version": "2.0.0",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "binary_classification",
        "target": "readmitted_within_30_days",
        "mlflow_experiment": experiment_name,
        "num_features_in": int(X_train.shape[1]),
        "num_features_transformed": int(len(transformed_feature_names)),
        "input_features": list(X_train.columns),
        "validation_status": "APPROVED",
        "models_benchmarked": list(results.keys()),
        "performance": {
            "champion_model": champion_name,
            "roc_auc": results[champion_name]["roc_auc"],
            "pr_auc": results[champion_name]["pr_auc"],
            "accuracy": results[champion_name]["accuracy"],
            "recall": results[champion_name]["recall"],
            "precision": results[champion_name]["precision"],
            "f1_score": results[champion_name]["f1_score"],
        },
        "comparison_table": [
            {
                "model": m_name,
                "roc_auc": results[m_name]["roc_auc"],
                "recall": results[m_name]["recall"],
                "precision": results[m_name]["precision"],
                "f1_score": results[m_name]["f1_score"],
                "accuracy": results[m_name]["accuracy"],
                "is_champion": bool(m_name == champion_name)
            }
            for m_name in results.keys()
        ],
        "recommended_clinical_threshold": 0.20,
        "artifact_path": "models/champion_model.pkl"
    }

    registry_path = MODEL_DIR / "model_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print("\n" + "=" * 70)
    print("3-MODEL COMPARISON SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model Architecture':<32} | {'Accuracy':<9} | {'Recall':<8} | {'Precision':<9} | {'F1':<7} | {'ROC-AUC':<8} | {'Status'}")
    print("-" * 90)
    for m_name, m_res in results.items():
        status = "CHAMPION" if m_name == champion_name else "Candidate"
        print(f"{m_name:<32} | {m_res['accuracy']*100:.2f}%   | {m_res['recall']*100:.2f}%  | {m_res['precision']*100:.2f}%    | {m_res['f1_score']*100:.2f}% | {m_res['roc_auc']:.4f}   | {status}")
    print("-" * 90)
    print(f"MLflow experiment tracked at: {MLRUNS_DIR}")
    print("Run `mlflow ui` or `run_mlflow_ui.bat` to view the interactive MLflow experiment dashboard.")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
