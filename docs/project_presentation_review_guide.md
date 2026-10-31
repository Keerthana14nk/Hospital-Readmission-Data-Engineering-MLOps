# Hospital Readmission Presentation Review Guide (Slide-by-Slide)

This guide maps all 6 required review pillars directly to your project presentation slides for the viva panel.

---

### Slide 1: Title Slide
- **Title**: Hospital Readmission Analytics and Prediction Using Public Healthcare Data
- **Subtitle**: Privacy-Preserving Data Engineering & Production MLOps Pipeline
- **Student**: V. Keerthana | B.E. CSE (AI & ML) | CMR Institute of Technology

---

### Slide 2: 1. Problem Understanding and Design
- **Clinical Problem**: Avoidable 30-day hospital readmissions compromise patient outcomes and trigger financial penalties under CMS's Hospital Readmissions Reduction Program (HRRP).
- **Project Goal**: Build an automated end-to-end pipeline to predict 30-day readmissions (<30 days), identify clinical risk factors, and serve real-time predictions to clinical staff.
- **Key Challenges Solved**:
  - Acute 1:8 class imbalance (11.16% readmissions).
  - Strict HIPAA-compliant de-identification without losing patient multi-visit lineage.
  - Zero target data leakage.

---

### Slide 3: 2. Data Source and Ingestion
- **Dataset**: UCI Diabetes 130-US Hospitals (1999–2008), Dataset ID: 296, DOI: 10.24432/C5230J.
- **Scale**: 101,766 inpatient encounters across 130 medical centers.
- **Features (50)**: Demographics, admission/discharge details, 23 diabetes medications, lab procedures, and ICD-9 diagnosis codes.
- **Ingestion (`src.ingestion.ingest`)**: Automated retrieval via `ucimlrepo`, immutable raw storage in `data/raw/`, and execution metadata logging.

---

### Slide 4: 3. Proposed End-to-End Architecture
- **Phase 1 (Data Engineering)**: Ingestion → Staging → Transformation → Data Quality Gate → PostgreSQL Data Mart.
- **Phase 2 (MLOps)**: Leakage-Free Stratified Split → 3-Model Training Engine → MLflow Experiment Tracking → Automated Quality Gate → Model Registry → Post-Deployment Drift Monitoring.
- **Phase 3 (Serving & Consumption)**: Real-time FastAPI REST API (`:8000`) and interactive Streamlit Clinical Platform (`:8501`).

---

### Slide 5: 4. ETL, Validation, and Storage
- **ETL Transformation (`src.transform.clean`)**:
  - Missing-value harmonization (`?`, `Unknown` mapped to `pd.NA`).
  - Cryptographic surrogate key generation using SHA-256 (`patient_key`).
  - ICD-9 diagnosis mapping into 8 clinical groups (Circulatory, Respiratory, Diabetes, etc.).
- **Data Quality Validation Gate (`src.quality.validate`)**:
  - 4 automated checks: Encounter ID not null, Patient Key not null, Length of stay 1–14 days, Valid readmission class.
  - Rejected records quarantined into `ops.rejected_records`.
- **PostgreSQL Data Warehouse (`src.load.postgres`)**:
  - Multi-schema star architecture (`staging`, `mart`, `ops`).
  - Tables: `mart.admission_fact`, `mart.diagnosis_summary`, `mart.patient_summary`, `mart.readmission_summary`.

---

### Slide 6: 5. Model Development & Leakage Prevention
- **Target Leakage Elimination**: Dropped `readmitted`, `readmission_class`, `readmitted_within_30_days`, `age_years_midpoint`, and IDs before training.
- **Stratified 80/20 Train-Test Split**:
  - Train: 81,412 encounters | Test: 20,354 encounters (11.16% positive class preserved).
- **Preprocessing Pipeline**: Continuous features (median imputer + `StandardScaler`), Categorical features (frequent imputer + `OneHotEncoder`).

---

### Slide 7: 6. 3-Model Benchmarking & Comparison
- Evaluated on holdout test set (20,354 encounters):

| Model | Accuracy | Recall (Class 1) | Precision | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Balanced)** | 64.33% | 54.87% | 16.65% | 25.55% | 0.6439 | Linear Baseline |
| **Random Forest (Balanced)** | 63.09% | 58.92% | 16.90% | 26.26% | 0.6580 | Bagging Ensemble |
| **XGBoost (Scale Pos Weight)** | **64.54%** | **60.55%** | **17.86%** | **27.59%** | **0.6815** | 🏆 **PROMOTED CHAMPION** |

- **Why XGBoost Won**: Highest discrimination capacity (ROC-AUC 0.6815) and highest sensitivity (catches 60.55% of readmitted patients).

---

### Slide 8: 7. MLflow Tracking & Model Governance
- **MLflow Workflow (`sqlite:///mlflow.db`)**:
  - Experiment: `hospital_readmission_prediction`.
  - Automatically logs parameters, metrics, confusion matrices, and ROC/PR curves.
  - Interactive UI at `http://localhost:5000` via `run_mlflow_ui.bat`.
- **Automated Operational Quality Gate (`ml.validate_model_gate`)**:
  - Asserts $\text{ROC-AUC} \ge 0.60$ and $\text{Recall} \ge 40\%$ before promotion.
- **Continuous Drift Monitoring (`ml.monitor_drift`)**:
  - Calculates Kolmogorov-Smirnov test and Population Stability Index (PSI) to detect feature distribution shifts.

---

### Slide 9: 8. Visualization & Interpretation (Streamlit Platform)
- **Tab 1: Healthcare Analytics & BI**: 5 required clinical views (Readmission by Age, Diagnosis, Length of Stay, Admission Patterns, High-Risk Cohort).
- **Tab 2: 3-Model Benchmark & MLflow**: Overlaid ROC curves, PR curves, side-by-side confusion matrices, top 15 feature importances.
- **Tab 3: Real-Time Bedside Risk Predictor**: Clinician intake form with algorithm toggle, risk gauge (0–100%), and tailored discharge intervention bundles.
- **Tab 4: Model Registry & Drift Monitoring**: Production health status and PSI drift metrics.

---

### Slide 10: 9. Real-Time Model Serving (FastAPI)
- **REST API Endpoint (`ml.serve_api`)**:
  - `GET /`: Service health and champion model version.
  - `GET /model/metadata`: Model registry specs.
  - `POST /predict`: Real-time scoring returning risk probability, risk category (🟢 Low / 🟡 Moderate / 🔴 High), and clinical actions.
  - Interactive Swagger docs at `http://localhost:8000/docs`.

---

### Slide 11: 10. Conclusion & Future Scope
- **Conclusion**: Successfully developed a privacy-preserving healthcare data engineering and production MLOps system that unifies ETL, automated validation, dimensional storage, multi-model benchmarking, and real-time clinical risk scoring.
- **Future Scope**:
  - Integration of unstructured clinical physician discharge notes via Large Language Models (LLMs) and NLP.
  - Integration of post-discharge remote patient monitoring (RPM) and wearable telemetry.
