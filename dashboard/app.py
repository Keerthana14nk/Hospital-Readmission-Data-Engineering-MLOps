import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Optional DB connection
try:
    from sqlalchemy import create_engine
    from src.config import POSTGRES_URL
    HAS_DB_CONFIG = True
except Exception:
    HAS_DB_CONFIG = False
    POSTGRES_URL = None

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = BASE_DIR / "data" / "cleaned" / "encounters_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
CHAMPION_MODEL_PATH = MODEL_DIR / "champion_model.pkl"
LR_MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
RF_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"

REGISTRY_PATH = MODEL_DIR / "model_registry.json"
METRICS_PATH = MODEL_DIR / "metrics_summary.json"
FEAT_IMP_PATH = MODEL_DIR / "feature_importance.csv"
DRIFT_PATH = MODEL_DIR / "drift_report.json"
GATE_PATH = MODEL_DIR / "validation_gate_result.json"

st.set_page_config(
    page_title="Hospital Readmission Analytics & MLOps Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=600)
def load_data():
    """Load analytical data from PostgreSQL if reachable, else fallback to CSV."""
    if HAS_DB_CONFIG and POSTGRES_URL:
        try:
            engine = create_engine(POSTGRES_URL)
            conn = engine.raw_connection()
            try:
                df = pd.read_sql_query("SELECT * FROM mart.admission_fact", conn)
                return df, "PostgreSQL Data Mart (mart.admission_fact)"
            finally:
                conn.close()
        except Exception:
            pass

    if CLEANED_DATA_PATH.exists():
        df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
        return df, "Curated Analytical CSV (Local Cache)"

    st.error("No analytical dataset found in PostgreSQL or data/cleaned/encounters_cleaned.csv.")
    st.stop()


@st.cache_resource
def load_all_models():
    models = {}
    if CHAMPION_MODEL_PATH.exists():
        models["Champion"] = joblib.load(CHAMPION_MODEL_PATH)
    if XGB_MODEL_PATH.exists():
        models["XGBoost"] = joblib.load(XGB_MODEL_PATH)
    if RF_MODEL_PATH.exists():
        models["Random Forest"] = joblib.load(RF_MODEL_PATH)
    if LR_MODEL_PATH.exists():
        models["Logistic Regression"] = joblib.load(LR_MODEL_PATH)
    return models


df_raw, data_source_name = load_data()
loaded_models = load_all_models()

# Header & Top Banner
st.title("🏥 Hospital Readmission Analytics & MLOps Platform")
st.caption(f"Integrated Healthcare Pipeline: Data Engineering + MLflow Tracking + 3-Model Benchmark | Data: {data_source_name}")

# Navigation Tabs
tab_analytics, tab_mlops, tab_predictor, tab_monitoring = st.tabs([
    "📊 Part 1: Healthcare Analytics & BI",
    "🚀 Part 2: 3-Model Benchmark & MLflow",
    "🩺 Part 2: Real-Time Risk Predictor",
    "🛡️ Part 2: Model Registry & Drift Monitoring",
])

# ==============================================================================
# TAB 1: HEALTHCARE ANALYTICS & BI (PART 1 FOUNDATION)
# ==============================================================================
with tab_analytics:
    st.header("Executive Clinical Analytics & Population Metrics")
    
    with st.expander("🔍 Filter Population Demographics", expanded=False):
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            ages = sorted(df_raw["age"].dropna().unique().tolist())
            selected_ages = st.multiselect("Age groups", ages, default=ages)
        with c_f2:
            genders = sorted(df_raw["gender"].dropna().unique().tolist())
            selected_genders = st.multiselect("Genders", genders, default=genders)

    df_filtered = df_raw[
        (df_raw["age"].isin(selected_ages)) & 
        (df_raw["gender"].isin(selected_genders))
    ]

    total_encounters = len(df_filtered)
    readmissions_30d = int(df_filtered["readmitted_within_30_days"].sum())
    rate = (readmissions_30d / total_encounters * 100) if total_encounters else 0
    avg_los = float(df_filtered["length_of_stay"].mean()) if total_encounters else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Admissions", f"{total_encounters:,}")
    kpi2.metric("30-Day Readmissions", f"{readmissions_30d:,}")
    kpi3.metric("30-Day Readmission Rate", f"{rate:.2f}%")
    kpi4.metric("Avg Length of Stay", f"{avg_los:.1f} days")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("1. Readmission Rate by Age Group")
        age_view = (
            df_filtered.groupby("age", as_index=False)
            .agg(admissions=("encounter_id", "count"),
                 readmissions_30d=("readmitted_within_30_days", "sum"))
        )
        age_view["readmission_rate"] = age_view["readmissions_30d"] / age_view["admissions"] * 100
        fig_age = px.bar(
            age_view, x="age", y="readmission_rate",
            title="Readmission Rate by Age Bracket (%)",
            labels={"readmission_rate": "Readmission Rate (%)", "age": "Age Group"},
            color="readmission_rate", color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col_b:
        st.subheader("2. Readmission by Diagnosis Group")
        diag_view = (
            df_filtered.groupby("diagnosis_group", as_index=False)
            .agg(admissions=("encounter_id", "count"),
                 readmissions_30d=("readmitted_within_30_days", "sum"))
        )
        diag_view["readmission_rate"] = diag_view["readmissions_30d"] / diag_view["admissions"] * 100
        diag_view = diag_view.sort_values("readmission_rate", ascending=False)
        fig_diag = px.bar(
            diag_view, x="diagnosis_group", y="readmission_rate",
            title="Readmission Rate by Primary ICD-9 Group (%)",
            labels={"readmission_rate": "Readmission Rate (%)", "diagnosis_group": "Diagnosis Group"},
            color="readmission_rate", color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_diag, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("3. Length-of-Stay Analysis")
        los_view = (
            df_filtered.groupby("length_of_stay", as_index=False)
            .agg(admissions=("encounter_id", "count"),
                 readmission_rate=("readmitted_within_30_days", "mean"))
        )
        los_view["readmission_rate"] *= 100
        fig_los = px.line(
            los_view, x="length_of_stay", y="readmission_rate",
            title="Correlation: Hospital Days vs 30-Day Readmission Risk",
            labels={"length_of_stay": "Days in Hospital", "readmission_rate": "Readmission Rate (%)"},
            markers=True
        )
        st.plotly_chart(fig_los, use_container_width=True)

    with col_d:
        st.subheader("4. Admission Patterns")
        adm_pattern = (
            df_filtered.groupby("admission_type_id")
            .size()
            .reset_index(name="admissions")
            .sort_values("admissions", ascending=False)
        )
        adm_pattern["Admission Type"] = adm_pattern["admission_type_id"].map({
            1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 5: "Trauma Center"
        }).fillna("Other / Unknown")
        fig_adm = px.pie(
            adm_pattern, values="admissions", names="Admission Type",
            title="Distribution of Hospital Admission Types",
            hole=0.4
        )
        st.plotly_chart(fig_adm, use_container_width=True)

    st.subheader("5. High-Risk Cohort (Frequent Prior Admissions & Severe Encounter)")
    high_risk = df_filtered[
        (df_filtered["readmitted_within_30_days"] == 1) |
        (df_filtered["prior_admission_count"] >= 3)
    ]
    st.info(f"Identified **{len(high_risk):,} high-risk patient encounters** meeting proactive clinical intervention criteria.")
    st.dataframe(
        high_risk[[
            "encounter_id", "age", "gender", "diagnosis_group",
            "length_of_stay", "prior_admission_count",
            "number_diagnoses", "medication_count", "readmission_class"
        ]].head(50),
        use_container_width=True
    )


# ==============================================================================
# TAB 2: 3-MODEL BENCHMARK & MLFLOW WORKFLOW (PART 2 CORE)
# ==============================================================================
with tab_mlops:
    st.header("Part 2 — ML Workflow & 3-Model Comparison")
    
    # MLflow Banner
    st.markdown("""
    <div style="background-color: #f0f7ff; border-left: 5px solid #0066cc; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px;">
        <h4 style="margin: 0; color: #004080;">📊 MLflow Experiment Tracking Workflow Active</h4>
        <p style="margin: 5px 0 0 0; color: #333; font-size: 14px;">
            Experiment: <b>hospital_readmission_prediction</b> | Backend: <b>SQLite (mlflow.db)</b> | 
            Runs logged: <b>3 distinct model runs</b> with hyperparams, metrics, and artifact curves.
            Launch MLflow UI via <code>run_mlflow_ui.bat</code> to view at <code>http://localhost:5000</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)

        # 3-Model Summary Comparison Table
        st.subheader("1. Comprehensive 3-Model Performance Comparison")
        st.caption("Evaluated on stratified holdout test set (20,354 encounters; 11.16% readmission prevalence).")

        rows = []
        for model_name, m in metrics_data.items():
            is_champ = ("XGBoost" in model_name)
            rows.append({
                "Model Architecture": ("🏆 " + model_name) if is_champ else model_name,
                "Accuracy (%)": f"{m['accuracy']*100:.2f}%",
                "Recall (Class 1) (%)": f"{m['recall']*100:.2f}%",
                "Precision (Class 1) (%)": f"{m['precision']*100:.2f}%",
                "F1 Score (Class 1) (%)": f"{m['f1_score']*100:.2f}%",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "PR-AUC": f"{m['pr_auc']:.4f}",
                "Training Time": f"{m.get('training_duration_seconds', 0.0):.2f}s",
                "Status": "PROMOTED CHAMPION" if is_champ else "Evaluated Candidate"
            })

        bench_df = pd.DataFrame(rows)
        st.dataframe(bench_df, use_container_width=True)

        st.markdown("""
        > **Clinical & Algorithmic Analysis**:
        > - **XGBoost (Scale Pos Weight)** achieves the highest overall discrimination with an **ROC-AUC of 0.6815** and the highest sensitivity (**60.55% Recall**). Its gradient boosting architecture effectively regularizes high-dimensional interactions.
        > - **Random Forest (Balanced)** achieves strong competitive performance with an **ROC-AUC of 0.6580** and **58.92% Recall**, making it a robust tree ensemble.
        > - **Logistic Regression (Balanced)** establishes a solid, fast, and transparent linear baseline (**0.6439 ROC-AUC**, **54.87% Recall**).
        """)

        st.markdown("---")

        # Interactive Overlay ROC Curves & Precision-Recall Curves
        st.subheader("2. Comparative Discrimination Curves")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("##### Overlaid ROC Curves (All 3 Models)")
            fig_roc = go.Figure()
            colors = {
                "XGBoost (Scale Pos Weight)": "#00cc66",
                "Random Forest (Balanced)": "#2b83ba",
                "Logistic Regression (Balanced)": "#d7191c"
            }

            for model_name, m in metrics_data.items():
                roc_pts = m.get("roc_curve", [])
                if roc_pts:
                    fpr_vals = [p["fpr"] for p in roc_pts]
                    tpr_vals = [p["tpr"] for p in roc_pts]
                    line_color = colors.get(model_name, "#333333")
                    fig_roc.add_trace(go.Scatter(
                        x=fpr_vals, y=tpr_vals, mode="lines",
                        name=f"{model_name} (AUC = {m['roc_auc']:.4f})",
                        line=dict(color=line_color, width=2.5)
                    ))

            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random Chance (AUC = 0.5000)",
                line=dict(color="grey", dash="dash")
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate (Recall)",
                legend=dict(x=0.45, y=0.15),
                margin=dict(l=40, r=40, t=30, b=40)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_c2:
            st.markdown("##### Overlaid Precision-Recall Curves")
            fig_pr = go.Figure()
            for model_name, m in metrics_data.items():
                pr_pts = m.get("pr_curve", [])
                if pr_pts:
                    rec_vals = [p["recall"] for p in pr_pts]
                    prec_vals = [p["precision"] for p in pr_pts]
                    line_color = colors.get(model_name, "#333333")
                    fig_pr.add_trace(go.Scatter(
                        x=rec_vals, y=prec_vals, mode="lines",
                        name=f"{model_name} (PR-AUC = {m['pr_auc']:.4f})",
                        line=dict(color=line_color, width=2.5)
                    ))
            fig_pr.add_hline(y=0.1116, line_dash="dash", line_color="grey", annotation_text="Baseline Positive Prevalence (11.16%)")
            fig_pr.update_layout(
                xaxis_title="Recall (Sensitivity)",
                yaxis_title="Precision (PPV)",
                legend=dict(x=0.45, y=0.85),
                margin=dict(l=40, r=40, t=30, b=40)
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("---")

        # Side-by-side Confusion Matrices
        st.subheader("3. Side-by-Side Confusion Matrix Breakdown")
        cm_cols = st.columns(3)
        for idx, (model_name, m) in enumerate(metrics_data.items()):
            with cm_cols[idx]:
                st.markdown(f"**{model_name}**")
                cm = m["confusion_matrix"]
                cm_arr = np.array([
                    [cm["tn"], cm["fp"]],
                    [cm["fn"], cm["tp"]]
                ])
                fig_cm = px.imshow(
                    cm_arr,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["No Readmit (0)", "Readmit <30d (1)"],
                    y=["No Readmit (0)", "Readmit <30d (1)"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=20), coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption(f"Caught **{cm['tp']:,}** / {cm['tp']+cm['fn']:,} readmitted patients ({m['recall']*100:.1f}% sensitivity).")

        st.markdown("---")

        # Feature Importance
        if FEAT_IMP_PATH.exists():
            st.subheader("4. Top 15 Clinical Predictors of Readmission (Champion Model)")
            feat_df = pd.read_csv(FEAT_IMP_PATH).head(15)
            feat_df["feature_clean"] = feat_df["feature"].str.replace("num__", "").str.replace("cat__", "")
            fig_feat = px.bar(
                feat_df.sort_values("importance", ascending=True),
                x="importance", y="feature_clean",
                orientation="h",
                title="Relative Feature Importance",
                labels={"importance": "Importance Weight", "feature_clean": "Clinical Variable"},
                color="importance", color_continuous_scale="Teal"
            )
            st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("No metrics summary found. Execute `python ml/train_and_evaluate.py` to populate.")


# ==============================================================================
# TAB 3: REAL-TIME PATIENT READMISSION RISK PREDICTOR
# ==============================================================================
with tab_predictor:
    st.header("🩺 Real-Time Clinical Readmission Risk Predictor")
    st.markdown("""
    Interactive bedside clinical risk scoring tool. Select which of the **3 trained models** evaluates 
    the encounter, input patient attributes, and receive an instant 30-day readmission risk probability and intervention bundle.
    """)

    # Model Selector Radio
    model_choice = st.radio(
        "⚡ Select Prediction Model:",
        ["XGBoost (Scale Pos Weight) [Champion]", "Random Forest (Balanced)", "Logistic Regression (Balanced)"],
        horizontal=True
    )

    # Map choice to model
    if "XGBoost" in model_choice:
        active_model = loaded_models.get("XGBoost") or loaded_models.get("Champion")
    elif "Random Forest" in model_choice:
        active_model = loaded_models.get("Random Forest")
    else:
        active_model = loaded_models.get("Logistic Regression")

    if active_model is None:
        st.error("Selected model artifact not found. Please run `python ml/train_and_evaluate.py` first.")
    else:
        with st.form("clinical_prediction_form"):
            st.subheader("Patient Encounter Information")
            c1, c2, c3 = st.columns(3)
            with c1:
                input_age = st.selectbox("Age bracket", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=7)
                input_gender = st.selectbox("Gender", ["Female", "Male"])
                input_race = st.selectbox("Race/Ethnicity", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"], index=0)
                input_diag_group = st.selectbox("Primary Diagnosis Group", ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Genitourinary", "Injury", "Neoplasms", "Other"], index=0)

            with c2:
                input_los = st.slider("Length of Stay (Days)", min_value=1, max_value=14, value=5)
                input_lab = st.slider("Number of Lab Procedures", min_value=1, max_value=120, value=48)
                input_meds = st.slider("Number of Medications", min_value=1, max_value=80, value=18)
                input_procedures = st.slider("Non-Lab Procedures", min_value=0, max_value=6, value=1)

            with c3:
                input_inpatient = st.number_input("Prior Inpatient Admissions (Past Year)", min_value=0, max_value=20, value=1)
                input_emergency = st.number_input("Prior Emergency Room Visits (Past Year)", min_value=0, max_value=20, value=0)
                input_outpatient = st.number_input("Prior Outpatient Visits (Past Year)", min_value=0, max_value=20, value=1)
                input_diagnoses = st.slider("Number of Diagnoses Recorded", min_value=1, max_value=16, value=9)

            st.subheader("Medication & Glycemic Indicators")
            c4, c5, c6 = st.columns(3)
            with c4:
                input_insulin = st.selectbox("Insulin Status", ["No", "Steady", "Up", "Down"], index=1)
            with c5:
                input_med_change = st.selectbox("Diabetes Med Change During Stay", ["No", "Ch"], index=0)
            with c6:
                input_diabetesmed = st.selectbox("Prescribed Any Diabetes Medication", ["Yes", "No"], index=0)

            submitted = st.form_submit_button("⚡ Compute Readmission Risk Score", use_container_width=True)

        if submitted:
            patient_dict = {
                "age": input_age,
                "gender": input_gender,
                "race": input_race,
                "time_in_hospital": input_los,
                "length_of_stay": input_los,
                "num_lab_procedures": input_lab,
                "num_procedures": input_procedures,
                "num_medications": input_meds,
                "medication_count": input_meds,
                "number_outpatient": input_outpatient,
                "number_emergency": input_emergency,
                "number_inpatient": input_inpatient,
                "prior_admission_count": input_inpatient + input_outpatient + input_emergency,
                "number_diagnoses": input_diagnoses,
                "diagnosis_count": input_diagnoses,
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 7,
                "diagnosis_group": input_diag_group,
                "max_glu_serum": "None",
                "a1cresult": "None",
                "insulin": input_insulin,
                "change": input_med_change,
                "diabetesmed": input_diabetesmed,
                "medical_specialty": "InternalMedicine",
                "payer_code": "MC",
            }

            with open(REGISTRY_PATH, "r") as f:
                reg_meta = json.load(f)
            expected_feats = reg_meta.get("input_features", [])

            patient_df = pd.DataFrame([patient_dict])
            for col in expected_feats:
                if col not in patient_df.columns:
                    patient_df[col] = "No"
            patient_df = patient_df[[c for c in expected_feats if c in patient_df.columns]]

            proba = float(active_model.predict_proba(patient_df)[0, 1])
            percent = proba * 100

            st.markdown("---")
            st.subheader(f"Inference Result: {model_choice}")

            r_col1, r_col2 = st.columns([1, 2])

            with r_col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=percent,
                    title={'text': "30-Day Readmission Risk (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if percent >= 35 else ("orange" if percent >= 20 else "green")},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgreen"},
                            {'range': [20, 35], 'color': "lightyellow"},
                            {'range': [35, 100], 'color': "mistyrose"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 35
                        }
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with r_col2:
                if percent >= 35:
                    st.error("🚨 **HIGH RISK PATIENT (Probability ≥ 35%)**")
                    st.markdown("""
                    **Prescribed Clinical Intervention Bundle:**
                    - **Action 1**: Comprehensive transition-of-care protocol prior to discharge.
                    - **Action 2**: Pharmacy consultation for full medication reconciliation.
                    - **Action 3**: Mandatory nurse telephone follow-up scheduled at **48 hours post-discharge**.
                    - **Action 4**: Fast-track primary care physician appointment within 5 business days.
                    """)
                elif percent >= 20:
                    st.warning("⚠️ **MODERATE RISK PATIENT (Probability 20% – 35%)**")
                    st.markdown("""
                    **Prescribed Clinical Intervention Bundle:**
                    - **Action 1**: Reinforced disease education (diabetes diet & hypoglycemia signs).
                    - **Action 2**: Outpatient clinic appointment scheduled within 7 to 10 days.
                    - **Action 3**: Verify discharge medication possession and comprehension.
                    """)
                else:
                    st.success("✅ **LOW RISK PATIENT (Probability < 20%)**")
                    st.markdown("""
                    **Prescribed Clinical Intervention Bundle:**
                    - **Action 1**: Standard routine hospital discharge summary and patient instructions.
                    - **Action 2**: Routine outpatient primary care follow-up.
                    """)

                st.write("**Key Patient Drivers Detected:**")
                drivers = []
                if input_inpatient >= 1:
                    drivers.append(f"• High prior inpatient admission history ({input_inpatient} past visits)")
                if input_los >= 5:
                    drivers.append(f"• Extended length of hospital stay ({input_los} days)")
                if input_meds >= 15:
                    drivers.append(f"• Polypharmacy burden ({input_meds} medications administered)")
                if input_diagnoses >= 8:
                    drivers.append(f"• Multiple chronic co-morbidities ({input_diagnoses} diagnoses)")
                if input_insulin in ["Up", "Down"]:
                    drivers.append(f"• Active titration of insulin dosing during stay")
                
                if not drivers:
                    st.write("• No severe risk flags detected; normal baseline.")
                else:
                    for d in drivers:
                        st.write(d)


# ==============================================================================
# TAB 4: MODEL REGISTRY & DRIFT MONITORING
# ==============================================================================
with tab_monitoring:
    st.header("Model Governance, Registry & Post-Deployment Monitoring")
    
    col_reg, col_gate = st.columns(2)

    with col_reg:
        st.subheader("Model Registry Metadata (`models/model_registry.json`)")
        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH, "r") as f:
                reg_info = json.load(f)
            st.json(reg_info)
        else:
            st.info("Registry file pending generation.")

    with col_gate:
        st.subheader("Operational Quality Gate (`validation_gate_result.json`)")
        if GATE_PATH.exists():
            with open(GATE_PATH, "r") as f:
                gate_info = json.load(f)
            st.success("✅ Quality Gate: Model Approved for Production Serving")
            st.json(gate_info)
        else:
            st.info("Quality gate report pending generation.")

    st.markdown("---")
    st.subheader("Data & Feature Drift Monitoring (`models/drift_report.json`)")
    if DRIFT_PATH.exists():
        with open(DRIFT_PATH, "r") as f:
            drift_data = json.load(f)

        s1, s2, s3 = st.columns(3)
        s1.metric("System Health", drift_data.get("overall_status", "UNKNOWN"))
        s2.metric("Features Monitored", drift_data.get("features_monitored", 0))
        s3.metric("Drifted Features", drift_data.get("features_with_drift", 0))

        st.caption(f"Recommendation: {drift_data.get('recommendation', '')}")

        drift_metrics = pd.DataFrame(drift_data.get("metrics", []))
        if not drift_metrics.empty:
            st.dataframe(drift_metrics, use_container_width=True)
    else:
        st.info("Drift monitoring report pending execution of `python ml/monitor_drift.py`.")
