"""
FastAPI Model Serving Endpoint for Hospital Readmission Risk Prediction.
Provides real-time scoring, input validation, risk stratification, and explainable recommendations.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "champion_model.pkl"
REGISTRY_PATH = MODEL_DIR / "model_registry.json"

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="MLOps Real-Time Patient 30-Day Readmission Risk Stratification Service",
    version="1.0.0",
)

# Load model and registry on startup
model = None
registry_metadata = {}

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            registry_metadata = json.load(f)
except Exception as e:
    print(f"Error loading model artifacts: {e}")


class PatientEncounterInput(BaseModel):
    age: str = Field(default="[70-80)", description="Age bracket e.g. [70-80)")
    gender: str = Field(default="Female", description="Gender (Female/Male)")
    race: str = Field(default="Caucasian", description="Race or ethnicity")
    time_in_hospital: int = Field(default=4, ge=1, le=14, description="Length of stay in days (1-14)")
    num_lab_procedures: int = Field(default=45, ge=0, le=150, description="Total lab procedures performed")
    num_procedures: int = Field(default=1, ge=0, le=20, description="Number of operating or other procedures")
    num_medications: int = Field(default=15, ge=1, le=100, description="Distinct medications administered")
    number_outpatient: int = Field(default=0, ge=0, description="Prior outpatient visits in past year")
    number_emergency: int = Field(default=0, ge=0, description="Prior emergency visits in past year")
    number_inpatient: int = Field(default=1, ge=0, description="Prior inpatient admissions in past year")
    number_diagnoses: int = Field(default=9, ge=1, le=25, description="Number of diagnoses recorded")
    admission_type_id: int = Field(default=1, description="Admission type code (1=Emergency, 2=Urgent, 3=Elective)")
    discharge_disposition_id: int = Field(default=1, description="Discharge disposition code (1=Home)")
    admission_source_id: int = Field(default=7, description="Admission source code (7=Emergency Room)")
    diagnosis_group: str = Field(default="Circulatory", description="Primary diagnosis category (Circulatory, Diabetes, etc.)")
    max_glu_serum: str = Field(default="None", description="Glucose serum test result")
    a1cresult: str = Field(default="None", description="HbA1c test result")
    insulin: str = Field(default="No", description="Insulin prescription status (No, Steady, Up, Down)")
    change: str = Field(default="No", description="Change in diabetic medication (No, Ch)")
    diabetesmed: str = Field(default="Yes", description="Any diabetic medication prescribed (Yes, No)")
    medical_specialty: str = Field(default="InternalMedicine", description="Admitting physician specialty")
    payer_code: str = Field(default="MC", description="Payer insurance code")


class PredictionResponse(BaseModel):
    prediction: int = Field(description="Binary prediction: 1 (Readmitted <30 days), 0 (No)")
    readmission_probability: float = Field(description="Probability of 30-day readmission (0.0 to 1.0)")
    risk_category: str = Field(description="Clinical risk tier: Low, Moderate, or High")
    recommended_action: str = Field(description="Clinical intervention recommendation")
    contributing_factors: List[str] = Field(description="Key encounter features contributing to the risk score")


def prepare_input_dataframe(encounter: PatientEncounterInput) -> pd.DataFrame:
    data = encounter.model_dump()
    df = pd.DataFrame([data])

    # Derive engineered features
    df["length_of_stay"] = df["time_in_hospital"]
    df["prior_admission_count"] = (
        df["number_inpatient"] + df["number_outpatient"] + df["number_emergency"]
    )
    df["diagnosis_count"] = df["number_diagnoses"]
    df["medication_count"] = df["num_medications"]

    # Align with expected training features
    expected_cols = registry_metadata.get("input_features", [])
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "No"  # Default for medication indicator columns

    # Keep exactly the features expected by the training pipeline
    if expected_cols:
        df = df[[c for c in expected_cols if c in df.columns]]

    return df


@app.get("/")
def root():
    return {
        "service": "Hospital Readmission Prediction MLOps API",
        "status": "online",
        "version": registry_metadata.get("version", "1.0.0"),
        "champion_model": registry_metadata.get("champion_model", "Random Forest (Balanced)"),
        "docs_url": "/docs"
    }


@app.get("/model/metadata")
def model_metadata():
    if not registry_metadata:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    return registry_metadata


@app.post("/predict", response_model=PredictionResponse)
def predict_encounter(encounter: PatientEncounterInput):
    global model
    if model is None:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=503, detail="Champion model artifact not found.")

    df_input = prepare_input_dataframe(encounter)

    proba = float(model.predict_proba(df_input)[0, 1])
    
    # Clinical decision threshold
    clinical_threshold = registry_metadata.get("recommended_clinical_threshold", 0.20)
    pred = 1 if proba >= clinical_threshold else 0

    if proba >= 0.35:
        risk_category = "High Risk"
        action = "High Priority: Initiate discharge bundle, schedule 48h nurse telephone follow-up, and conduct medication reconciliation."
    elif proba >= 0.20:
        risk_category = "Moderate Risk"
        action = "Moderate Priority: Schedule outpatient clinic visit within 7 days and verify patient comprehension of discharge plan."
    else:
        risk_category = "Low Risk"
        action = "Standard Routine Care: Standard discharge instructions and routine primary care follow-up."

    # Identify patient-specific drivers
    factors = []
    if encounter.number_inpatient >= 1:
        factors.append(f"History of prior inpatient admissions ({encounter.number_inpatient} in past year)")
    if encounter.time_in_hospital >= 5:
        factors.append(f"Extended length of stay ({encounter.time_in_hospital} days)")
    if encounter.num_medications >= 15:
        factors.append(f"High polypharmacy load ({encounter.num_medications} medications)")
    if encounter.num_lab_procedures >= 50:
        factors.append(f"High diagnostic complexity ({encounter.num_lab_procedures} lab tests)")
    if encounter.number_diagnoses >= 8:
        factors.append(f"Multiple chronic comorbidities ({encounter.number_diagnoses} diagnoses)")
    if not factors:
        factors.append("Low overall clinical risk indicators")

    return PredictionResponse(
        prediction=pred,
        readmission_probability=round(proba, 4),
        risk_category=risk_category,
        recommended_action=action,
        contributing_factors=factors
    )
