@echo off
echo =========================================================================
echo Hospital Readmission: Full Integrated Data Engineering + MLOps Pipeline
echo =========================================================================

python -m src.pipeline --stage all

echo =========================================================================
echo Pipeline Execution Complete. Launching Streamlit Healthcare Platform...
echo =========================================================================
streamlit run dashboard/app.py
