@echo off
echo ==========================================================
echo Starting Hospital Readmission Part 2 MLOps Pipeline...
echo ==========================================================

echo [1/4] Splitting feature dataset...
python -m ml.split_data

echo [2/4] Training and benchmarking models...
python -m ml.train_and_evaluate

echo [3/4] Running automated model validation gate...
python -m ml.validate_model_gate

echo [4/4] Running data drift monitoring...
python -m ml.monitor_drift

echo ==========================================================
echo MLOps Pipeline Complete. Launching Streamlit Platform...
echo ==========================================================
streamlit run dashboard/app.py
