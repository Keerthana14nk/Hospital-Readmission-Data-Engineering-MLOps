@echo off
echo Starting MLflow Experiment Tracking UI...
echo Access the UI at: http://localhost:5000
echo Press Ctrl+C to stop the server.
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
