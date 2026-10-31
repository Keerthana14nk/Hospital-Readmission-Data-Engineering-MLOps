@echo off
echo Starting Hospital Readmission FastAPI Serving Service...
echo Swagger Docs available at: http://localhost:8000/docs
uvicorn ml.serve_api:app --host 0.0.0.0 --port 8000
