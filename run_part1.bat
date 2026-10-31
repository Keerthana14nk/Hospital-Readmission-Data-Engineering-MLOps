@echo off
python -m src.ingestion.ingest
python -m src.transform.clean
python -m src.quality.validate
python -m src.load.postgres
streamlit run dashboard/app.py
