# Hospital Readmission Analytics and Prediction

## 1. Introduction
Explain the healthcare readmission problem and project objective.

## 2. Problem Statement
Build a privacy-conscious data pipeline for hospital admission/readmission analytics using anonymized public data.

## 3. Dataset and Source
Describe the UCI Diabetes 130-US Hospitals dataset, its time period, record type, target/readmission labels, and source/access instructions.

## 4. System Architecture
Insert the architecture diagram and explain source, ingestion, raw, staging, validation, cleaned, PostgreSQL, and dashboard layers.

## 5. Data Ingestion
Explain Python/Pandas ingestion, extraction metadata, repeatability, and error handling.

## 6. ETL and Data Cleaning
Discuss missing-value handling, duplicate removal, invalid encounter handling, surrogate patient keys, diagnosis grouping, length-of-stay, prior-admission, diagnosis-count, and medication-count features.

## 7. Database Design
Describe PostgreSQL schemas:
- staging
- mart
- ops

Explain the analytical tables.

## 8. Data Quality
Document validation rules and rejected-record logging. Add screenshots.

## 9. Dashboard
Explain the five dashboard views:
1. Readmission rate by age group
2. Readmission by diagnosis
3. Length-of-stay analysis
4. Admission/discharge patterns
5. High-risk cohort

Add screenshots and interpretations.

## 10. End-to-End Execution
Show:
Dataset → ingestion → Airflow → staging → validation → cleaned data → PostgreSQL → Streamlit.

## 11. Privacy and Limitations
Explain that the dataset is public/de-identified and that the project does not add personal information. Discuss limitations of retrospective public data and broad diagnosis grouping.

## 12. Conclusion
Summarize what the pipeline achieves and how Part 2 could extend it with MLOps.

## References
1. UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999–2008.
2. Course assignment document.
