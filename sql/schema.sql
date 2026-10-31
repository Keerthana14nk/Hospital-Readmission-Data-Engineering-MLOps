CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS ops;

-- The Python loader creates/populates these tables.
-- This file documents the expected analytical schema.

-- mart.admission_fact
-- One row per valid hospital encounter.

-- mart.diagnosis_summary
-- Aggregated admission, 30-day readmission, and length-of-stay metrics by diagnosis group.

-- mart.patient_summary
-- Patient-safe aggregated summary using the surrogate patient_key.

-- mart.readmission_summary
-- Readmission class by age group.

-- ops.rejected_records
-- Validation failures with the rule that caused rejection.
