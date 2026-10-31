# Data Dictionary — Part 1

| Field | Meaning / use |
|---|---|
| encounter_id | Unique encounter identifier from source |
| patient_key | Deterministic privacy-safe surrogate patient key |
| race | Source demographic category |
| gender | Source gender category |
| age | Source age band |
| age_years_midpoint | Numeric midpoint derived from age band |
| admission_type_id | Source admission type code |
| discharge_disposition_id | Source discharge disposition code |
| admission_source_id | Source admission source code |
| time_in_hospital | Source length of stay in days |
| length_of_stay | Engineered copy of time_in_hospital |
| num_lab_procedures | Number of lab procedures |
| num_procedures | Number of procedures |
| num_medications | Number of medications |
| number_outpatient | Outpatient visits before encounter |
| number_emergency | Emergency visits before encounter |
| number_inpatient | Inpatient visits before encounter |
| number_diagnoses | Number of diagnoses |
| prior_admission_count | Sum of prior inpatient/outpatient/emergency visits |
| diagnosis_count | Engineered diagnosis count |
| medication_count | Engineered medication count |
| diagnosis_group | Broad group from primary diagnosis |
| readmitted | Dataset-defined class: `<30`, `>30`, `NO` |
| readmitted_within_30_days | Binary indicator: 1 when readmitted `<30` |
| admission_date | Source field if present |
| discharge_date | Source field if present |

## Validation rules
1. encounter_id must not be null.
2. patient_key must not be null.
3. length of stay must be between 1 and 14 days.
4. readmission class must be one of `<30`, `>30`, `NO`.
5. Duplicate encounter IDs are removed.
6. Special missing-value tokens are converted to nulls.
