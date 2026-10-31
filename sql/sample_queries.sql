-- 1. Overall 30-day readmission rate
SELECT
    COUNT(*) AS admissions,
    SUM(readmitted_within_30_days) AS readmissions_30d,
    ROUND(100.0 * AVG(readmitted_within_30_days), 2) AS readmission_rate_pct
FROM mart.admission_fact;

-- 2. By age
SELECT
    age,
    COUNT(*) AS admissions,
    ROUND(100.0 * AVG(readmitted_within_30_days), 2) AS readmission_rate_pct
FROM mart.admission_fact
GROUP BY age
ORDER BY age;

-- 3. By diagnosis group
SELECT *
FROM mart.diagnosis_summary
ORDER BY readmission_rate_30d DESC;

-- 4. Length of stay
SELECT
    length_of_stay,
    COUNT(*) AS admissions,
    ROUND(100.0 * AVG(readmitted_within_30_days), 2) AS readmission_rate_pct
FROM mart.admission_fact
GROUP BY length_of_stay
ORDER BY length_of_stay;

-- 5. High prior-utilization cohort
SELECT
    age,
    diagnosis_group,
    COUNT(*) AS admissions,
    ROUND(100.0 * AVG(readmitted_within_30_days), 2) AS readmission_rate_pct
FROM mart.admission_fact
WHERE prior_admission_count >= 3
GROUP BY age, diagnosis_group
ORDER BY readmission_rate_pct DESC;
