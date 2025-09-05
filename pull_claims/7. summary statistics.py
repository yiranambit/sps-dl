# Databricks notebook source
# MAGIC %md
# MAGIC #### 1.1 code frequency

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SPS Descriptive Statistics Analysis with 5-digit truncated codes
# MAGIC   -- Compares SPS patient code frequencies in cumulative 6m/12m/24m/36m intervals before index diagnosis
# MAGIC   -- with background frequencies from SPS patients' entire trajectory
# MAGIC
# MAGIC   -- Step 1: Get SPS patients and their index dates
# MAGIC   WITH sps_patients AS (
# MAGIC     SELECT
# MAGIC       FORIAN_PATIENT_ID,
# MAGIC       index_diagnosis_date,
# MAGIC       sps_flag
# MAGIC     FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC     WHERE sps_flag = 1
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 2: Calculate SPS code frequencies by cumulative time intervals
# MAGIC   sps_6m AS (
# MAGIC     SELECT
# MAGIC       '6m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(s.index_diagnosis_date, 180)  -- 6 months = 180 days
# MAGIC       AND c.DATE_OF_SERVICE < s.index_diagnosis_date
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   sps_12m AS (
# MAGIC     SELECT
# MAGIC       '12m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(s.index_diagnosis_date, 365)  -- 12 months = 365 days
# MAGIC       AND c.DATE_OF_SERVICE < s.index_diagnosis_date
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   sps_24m AS (
# MAGIC     SELECT
# MAGIC       '24m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(s.index_diagnosis_date, 730)  -- 24 months = 730 days
# MAGIC       AND c.DATE_OF_SERVICE < s.index_diagnosis_date
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   sps_36m AS (
# MAGIC     SELECT
# MAGIC       '36m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(s.index_diagnosis_date, 1095)  -- 36 months = 1095 days
# MAGIC       AND c.DATE_OF_SERVICE < s.index_diagnosis_date
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 3: Calculate background frequencies from SPS patients' entire trajectory
# MAGIC   background_frequencies AS (
# MAGIC     SELECT
# MAGIC       'background' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 4: Union all intervals including background
# MAGIC   all_frequencies AS (
# MAGIC     SELECT * FROM sps_6m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM sps_12m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM sps_24m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM sps_36m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM background_frequencies
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 5: Get top 20 codes for each interval (including background)
# MAGIC   top_codes_all_intervals AS (
# MAGIC     SELECT
# MAGIC       interval_period,
# MAGIC       diagnosis_code_5digit,
# MAGIC       frequency,
# MAGIC       code_count,
# MAGIC       ROW_NUMBER() OVER (PARTITION BY interval_period ORDER BY frequency
# MAGIC   DESC) as frequency_rank
# MAGIC     FROM all_frequencies
# MAGIC   )
# MAGIC
# MAGIC   -- Final output: Top 20 codes for each interval including background
# MAGIC   SELECT
# MAGIC     interval_period,
# MAGIC     diagnosis_code_5digit,
# MAGIC     ROUND(frequency, 6) as frequency,
# MAGIC     code_count,
# MAGIC     frequency_rank
# MAGIC   FROM top_codes_all_intervals
# MAGIC   WHERE frequency_rank <= 20
# MAGIC   ORDER BY
# MAGIC     CASE interval_period
# MAGIC       WHEN '6m' THEN 1
# MAGIC       WHEN '12m' THEN 2
# MAGIC       WHEN '24m' THEN 3
# MAGIC       WHEN '36m' THEN 4
# MAGIC       WHEN 'background' THEN 5
# MAGIC     END,
# MAGIC     frequency_rank;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1.1 code frequency exclusive window

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SPS Descriptive Statistics Analysis with 5-digit truncated codes
# MAGIC   -- Compares SPS patient code frequencies in 0-6m/6-12m/12-24m/24-36m intervals before index diagnosis
# MAGIC   -- with background frequencies from SPS patients' entire trajectory
# MAGIC
# MAGIC   -- Step 1: Get SPS patients and their index dates
# MAGIC   WITH sps_patients AS (
# MAGIC     SELECT
# MAGIC       FORIAN_PATIENT_ID,
# MAGIC       index_diagnosis_date,
# MAGIC       sps_flag
# MAGIC     FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC     WHERE sps_flag = 1
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 2: Calculate SPS code frequencies by time intervals
# MAGIC   sps_0_6m AS (
# MAGIC     SELECT
# MAGIC       '0-6m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(s.index_diagnosis_date, 180)  -- 0-6 months = 180 days
# MAGIC       AND c.DATE_OF_SERVICE < s.index_diagnosis_date
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   sps_6_12m AS (
# MAGIC     SELECT
# MAGIC       '6-12m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(s.index_diagnosis_date, 365)  --  6-12 months
# MAGIC       AND c.DATE_OF_SERVICE < DATE_SUB(s.index_diagnosis_date, 180)
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   sps_12_24m AS (
# MAGIC     SELECT
# MAGIC       '12-24m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(s.index_diagnosis_date, 730)  -- 12-24 months
# MAGIC       AND c.DATE_OF_SERVICE < DATE_SUB(s.index_diagnosis_date, 365)
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   sps_24_36m AS (
# MAGIC     SELECT
# MAGIC       '24-36m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(s.index_diagnosis_date, 1095)  -- 24-36 months
# MAGIC       AND c.DATE_OF_SERVICE < DATE_SUB(s.index_diagnosis_date, 730)
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 3: Calculate background frequencies from SPS patients' entire trajectory
# MAGIC   background_frequencies AS (
# MAGIC     SELECT
# MAGIC       'background' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 4: Union all intervals including background
# MAGIC   all_frequencies AS (
# MAGIC     SELECT * FROM sps_0_6m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM sps_6_12m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM sps_12_24m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM sps_24_36m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM background_frequencies
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 5: Get top 20 codes for each interval (including background)
# MAGIC   top_codes_all_intervals AS (
# MAGIC     SELECT
# MAGIC       interval_period,
# MAGIC       diagnosis_code_5digit,
# MAGIC       frequency,
# MAGIC       code_count,
# MAGIC       ROW_NUMBER() OVER (PARTITION BY interval_period ORDER BY frequency
# MAGIC   DESC) as frequency_rank
# MAGIC     FROM all_frequencies
# MAGIC   )
# MAGIC
# MAGIC   -- Final output: Top 20 codes for each interval including background
# MAGIC   SELECT
# MAGIC     interval_period,
# MAGIC     diagnosis_code_5digit,
# MAGIC     ROUND(frequency, 6) as frequency,
# MAGIC     code_count,
# MAGIC     frequency_rank
# MAGIC   FROM top_codes_all_intervals
# MAGIC   WHERE frequency_rank <= 20
# MAGIC   ORDER BY
# MAGIC     CASE interval_period
# MAGIC       WHEN '0-6m' THEN 1
# MAGIC       WHEN '6-12m' THEN 2
# MAGIC       WHEN '12-24m' THEN 3
# MAGIC       WHEN '24-36m' THEN 4
# MAGIC       WHEN 'background' THEN 5
# MAGIC     END,
# MAGIC     frequency_rank;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2 age + sex

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Age and Sex Statistics for SPS Patients vs Controls
# MAGIC
# MAGIC   WITH patient_demographics AS (
# MAGIC     SELECT
# MAGIC       FORIAN_PATIENT_ID,
# MAGIC       index_diagnosis_date,
# MAGIC       sps_flag,
# MAGIC       BIRTH_YEAR,
# MAGIC       GENDER_CODE,
# MAGIC       -- Calculate age at index diagnosis for SPS patients, or current age for controls
# MAGIC       CASE
# MAGIC         WHEN sps_flag = 1 THEN YEAR(CURRENT_DATE()) - BIRTH_YEAR
# MAGIC         ELSE YEAR(CURRENT_DATE()) - BIRTH_YEAR
# MAGIC       END as age_at_index
# MAGIC     FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC   ),
# MAGIC
# MAGIC   -- Age statistics by group
# MAGIC   age_stats AS (
# MAGIC     SELECT
# MAGIC       CASE WHEN sps_flag = 1 THEN 'SPS' ELSE 'Control' END as patient_type,
# MAGIC       COUNT(*) as total_patients,
# MAGIC       ROUND(AVG(age_at_index), 1) as mean_age,
# MAGIC       ROUND(STDDEV(age_at_index), 1) as std_age,
# MAGIC       MIN(age_at_index) as min_age,
# MAGIC       MAX(age_at_index) as max_age,
# MAGIC       PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY age_at_index) as q1_age,
# MAGIC       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY age_at_index) as
# MAGIC   median_age,
# MAGIC       PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY age_at_index) as q3_age
# MAGIC     FROM patient_demographics
# MAGIC     WHERE age_at_index IS NOT NULL AND age_at_index > 0 AND age_at_index <
# MAGIC   120
# MAGIC     GROUP BY sps_flag
# MAGIC   ),
# MAGIC
# MAGIC   -- Gender statistics by group
# MAGIC   gender_stats AS (
# MAGIC     SELECT
# MAGIC       CASE WHEN sps_flag = 1 THEN 'SPS' ELSE 'Control' END as patient_type,
# MAGIC       GENDER_CODE,
# MAGIC       COUNT(*) as count,
# MAGIC       COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY sps_flag) as
# MAGIC   percentage
# MAGIC     FROM patient_demographics
# MAGIC     WHERE GENDER_CODE IS NOT NULL
# MAGIC     GROUP BY sps_flag, GENDER_CODE
# MAGIC   ),
# MAGIC
# MAGIC   -- Age group distribution
# MAGIC   age_group_stats AS (
# MAGIC     SELECT
# MAGIC       CASE WHEN sps_flag = 1 THEN 'SPS' ELSE 'Control' END as patient_type,
# MAGIC       CASE
# MAGIC         WHEN age_at_index < 18 THEN '0-17'
# MAGIC         WHEN age_at_index < 30 THEN '18-29'
# MAGIC         WHEN age_at_index < 40 THEN '30-39'
# MAGIC         WHEN age_at_index < 50 THEN '40-49'
# MAGIC         WHEN age_at_index < 60 THEN '50-59'
# MAGIC         WHEN age_at_index < 70 THEN '60-69'
# MAGIC         WHEN age_at_index < 80 THEN '70-79'
# MAGIC         ELSE '80+'
# MAGIC       END as age_group,
# MAGIC       COUNT(*) as count,
# MAGIC       COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY sps_flag) as
# MAGIC   percentage
# MAGIC     FROM patient_demographics
# MAGIC     WHERE age_at_index IS NOT NULL AND age_at_index > 0 AND age_at_index <
# MAGIC   120
# MAGIC     GROUP BY sps_flag,
# MAGIC       CASE
# MAGIC         WHEN age_at_index < 18 THEN '0-17'
# MAGIC         WHEN age_at_index < 30 THEN '18-29'
# MAGIC         WHEN age_at_index < 40 THEN '30-39'
# MAGIC         WHEN age_at_index < 50 THEN '40-49'
# MAGIC         WHEN age_at_index < 60 THEN '50-59'
# MAGIC         WHEN age_at_index < 70 THEN '60-69'
# MAGIC         WHEN age_at_index < 80 THEN '70-79'
# MAGIC         ELSE '80+'
# MAGIC       END
# MAGIC   )
# MAGIC
# MAGIC   -- Output 1: Age Statistics Summary
# MAGIC   SELECT
# MAGIC     'Age Statistics' as analysis_type,
# MAGIC     patient_type,
# MAGIC     CAST(total_patients AS STRING) as category,
# MAGIC     total_patients as count,
# MAGIC     mean_age as value,
# MAGIC     std_age as additional_value1,
# MAGIC     min_age as additional_value2,
# MAGIC     max_age as additional_value3,
# MAGIC     q1_age as additional_value4,
# MAGIC     median_age as additional_value5,
# MAGIC     q3_age as additional_value6
# MAGIC   FROM age_stats
# MAGIC
# MAGIC   UNION ALL
# MAGIC
# MAGIC   -- Output 2: Gender Distribution
# MAGIC   SELECT
# MAGIC     'Gender Distribution' as analysis_type,
# MAGIC     patient_type,
# MAGIC     GENDER_CODE as category,
# MAGIC     count,
# MAGIC     percentage as value,
# MAGIC     NULL as additional_value1,
# MAGIC     NULL as additional_value2,
# MAGIC     NULL as additional_value3,
# MAGIC     NULL as additional_value4,
# MAGIC     NULL as additional_value5,
# MAGIC     NULL as additional_value6
# MAGIC   FROM gender_stats
# MAGIC
# MAGIC   UNION ALL
# MAGIC
# MAGIC   -- Output 3: Age Group Distribution  
# MAGIC   SELECT
# MAGIC     'Age Group Distribution' as analysis_type,
# MAGIC     patient_type,
# MAGIC     age_group as category,
# MAGIC     count,
# MAGIC     percentage as value,
# MAGIC     NULL as additional_value1,
# MAGIC     NULL as additional_value2,
# MAGIC     NULL as additional_value3,
# MAGIC     NULL as additional_value4,
# MAGIC     NULL as additional_value5,
# MAGIC     NULL as additional_value6
# MAGIC   FROM age_group_stats
# MAGIC
# MAGIC   ORDER BY analysis_type, patient_type, category;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. HCP specialty

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 HCP spetialty -- index diagnosis only

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH sps_patients AS (
# MAGIC     SELECT
# MAGIC       FORIAN_PATIENT_ID,
# MAGIC       index_diagnosis_date
# MAGIC     FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC     WHERE sps_flag = 1
# MAGIC   ),
# MAGIC
# MAGIC   -- Get index diagnosis claims for SPS patients
# MAGIC   index_diagnosis_claims AS (
# MAGIC     SELECT
# MAGIC       s.FORIAN_PATIENT_ID,
# MAGIC       c.CLAIM_NUMBER,
# MAGIC       c.DATE_OF_SERVICE
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC       AND c.DATE_OF_SERVICE = s.index_diagnosis_date
# MAGIC     WHERE c.D_DIAGNOSIS_CODE = 'G2582'
# MAGIC   ),
# MAGIC     claim_providers AS (
# MAGIC     SELECT
# MAGIC       idc.FORIAN_PATIENT_ID,
# MAGIC       cp.PROVIDER_NPI,
# MAGIC       cp.PROVIDER_ROLE
# MAGIC     FROM index_diagnosis_claims idc
# MAGIC     JOIN
# MAGIC   bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_provider cp
# MAGIC       ON idc.CLAIM_NUMBER = cp.CLAIM_NUMBER
# MAGIC       AND cp.
# MAGIC     WHERE cp.PROVIDER_NPI IS NOT NULL
# MAGIC   ),
# MAGIC     provider_specialty_names AS (
# MAGIC     SELECT
# MAGIC       cp.FORIAN_PATIENT_ID,
# MAGIC       cp.PROVIDER_ROLE,
# MAGIC       sp.CATEGORY
# MAGIC     FROM claim_providers cp
# MAGIC     JOIN ambit_analytics.forian_reference.npi_reference_nppes sp
# MAGIC       ON cp.PROVIDER_NPI = sp.NPI
# MAGIC   )
# MAGIC
# MAGIC     -- Final output: Count by specialty and role
# MAGIC   SELECT
# MAGIC     category,
# MAGIC     PROVIDER_ROLE,
# MAGIC     COUNT(DISTINCT FORIAN_PATIENT_ID) as patient_count
# MAGIC   FROM provider_specialty_names
# MAGIC   GROUP BY category, PROVIDER_ROLE
# MAGIC   ORDER BY category, patient_count DESC;
# MAGIC   
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH sps_patients AS (
# MAGIC     SELECT
# MAGIC       FORIAN_PATIENT_ID,
# MAGIC       index_diagnosis_date
# MAGIC     FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC     WHERE sps_flag = 1
# MAGIC   ),
# MAGIC
# MAGIC   -- Get index diagnosis claims for SPS patients
# MAGIC   index_diagnosis_claims AS (
# MAGIC     SELECT
# MAGIC       s.FORIAN_PATIENT_ID,
# MAGIC       c.CLAIM_NUMBER,
# MAGIC       c.DATE_OF_SERVICE,
# MAGIC       c.D_DIAGNOSIS_CODE
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC       AND c.DATE_OF_SERVICE = s.index_diagnosis_date
# MAGIC     WHERE c.D_DIAGNOSIS_CODE = 'G2582'
# MAGIC   )
# MAGIC   select * from index_diagnosis_claims
# MAGIC   order by forian_patient_id, date_of_service

# COMMAND ----------

# MAGIC %md
# MAGIC HCP spetialty -- all g2582 diagnosis

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH sps_patients AS (
# MAGIC     SELECT
# MAGIC       FORIAN_PATIENT_ID,
# MAGIC       index_diagnosis_date
# MAGIC     FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC     WHERE sps_flag = 1
# MAGIC   ),
# MAGIC
# MAGIC   -- Get index diagnosis claims for SPS patients
# MAGIC   index_diagnosis_claims AS (
# MAGIC     SELECT
# MAGIC       s.FORIAN_PATIENT_ID,
# MAGIC       c.CLAIM_NUMBER,
# MAGIC       c.DATE_OF_SERVICE
# MAGIC     FROM sps_patients s
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON s.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.D_DIAGNOSIS_CODE = 'G2582'
# MAGIC   ),
# MAGIC     claim_providers AS (
# MAGIC     SELECT
# MAGIC       idc.FORIAN_PATIENT_ID,
# MAGIC       cp.PROVIDER_NPI,
# MAGIC       cp.PROVIDER_ROLE
# MAGIC     FROM index_diagnosis_claims idc
# MAGIC     JOIN
# MAGIC   bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_provider cp
# MAGIC       ON idc.CLAIM_NUMBER = cp.CLAIM_NUMBER
# MAGIC     WHERE cp.PROVIDER_NPI IS NOT NULL
# MAGIC   ),
# MAGIC     provider_specialty_names AS (
# MAGIC     SELECT
# MAGIC       cp.FORIAN_PATIENT_ID,
# MAGIC       cp.PROVIDER_ROLE,
# MAGIC       sp.CATEGORY
# MAGIC     FROM claim_providers cp
# MAGIC     JOIN ambit_analytics.forian_reference.npi_reference_nppes sp
# MAGIC       ON cp.PROVIDER_NPI = sp.NPI
# MAGIC   )
# MAGIC
# MAGIC     -- Final output: Count by specialty and role
# MAGIC   SELECT
# MAGIC     category,
# MAGIC     PROVIDER_ROLE,
# MAGIC     COUNT(DISTINCT FORIAN_PATIENT_ID) as patient_count
# MAGIC   FROM provider_specialty_names
# MAGIC   GROUP BY category, PROVIDER_ROLE
# MAGIC   ORDER BY category, patient_count DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 ctr code frequency with non-exclusive windows

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Control Patient Descriptive Statistics Analysis with 5-digit truncated codes  
# MAGIC   -- Analyzes control patient code frequencies in cumulative 6m/12m/24m/36m intervals before index diagnosis
# MAGIC   -- with background frequencies from control patients' entire trajectory
# MAGIC
# MAGIC   -- Step 1: Get control patients and their index dates
# MAGIC   WITH control_patients AS (
# MAGIC     SELECT
# MAGIC       FORIAN_PATIENT_ID,
# MAGIC       index_diagnosis_date,
# MAGIC       sps_flag
# MAGIC     FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC     WHERE sps_flag = 0
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 2: Calculate control code frequencies by cumulative time intervals
# MAGIC   control_6m AS (
# MAGIC     SELECT
# MAGIC       '6m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM control_patients ctrl
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON ctrl.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(ctrl.index_diagnosis_date, 180)  -- 6 months = 180 days
# MAGIC       AND c.DATE_OF_SERVICE < ctrl.index_diagnosis_date
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   control_12m AS (
# MAGIC     SELECT
# MAGIC       '12m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM control_patients ctrl
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON ctrl.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(ctrl.index_diagnosis_date, 365)  -- 12 months = 365 days
# MAGIC       AND c.DATE_OF_SERVICE < ctrl.index_diagnosis_date
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   control_24m AS (
# MAGIC     SELECT
# MAGIC       '24m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM control_patients ctrl
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON ctrl.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(ctrl.index_diagnosis_date, 730)  -- 24 months = 730 days
# MAGIC       AND c.DATE_OF_SERVICE < ctrl.index_diagnosis_date
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   control_36m AS (
# MAGIC     SELECT
# MAGIC       '36m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM control_patients ctrl
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON ctrl.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(ctrl.index_diagnosis_date, 1095) -- 36 months = 1095 days
# MAGIC       AND c.DATE_OF_SERVICE < ctrl.index_diagnosis_date
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 3: Calculate background frequencies from control patients' entire trajectory
# MAGIC   background_frequencies AS (
# MAGIC     SELECT
# MAGIC       'background' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM control_patients ctrl
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON ctrl.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 4: Union all intervals including background
# MAGIC   all_frequencies AS (
# MAGIC     SELECT * FROM control_6m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM control_12m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM control_24m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM control_36m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM background_frequencies
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 5: Get top 20 codes for each interval (including background)
# MAGIC   top_codes_all_intervals AS (
# MAGIC     SELECT
# MAGIC       interval_period,
# MAGIC       diagnosis_code_5digit,
# MAGIC       frequency,
# MAGIC       code_count,
# MAGIC       ROW_NUMBER() OVER (PARTITION BY interval_period ORDER BY frequency
# MAGIC   DESC) as frequency_rank
# MAGIC     FROM all_frequencies
# MAGIC   )
# MAGIC
# MAGIC   -- Final output: Top 20 codes for each interval including background
# MAGIC   SELECT
# MAGIC     interval_period,
# MAGIC     diagnosis_code_5digit,
# MAGIC     ROUND(frequency, 6) as frequency,
# MAGIC     code_count,
# MAGIC     frequency_rank
# MAGIC   FROM top_codes_all_intervals
# MAGIC   WHERE frequency_rank <= 20
# MAGIC   ORDER BY
# MAGIC     CASE interval_period
# MAGIC       WHEN '6m' THEN 1
# MAGIC       WHEN '12m' THEN 2
# MAGIC       WHEN '24m' THEN 3
# MAGIC       WHEN '36m' THEN 4
# MAGIC       WHEN 'background' THEN 5
# MAGIC     END,
# MAGIC     frequency_rank;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1.1 ctr code frequencies with exclusive windows

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Control Patient Descriptive Statistics Analysis with 5-digit truncated codes  
# MAGIC   -- Analyzes control patient code frequencies in exclusive 0-6m/6-12m/12-24m/24-36m intervals before index diagnosis
# MAGIC   -- with background frequencies from control patients' entire trajectory
# MAGIC
# MAGIC   -- Step 1: Get control patients and their index dates
# MAGIC   WITH control_patients AS (
# MAGIC     SELECT
# MAGIC       FORIAN_PATIENT_ID,
# MAGIC       index_diagnosis_date,
# MAGIC       sps_flag
# MAGIC     FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC     WHERE sps_flag = 0
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 2: Calculate control code frequencies by exclusive time intervals
# MAGIC   control_0_6m AS (
# MAGIC     SELECT
# MAGIC       '0-6m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM control_patients ctrl
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON ctrl.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(ctrl.index_diagnosis_date, 180)  -- 0-6 months
# MAGIC       AND c.DATE_OF_SERVICE < ctrl.index_diagnosis_date
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   control_6_12m AS (
# MAGIC     SELECT
# MAGIC       '6-12m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM control_patients ctrl
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON ctrl.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(ctrl.index_diagnosis_date, 365)  -- 6-12 months
# MAGIC       AND c.DATE_OF_SERVICE < DATE_SUB(ctrl.index_diagnosis_date, 180)
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   control_12_24m AS (
# MAGIC     SELECT
# MAGIC       '12-24m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM control_patients ctrl
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON ctrl.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(ctrl.index_diagnosis_date, 730)  -- 12-24 months
# MAGIC       AND c.DATE_OF_SERVICE < DATE_SUB(ctrl.index_diagnosis_date, 365)
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   control_24_36m AS (
# MAGIC     SELECT
# MAGIC       '24-36m' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM control_patients ctrl
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON ctrl.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.DATE_OF_SERVICE >= DATE_SUB(ctrl.index_diagnosis_date, 1095) -- 24-36 months
# MAGIC       AND c.DATE_OF_SERVICE < DATE_SUB(ctrl.index_diagnosis_date, 730)
# MAGIC       AND c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 3: Calculate background frequencies from control patients' entire trajectory
# MAGIC   background_frequencies AS (
# MAGIC     SELECT
# MAGIC       'background' as interval_period,
# MAGIC       LEFT(c.D_DIAGNOSIS_CODE, 5) as diagnosis_code_5digit,
# MAGIC       COUNT(*) as code_count,
# MAGIC       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () as frequency
# MAGIC     FROM control_patients ctrl
# MAGIC     JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis c
# MAGIC       ON ctrl.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC     WHERE c.D_DIAGNOSIS_CODE IS NOT NULL
# MAGIC       AND LENGTH(c.D_DIAGNOSIS_CODE) >= 5
# MAGIC     GROUP BY LEFT(c.D_DIAGNOSIS_CODE, 5)
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 4: Union all intervals including background
# MAGIC   all_frequencies AS (
# MAGIC     SELECT * FROM control_0_6m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM control_6_12m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM control_12_24m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM control_24_36m
# MAGIC     UNION ALL
# MAGIC     SELECT * FROM background_frequencies
# MAGIC   ),
# MAGIC
# MAGIC   -- Step 5: Get top 20 codes for each interval (including background)
# MAGIC   top_codes_all_intervals AS (
# MAGIC     SELECT
# MAGIC       interval_period,
# MAGIC       diagnosis_code_5digit,
# MAGIC       frequency,
# MAGIC       code_count,
# MAGIC       ROW_NUMBER() OVER (PARTITION BY interval_period ORDER BY frequency
# MAGIC   DESC) as frequency_rank
# MAGIC     FROM all_frequencies
# MAGIC   )
# MAGIC
# MAGIC   -- Final output: Top 20 codes for each interval including background
# MAGIC   SELECT
# MAGIC     interval_period,
# MAGIC     diagnosis_code_5digit,
# MAGIC     ROUND(frequency, 6) as frequency,
# MAGIC     code_count,
# MAGIC     frequency_rank
# MAGIC   FROM top_codes_all_intervals
# MAGIC   WHERE frequency_rank <= 20
# MAGIC   ORDER BY
# MAGIC     CASE interval_period
# MAGIC       WHEN '0-6m' THEN 1
# MAGIC       WHEN '6-12m' THEN 2
# MAGIC       WHEN '12-24m' THEN 3
# MAGIC       WHEN '24-36m' THEN 4
# MAGIC       WHEN 'background' THEN 5
# MAGIC     END,
# MAGIC     frequency_rank;