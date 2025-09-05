# Databricks notebook source
# MAGIC %md
# MAGIC For V2, the main changes are:
# MAGIC Continous enrollment definition changed: have claims every 6m/12m for consecutive years
# MAGIC 50 claims lower limit may be removed
# MAGIC 1. Less restricted for sps patients: 
# MAGIC   one claim instead of 2 claims > 30 days apart
# MAGIC   1 year look forward filter using new mimic code list
# MAGIC
# MAGIC 2. More restricted for control cohort:
# MAGIC   less mimic code list

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. SPS patient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Starting from step 2 of version 1

# COMMAND ----------

# MAGIC %sql
# MAGIC -- V1: Step 2: G2582 Index Diagnoses
# MAGIC -- V2: Step 2: G2582 Index Diagnoses + date filter so can skip step 3
# MAGIC CREATE OR REPLACE TABLE ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2 AS
# MAGIC SELECT 
# MAGIC     d.FORIAN_PATIENT_ID,
# MAGIC     MIN(d.DATE_OF_SERVICE) as index_diagnosis_date
# MAGIC FROM bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC INNER JOIN ambit_analytics.sps_patient_journey.patients_18_plus_at_first_record p
# MAGIC     ON d.FORIAN_PATIENT_ID = p.FORIAN_PATIENT_ID
# MAGIC WHERE d.D_DIAGNOSIS_CODE = 'G2582'
# MAGIC and d.DATE_OF_SERVICE < '2024-06-01'
# MAGIC GROUP BY d.FORIAN_PATIENT_ID;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 skipped

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 Continous enrollment check

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Combo 1: 0.5y + 0y every 6 month

# COMMAND ----------

# MAGIC %sql
# MAGIC -- V1: Step 4: Continuous Enrollment Check
# MAGIC -- V2: Step 4: Updated Continuous Enrollment Check Criteria: have 1+ claim every 6m/12m for x years look back, and y year look forward
# MAGIC CREATE OR REPLACE TABLE
# MAGIC
# MAGIC   ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_6m_0y
# MAGIC   AS
# MAGIC     WITH
# MAGIC     -- Calculate activity by 6-month intervals with FLOOR
# MAGIC     patient_activity AS (
# MAGIC         SELECT
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             -- Use FLOOR for 6-month intervals
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6) as interval_6m,
# MAGIC             COUNT(DISTINCT d.DATE_OF_SERVICE) as claim_count
# MAGIC         FROM ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2
# MAGIC   idx
# MAGIC         INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC             ON idx.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC         WHERE
# MAGIC             -- 6 months before only (excluding index date)
# MAGIC             d.DATE_OF_SERVICE > ADD_MONTHS(idx.index_diagnosis_date, -6)
# MAGIC             AND d.DATE_OF_SERVICE < idx.index_diagnosis_date
# MAGIC         GROUP BY
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6)
# MAGIC     ),
# MAGIC
# MAGIC     -- Count intervals with activity
# MAGIC     interval_summary AS (
# MAGIC         SELECT
# MAGIC             FORIAN_PATIENT_ID,
# MAGIC             index_diagnosis_date,
# MAGIC             -- Count 6-month periods BEFORE index (interval 0 represents the 1 period in 6 months)
# MAGIC             COUNT(DISTINCT CASE WHEN interval_6m = 0 THEN interval_6m END)
# MAGIC   as n_6m_intervals_before
# MAGIC         FROM patient_activity
# MAGIC         GROUP BY FORIAN_PATIENT_ID, index_diagnosis_date
# MAGIC     )
# MAGIC
# MAGIC     -- Final selection
# MAGIC     SELECT
# MAGIC         FORIAN_PATIENT_ID,
# MAGIC         index_diagnosis_date,
# MAGIC         n_6m_intervals_before,
# MAGIC         CASE
# MAGIC             WHEN n_6m_intervals_before = 1 THEN 1
# MAGIC             ELSE 0
# MAGIC         END as continuous_enrollment_flag
# MAGIC     FROM interval_summary
# MAGIC     WHERE n_6m_intervals_before = 1;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(DISTINCT FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_6m_0y
# MAGIC     

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Combo 2: 1y + 0y every 6 month

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC
# MAGIC   ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_1y_0y
# MAGIC   AS
# MAGIC     WITH
# MAGIC     -- Calculate activity by 6-month intervals with FLOOR
# MAGIC     patient_activity AS (
# MAGIC         SELECT
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             -- Use FLOOR for 6-month intervals
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6) as interval_6m,
# MAGIC             COUNT(DISTINCT d.DATE_OF_SERVICE) as claim_count
# MAGIC         FROM ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2
# MAGIC   idx
# MAGIC         INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC             ON idx.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC         WHERE
# MAGIC             -- 1 year before only (excluding index date)
# MAGIC             d.DATE_OF_SERVICE > ADD_MONTHS(idx.index_diagnosis_date, -12)
# MAGIC             AND d.DATE_OF_SERVICE < idx.index_diagnosis_date
# MAGIC         GROUP BY
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6)
# MAGIC     ),
# MAGIC
# MAGIC     -- Count intervals with activity
# MAGIC     interval_summary AS (
# MAGIC         SELECT
# MAGIC             FORIAN_PATIENT_ID,
# MAGIC             index_diagnosis_date,
# MAGIC             -- Count 6-month periods BEFORE index (intervals 0-1 represent the 2 periods in 1 year)
# MAGIC             COUNT(DISTINCT CASE WHEN interval_6m >= 0 AND interval_6m <= 1
# MAGIC   THEN interval_6m END) as n_6m_intervals_before
# MAGIC         FROM patient_activity
# MAGIC         GROUP BY FORIAN_PATIENT_ID, index_diagnosis_date
# MAGIC     )
# MAGIC
# MAGIC     -- Final selection
# MAGIC     SELECT
# MAGIC         FORIAN_PATIENT_ID,
# MAGIC         index_diagnosis_date,
# MAGIC         n_6m_intervals_before,
# MAGIC         CASE
# MAGIC             WHEN n_6m_intervals_before = 2 THEN 1
# MAGIC             ELSE 0
# MAGIC         END as continuous_enrollment_flag
# MAGIC     FROM interval_summary
# MAGIC     WHERE n_6m_intervals_before = 2;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(DISTINCT FORIAN_PATIENT_ID) 
# MAGIC from ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_1y_0y

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Combo 3: 1.5y + 0y every 6 month

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC
# MAGIC   ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_18m_0y
# MAGIC    AS
# MAGIC     WITH
# MAGIC     -- Calculate activity by 6-month intervals with FLOOR
# MAGIC     patient_activity AS (
# MAGIC         SELECT
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             -- Use FLOOR for 6-month intervals
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6) as interval_6m,
# MAGIC             COUNT(DISTINCT d.DATE_OF_SERVICE) as claim_count
# MAGIC         FROM ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2
# MAGIC   idx
# MAGIC         INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC             ON idx.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC         WHERE
# MAGIC             -- 18 months (1.5 years) before only (excluding index date)
# MAGIC             d.DATE_OF_SERVICE > ADD_MONTHS(idx.index_diagnosis_date, -18)
# MAGIC             AND d.DATE_OF_SERVICE < idx.index_diagnosis_date
# MAGIC         GROUP BY
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6)
# MAGIC     ),
# MAGIC
# MAGIC     -- Count intervals with activity
# MAGIC     interval_summary AS (
# MAGIC         SELECT
# MAGIC             FORIAN_PATIENT_ID,
# MAGIC             index_diagnosis_date,
# MAGIC             -- Count 6-month periods BEFORE index (intervals 0-2 represent the 3 periods in 18 months)
# MAGIC             COUNT(DISTINCT CASE WHEN interval_6m >= 0 AND interval_6m <= 2
# MAGIC   THEN interval_6m END) as n_6m_intervals_before
# MAGIC         FROM patient_activity
# MAGIC         GROUP BY FORIAN_PATIENT_ID, index_diagnosis_date
# MAGIC     )
# MAGIC
# MAGIC     -- Final selection
# MAGIC     SELECT
# MAGIC         FORIAN_PATIENT_ID,
# MAGIC         index_diagnosis_date,
# MAGIC         n_6m_intervals_before,
# MAGIC         CASE
# MAGIC             WHEN n_6m_intervals_before = 3 THEN 1
# MAGIC             ELSE 0
# MAGIC         END as continuous_enrollment_flag
# MAGIC     FROM interval_summary
# MAGIC     WHERE n_6m_intervals_before = 3;
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_18m_0y

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3. Combo 3.3: 2y + 0y every 6 month

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC
# MAGIC   ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_2y_0y
# MAGIC   AS
# MAGIC     WITH
# MAGIC     -- Calculate activity by 6-month intervals with FLOOR
# MAGIC     patient_activity AS (
# MAGIC         SELECT
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             -- Use FLOOR for 6-month intervals
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6) as interval_6m,
# MAGIC             COUNT(DISTINCT d.DATE_OF_SERVICE) as claim_count
# MAGIC         FROM ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2
# MAGIC   idx
# MAGIC         INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC             ON idx.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC         WHERE
# MAGIC             -- 2 years before only (excluding index date)
# MAGIC             d.DATE_OF_SERVICE > ADD_MONTHS(idx.index_diagnosis_date, -24)
# MAGIC             AND d.DATE_OF_SERVICE < idx.index_diagnosis_date
# MAGIC         GROUP BY
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6)
# MAGIC     ),
# MAGIC
# MAGIC     -- Count intervals with activity
# MAGIC     interval_summary AS (
# MAGIC         SELECT
# MAGIC             FORIAN_PATIENT_ID,
# MAGIC             index_diagnosis_date,
# MAGIC             -- Count 6-month periods BEFORE index (intervals 0-3 represent the 4 periods in 2 years)
# MAGIC             COUNT(DISTINCT CASE WHEN interval_6m >= 0 AND interval_6m <= 3
# MAGIC   THEN interval_6m END) as n_6m_intervals_before
# MAGIC         FROM patient_activity
# MAGIC         GROUP BY FORIAN_PATIENT_ID, index_diagnosis_date
# MAGIC     )
# MAGIC
# MAGIC     -- Final selection
# MAGIC     SELECT
# MAGIC         FORIAN_PATIENT_ID,
# MAGIC         index_diagnosis_date,
# MAGIC         n_6m_intervals_before,
# MAGIC         CASE
# MAGIC             WHEN n_6m_intervals_before = 4 THEN 1
# MAGIC             ELSE 0
# MAGIC         END as continuous_enrollment_flag
# MAGIC     FROM interval_summary
# MAGIC     WHERE n_6m_intervals_before = 4;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_2y_0y

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.4. Combo 3.4: 3y + 0y every 6 month

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC
# MAGIC   ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_3y_0y
# MAGIC   AS
# MAGIC     WITH
# MAGIC     -- Calculate activity by 6-month intervals with FLOOR
# MAGIC     patient_activity AS (
# MAGIC         SELECT
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             -- Use FLOOR for 6-month intervals
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6) as interval_6m,
# MAGIC             COUNT(DISTINCT d.DATE_OF_SERVICE) as claim_count
# MAGIC         FROM ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2
# MAGIC   idx
# MAGIC         INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC             ON idx.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC         WHERE
# MAGIC             -- 3 years before only (excluding index date)
# MAGIC             d.DATE_OF_SERVICE > ADD_MONTHS(idx.index_diagnosis_date, -36)
# MAGIC             AND d.DATE_OF_SERVICE < idx.index_diagnosis_date
# MAGIC         GROUP BY
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6)
# MAGIC     ),
# MAGIC
# MAGIC     -- Count intervals with activity
# MAGIC     interval_summary AS (
# MAGIC         SELECT
# MAGIC             FORIAN_PATIENT_ID,
# MAGIC             index_diagnosis_date,
# MAGIC             -- Count 6-month periods BEFORE index (intervals 0-5 represent the 6 periods in 3 years)
# MAGIC             COUNT(DISTINCT CASE WHEN interval_6m >= 0 AND interval_6m <= 5
# MAGIC   THEN interval_6m END) as n_6m_intervals_before
# MAGIC         FROM patient_activity
# MAGIC         GROUP BY FORIAN_PATIENT_ID, index_diagnosis_date
# MAGIC     )
# MAGIC
# MAGIC     -- Final selection
# MAGIC     SELECT
# MAGIC         FORIAN_PATIENT_ID,
# MAGIC         index_diagnosis_date,
# MAGIC         n_6m_intervals_before,
# MAGIC         CASE
# MAGIC             WHEN n_6m_intervals_before = 6 THEN 1
# MAGIC             ELSE 0
# MAGIC         END as continuous_enrollment_flag
# MAGIC     FROM interval_summary
# MAGIC     WHERE n_6m_intervals_before = 6;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_3y_0y

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.5. Combo 3.5: 4y + 0y every 6 month

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC
# MAGIC   ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_4y_0y
# MAGIC   AS
# MAGIC     WITH
# MAGIC     -- Calculate activity by 6-month intervals with FLOOR
# MAGIC     patient_activity AS (
# MAGIC         SELECT
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             -- Use FLOOR for 6-month intervals
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6) as interval_6m,
# MAGIC             COUNT(DISTINCT d.DATE_OF_SERVICE) as claim_count
# MAGIC         FROM ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2
# MAGIC   idx
# MAGIC         INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC             ON idx.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC         WHERE
# MAGIC             -- 4 years (48 months) before only (excluding index date)
# MAGIC             d.DATE_OF_SERVICE > ADD_MONTHS(idx.index_diagnosis_date, -48)
# MAGIC             AND d.DATE_OF_SERVICE < idx.index_diagnosis_date
# MAGIC         GROUP BY
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 6)
# MAGIC     ),
# MAGIC
# MAGIC     -- Count intervals with activity
# MAGIC     interval_summary AS (
# MAGIC         SELECT
# MAGIC             FORIAN_PATIENT_ID,
# MAGIC             index_diagnosis_date,
# MAGIC             -- Count 6-month periods BEFORE index (intervals 0-7 represent the 8 periods in 4 years)
# MAGIC             COUNT(DISTINCT CASE WHEN interval_6m >= 0 AND interval_6m <= 7
# MAGIC   THEN interval_6m END) as n_6m_intervals_before
# MAGIC         FROM patient_activity
# MAGIC         GROUP BY FORIAN_PATIENT_ID, index_diagnosis_date
# MAGIC     )
# MAGIC
# MAGIC     -- Final selection
# MAGIC     SELECT
# MAGIC         FORIAN_PATIENT_ID,
# MAGIC         index_diagnosis_date,
# MAGIC         n_6m_intervals_before,
# MAGIC         CASE
# MAGIC             WHEN n_6m_intervals_before = 8 THEN 1
# MAGIC             ELSE 0
# MAGIC         END as continuous_enrollment_flag
# MAGIC     FROM interval_summary
# MAGIC     WHERE n_6m_intervals_before = 8;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.continuous_enrollment_check_6m_4y_0y

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Combo 4: 1y + 0y every 12 month

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC
# MAGIC   ambit_analytics.sps_patient_journey.continuous_enrollment_check_12m_1y_0y
# MAGIC    AS
# MAGIC     WITH
# MAGIC     -- Calculate activity by 12-month intervals with FLOOR
# MAGIC     patient_activity AS (
# MAGIC         SELECT
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             -- Use FLOOR for 12-month intervals
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 12) as interval_12m,
# MAGIC             COUNT(DISTINCT d.DATE_OF_SERVICE) as claim_count
# MAGIC         FROM ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2
# MAGIC   idx
# MAGIC         INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC             ON idx.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC         WHERE
# MAGIC             -- 1 year before only (excluding index date)
# MAGIC             d.DATE_OF_SERVICE > ADD_MONTHS(idx.index_diagnosis_date, -12)
# MAGIC             AND d.DATE_OF_SERVICE < idx.index_diagnosis_date
# MAGIC         GROUP BY
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 12)
# MAGIC     ),
# MAGIC
# MAGIC     -- Count intervals with activity
# MAGIC     interval_summary AS (
# MAGIC         SELECT
# MAGIC             FORIAN_PATIENT_ID,
# MAGIC             index_diagnosis_date,
# MAGIC             -- Count 12-month periods BEFORE index (interval 0 represents the 1 period in 1 year)
# MAGIC             COUNT(DISTINCT CASE WHEN interval_12m = 0 THEN interval_12m
# MAGIC   END) as n_12m_intervals_before
# MAGIC         FROM patient_activity
# MAGIC         GROUP BY FORIAN_PATIENT_ID, index_diagnosis_date
# MAGIC     )
# MAGIC
# MAGIC     -- Final selection
# MAGIC     SELECT
# MAGIC         FORIAN_PATIENT_ID,
# MAGIC         index_diagnosis_date,
# MAGIC         n_12m_intervals_before,
# MAGIC         CASE
# MAGIC             WHEN n_12m_intervals_before = 1 THEN 1
# MAGIC             ELSE 0
# MAGIC         END as continuous_enrollment_flag
# MAGIC     FROM interval_summary
# MAGIC     WHERE n_12m_intervals_before = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.continuous_enrollment_check_12m_1y_0y

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. combo 5: 2y + 0y every 12 month

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC
# MAGIC   ambit_analytics.sps_patient_journey.continuous_enrollment_check_12m_2y_0y
# MAGIC    AS
# MAGIC     WITH
# MAGIC     -- Calculate activity by 12-month intervals with FLOOR
# MAGIC     patient_activity AS (
# MAGIC         SELECT
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             -- Use FLOOR for 12-month intervals
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 12) as interval_12m,
# MAGIC             COUNT(DISTINCT d.DATE_OF_SERVICE) as claim_count
# MAGIC         FROM ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2
# MAGIC   idx
# MAGIC         INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC             ON idx.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC         WHERE
# MAGIC             -- 2 years before only (excluding index date)
# MAGIC             d.DATE_OF_SERVICE > ADD_MONTHS(idx.index_diagnosis_date, -24)
# MAGIC             AND d.DATE_OF_SERVICE < idx.index_diagnosis_date
# MAGIC         GROUP BY
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 12)
# MAGIC     ),
# MAGIC
# MAGIC     -- Count intervals with activity
# MAGIC     interval_summary AS (
# MAGIC         SELECT
# MAGIC             FORIAN_PATIENT_ID,
# MAGIC             index_diagnosis_date,
# MAGIC             -- Count 12-month periods BEFORE index (intervals 0-1 represent the 2 periods in 2 years)
# MAGIC             COUNT(DISTINCT CASE WHEN interval_12m >= 0 AND interval_12m <=
# MAGIC   1 THEN interval_12m END) as n_12m_intervals_before
# MAGIC         FROM patient_activity
# MAGIC         GROUP BY FORIAN_PATIENT_ID, index_diagnosis_date
# MAGIC     )
# MAGIC
# MAGIC     -- Final selection
# MAGIC     SELECT
# MAGIC         FORIAN_PATIENT_ID,
# MAGIC         index_diagnosis_date,
# MAGIC         n_12m_intervals_before,
# MAGIC         CASE
# MAGIC             WHEN n_12m_intervals_before = 2 THEN 1
# MAGIC             ELSE 0
# MAGIC         END as continuous_enrollment_flag
# MAGIC     FROM interval_summary
# MAGIC     WHERE n_12m_intervals_before = 2;
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.continuous_enrollment_check_12m_2y_0y

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6. combo 6: 3y + 0y every **12** month

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC
# MAGIC   ambit_analytics.sps_patient_journey.continuous_enrollment_check_12m_3y_0y
# MAGIC    AS
# MAGIC     WITH
# MAGIC     -- Calculate activity by 12-month intervals with FLOOR
# MAGIC     patient_activity AS (
# MAGIC         SELECT
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             -- Use FLOOR for 12-month intervals
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 12) as interval_12m,
# MAGIC             COUNT(DISTINCT d.DATE_OF_SERVICE) as claim_count
# MAGIC         FROM ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2
# MAGIC   idx
# MAGIC         INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC             ON idx.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC         WHERE
# MAGIC             -- 3 years before only (excluding index date)
# MAGIC             d.DATE_OF_SERVICE > ADD_MONTHS(idx.index_diagnosis_date, -36)
# MAGIC             AND d.DATE_OF_SERVICE < idx.index_diagnosis_date
# MAGIC         GROUP BY
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 12)
# MAGIC     ),
# MAGIC
# MAGIC     -- Count intervals with activity
# MAGIC     interval_summary AS (
# MAGIC         SELECT
# MAGIC             FORIAN_PATIENT_ID,
# MAGIC             index_diagnosis_date,
# MAGIC             -- Count 12-month periods BEFORE index (intervals 0-2 represent the 3 periods in 3 years)
# MAGIC             COUNT(DISTINCT CASE WHEN interval_12m >= 0 AND interval_12m <=
# MAGIC   2 THEN interval_12m END) as n_12m_intervals_before
# MAGIC         FROM patient_activity
# MAGIC         GROUP BY FORIAN_PATIENT_ID, index_diagnosis_date
# MAGIC     )
# MAGIC
# MAGIC     -- Final selection
# MAGIC     SELECT
# MAGIC         FORIAN_PATIENT_ID,
# MAGIC         index_diagnosis_date,
# MAGIC         n_12m_intervals_before,
# MAGIC         CASE
# MAGIC             WHEN n_12m_intervals_before = 3 THEN 1
# MAGIC             ELSE 0
# MAGIC         END as continuous_enrollment_flag
# MAGIC     FROM interval_summary
# MAGIC     WHERE n_12m_intervals_before = 3;
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.continuous_enrollment_check_12m_3y_0y

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7. combo 7: 4y + 0y every 12 month

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC
# MAGIC   ambit_analytics.sps_patient_journey.continuous_enrollment_check_12m_4y_0y
# MAGIC    AS
# MAGIC     WITH
# MAGIC     -- Calculate activity by 12-month intervals with FLOOR
# MAGIC     patient_activity AS (
# MAGIC         SELECT
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             -- Use FLOOR for 12-month intervals
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 12) as interval_12m,
# MAGIC             COUNT(DISTINCT d.DATE_OF_SERVICE) as claim_count
# MAGIC         FROM ambit_analytics.sps_patient_journey.g2582_index_diagnoses_v2
# MAGIC   idx
# MAGIC         INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC             ON idx.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC         WHERE
# MAGIC             -- 4 years before only (excluding index date)
# MAGIC             d.DATE_OF_SERVICE > ADD_MONTHS(idx.index_diagnosis_date, -48)
# MAGIC             AND d.DATE_OF_SERVICE < idx.index_diagnosis_date
# MAGIC         GROUP BY
# MAGIC             idx.FORIAN_PATIENT_ID,
# MAGIC             idx.index_diagnosis_date,
# MAGIC             FLOOR(MONTHS_BETWEEN(idx.index_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 12)
# MAGIC     ),
# MAGIC
# MAGIC     -- Count intervals with activity
# MAGIC     interval_summary AS (
# MAGIC         SELECT
# MAGIC             FORIAN_PATIENT_ID,
# MAGIC             index_diagnosis_date,
# MAGIC             -- Count 12-month periods BEFORE index (intervals 0-3 represent the 4 periods in 4 years)
# MAGIC             COUNT(DISTINCT CASE WHEN interval_12m >= 0 AND interval_12m <=
# MAGIC   3 THEN interval_12m END) as n_12m_intervals_before
# MAGIC         FROM patient_activity
# MAGIC         GROUP BY FORIAN_PATIENT_ID, index_diagnosis_date
# MAGIC     )
# MAGIC
# MAGIC     -- Final selection
# MAGIC     SELECT
# MAGIC         FORIAN_PATIENT_ID,
# MAGIC         index_diagnosis_date,
# MAGIC         n_12m_intervals_before,
# MAGIC         CASE
# MAGIC             WHEN n_12m_intervals_before = 4 THEN 1
# MAGIC             ELSE 0
# MAGIC         END as continuous_enrollment_flag
# MAGIC     FROM interval_summary
# MAGIC     WHERE n_12m_intervals_before = 4;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1y+ record + no mimic code after index diagnosis

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC       ambit_analytics.sps_patient_journey.final_sps_v2 AS
# MAGIC   SELECT c.*
# MAGIC   FROM
# MAGIC   ambit_analytics.sps_patient_journey.continuous_enrollment_check_12m_2y_0y
# MAGIC    c
# MAGIC   WHERE NOT EXISTS (
# MAGIC       SELECT 1
# MAGIC       FROM bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC       INNER JOIN ambit_analytics.sps_patient_journey.control_ref_less cr
# MAGIC           ON d.D_DIAGNOSIS_CODE = cr.code
# MAGIC       WHERE d.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC           AND d.DATE_OF_SERVICE >= c.index_diagnosis_date
# MAGIC   )
# MAGIC   AND EXISTS (
# MAGIC       SELECT 1
# MAGIC       FROM bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC       WHERE d.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC           AND d.DATE_OF_SERVICE >= ADD_MONTHS(c.index_diagnosis_date, 12)
# MAGIC   );

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.final_sps_v2

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ambit_analytics.sps_patient_journey.final_sps_v3 AS
# MAGIC SELECT DISTINCT
# MAGIC     p.FORIAN_PATIENT_ID,
# MAGIC     1 AS gad65_within_6mo_flag
# MAGIC FROM ambit_analytics.sps_patient_journey.continuous_enrollment_check_12m_1y_0y p
# MAGIC JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_procedure px
# MAGIC     ON p.FORIAN_PATIENT_ID = px.FORIAN_PATIENT_ID
# MAGIC WHERE px.D_PROCEDURE_CODE IN ('86294', '86341')
# MAGIC   AND px.DATE_OF_SERVICE BETWEEN DATEADD(MONTH, -6, p.index_diagnosis_date) AND DATEADD(MONTH, 6, p.index_diagnosis_date);

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.final_sps_v3

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.Control patient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Count the number of patients for each code

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. 1: Patients with ≥2 Control Condition 30+d apart with updated code list

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC     ambit_analytics.sps_patient_journey.patients_with_control_conditions_v2_2plus_code
# MAGIC       AS
# MAGIC       WITH control_diagnoses_info AS (
# MAGIC           SELECT
# MAGIC               d.FORIAN_PATIENT_ID,
# MAGIC               COUNT(*) as num_control_diagnoses,
# MAGIC               MIN(d.DATE_OF_SERVICE) as first_control_diagnosis_date,
# MAGIC               MAX(CASE WHEN rn = 2 THEN d.DATE_OF_SERVICE END) as
# MAGIC       second_control_diagnosis_date
# MAGIC           FROM (
# MAGIC               SELECT
# MAGIC                   d.FORIAN_PATIENT_ID,
# MAGIC                   d.DATE_OF_SERVICE,
# MAGIC                   ROW_NUMBER() OVER (PARTITION BY d.FORIAN_PATIENT_ID ORDER
# MAGIC     BY
# MAGIC       d.DATE_OF_SERVICE) as rn
# MAGIC               FROM bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC               INNER JOIN
# MAGIC       ambit_analytics.sps_patient_journey.patients_18_plus_at_first_record
# MAGIC   p
# MAGIC                   ON d.FORIAN_PATIENT_ID = p.FORIAN_PATIENT_ID
# MAGIC               INNER JOIN
# MAGIC   ambit_analytics.sps_patient_journey.control_ref_less
# MAGIC       cr
# MAGIC                   ON d.D_DIAGNOSIS_CODE = cr.code) d
# MAGIC           GROUP BY d.FORIAN_PATIENT_ID
# MAGIC       )
# MAGIC       SELECT
# MAGIC           FORIAN_PATIENT_ID,
# MAGIC           num_control_diagnoses,
# MAGIC           first_control_diagnosis_date,
# MAGIC           second_control_diagnosis_date
# MAGIC       FROM control_diagnoses_info
# MAGIC       WHERE num_control_diagnoses >= 2
# MAGIC         AND DATEDIFF(second_control_diagnosis_date,
# MAGIC   first_control_diagnosis_date) >= 30;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct forian_patient_id)
# MAGIC from ambit_analytics.sps_patient_journey.patients_with_control_conditions_v2_2plus_code

# COMMAND ----------

# MAGIC %md
# MAGIC ## Skip step 2: filter > 50 codes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.1: based on 1.1 - patient has at least 2 year+ record following first diagnosis date
# MAGIC
# MAGIC ## Goal: we give HCPs one years to confirm this is not a misdiagnosed sps patient since the index diagnosis

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC   ambit_analytics.sps_patient_journey.patients_with_control_2plus_record_v2_2plus_code AS
# MAGIC   SELECT
# MAGIC       c.FORIAN_PATIENT_ID,
# MAGIC       c.first_control_diagnosis_date,
# MAGIC       c.second_control_diagnosis_date
# MAGIC   FROM
# MAGIC
# MAGIC   ambit_analytics.sps_patient_journey.patients_with_control_conditions_v2_2plus_code c
# MAGIC   WHERE EXISTS (
# MAGIC       SELECT 1
# MAGIC       FROM bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC       WHERE d.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC         AND d.DATE_OF_SERVICE >=
# MAGIC   ADD_MONTHS(c.first_control_diagnosis_date, 24)
# MAGIC   )
# MAGIC   ORDER BY c.FORIAN_PATIENT_ID;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct forian_patient_id)
# MAGIC from ambit_analytics.sps_patient_journey.patients_with_control_2plus_record_v2_2plus_code

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.1: based on 3.1 - Continuous enrollent check

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC       ambit_analytics.sps_patient_journey.control_continuous_enrollment_check_12m_2y_0y_2plus_code AS
# MAGIC       WITH
# MAGIC       -- Calculate activity by 12-month intervals with FLOOR
# MAGIC       patient_activity AS (
# MAGIC           SELECT
# MAGIC               ctrl.FORIAN_PATIENT_ID,
# MAGIC               ctrl.first_control_diagnosis_date,
# MAGIC               ctrl.second_control_diagnosis_date,
# MAGIC               -- Use FLOOR for 12-month intervals
# MAGIC               FLOOR(MONTHS_BETWEEN(ctrl.first_control_diagnosis_date, d.DATE_OF_SERVICE) / 12) as interval_12m,
# MAGIC               COUNT(DISTINCT d.DATE_OF_SERVICE) as claim_count
# MAGIC           FROM ambit_analytics.sps_patient_journey.patients_with_control_2plus_record_v2_2plus_code
# MAGIC   ctrl
# MAGIC           INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC               ON ctrl.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC           WHERE
# MAGIC               -- 2 years before control diagnosis (excluding diagnosis date)
# MAGIC               d.DATE_OF_SERVICE >
# MAGIC   ADD_MONTHS(ctrl.first_control_diagnosis_date, -24)
# MAGIC               AND d.DATE_OF_SERVICE < ctrl.first_control_diagnosis_date
# MAGIC           GROUP BY
# MAGIC               ctrl.FORIAN_PATIENT_ID,
# MAGIC               ctrl.first_control_diagnosis_date,
# MAGIC               ctrl.second_control_diagnosis_date,
# MAGIC               FLOOR(MONTHS_BETWEEN(ctrl.first_control_diagnosis_date,
# MAGIC   d.DATE_OF_SERVICE) / 12)
# MAGIC       ),
# MAGIC
# MAGIC       -- Count intervals with activity
# MAGIC       interval_summary AS (
# MAGIC           SELECT
# MAGIC               FORIAN_PATIENT_ID,
# MAGIC               first_control_diagnosis_date,
# MAGIC               second_control_diagnosis_date,
# MAGIC               -- Count 12-month periods BEFORE index (intervals 0-1 represent the 2 periods in 2 years)
# MAGIC               COUNT(DISTINCT CASE WHEN interval_12m >= 0 AND interval_12m
# MAGIC   <= 1 THEN interval_12m END) as n_12m_intervals_before
# MAGIC           FROM patient_activity
# MAGIC           GROUP BY FORIAN_PATIENT_ID, first_control_diagnosis_date, second_control_diagnosis_date
# MAGIC       )
# MAGIC
# MAGIC       -- Final selection
# MAGIC       SELECT
# MAGIC           FORIAN_PATIENT_ID,
# MAGIC           first_control_diagnosis_date,
# MAGIC           second_control_diagnosis_date,
# MAGIC           n_12m_intervals_before,
# MAGIC           CASE
# MAGIC               WHEN n_12m_intervals_before = 2 THEN 1
# MAGIC               ELSE 0
# MAGIC           END as continuous_enrollment_flag
# MAGIC       FROM interval_summary
# MAGIC       WHERE n_12m_intervals_before = 2;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.control_continuous_enrollment_check_12m_2y_0y_2plus_code

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5.1: based on 4.1 - exclude patient who ever had G2582 diagnosis

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC     ambit_analytics.sps_patient_journey.final_ctr_v2 AS
# MAGIC     SELECT
# MAGIC         c.*
# MAGIC     FROM ambit_analytics.sps_patient_journey.control_continuous_enrollment_check_12m_2y_0y_2plus_code c
# MAGIC     WHERE NOT EXISTS (
# MAGIC         SELECT 1
# MAGIC         FROM bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC         WHERE d.FORIAN_PATIENT_ID = c.FORIAN_PATIENT_ID
# MAGIC           AND d.D_DIAGNOSIS_CODE = 'G2582'
# MAGIC     );

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.final_ctr_v2

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from ambit_analytics.sps_patient_journey.final_ctr_v2
# MAGIC INNER JOIN ambit_analytics.sps_patient_journey.final_sps_v2
# MAGIC on ambit_analytics.sps_patient_journey.final_ctr_v2.FORIAN_PATIENT_ID = ambit_analytics.sps_patient_journey.final_sps_v2.FORIAN_PATIENT_ID