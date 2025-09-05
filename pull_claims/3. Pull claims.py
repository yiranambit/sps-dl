# Databricks notebook source
# MAGIC %md
# MAGIC ## 1. Pull claims

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1 sps truncate before index + ctr 2 year before end of trajectory

# COMMAND ----------

# MAGIC %sql
# MAGIC -- age_sex_check_step2 has already remove duplicate age OR sex
# MAGIC -- impute 01-01 + 5 digit truncation
# MAGIC -- Pull claims prior to index date for sps
# MAGIC -- Pull claims 2 year before trajectory for ctr
# MAGIC
# MAGIC CREATE OR REPLACE TABLE
# MAGIC       ambit_analytics.sps_patient_journey.all_claims_split_v3 AS
# MAGIC   WITH patient_end_dates AS (
# MAGIC       -- Calculate end of trajectory date for each patient
# MAGIC       SELECT
# MAGIC           FORIAN_PATIENT_ID,
# MAGIC           MAX(DATE_OF_SERVICE) as end_of_trajectory_date
# MAGIC       FROM bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis
# MAGIC       GROUP BY FORIAN_PATIENT_ID
# MAGIC   )
# MAGIC   SELECT
# MAGIC       p.FORIAN_PATIENT_ID,
# MAGIC       p.sps_flag,
# MAGIC       p.split_flag,
# MAGIC       p.index_diagnosis_date,
# MAGIC       CONCAT(CAST(p.BIRTH_YEAR AS STRING), '-01-01') AS dob,
# MAGIC       d.DATE_OF_SERVICE,
# MAGIC       SUBSTR(d.D_DIAGNOSIS_CODE, 1, 5) as D_DIAGNOSIS_CODE,
# MAGIC       d.L_CLAIM_TYPE_CODE
# MAGIC   FROM ambit_analytics.sps_patient_journey.age_sex_check_step2 p
# MAGIC   INNER JOIN bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC       ON p.FORIAN_PATIENT_ID = d.FORIAN_PATIENT_ID
# MAGIC   INNER JOIN patient_end_dates e
# MAGIC       ON p.FORIAN_PATIENT_ID = e.FORIAN_PATIENT_ID
# MAGIC   WHERE
# MAGIC       (
# MAGIC           -- For SPS patients: only claims before index diagnosis
# MAGIC           (p.sps_flag = 1 AND d.DATE_OF_SERVICE < p.index_diagnosis_date)
# MAGIC           OR
# MAGIC           -- For control patients: claims up to 1 year before end of trajectory
# MAGIC           (p.sps_flag = 0 AND d.DATE_OF_SERVICE <
# MAGIC   DATE_ADD(e.end_of_trajectory_date, -730))
# MAGIC       )
# MAGIC   ORDER BY p.FORIAN_PATIENT_ID, d.DATE_OF_SERVICE;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from ambit_analytics.sps_patient_journey.all_claims_split_v3
# MAGIC where sps_flag = 0
# MAGIC and D_DIAGNOSIS_CODE = 'G2582'