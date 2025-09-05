-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## 1. Construct input demographic (include sex)

-- COMMAND ----------

-- construct input full cohort: demographic
CREATE OR REPLACE TABLE
  ambit_analytics.sps_patient_journey.model_input_demographic_v3 AS
  SELECT
      CONCAT('PAT_', LPAD(ROW_NUMBER() OVER (ORDER BY FORIAN_PATIENT_ID),
  6, '0')) as PATIENT_ID,
      DATE(CONCAT(BIRTH_YEAR, '-01-01')) as DATE_OF_BIRTH,
      CASE WHEN sps_flag = 1 THEN 'true' ELSE 'false' END as IS_SPS,
      GENDER_CODE as GENDER,
      split_flag
  FROM (
      SELECT DISTINCT FORIAN_PATIENT_ID, BIRTH_YEAR, sps_flag, GENDER_CODE, split_flag
      FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
      WHERE BIRTH_YEAR IS NOT NULL
        AND GENDER_CODE IN ('M', 'F')
  ) t;



-- COMMAND ----------

select *
from ambit_analytics.sps_patient_journey.model_input_demographic_v3


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. Construct input claims

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### With g2582

-- COMMAND ----------

CREATE OR REPLACE TABLE
  ambit_analytics.sps_patient_journey.model_input_claims_v3 AS
  WITH patient_mapping AS (
      SELECT
          FORIAN_PATIENT_ID,
          CONCAT('PAT_', LPAD(ROW_NUMBER() OVER (ORDER BY
  FORIAN_PATIENT_ID), 6, '0')) as PATIENT_ID
      FROM (
          SELECT DISTINCT FORIAN_PATIENT_ID
          FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
          WHERE BIRTH_YEAR IS NOT NULL
            AND GENDER_CODE IN ('M', 'F')  
      )
  )
  SELECT
      pm.PATIENT_ID,
      c.DATE_OF_SERVICE,
      c.D_DIAGNOSIS_CODE as DIAGNOSIS_CODE
  FROM ambit_analytics.sps_patient_journey.all_claims_split_v3 c
  JOIN patient_mapping pm ON c.FORIAN_PATIENT_ID = pm.FORIAN_PATIENT_ID
  WHERE c.D_DIAGNOSIS_CODE IS NOT NULL
  ORDER BY pm.PATIENT_ID, c.DATE_OF_SERVICE;


-- COMMAND ----------

select * from ambit_analytics.sps_patient_journey.model_input_claims_v3
order by PATIENT_ID, DATE_OF_SERVICE

-- COMMAND ----------

select count(distinct PATIENT_ID)
from ambit_analytics.sps_patient_journey.model_input_demographic_v3 c
where c.IS_SPS = 'true'


-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### input claims without G2582

-- COMMAND ----------

select * from ambit_analytics.sps_patient_journey.model_input_claims_v3
where DIAGNOSIS_CODE != 'G2582'
order by PATIENT_ID, DATE_OF_SERVICE

-- COMMAND ----------

select count(distinct PATIENT_ID)
from ambit_analytics.sps_patient_journey.model_input_claims_v3;

  -- Check demographics table
  SELECT
      COUNT(*) as total_patients,
      SUM(CASE WHEN IS_SPS = 'true' THEN 1 ELSE 0 END) as sps_patients,
      SUM(CASE WHEN IS_SPS = 'false' THEN 1 ELSE 0 END) as control_patients
  FROM ambit_analytics.sps_patient_journey.model_input_demographic_v2;

  -- Check claims table
  SELECT
      COUNT(*) as total_claims,
      COUNT(DISTINCT PATIENT_ID) as unique_patients,
      MIN(DATE_OF_SERVICE) as earliest_date,
      MAX(DATE_OF_SERVICE) as latest_date
  FROM ambit_analytics.sps_patient_journey.model_input_claims_v3;

  -- Check events per patient distribution
  SELECT
      MIN(event_count) as min_events,
      MAX(event_count) as max_events,
      AVG(event_count) as avg_events,
      PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY event_count) as median_events,
      SUM(CASE WHEN event_count >= 10 THEN 1 ELSE 0 END) as patients_with_10plus_events,
      SUM(CASE WHEN event_count < 10 THEN 1 ELSE 0 END) as patients_under_10_events
  FROM (
      SELECT PATIENT_ID, COUNT(*) as event_count
      FROM ambit_analytics.sps_patient_journey.model_input_claims_v3
      GROUP BY PATIENT_ID
  );