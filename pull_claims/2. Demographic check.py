# Databricks notebook source
# MAGIC %md
# MAGIC ## Combine patients
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Train test split - patient level data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train test split on positive

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC   ambit_analytics.sps_patient_journey.train_test_split_sps_v2 AS
# MAGIC   WITH sps_with_random AS (
# MAGIC       SELECT
# MAGIC           FORIAN_PATIENT_ID,
# MAGIC           index_diagnosis_date,
# MAGIC           RAND(42) as random_seed,
# MAGIC           ROW_NUMBER() OVER (ORDER BY RAND(42)) as rn,
# MAGIC           COUNT(*) OVER () as total_count
# MAGIC       FROM ambit_analytics.sps_patient_journey.final_sps_v2
# MAGIC   )
# MAGIC   SELECT
# MAGIC       FORIAN_PATIENT_ID,
# MAGIC       index_diagnosis_date,
# MAGIC       1 as sps_flag,
# MAGIC       CASE
# MAGIC           WHEN rn <= FLOOR(total_count * 0.8) THEN 'train'
# MAGIC           WHEN rn <= FLOOR(total_count * 0.9) THEN 'val'
# MAGIC           ELSE 'test'
# MAGIC       END as split_flag
# MAGIC   FROM sps_with_random;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train test split on negative

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE
# MAGIC   ambit_analytics.sps_patient_journey.train_test_split_ctr_v2 AS
# MAGIC   WITH control_with_random AS (
# MAGIC       SELECT
# MAGIC           FORIAN_PATIENT_ID,
# MAGIC           first_control_diagnosis_date,
# MAGIC           RAND(42) as random_seed,
# MAGIC           ROW_NUMBER() OVER (ORDER BY RAND(42)) as rn,
# MAGIC           COUNT(*) OVER () as total_count
# MAGIC       FROM ambit_analytics.sps_patient_journey.final_ctr_v2
# MAGIC   )
# MAGIC   SELECT
# MAGIC       FORIAN_PATIENT_ID,
# MAGIC       first_control_diagnosis_date as index_diagnosis_date,
# MAGIC       0 as sps_flag,
# MAGIC       CASE
# MAGIC           WHEN rn <= FLOOR(total_count * 0.8) THEN 'train'
# MAGIC           WHEN rn <= FLOOR(total_count * 0.9) THEN 'val'
# MAGIC           ELSE 'test'
# MAGIC       END as split_flag
# MAGIC   FROM control_with_random;

# COMMAND ----------

# MAGIC %sql
# MAGIC select train_test_split_ctr_v2.*
# MAGIC from ambit_analytics.sps_patient_journey.train_test_split_ctr_v2
# MAGIC left join bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_diagnosis d
# MAGIC on d.FORIAN_PATIENT_ID = train_test_split_ctr_v2.FORIAN_PATIENT_ID
# MAGIC where d.D_DIAGNOSIS_CODE = "G2582"

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC where sps_flag = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Demographic comparison

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Combine patients

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace table
# MAGIC   ambit_analytics.sps_patient_journey.final_combined_pts_v2 AS
# MAGIC select * from ambit_analytics.sps_patient_journey.train_test_split_sps_v2
# MAGIC UNION ALL 
# MAGIC select * from ambit_analytics.sps_patient_journey.train_test_split_ctr_v2

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct FORIAN_PATIENT_ID) - 324748 - 4803
# MAGIC from ambit_analytics.sps_patient_journey.final_combined_pts_v2 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2 Age check + sex check - impute dob & age + remove duplicate dob OR age

# COMMAND ----------

# MAGIC %sql
# MAGIC -- create YOB & gender code
# MAGIC CREATE OR REPLACE TABLE
# MAGIC       ambit_analytics.sps_patient_journey.age_sex_check_step1 AS
# MAGIC       SELECT
# MAGIC           a.*,
# MAGIC           b.BIRTH_YEAR,
# MAGIC           b.GENDER_CODE
# MAGIC       FROM ambit_analytics.sps_patient_journey.final_combined_pts_v2 a
# MAGIC       INNER JOIN (
# MAGIC           SELECT 
# MAGIC               DISTINCT
# MAGIC               FORIAN_PATIENT_ID,
# MAGIC               BIRTH_YEAR,
# MAGIC               GENDER_CODE
# MAGIC           FROM
# MAGIC   bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_patient
# MAGIC       ) b
# MAGIC       ON a.FORIAN_PATIENT_ID = b.FORIAN_PATIENT_ID;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(DISTINCT FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.age_sex_check_step1
# MAGIC where 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- remove duplicate yob OR sex
# MAGIC CREATE OR REPLACE TABLE
# MAGIC       ambit_analytics.sps_patient_journey.age_sex_check_step2 AS
# MAGIC       SELECT *
# MAGIC       FROM ambit_analytics.sps_patient_journey.age_sex_check_step1
# MAGIC       WHERE FORIAN_PATIENT_ID NOT IN (
# MAGIC           SELECT FORIAN_PATIENT_ID
# MAGIC           FROM
# MAGIC   ambit_analytics.sps_patient_journey.age_sex_check_step1
# MAGIC           GROUP BY FORIAN_PATIENT_ID
# MAGIC           HAVING COUNT(DISTINCT BIRTH_YEAR) > 1 OR COUNT(DISTINCT GENDER_CODE) > 1
# MAGIC       );
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(FORIAN_PATIENT_ID)
# MAGIC from ambit_analytics.sps_patient_journey.age_sex_check_step2

# COMMAND ----------

# MAGIC %sql
# MAGIC -- age dist comparison
# MAGIC SELECT
# MAGIC       sps_flag,
# MAGIC       CASE WHEN sps_flag = 1 THEN 'SPS' ELSE 'Control' END as group_name,
# MAGIC       COUNT(*) as patient_count,
# MAGIC       ROUND(AVG(YEAR(index_diagnosis_date) - BIRTH_YEAR), 2) as mean_age,
# MAGIC       MIN(YEAR(index_diagnosis_date) - BIRTH_YEAR) as min_age,
# MAGIC       MAX(YEAR(index_diagnosis_date) - BIRTH_YEAR) as max_age,
# MAGIC       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
# MAGIC   YEAR(index_diagnosis_date) - BIRTH_YEAR) as median_age
# MAGIC   FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC   WHERE BIRTH_YEAR IS NOT NULL
# MAGIC   GROUP BY sps_flag
# MAGIC   ORDER BY sps_flag DESC;
# MAGIC
# MAGIC   

# COMMAND ----------

# MAGIC %sql
# MAGIC -- sex dist check
# MAGIC SELECT
# MAGIC       sps_flag,
# MAGIC       CASE WHEN sps_flag = 1 THEN 'SPS' ELSE 'Control' END as group_name,
# MAGIC       GENDER_CODE,
# MAGIC       COUNT(*) as patient_count,
# MAGIC       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY sps_flag),
# MAGIC   2) as percentage_within_group
# MAGIC   FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC   WHERE GENDER_CODE IN ('M', 'F')
# MAGIC   GROUP BY sps_flag, GENDER_CODE
# MAGIC   ORDER BY sps_flag DESC, GENDER_CODE;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- standardized mean difference
# MAGIC WITH age_stats AS (
# MAGIC       SELECT
# MAGIC           sps_flag,
# MAGIC           AVG(YEAR(index_diagnosis_date) - BIRTH_YEAR) as mean_age,
# MAGIC           VAR_SAMP(YEAR(index_diagnosis_date) - BIRTH_YEAR) as var_age,
# MAGIC           COUNT(*) as n_age
# MAGIC       FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC       WHERE BIRTH_YEAR IS NOT NULL
# MAGIC       GROUP BY sps_flag
# MAGIC   ),
# MAGIC   sex_stats AS (
# MAGIC       SELECT
# MAGIC           sps_flag,
# MAGIC           SUM(CASE WHEN GENDER_CODE = 'M' THEN 1 ELSE 0 END) / COUNT(*) as
# MAGIC   prop_male,
# MAGIC           COUNT(*) as n_sex
# MAGIC       FROM ambit_analytics.sps_patient_journey.age_sex_check_step2
# MAGIC       WHERE GENDER_CODE IN ('M', 'F')
# MAGIC       GROUP BY sps_flag
# MAGIC   ),
# MAGIC   age_smd AS (
# MAGIC       SELECT
# MAGIC           'Age' as variable,
# MAGIC           ABS(
# MAGIC               (SELECT mean_age FROM age_stats WHERE sps_flag = 1) -
# MAGIC               (SELECT mean_age FROM age_stats WHERE sps_flag = 0)
# MAGIC           ) / SQRT(
# MAGIC               (
# MAGIC                   (SELECT var_age FROM age_stats WHERE sps_flag = 1) +
# MAGIC                   (SELECT var_age FROM age_stats WHERE sps_flag = 0)
# MAGIC               ) / 2
# MAGIC           ) as smd
# MAGIC   ),
# MAGIC   sex_smd AS (
# MAGIC       SELECT
# MAGIC           'Sex (% Male)' as variable,
# MAGIC           ABS(
# MAGIC               (SELECT prop_male FROM sex_stats WHERE sps_flag = 1) -
# MAGIC               (SELECT prop_male FROM sex_stats WHERE sps_flag = 0)
# MAGIC           ) / SQRT(
# MAGIC               (
# MAGIC                   (SELECT prop_male * (1 - prop_male) FROM sex_stats WHERE
# MAGIC   sps_flag = 1) +
# MAGIC                   (SELECT prop_male * (1 - prop_male) FROM sex_stats WHERE
# MAGIC   sps_flag = 0)
# MAGIC               ) / 2
# MAGIC           ) as smd
# MAGIC   )
# MAGIC   SELECT
# MAGIC       variable,
# MAGIC       ROUND(smd, 4) as standardized_mean_difference,
# MAGIC       CASE
# MAGIC           WHEN smd < 0.1 THEN 'Small difference'
# MAGIC           WHEN smd < 0.2 THEN 'Medium difference'
# MAGIC           ELSE 'Large difference'
# MAGIC       END as interpretation
# MAGIC   FROM age_smd
# MAGIC   UNION ALL
# MAGIC   SELECT
# MAGIC       variable,
# MAGIC       ROUND(smd, 4) as standardized_mean_difference,
# MAGIC       CASE
# MAGIC           WHEN smd < 0.1 THEN 'Small difference'
# MAGIC           WHEN smd < 0.2 THEN 'Medium difference'
# MAGIC           ELSE 'Large difference'
# MAGIC       END as interpretation
# MAGIC       FROM sex_smd;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3 Payer type check

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH patient_payer_counts AS (
# MAGIC       SELECT
# MAGIC           a.FORIAN_PATIENT_ID,
# MAGIC           a.sps_flag,
# MAGIC           p.L_TYPE_OF_COVERAGE_CODE,
# MAGIC           COUNT(*) as payer_claim_count
# MAGIC       FROM ambit_analytics.sps_patient_journey.final_combined_pts_v2 a
# MAGIC       INNER JOIN
# MAGIC   bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_payer p
# MAGIC           ON a.FORIAN_PATIENT_ID = p.FORIAN_PATIENT_ID
# MAGIC       WHERE p.L_TYPE_OF_COVERAGE_CODE IS NOT NULL
# MAGIC       GROUP BY a.FORIAN_PATIENT_ID, a.sps_flag, p.L_TYPE_OF_COVERAGE_CODE
# MAGIC   ),
# MAGIC   patient_most_common_payer AS (
# MAGIC       SELECT
# MAGIC           FORIAN_PATIENT_ID,
# MAGIC           sps_flag,
# MAGIC           L_TYPE_OF_COVERAGE_CODE as most_common_payer,
# MAGIC           payer_claim_count,
# MAGIC           ROW_NUMBER() OVER (PARTITION BY FORIAN_PATIENT_ID ORDER BY
# MAGIC   payer_claim_count DESC, L_TYPE_OF_COVERAGE_CODE) as rn
# MAGIC       FROM patient_payer_counts
# MAGIC   )
# MAGIC   -- Compare payer distribution between SPS and control groups
# MAGIC   SELECT
# MAGIC       sps_flag,
# MAGIC       CASE WHEN sps_flag = 1 THEN 'SPS' ELSE 'Control' END as group_name,
# MAGIC       most_common_payer,
# MAGIC       COUNT(*) as patient_count,
# MAGIC       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY sps_flag),
# MAGIC   2) as percentage_within_group
# MAGIC   FROM patient_most_common_payer
# MAGIC   WHERE rn = 1
# MAGIC   GROUP BY sps_flag, most_common_payer
# MAGIC   ORDER BY sps_flag DESC, patient_count DESC;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- standardized mean difference
# MAGIC WITH patient_payer_counts AS (
# MAGIC       SELECT
# MAGIC           a.FORIAN_PATIENT_ID,
# MAGIC           a.sps_flag,
# MAGIC           p.L_TYPE_OF_COVERAGE_CODE,
# MAGIC           COUNT(*) as payer_claim_count
# MAGIC       FROM ambit_analytics.sps_patient_journey.final_combined_pts_v2 a
# MAGIC       INNER JOIN
# MAGIC   bronze.forian.new_mx_source_jan_2025_open_medical_hospital_claim_payer p
# MAGIC           ON a.FORIAN_PATIENT_ID = p.FORIAN_PATIENT_ID
# MAGIC       WHERE p.L_TYPE_OF_COVERAGE_CODE IS NOT NULL
# MAGIC       GROUP BY a.FORIAN_PATIENT_ID, a.sps_flag, p.L_TYPE_OF_COVERAGE_CODE
# MAGIC   ),
# MAGIC   patient_most_common_payer AS (
# MAGIC       SELECT
# MAGIC           FORIAN_PATIENT_ID,
# MAGIC           sps_flag,
# MAGIC           L_TYPE_OF_COVERAGE_CODE as most_common_payer,
# MAGIC           ROW_NUMBER() OVER (PARTITION BY FORIAN_PATIENT_ID ORDER BY
# MAGIC   payer_claim_count DESC, L_TYPE_OF_COVERAGE_CODE) as rn
# MAGIC       FROM patient_payer_counts
# MAGIC   ),
# MAGIC   payer_props AS (
# MAGIC       SELECT
# MAGIC           sps_flag,
# MAGIC           most_common_payer,
# MAGIC           COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY sps_flag) as
# MAGIC   proportion
# MAGIC       FROM patient_most_common_payer
# MAGIC       WHERE rn = 1
# MAGIC       GROUP BY sps_flag, most_common_payer
# MAGIC   ),
# MAGIC   payer_smd AS (
# MAGIC       SELECT
# MAGIC           p1.most_common_payer as payer_type,
# MAGIC           COALESCE(p1.proportion, 0) as prop_sps,
# MAGIC           COALESCE(p2.proportion, 0) as prop_control,
# MAGIC           ABS(COALESCE(p1.proportion, 0) - COALESCE(p2.proportion, 0)) /
# MAGIC           SQRT(
# MAGIC               (COALESCE(p1.proportion, 0) * (1 - COALESCE(p1.proportion,
# MAGIC   0)) +
# MAGIC                COALESCE(p2.proportion, 0) * (1 - COALESCE(p2.proportion,
# MAGIC   0))) / 2
# MAGIC           ) as smd
# MAGIC       FROM (SELECT most_common_payer, proportion FROM payer_props WHERE
# MAGIC   sps_flag = 1) p1
# MAGIC       FULL OUTER JOIN (SELECT most_common_payer, proportion FROM
# MAGIC   payer_props WHERE sps_flag = 0) p2
# MAGIC           ON p1.most_common_payer = p2.most_common_payer
# MAGIC   )
# MAGIC   SELECT
# MAGIC       payer_type,
# MAGIC       ROUND(prop_sps * 100, 2) as sps_percentage,
# MAGIC       ROUND(prop_control * 100, 2) as control_percentage,
# MAGIC       ROUND(smd, 4) as standardized_mean_difference,
# MAGIC       CASE
# MAGIC           WHEN smd < 0.1 THEN 'Small difference'
# MAGIC           WHEN smd < 0.2 THEN 'Medium difference'
# MAGIC           ELSE 'Large difference'
# MAGIC       END as interpretation
# MAGIC   FROM payer_smd
# MAGIC   WHERE smd > 0
# MAGIC   ORDER BY smd DESC;