# Databricks notebook source
# MAGIC %md
# MAGIC ## 1. Get all claims before index diagnosis from train set only (before 50-225 truncation)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Combine training sps + ctr claims
# MAGIC create or replace table ambit_analytics.sps_patient_journey.all_claims_b_index_training
# MAGIC select *
# MAGIC from ambit_analytics.sps_patient_journey.all_claims_split_v3
# MAGIC where split_flag = 'train'

# COMMAND ----------

# MAGIC %sql
# MAGIC -- find t2k codes from train set
# MAGIC -- Calculate code frequency for training set only with filtered discriminative score
# MAGIC   CREATE OR REPLACE TABLE
# MAGIC     ambit_analytics.sps_patient_journey.code_frequencies_train AS
# MAGIC
# MAGIC     WITH code_patient_counts AS (
# MAGIC         SELECT
# MAGIC             D_DIAGNOSIS_CODE as code,
# MAGIC             COUNT(DISTINCT CASE WHEN sps_flag = 1 THEN FORIAN_PATIENT_ID
# MAGIC   END)
# MAGIC      as n_sps_pts,
# MAGIC             COUNT(DISTINCT CASE WHEN sps_flag = 0 THEN FORIAN_PATIENT_ID
# MAGIC   END)
# MAGIC      as n_control_pts,
# MAGIC             COUNT(DISTINCT FORIAN_PATIENT_ID) as n_total_pts
# MAGIC         FROM
# MAGIC   ambit_analytics.sps_patient_journey.all_claims_b_index_training
# MAGIC         GROUP BY D_DIAGNOSIS_CODE
# MAGIC     ),
# MAGIC
# MAGIC     patient_totals AS (
# MAGIC         SELECT
# MAGIC             COUNT(DISTINCT CASE WHEN sps_flag = 1 THEN FORIAN_PATIENT_ID
# MAGIC   END)
# MAGIC      as total_sps_pts,
# MAGIC             COUNT(DISTINCT CASE WHEN sps_flag = 0 THEN FORIAN_PATIENT_ID
# MAGIC   END)
# MAGIC      as total_control_pts
# MAGIC         FROM
# MAGIC   ambit_analytics.sps_patient_journey.all_claims_b_index_training
# MAGIC     ),
# MAGIC
# MAGIC     code_with_frequencies AS (
# MAGIC         SELECT
# MAGIC             c.code,
# MAGIC             c.n_sps_pts,
# MAGIC             c.n_control_pts,
# MAGIC             c.n_total_pts,
# MAGIC             ROUND(c.n_sps_pts * 100.0 / t.total_sps_pts, 4) as
# MAGIC   freq_sps_pct,
# MAGIC             ROUND(c.n_control_pts * 100.0 / t.total_control_pts, 4) as
# MAGIC     freq_control_pct,
# MAGIC             LOG2((c.n_sps_pts + 1.0) / (c.n_control_pts + 1.0) *
# MAGIC   t.total_control_pts / t.total_sps_pts) as log2_relative_freq,
# MAGIC             ABS(LOG2((c.n_sps_pts + 1.0) / (c.n_control_pts + 1.0) *
# MAGIC   t.total_control_pts / t.total_sps_pts)) as abs_log2_relative_freq,
# MAGIC             -- Modified penalized discriminative score with filtering condition and discriminative threshold
# MAGIC             CASE
# MAGIC                 WHEN c.n_sps_pts <= 5 AND c.n_control_pts <= c.n_sps_pts *
# MAGIC   100 THEN 0
# MAGIC                 WHEN ABS(LOG2((c.n_sps_pts + 1.0) / (c.n_control_pts + 1.0)
# MAGIC    * t.total_control_pts / t.total_sps_pts)) < 0.3 THEN 0
# MAGIC                 ELSE ABS(LOG2((c.n_sps_pts + 1.0) / (c.n_control_pts + 1.0)
# MAGIC    * t.total_control_pts / t.total_sps_pts))
# MAGIC                      * SQRT(GREATEST(c.n_sps_pts * 10, c.n_control_pts / 10))
# MAGIC             END as penalized_discriminative_score,
# MAGIC             t.total_sps_pts,
# MAGIC             t.total_control_pts
# MAGIC         FROM code_patient_counts c
# MAGIC         CROSS JOIN patient_totals t
# MAGIC         WHERE c.n_total_pts >= 5
# MAGIC     ),
# MAGIC
# MAGIC     ranked_codes AS (
# MAGIC         SELECT *,
# MAGIC             ROW_NUMBER() OVER (ORDER BY n_total_pts DESC) as
# MAGIC   rank_total_freq,
# MAGIC             ROW_NUMBER() OVER (ORDER BY n_sps_pts DESC) as rank_sps_freq,
# MAGIC             ROW_NUMBER() OVER (ORDER BY n_control_pts DESC) as
# MAGIC     rank_control_freq,
# MAGIC             ROW_NUMBER() OVER (ORDER BY abs_log2_relative_freq DESC) as
# MAGIC     rank_abs_log2_relative_freq,
# MAGIC             -- Handle NULL values in ranking by placing them at the end
# MAGIC             ROW_NUMBER() OVER (ORDER BY penalized_discriminative_score DESC
# MAGIC     NULLS LAST) as rank_penalized_discriminative
# MAGIC         FROM code_with_frequencies
# MAGIC     )
# MAGIC
# MAGIC     SELECT
# MAGIC         code,
# MAGIC         n_sps_pts,
# MAGIC         n_control_pts,
# MAGIC         n_total_pts,
# MAGIC         freq_sps_pct,
# MAGIC         freq_control_pct,
# MAGIC         log2_relative_freq,
# MAGIC         abs_log2_relative_freq,
# MAGIC         penalized_discriminative_score,
# MAGIC         rank_total_freq,
# MAGIC         rank_sps_freq,
# MAGIC         rank_control_freq,
# MAGIC         rank_abs_log2_relative_freq,
# MAGIC         rank_penalized_discriminative,
# MAGIC         CASE
# MAGIC             WHEN (rank_total_freq <= 500 OR rank_sps_freq <= 500 OR
# MAGIC                   rank_control_freq <= 500 OR
# MAGIC   rank_penalized_discriminative
# MAGIC     <= 500)
# MAGIC             THEN 1
# MAGIC             ELSE 0
# MAGIC         END as selected_for_vocab,
# MAGIC         total_sps_pts,
# MAGIC         total_control_pts
# MAGIC     FROM ranked_codes
# MAGIC     ORDER BY rank_penalized_discriminative;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ambit_analytics.sps_patient_journey.code_frequencies_train

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct code) from ambit_analytics.sps_patient_journey.code_frequencies_train where selected_for_vocab = 1