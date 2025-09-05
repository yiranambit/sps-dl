# Databricks notebook source
# MAGIC %sql
# MAGIC with filtered_claims as (
# MAGIC         select
# MAGIC           a.FORIAN_PATIENT_ID,
# MAGIC           a.DATE_OF_SERVICE,
# MAGIC           a.D_DIAGNOSIS_CODE,
# MAGIC           p.sps_flag
# MAGIC         from ambit_analytics.sps_patient_journey.all_claims_split_v3 a
# MAGIC         inner join ambit_analytics.sps_patient_journey.age_sex_check_step2 p
# MAGIC           on a.FORIAN_PATIENT_ID = p.FORIAN_PATIENT_ID
# MAGIC         where a.D_DIAGNOSIS_CODE in (
# MAGIC           select code
# MAGIC           from ambit_analytics.sps_patient_journey.code_frequencies_train
# MAGIC           where selected_for_vocab = 1
# MAGIC         )
# MAGIC         and a.D_DIAGNOSIS_CODE not in (
# MAGIC           select code
# MAGIC           from ambit_analytics.sps_patient_journey.control_ref_less
# MAGIC         )
# MAGIC         and a.D_DIAGNOSIS_CODE != 'G2582'
# MAGIC         group by a.FORIAN_PATIENT_ID, a.DATE_OF_SERVICE, 
# MAGIC   a.D_DIAGNOSIS_CODE,
# MAGIC     p.sps_flag
# MAGIC       ),
# MAGIC       patient_lengths as (
# MAGIC         select
# MAGIC           FORIAN_PATIENT_ID,
# MAGIC           sps_flag,
# MAGIC           count(*) as trajectory_length
# MAGIC         from filtered_claims
# MAGIC         group by FORIAN_PATIENT_ID, sps_flag
# MAGIC       ),
# MAGIC       valid_patients as (
# MAGIC         select
# MAGIC           FORIAN_PATIENT_ID,
# MAGIC           sps_flag,
# MAGIC           case
# MAGIC             when trajectory_length > 200 then 200
# MAGIC             else trajectory_length
# MAGIC           end as capped_length
# MAGIC         from patient_lengths
# MAGIC         where trajectory_length >= 30
# MAGIC       )
# MAGIC       select
# MAGIC         case when sps_flag = 1 then 'SPS' else 'Control' end as 
# MAGIC   patient_type,
# MAGIC         count(*) as num_patients,
# MAGIC         avg(capped_length) as avg_trajectory_length,
# MAGIC         min(capped_length) as min_length,
# MAGIC         max(capped_length) as max_length
# MAGIC       from valid_patients
# MAGIC       group by sps_flag
# MAGIC
# MAGIC       union all
# MAGIC
# MAGIC       select
# MAGIC         'Total SPS' as patient_type,
# MAGIC         sum(case when sps_flag = 1 then 1 else 0 end) as num_patients,
# MAGIC         null as avg_trajectory_length,
# MAGIC         null as min_length,
# MAGIC         null as max_length
# MAGIC       from valid_patients
# MAGIC
# MAGIC       order by patient_type 
# MAGIC   desc
# MAGIC