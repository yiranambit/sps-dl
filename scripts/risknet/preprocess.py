# -*- encoding: utf-8 -*-
"""
Preprocessing script for RiskNet.
"""

from __future__ import annotations

import os

os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import logging

import numpy as np
import polars as pl
import typing as t

from joblib import delayed
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from lpm.data.datasets.risknet import (
    Patient,
    PatientCollection,
    TrajectoryValidator,
)
from lpm.data.datasets.risknet.dataset import serialize_patients_parallel
from lpm.utils.progress import ParallelTqdm


log = logging.getLogger(__name__)


def load_patient_info(file_path: Path, config: DictConfig = None) -> pl.DataFrame:
    """Reads the raw patient info data."""
    # cols = ["PATIENT_ID", "GENDER", "DIAGNOSIS", "DATE_OF_BIRTH", "DIAGNOSIS_DATE"]

    data = pl.read_csv(file_path, try_parse_dates=True)
    data = data.rename(lambda x: x.upper())

    filter_expr = ""

    if config is not None:
        filter_expr = config.params.filter_expr.patient_info
    else:
        filter_expr = "true"

    data = data.filter(pl.sql_expr(filter_expr))

    return (
        data.select(
            ("PATIENT_ID"),
            (pl.col("DATE_OF_BIRTH").cast(pl.Datetime)),
            (pl.col("DIAGNOSIS") == config.indication).cast(pl.Int64).fill_null(0),
            (pl.col("DIAGNOSIS_DATE").cast(pl.Datetime)),
        )
        .unique("PATIENT_ID")
        .with_columns((pl.col("DIAGNOSIS") == 1).cast(int).alias("HAS_DIAGNOSIS"))
    )


def load_patient_claims(file_path: Path, config: DictConfig = None) -> pl.DataFrame:
    """Reads the raw patient claims data."""
    # cols = ["PATIENT_ID", "DATE_OF_SERVICE", "DIAGNOSIS_CODE"]

    if "feather" in file_path:  # pyright: ignore[reportOperatorIssue]
        data = pl.read_ipc(file_path)
    else:
        data = pl.read_csv(file_path, try_parse_dates=True)

    data = data.rename(lambda x: x.upper())

    if config is not None:
        filter_expr = config.params.filter_expr.patient_claims
    else:
        filter_expr = "true"

    data = data.filter(pl.sql_expr(filter_expr)).filter(
        pl.col("DIAGNOSIS_CODE").str.contains("^[A-Z]")
    )

    if config is not None and config.paths.code_inclusion_list is not None:
        code_inclusion = pl.read_csv(config.paths.code_inclusion_list)

        if config.params.filter_expr.code_inclusions is not None:
            code_inclusion = code_inclusion.filter(
                pl.sql_expr(config.params.filter_expr.code_inclusions)
            )

        code_inclusion = code_inclusion["diagnosis_code"].to_list()

        data = data.filter(pl.col("DIAGNOSIS_CODE").is_in(code_inclusion))

    return data.select(
        "PATIENT_ID", pl.col("DATE_OF_SERVICE").cast(pl.Datetime), "DIAGNOSIS_CODE"
    )


def load_diagnosis_code_info(file_path: Path) -> pl.DataFrame:
    """"""
    return pl.read_csv(file_path, columns=["CODE", "DESCRIPTION"])


def get_status_epilepticus_codes(code_info: pl.DataFrame) -> t.List[str]:
    """"""
    # FIXME: confirm that this is not case sensitive
    filter_ = pl.col("DESCRIPTION").str.contains("status epilepticus")
    return code_info.filter(filter_)["CODE"].to_list()


def _sanitize_icd10_code(code: str) -> str:
    """Sanitize an ICD10 code."""
    return "".join([c for c in code if c.isalnum()]).upper()


def process_patient_claims(
    patient_claims: pl.DataFrame,
    code_num_chars: int,
    params: DictConfig | None = None,
) -> pl.DataFrame:
    """Processes the raw patient claims data."""
    if params.code_blacklist is None:
        code_blacklist = []
    else:
        code_blacklist = params.code_blacklist

    code_blacklist = [_sanitize_icd10_code(code) for code in code_blacklist]

    def filter_same_code_claims(df):
        if params.minimal_days_between_same_code_claims is not None:
            df = (
                df.with_columns(
                    PREV_ICD_CODE_DATE=pl.col("DATE_OF_SERVICE")
                    .shift(1)
                    .over(
                        partition_by=[pl.col("PATIENT_ID"), pl.col("DIAGNOSIS_CODE")],
                        order_by=pl.col("DATE_OF_SERVICE"),
                    )
                )
                .with_columns(
                    DATE_BETWEEN_TWO_ICD_CLAIMS=(
                        pl.col("DATE_OF_SERVICE") - pl.col("PREV_ICD_CODE_DATE")
                    ).dt.total_days()
                )
                .filter(
                    (
                        pl.col("DATE_BETWEEN_TWO_ICD_CLAIMS")
                        > params.minimal_days_between_same_code_claims
                    )
                    | pl.col("DATE_BETWEEN_TWO_ICD_CLAIMS").is_null()
                )
            )
            return df

        else:
            return df

    return (
        patient_claims.select(
            ("PATIENT_ID"),
            ("DATE_OF_SERVICE"),
            # filter non-alphanumberic characters
            (
                pl.col("DIAGNOSIS_CODE")
                .str.replace_all(r"[^0-9A-Za-z_]", "")
                .str.to_uppercase()
                .str.strip_chars()
            ),
        )
        .drop_nulls("DIAGNOSIS_CODE")
        .filter(~pl.col("DIAGNOSIS_CODE").is_in(code_blacklist))
        .select(
            ("PATIENT_ID"),
            ("DATE_OF_SERVICE"),
            # restrict to max characters in the ICD10 codes
            (
                pl.when(pl.col("DIAGNOSIS_CODE").str.contains("^(PX)|(RX)"))
                .then(pl.col("DIAGNOSIS_CODE"))
                .otherwise(pl.col("DIAGNOSIS_CODE").str.slice(0, code_num_chars))
            ).alias("DIAGNOSIS_CODE"),
        )
        .unique(["PATIENT_ID", "DATE_OF_SERVICE", "DIAGNOSIS_CODE"])
        .sort(["PATIENT_ID", "DATE_OF_SERVICE"])
        .pipe(filter_same_code_claims)
        .select("PATIENT_ID", "DATE_OF_SERVICE", "DIAGNOSIS_CODE")
        .group_by("PATIENT_ID")
        .agg("DATE_OF_SERVICE", "DIAGNOSIS_CODE")
    )


def make_patient_parser(
    trajectory_validator: TrajectoryValidator,
) -> t.Callable[[t.Tuple[t.Any]], Patient]:
    """"""

    def struct_parser(struct: t.Tuple[t.Any]) -> Patient:
        """Parse raw patient data from a struct of a DataFrame."""
        code_sequence = struct["DIAGNOSIS_CODE"]
        date_sequence = struct["DATE_OF_SERVICE"]

        # use the diagnosis date as the outcome date if the patient has a diagnosis
        outcome_date = (
            struct["DIAGNOSIS_DATE"]
            if struct["HAS_DIAGNOSIS"] == 1
            else date_sequence[-1]
        )

        return Patient(
            id=struct["PATIENT_ID"],
            dob=struct["DATE_OF_BIRTH"],
            outcome_date=outcome_date,
            future_diagnosis=bool(struct["HAS_DIAGNOSIS"]),
            events=list(zip(date_sequence, code_sequence)),
            trajectory_validator=trajectory_validator,
        )

    return struct_parser


def get_age_at_outcome(patient: Patient, split=0.8) -> float:
    """Get the age of a patient at the outcome date."""
    outcome_age = (patient.outcome_date - patient.dob).days / 365
    return np.floor(outcome_age * split) / split


def mask_low_frequency_values(
    x: np.ndarray, min_freq: int = 5, mask_value: int = -1
) -> np.ndarray:
    """Mask low-frequency values in an array."""
    uniq, counts = np.unique(x, return_counts=True)
    mask_values = uniq[counts < min_freq]
    return np.where(np.isin(x, mask_values), mask_value, x)


def downsample_negative_patients(
    patients: PatientCollection, num_negative_samples: int, seed: int
) -> PatientCollection:
    """Downsample negative patients."""
    rng = np.random.default_rng(seed)
    pos_patients = patients.positive_patients
    neg_patients = rng.choice(
        patients.negative_patients,
        num_negative_samples,
        replace=(len(patients.negative_patients) < num_negative_samples),
    )
    return PatientCollection(*pos_patients, *list(neg_patients))


def parse_patients_parallel(
    patient_structs: pl.Series,
    trajectory_validator: TrajectoryValidator,
    num_processes: int,
    **kwargs,
) -> PatientCollection:
    """Parse patients from a DataFrame."""
    parser = make_patient_parser(trajectory_validator)

    parallel = ParallelTqdm(
        total_tasks=len(patient_structs), n_jobs=num_processes, **kwargs
    )

    patients = parallel(delayed(parser)(s) for s in patient_structs)
    patients = filter(lambda pt: pt.has_valid_trajectories, patients)

    return PatientCollection(*patients)


@hydra.main(
    version_base=None, config_path="../../config/risknet", config_name="preprocess"
)
def preprocess(cfg: DictConfig) -> None:
    """Runs data preprocessing for RiskNet."""
    log.info("Loading data...")

    paths = cfg.paths
    params = cfg.params

    patient_info = load_patient_info(paths.patient_info, cfg)
    patient_claims = pl.concat(
        list(map(lambda x: load_patient_claims(x, cfg), paths.patient_claims))
    )

    # add separate status epilepticus claims
    if params.status_epilepticus_claims:
        code_info = load_diagnosis_code_info(paths.code_info)
        se_codes = get_status_epilepticus_codes(code_info)
        se_claims = patient_claims.filter(pl.col("DIAGNOSIS_CODE").is_in(se_codes))
        se_claims = se_claims.select(
            ["PATIENT_ID", "DATE_OF_SERVICE", pl.lit("ZZZZZ").alias("DIAGNOSIS_CODE")]
        )
        patient_claims = pl.concat([patient_claims, se_claims])

    patient_claims = process_patient_claims(
        patient_claims, params.code_num_chars, params
    )

    patient_structs = patient_info.join(patient_claims, on="PATIENT_ID", how="inner")
    patient_structs = patient_structs.select(pl.struct(pl.all()).alias("STRUCT"))

    # downsample negative patients
    pos_structs = patient_structs.filter(pl.col("STRUCT").struct["HAS_DIAGNOSIS"] == 1)
    neg_structs = patient_structs.filter(pl.col("STRUCT").struct["HAS_DIAGNOSIS"] == 0)
    neg_structs = neg_structs.sample(params.num_negative_samples * 3, seed=params.seed)

    patient_structs = pl.concat([pos_structs, neg_structs])

    log.info("Parsing patients from data...")

    trajectory_validator = TrajectoryValidator(cfg.data)

    patients = parse_patients_parallel(
        patient_structs["STRUCT"],
        trajectory_validator=trajectory_validator,
        num_processes=params.num_processes,
    )
    patients = downsample_negative_patients(
        patients, params.num_negative_samples, seed=params.seed
    )

    log.info("Splitting into train/test...")

    pos_outcome_ages = [get_age_at_outcome(pt) for pt in patients.positive_patients]
    neg_outcome_ages = [get_age_at_outcome(pt) for pt in patients.negative_patients]

    # mask low-frequency values to prevent errors with stratification
    pos_outcome_ages = mask_low_frequency_values(pos_outcome_ages)
    neg_outcome_ages = mask_low_frequency_values(neg_outcome_ages)

    train_pos_patients, test_pos_patients = train_test_split(
        patients.positive_patients,
        test_size=cfg.params.test_size,
        random_state=cfg.params.seed,
        stratify=pos_outcome_ages,
    )

    train_neg_patients, test_neg_patients = train_test_split(
        patients.negative_patients,
        test_size=cfg.params.test_size,
        random_state=cfg.params.seed,
        stratify=neg_outcome_ages,
    )

    log.info("Serializing patients...")

    # FIXME: add cleanup of past files in directories

    serialize_patients_parallel(
        train_pos_patients, Path("./train/pos"), params.num_processes
    )

    serialize_patients_parallel(
        train_neg_patients, Path("./train/neg"), params.num_processes
    )

    serialize_patients_parallel(
        test_pos_patients, Path("./test/pos"), params.num_processes
    )

    serialize_patients_parallel(
        test_neg_patients, Path("./test/neg"), params.num_processes
    )


if __name__ == "__main__":
    preprocess()
