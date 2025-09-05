# -*- encoding: utf-8 -*-
"""
Training script for RiskNet.
"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import functools
import json
import logging

import numpy as np
import typing as t
import tensorflow as tf

from hydra.core.config_store import ConfigStore
from joblib import delayed
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from keras import layers, callbacks, optimizers

from lpm.data.datasets.risknet import (
    Patient,
    BalancedRiskNetBatchGenerator,
    PatientCollection,
    RiskNetBatchGenerator,
    RiskNetConfig,
    TrajectoryValidator,
)
from lpm.data.datasets.risknet.dataset import deserialize_patient
from lpm.model import RiskNet
from lpm.model.risknet import get_eval_scores
from lpm.utils.progress import ParallelTqdm


log = logging.getLogger(__name__)


def deserialize_patients_parallel(
    file_list: t.List[str | Path],
    trajectory_validator: TrajectoryValidator,
    num_processes: int,
) -> PatientCollection:
    """Deserialize patients from JSON."""

    parallel = ParallelTqdm(total_tasks=len(file_list), n_jobs=num_processes)

    patients = parallel(
        delayed(deserialize_patient)(f, trajectory_validator) for f in file_list
    )

    return PatientCollection(*patients)


def load_dataset(
    data_dir: Path, trajectory_validator: TrajectoryValidator, num_processes: int
):
    """Load a dataset from disk."""
    deserializer = functools.partial(
        deserialize_patients_parallel,
        trajectory_validator=trajectory_validator,
        num_processes=num_processes,
    )

    train_pos = deserializer(list(data_dir.glob("train/pos/*.json")))
    train_neg = deserializer(list(data_dir.glob("train/neg/*.json")))
    test_pos = deserializer(list(data_dir.glob("test/pos/*.json")))
    test_neg = deserializer(list(data_dir.glob("test/neg/*.json")))

    return train_pos, train_neg, test_pos, test_neg


def build_vocab_from_patients(patients: PatientCollection) -> np.ndarray:
    """Build a vocabulary from a list of patients."""
    all_codes = []
    for pt in patients:
        codes = [event[1] for event in pt.events]
        all_codes.extend(codes)
    return np.unique(all_codes)


cs = ConfigStore.instance()
cs.store(name="config", node=RiskNetConfig)


def train_and_evaluate_fold(
    train_pos: PatientCollection,
    train_neg: PatientCollection,
    val_pos: PatientCollection,
    val_neg: PatientCollection,
    cfg: RiskNetConfig,
    fold: int,
) -> t.Dict[str, float]:
    """Train and evaluate a fold of the model."""

    log.info(f"Training fold {fold}...")

    train_vocab = build_vocab_from_patients(train_pos + train_neg)
    tokenizer = layers.StringLookup(vocabulary=train_vocab, mask_token="")
    vocab_size = tokenizer.vocabulary_size()

    train_gen = BalancedRiskNetBatchGenerator(
        cfg.model.batch_size,
        cfg.model.max_sequence_length,
        tokenizer=tokenizer,
        n_trajectories=cfg.model.n_trajectories,
    )

    val_gen = RiskNetBatchGenerator(
        cfg.model.eval_batch_size,
        cfg.model.max_sequence_length,
        tokenizer=tokenizer,
        n_trajectories=cfg.model.eval_n_trajectories,
    )

    model = RiskNet(
        input_dim=cfg.model.max_sequence_length,
        vocab_size=vocab_size,
        token_embed_dim=cfg.model.token_embed_dim,
        age_embed_dim=cfg.model.age_embed_dim,
        num_heads=cfg.model.num_heads,
        num_blocks=cfg.model.num_blocks,
        output_dim=len(cfg.data.month_endpoints),
        min_time_embed_period=cfg.data.min_time_embed_period_in_days,
        max_time_embed_period=cfg.data.max_time_embed_period_in_days,
    )

    model.compile(optimizer=optimizers.Adam(cfg.model.learning_rate))

    train_seq = train_gen.flow(
        train_pos, train_neg, shuffle=True, seed=cfg.preprocess.seed
    )
    val_seq = val_gen.flow(val_pos, val_neg, seed=cfg.preprocess.seed)

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=30,
        restore_best_weights=True,
        start_from_epoch=20,
    )

    model.fit(
        train_seq,
        epochs=cfg.model.epochs,
        validation_data=val_seq,
        callbacks=[early_stopping],
    )

    eval_scores = get_eval_scores(model, val_seq, cfg.data.month_endpoints)

    return eval_scores


def get_age_at_outcome(patient: Patient) -> float:
    """Get the age of a patient at the outcome date."""
    outcome_age = (patient.outcome_date - patient.dob).days / 365
    return np.floor(outcome_age * 2) / 2


@hydra.main(version_base=None, config_path="../../config/risknet", config_name="hyper")
def hyper(cfg: RiskNetConfig) -> None:
    """Trains the RiskNet model."""

    log.info("Loading dataset...")

    trajectory_validator = TrajectoryValidator(cfg.data)
    train_pos, train_neg, *_ = load_dataset(
        Path(cfg.paths.raw), trajectory_validator, cfg.preprocess.num_processes
    )

    skf_pos = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.preprocess.seed)
    skf_neg = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.preprocess.seed)

    pos_outcome_ages = [get_age_at_outcome(pt) for pt in train_pos]
    _, pos_outcome_age_enc = np.unique(pos_outcome_ages, return_inverse=True)
    neg_outcome_ages = [get_age_at_outcome(pt) for pt in train_neg]
    _, neg_outcome_age_enc = np.unique(neg_outcome_ages, return_inverse=True)

    pos_split_gen = skf_pos.split(train_pos, pos_outcome_age_enc)
    neg_split_gen = skf_neg.split(train_neg, neg_outcome_age_enc)

    split_gen = enumerate(zip(pos_split_gen, neg_split_gen))

    losses = []
    scores = []
    eval_scores = {}
    for i, ((pos_train_idx, pos_val_idx), (neg_train_idx, neg_val_idx)) in split_gen:
        train_pos_fold = PatientCollection(*[train_pos[i] for i in pos_train_idx])
        val_pos_fold = PatientCollection(*[train_pos[i] for i in pos_val_idx])

        train_neg_fold = PatientCollection(*[train_neg[i] for i in neg_train_idx])
        val_neg_fold = PatientCollection(*[train_neg[i] for i in neg_val_idx])

        fold_eval_scores = train_and_evaluate_fold(
            train_pos_fold, train_neg_fold, val_pos_fold, val_neg_fold, cfg, i
        )

        eval_scores[i] = fold_eval_scores
        losses.append(fold_eval_scores["loss"])
        scores.append(
            fold_eval_scores["endpoint_metrics"][cfg.hyper.endpoint][cfg.hyper.metric]
        )

        tf.keras.backend.clear_session()

        if i + 1 >= cfg.hyper.max_splits:
            break

    with open("scores.json", "w", encoding="utf-8") as fh:
        json.dump(eval_scores, fh, ensure_ascii=False, indent=4)

    return np.mean(scores)


if __name__ == "__main__":
    hyper()
