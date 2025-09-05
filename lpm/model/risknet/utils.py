"""RiskNet utilities."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import typing as t
import sklearn.metrics as skm

from tensorflow import keras

from keras import layers, ops

from lpm.data.datasets.risknet import Patient
from lpm.data.datasets.risknet.sequence import encode_trajectories


if t.TYPE_CHECKING:
    from .model import RiskNet
    from lpm.data.datasets.risknet import RiskNetSequence


def create_embedding_model(model: RiskNet) -> keras.Model:
    """"""
    emb_input = model.encoder.input
    emb_output = [model.layers[2].layers[-2].output, model.encoder.output]

    return keras.Model(inputs=emb_input, outputs=emb_output)


def get_patient_predictions_and_embeddings(
    model: RiskNet,
    patient: Patient,
    max_codes: int,
    max_trajectories: int | None = None,
    return_probs: bool = True,
) -> pd.DataFrame:
    """Generate predictions for all possible trajectories of a patient."""
    emb_model = create_embedding_model(model)

    trajectories = patient.get_trajectories()
    trajectory_ids = list(range(len(trajectories)))

    if max_trajectories is not None and len(trajectories) > max_trajectories:
        trajectory_ids = random.sample(trajectory_ids, max_trajectories)
        trajectories = [trajectories[i] for i in trajectory_ids]

    x, (y_true, _) = encode_trajectories(trajectories, max_codes, model.tokenizer)

    y_pred_emb, y_pred = emb_model(list(x), training=False)

    if return_probs:
        y_pred = ops.sigmoid(y_pred)

    y_pred = pd.DataFrame(y_pred)
    y_true = pd.DataFrame(y_true)
    y_pred_emb = pd.DataFrame(y_pred_emb)

    y_pred.columns = [f"y_pred_{i}" for i in range(y_pred.shape[1])]
    y_true.columns = [f"y_true_{i}" for i in range(y_true.shape[1])]
    y_pred_emb.columns = [f"y_emb_{i}" for i in range(y_pred_emb.shape[1])]

    y_info = pd.DataFrame(
        {
            "patient_id": patient.id,
            "diagnosis_age": (patient.outcome_date - patient.dob).days / 365.25,
            "trajectory_age": x[1][:, -1] / 365.25,
            "trajectory_len": (x[1][:, -1] - x[1][:, 0]) / 365.25,
            "trajectory_id": trajectory_ids,
        }
    )

    return pd.concat([y_info, y_true, y_pred, y_pred_emb], axis=1)


def get_patient_predictions(
    model: RiskNet,
    patient: Patient,
    max_codes: int,
    max_trajectories: int | None = None,
    return_probs: bool = True,
) -> pd.DataFrame:
    """Generate predictions for all possible trajectories of a patient."""
    trajectories = patient.get_trajectories()
    trajectory_ids = list(range(len(trajectories)))

    if max_trajectories is not None and len(trajectories) > max_trajectories:
        trajectory_ids = random.sample(trajectory_ids, max_trajectories)
        trajectories = [trajectories[i] for i in trajectory_ids]

    x, (y_true, _) = encode_trajectories(trajectories, max_codes, model.tokenizer)

    y_pred = model(list(x), training=False)

    if return_probs:
        y_pred = ops.sigmoid(y_pred)

    y_pred = pd.DataFrame(y_pred, columns=[f"y_pred_{i}" for i in range(y_pred.shape[1])])
    y_true = pd.DataFrame(y_true, columns=[f"y_true_{i}" for i in range(y_true.shape[1])])

    y_info = pd.DataFrame(
        {
            "patient_id": patient.id,
            "diagnosis_age": (patient.outcome_date - patient.dob).days / 365.25,
            "trajectory_age": x[1][:, -1] / 365.25,
            "trajectory_len": (x[1][:, -1] - x[1][:, 0]) / 365.25,
            "trajectory_id": trajectory_ids,
        }
    )

    return pd.concat([y_info, y_true, y_pred], axis=1)


def get_predictions_and_labels(
    model: keras.Model, seq: keras.utils.Sequence, verbose: int = 0
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Get model predictions and labels for a list of patients."""

    pbar = keras.utils.Progbar(target=len(seq), verbose=verbose)

    y_true = []
    y_pred = []
    for step in range(len(seq)):
        x, (y_true_batch, _) = seq[step]
        y_pred_batch = model(x, training=False)
        y_true.append(y_true_batch)
        y_pred.append(y_pred_batch)
        pbar.update(step + 1)

    if verbose > 0:
        pbar.update(step + 1, finalize=True)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return y_true, y_pred


def get_best_threshold(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Computes the optimal classification threshold using Youden's J statistic."""
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_pred)
    return thresholds[np.argmax(tpr - fpr)]


def get_pr_auc_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    p, r, _ = skm.precision_recall_curve(y_true, y_pred)
    return skm.auc(r, p)


def compute_endpoint_metrics(y_true: pd.Series, y_pred: pd.Series) -> t.Dict[str, t.Any]:
    """"""
    balance = np.mean(y_true).astype(np.float64)
    roc_auc_score = skm.roc_auc_score(y_true, y_pred)
    pr_auc_score = get_pr_auc_score(y_true, y_pred)
    average_precision = skm.average_precision_score(y_true, y_pred)

    # thresholded metrics
    # FIXME: threshold should be provided as a param so we can derive it from training data
    threshold = get_best_threshold(y_true, y_pred)
    y_pred_class = y_pred > threshold

    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred_class).ravel().astype(np.float64)

    return {
        "balance": balance,
        "auROC": roc_auc_score,
        "auPRC": pr_auc_score,
        "average_precision": average_precision,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tpr": tp / (tp + fn),
        "fpr": fp / (fp + tn),
    }


def get_eval_scores(
    model: RiskNet,
    batch_seq: RiskNetSequence,
    class_labels: t.List[t.Any] | None = None,
    verbose: int = 0,
) -> t.Dict[str, t.Any]:
    """"""
    loss, *_ = model.evaluate(batch_seq, verbose=verbose)
    scores = {"loss": np.float64(loss)}

    y_true, y_pred = get_predictions_and_labels(model, batch_seq, verbose=verbose)

    class_inds = range(y_true.shape[1])
    y_true = pd.DataFrame(y_true, columns=[f"y_true_{i}" for i in class_inds])
    y_pred = pd.DataFrame(y_pred, columns=[f"y_pred_{i}" for i in class_inds])

    endpoint_metrics = {}
    for idx in class_inds:
        y_true_idx = y_true[f"y_true_{idx}"]
        y_pred_idx = y_pred[f"y_pred_{idx}"]
        if class_labels is not None:
            idx = class_labels[idx]
        endpoint_metrics[idx] = compute_endpoint_metrics(y_true_idx, y_pred_idx)

    scores["endpoint_metrics"] = endpoint_metrics

    return scores
