#!/usr/bin/env python
"""
Runs integrated gradients on a trained RiskNet model.

Usage
-----
>>> python scripts/risknet/utils/run_integrated_gradients.py \
    --run-dir /path/to/run \
    --code-info /path/to/code_info.csv \
    --max-trajectories 5 \
    --max-patients 2 \
    --class-idx -1 \
    --n-steps 50
"""

import argparse
import keras

import numpy as np
import pandas as pd
import tensorflow as tf
import typing as t

from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

from lpm.data.datasets.risknet import Patient
from lpm.data.datasets.risknet import TrajectoryValidator
from lpm.data.datasets.risknet.dataset import load_test_dataset
from lpm.data.datasets.risknet.sequence import encode_trajectories
from lpm.interp import IntegratedGradients
from lpm.model import RiskNet
from lpm.model.risknet import get_patient_predictions


def parse_args() -> argparse.Namespace:
    """"""
    parser = argparse.ArgumentParser(description="Run integrated gradients for RiskNet.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument(
        "--code-info", type=str, required=True, help="Code info file path"
    )
    parser.add_argument(
        "--max-trajectories", type=int, default=10, help="Max trajectories per patient"
    )
    parser.add_argument(
        "--max-patients", type=int, default=50, help="Max patients to process"
    )
    parser.add_argument("--class-idx", type=int, default=-1, help="Predicted class index")
    parser.add_argument("--n-steps", type=int, default=50, help="Number of steps")
    args = parser.parse_args()
    return args


def load_diagnosis_code_info(file_path: str | Path) -> pd.DataFrame:
    """"""
    return pd.read_csv(file_path, usecols=["CODE", "DESCRIPTION"])


def get_code_to_desc(code_info: pd.DataFrame) -> t.Dict[str, str]:
    """"""
    return (
        code_info.assign(code=lambda df: df["CODE"].str[:4])
        .sort_values("CODE")
        .drop_duplicates(subset="code", keep="first")
        .set_index("code")["DESCRIPTION"]
        .to_dict()
    )


def load_run(run_dir: str | Path) -> t.Tuple[RiskNet, OmegaConf]:
    """"""
    if not isinstance(run_dir, Path):
        run_dir = Path(run_dir)

    cfg = OmegaConf.load(run_dir / ".hydra/config.yaml")
    model: RiskNet = keras.models.load_model(run_dir / "risknet.keras")

    return model, cfg


def downsample_trajectories(pred_df: pd.DataFrame, n_trajectories: int) -> pd.DataFrame:
    """"""
    bins = pd.qcut(pred_df["y_pred_4"], n_trajectories)
    return (
        pred_df.groupby(bins, observed=True, as_index=False)
        .apply(lambda g: g.loc[g["y_pred_4"].idxmax()])
        .reset_index(drop=True)
    )


def run_patient_integrated_gradients(
    pt: Patient,
    cfg: OmegaConf,
    model: RiskNet,
    igrad: IntegratedGradients,
    trajectory_ids: t.List[int] | None = None,
) -> pd.DataFrame:
    """"""
    if trajectory_ids is None:
        trajectory_ids = list(range(len(pt.valid_trajectory_indices)))

    trajectories = pt.get_trajectories()
    emb_layer = model.layers[-1].layers[2]

    trajectory_attrs = []
    for trajectory_id in trajectory_ids:
        trajectory = trajectories[trajectory_id]
        x, _ = encode_trajectories(
            [trajectory], cfg.model.max_sequence_length, model.tokenizer
        )
        x_emb = tf.squeeze(emb_layer(x), 0)
        x_mask = tf.squeeze(tf.cast(x[0] != 0, dtype=tf.float32), 0)

        attrs = igrad.run(x_emb, mask=x_mask)
        attrs = np.take(attrs, np.where(x_mask.numpy() != 0)).squeeze()

        codes = trajectory.code_seq[-cfg.model.max_sequence_length :]
        ages = [x / 365.25 for x in trajectory.age_seq[-cfg.model.max_sequence_length :]]

        trajectory_attrs.append(
            {"trajectory_id": trajectory_id, "code": codes, "age": ages, "attr": attrs}
        )

    return (
        pd.concat(map(pd.DataFrame, trajectory_attrs))
        .rename_axis(index="event_idx")
        .reset_index()
    )


def main(args: argparse.Namespace) -> None:
    """"""
    model, cfg = load_run(args.run_dir)
    trajectory_validator = TrajectoryValidator(cfg.data)

    out_root = Path(args.run_dir) / "integrated_gradients"
    out_root.mkdir(exist_ok=True)

    dataset = load_test_dataset(
        Path(cfg.paths.raw), trajectory_validator, cfg.preprocess.num_processes
    )

    code_info = load_diagnosis_code_info(args.code_info)
    code_to_desc = get_code_to_desc(code_info)

    ig = IntegratedGradients(
        model=keras.models.Sequential(model.encoder.layers[3:]),
        steps=args.n_steps,
        class_idx=args.class_idx,
    )

    for pt in tqdm(dataset.pos[: args.max_patients], desc="Running integrated gradients"):
        pt_out_dir = out_root / pt.id
        pt_out_dir.mkdir(exist_ok=True)

        preds = get_patient_predictions(model, pt, cfg.model.max_sequence_length)
        if preds.shape[0] > args.max_trajectories:
            preds = downsample_trajectories(preds, args.max_trajectories)

        attrs = run_patient_integrated_gradients(
            pt, cfg, model, ig, trajectory_ids=preds["trajectory_id"].to_list()
        )
        attrs["desc"] = attrs["code"].map(code_to_desc)

        preds.to_csv(pt_out_dir / "predictions.csv", index=False)
        attrs.to_csv(pt_out_dir / "attributions.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
