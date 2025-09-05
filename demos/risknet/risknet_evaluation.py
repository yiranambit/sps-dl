# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# ## TODO
#
# - [ ] Double check the data processing and make sure pre-Dx intervals are correct
# - [ ] Pre-Dx intervals (earliest time before Dx at which we can predict positive for each positive patient)
# - [ ] Relative risk plot
#
# ## Notes
#
# - The trajectories here are much longer and span more heterogeneous time periods than CancerRiskNet
# - We need to think about how to adapt this to better suit our rare disease use case

ts = "2025-02-25"

# +
import functools
import json
import keras
import random

import altair as alt
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import tensorflow as tf
import typing as t

from IPython.display import display, HTML
from joblib import delayed
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy import stats

from keras import layers, ops

from lpm.data.datasets.risknet import Patient
from lpm.data.datasets.risknet import (
    BalancedRiskNetBatchGenerator,
    PatientCollection,
    RiskNetBatchGenerator,
    RiskNetConfig,
    TrajectoryValidator,
)
from lpm.data.datasets.risknet.dataset import deserialize_patient
from lpm.data.datasets.risknet.sequence import encode_trajectories
from lpm.model import RiskNet
from lpm.model.risknet import (
    get_eval_scores,
    get_predictions_and_labels,
    predict_patient,
)
from lpm.model.risknet.utils import compute_endpoint_metrics
from lpm.utils.progress import ParallelTqdm

# -

alt.data_transformers.enable("vegafusion")


def load_diagnosis_code_info(file_path: Path) -> pd.DataFrame:
    """"""
    return pd.read_csv(file_path, usecols=["CODE", "DESCRIPTION"])


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
) -> t.Tuple[
    PatientCollection, PatientCollection, PatientCollection, PatientCollection
]:
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


# +
run_dir = Path("../../data/outputs/run/Broad-2025-02-25_20-23-54")

cfg = OmegaConf.load(run_dir / ".hydra/config.yaml")
model: RiskNet = keras.models.load_model(run_dir / "risknet.keras")

# +
trajectory_validator = TrajectoryValidator(cfg.data)

train_pos, train_neg, test_pos, test_neg = load_dataset(
    Path(cfg.paths.raw), trajectory_validator, cfg.preprocess.num_processes
)
# -

batch_gen = RiskNetBatchGenerator(
    batch_size=25,
    tokenizer=model.tokenizer,
    max_codes=cfg.model.max_sequence_length,
    n_trajectories=20,
)

eval_scores = get_eval_scores(
    model,
    batch_gen.flow(test_pos, test_neg, shuffle=False),
    class_labels=cfg.data.month_endpoints,
)

pd.DataFrame.from_dict(eval_scores["endpoint_metrics"], orient="index")

y_true, y_pred = get_predictions_and_labels(
    model, batch_gen.flow(test_pos, test_neg, shuffle=False)
)
y_true = pd.DataFrame(y_true, columns=[f"y_true_{i}" for i in range(y_true.shape[1])])
y_pred = pd.DataFrame(y_pred, columns=[f"y_pred_{i}" for i in range(y_pred.shape[1])])

# ## Basic Performance Metrics

endpoint_to_idx = {ep: i for i, ep in enumerate(cfg.data.month_endpoints)}
idx_to_endpoint = {i: ep for i, ep in enumerate(cfg.data.month_endpoints)}

# +
# plot roc curves

roc_data = []
for ep_idx in range(y_true.shape[1]):
    ep_name = idx_to_endpoint[ep_idx]

    y_true_var = f"y_true_{ep_idx}"
    y_pred_var = f"y_pred_{ep_idx}"
    fpr_ep, tpr_ep, _ = skm.roc_curve(y_true[y_true_var], y_pred[y_pred_var])

    roc_data_ep = pd.DataFrame({"fpr": fpr_ep, "tpr": tpr_ep, "endpoint": ep_name})
    roc_data.append(roc_data_ep)

roc_data = pd.concat(roc_data)

# url = "../../data/temp/roc_data.json"
# roc_data.to_json(url, orient="records")

roc_chart = (
    alt.Chart(roc_data, width=300, height=300)
    .mark_line(interpolate="step-after")
    .encode(
        alt.X("fpr:Q").axis(grid=False, tickCount=5).title("False Positive Rate (FPR)"),
        alt.Y("tpr:Q").axis(grid=False, tickCount=5).title("True Positive Rate (TPR)"),
        alt.Color("endpoint:O").legend().title(None),
    )
)

# +
pr_data = []
for ep_idx in range(y_true.shape[1]):
    ep_name = idx_to_endpoint[ep_idx]

    y_true_var = f"y_true_{ep_idx}"
    y_pred_var = f"y_pred_{ep_idx}"
    p_ep, r_ep, _ = skm.precision_recall_curve(y_true[y_true_var], y_pred[y_pred_var])

    pr_data_ep = pd.DataFrame({"precision": p_ep, "recall": r_ep, "endpoint": ep_name})
    pr_data.append(pr_data_ep)

pr_data = pd.concat(pr_data)

#  = "../../data/temp/pr_data.json"
# pr_data.to_json(url, orient="records")

pr_chart = (
    alt.Chart(pr_data, width=300, height=300)
    .mark_line(interpolate="step-after")
    .encode(
        alt.X("recall:Q").axis(grid=False, tickCount=5).title("Recall"),
        alt.Y("precision:Q").axis(grid=False, tickCount=5).title("Precision"),
        alt.Color("endpoint:O").legend().title(None),
    )
)

# +
chart = (
    alt.hconcat(roc_chart, pr_chart)
    .resolve_scale(color="independent")
    .configure_view(strokeOpacity=0)
    .configure_axis(titlePadding=10)
)

chart.display()
# -

# ## Visualize ICD10 Code Embeddings

code_embeddings = model.embedding.token_emb.get_weights()[0]
code_embeddings = pd.DataFrame(code_embeddings, index=model.tokenizer.get_vocabulary())
code_embeddings = code_embeddings.iloc[2:]  # remove the padding and UNK tokens
code_embeddings.head()

# +
reducer = TSNE(n_components=2, random_state=42)
code_embeddings_2d = reducer.fit_transform(code_embeddings)

code_embeddings_2d = pd.DataFrame(code_embeddings_2d, columns=["tsne_1", "tsne_2"])
code_embeddings_2d["code"] = code_embeddings.index

# +
pos_events = [(i, e[1]) for i, pt in enumerate(train_pos) for e in pt.events]
neg_events = [(i, e[1]) for i, pt in enumerate(train_neg) for e in pt.events]

pos_events = pd.DataFrame(pos_events, columns=["patient", "code"])
neg_events = pd.DataFrame(neg_events, columns=["patient", "code"])

pos_freqs = pos_events.groupby("code")["patient"].nunique().to_frame(name="pos_count")
pos_freqs["pos_freq"] = pos_freqs["pos_count"] / pos_events["patient"].nunique()

neg_freqs = neg_events.groupby("code")["patient"].nunique().to_frame(name="neg_count")
neg_freqs["neg_freq"] = neg_freqs["neg_count"] / neg_events["patient"].nunique()

code_embeddings_2d = code_embeddings_2d.merge(
    pos_freqs, left_on="code", right_index=True, how="left"
)
code_embeddings_2d = code_embeddings_2d.merge(
    neg_freqs, left_on="code", right_index=True, how="left"
)

code_embeddings_2d["rel_freq"] = (
    code_embeddings_2d["pos_freq"] / code_embeddings_2d["neg_freq"]
)
code_embeddings_2d["log_rel_freq"] = np.log2(code_embeddings_2d["rel_freq"])

code_embeddings_2d["pos_count"] = code_embeddings_2d["pos_count"].fillna(0)
code_embeddings_2d["neg_count"] = code_embeddings_2d["neg_count"].fillna(0)

code_embeddings_2d["total_count"] = (
    code_embeddings_2d["pos_count"] + code_embeddings_2d["neg_count"]
)
# -

kmeans = KMeans(n_clusters=6, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(code_embeddings)
code_embeddings_2d["cluster"] = clusters

# +
code_info_path = "../../data/diagnosis_codes.csv"
code_info = load_diagnosis_code_info(code_info_path)

code_to_description = dict(zip(code_info["CODE"], code_info["DESCRIPTION"]))
code_embeddings_2d["pheno"] = code_embeddings_2d["code"].map(code_to_description)

# +
# url = "../../data/temp/code_embeddings.json"

sorted_idx = (
    code_embeddings_2d["log_rel_freq"].abs().sort_values(na_position="first").index
)

# (
#     code_embeddings_2d.loc[sorted_idx]
#     .reset_index(drop=True)
#     .to_json(url, orient="records")
# )

icd_alt_data = code_embeddings_2d.loc[sorted_idx].reset_index(drop=True)


# +
z_max = np.ceil(code_embeddings_2d["log_rel_freq"].abs().max())
z_min = -z_max

icd_cluster = (
    alt.Chart(icd_alt_data)
    .transform_filter(alt.datum.total_count >= 30)
    .mark_circle()
    .encode(
        alt.X("tsne_1:Q").axis(None).title(None),
        alt.Y("tsne_2:Q").axis(None).title(None),
        alt.Color("log_rel_freq:Q").scale(scheme="redblue", domain=(z_min, 0, z_max)),
        alt.Size("log_rel_freq:Q")
        .scale(range=(200, 50, 200), domain=(z_min, 0, z_max))
        .legend(None),
        tooltip=[
            alt.Tooltip("code:N", title="ICD10 Code"),
            alt.Tooltip("pheno:N", title="ICD10 Description"),
            alt.Tooltip(
                "log_rel_freq:Q", format=".2f", title="Log2 Relative Frequency"
            ),
        ],
    )
    .properties(width=800, height=500)
)

icd_cluster.display()
# -


# ## Trajectory Embeddings
#
# Trajectory embeddings correspond to patient-level embeddings

# +
# hack to get inputs in the model
# seq = batch_gen.flow(test_pos[:2], test_neg[:2], shuffle=False)
# model(seq[0][0])

# +
emb_input = model.encoder.input
emb_output = [model.layers[2].layers[-2].output, model.encoder.output]

emb_model = keras.Model(inputs=emb_input, outputs=emb_output)


# -


def predict_patient(
    model: keras.Model,
    patient: Patient,
    max_codes: int,
    tokenizer: layers.StringLookup,
    max_trajectories: int | None = None,
) -> pd.DataFrame:
    """Generate predictions for all possible trajectories of a patient."""
    trajectories = patient.get_trajectories()

    if max_trajectories is not None and len(trajectories) > max_trajectories:
        trajectories = random.sample(trajectories, max_trajectories)

    x, (y_true, _) = encode_trajectories(trajectories, max_codes, tokenizer)

    y_pred_emb, y_pred = model(x, training=False)

    y_pred_emb = pd.DataFrame(y_pred_emb)
    y_pred = pd.DataFrame(y_pred)

    y_pred_emb.columns = [f"y_emb_{i}" for i in range(y_pred_emb.shape[1])]
    y_pred.columns = [f"y_pred_{i}" for i in range(y_pred.shape[1])]

    y_true = pd.DataFrame(y_true)
    y_true.columns = [f"y_true_{i}" for i in range(y_true.shape[1])]

    y_info = pd.DataFrame(
        {
            "patient_id": patient.id,
            "diagnosis_age": (patient.outcome_date - patient.dob).days / 365,
            "trajectory_age": x[1][:, -1] / 365,
            "trajectory_duration": (x[1][:, -1] - x[1][:, 0]) / 365,
            # -1 is used to pad the sequence, so we want to get the duration length by removing the padding placeholder
            "trajectory_len": (x[1] > -1).sum(axis=1),
        }
    )

    return pd.concat([y_info, y_true, y_pred, y_pred_emb], axis=1)


# +
pos_results = []
for pos_pt in tqdm(test_pos):
    pos_results.append(
        predict_patient(
            emb_model,
            pos_pt,
            cfg.model.max_sequence_length,
            model.tokenizer,
            max_trajectories=5,
        )
    )
pos_results = pd.concat(pos_results)

pos_results.head()

# +
neg_results = []
for neg_pt in tqdm(test_neg):
    neg_results.append(
        predict_patient(
            emb_model,
            neg_pt,
            cfg.model.max_sequence_length,
            model.tokenizer,
            max_trajectories=5,
        )
    )
neg_results = pd.concat(neg_results)

neg_results.head()

# +
results = pd.concat([pos_results, neg_results])

embeddings = results.filter(like="y_emb")
embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
embeddings_2d = pd.DataFrame(embeddings_2d, columns=["tsne_1", "tsne_2"])

# +
info_cols = ["patient_id", "diagnosis_age", "trajectory_age", "trajectory_len"]
results_info = results[info_cols].reset_index(drop=True)
results_y_true = results.filter(like="y_true").reset_index(drop=True)
results_y_pred = results.filter(like="y_pred").reset_index(drop=True)

embeddings_2d = pd.concat(
    [results_info, results_y_true, results_y_pred, embeddings_2d], axis=1
)

embeddings_2d = embeddings_2d.sort_values(["y_true_4", "patient_id"])
embeddings_2d["age_bin"] = np.floor(embeddings_2d["trajectory_age"])

embeddings_2d.head()

# +
# url = "../../data/temp/patient_embeddings.json"
# embeddings_2d.to_json(url, orient="records")

# +
# chart colored by observed label at 60 months
TOOLTIP = [
    alt.Tooltip("patient_id:N").title("Patient ID"),
    alt.Tooltip("diagnosis_age:Q", format=".2f").title("Diagnosis Age (yrs)"),
    alt.Tooltip("trajectory_age:Q", format=".2f").title("Trajectory Age (yrs)"),
    alt.Tooltip("trajectory_len:Q", format=".2f").title("Trajectory Length (yrs)"),
    alt.Tooltip("y_pred_4:Q", format=".2f").title("Predicted Risk (60 mo.)"),
]

chart = (
    alt.Chart(embeddings_2d)
    .mark_circle()
    .encode(
        alt.X("tsne_1:Q").axis(None).title(None),
        alt.Y("tsne_2:Q").axis(None).title(None),
        alt.Color("y_true_4:N")
        .scale(domain=(1, 0), range=("#4C78A8", "lightgray"))
        .title("Observed Diagnosis (60 mo.)"),
        tooltip=TOOLTIP,
    )
    .properties(width=800, height=500)
)

chart.configure_view(strokeOpacity=0)

# +
# predicted risk at 60 months
TOOLTIP = [
    alt.Tooltip("patient_id:N").title("Patient ID"),
    alt.Tooltip("diagnosis_age:Q", format=".2f").title("Diagnosis Age (yrs)"),
    alt.Tooltip("trajectory_age:Q", format=".2f").title("Trajectory Age (yrs)"),
    alt.Tooltip("trajectory_len:Q", format=".2f").title("Trajectory Length (yrs)"),
    alt.Tooltip("y_pred_4:Q", format=".2f").title("Predicted Risk (60 mo.)"),
]

chart = (
    alt.Chart(embeddings_2d)
    .mark_circle()
    .encode(
        alt.X("tsne_1:Q").axis(None).title(None),
        alt.Y("tsne_2:Q").axis(None).title(None),
        alt.Color("y_pred_4:Q")
        .scale(scheme="redblue", reverse=True, domainMid=0)
        .title("Predicted Risk (60 mo.)"),
        tooltip=TOOLTIP,
    )
    .properties(width=800, height=500)
)

chart.configure_view(strokeOpacity=0)

# +
# chart colored by the patient's age at the last claim in the trajectory
TOOLTIP = [
    alt.Tooltip("patient_id:N").title("Patient ID"),
    alt.Tooltip("diagnosis_age:Q", format=".2f").title("Diagnosis Age (yrs)"),
    alt.Tooltip("trajectory_age:Q", format=".2f").title("Trajectory Age (yrs)"),
    alt.Tooltip("trajectory_len:Q", format=".2f").title("Trajectory Length (yrs)"),
    alt.Tooltip("y_pred_4:Q", format=".2f").title("Predicted Risk (60 mo.)"),
]

chart = (
    alt.Chart(embeddings_2d)
    .mark_circle(opacity=0.5)
    .encode(
        alt.X("tsne_1:Q").axis(None).title(None),
        alt.Y("tsne_2:Q").axis(None).title(None),
        alt.Color("trajectory_len:Q").title("Trajectory Length (yrs)"),
        tooltip=TOOLTIP,
    )
    .properties(width=800, height=500)
)

chart.configure_view(strokeOpacity=0)

# +
# chart colored by the patient's age at the last claim in the trajectory
TOOLTIP = [
    alt.Tooltip("patient_id:N").title("Patient ID"),
    alt.Tooltip("diagnosis_age:Q", format=".2f").title("Diagnosis Age (yrs)"),
    alt.Tooltip("trajectory_age:Q", format=".2f").title("Trajectory Age (yrs)"),
    alt.Tooltip("trajectory_len:Q", format=".2f").title("Trajectory Length (yrs)"),
    alt.Tooltip("y_pred_4:Q", format=".2f").title("Predicted Risk (60 mo.)"),
]

chart = (
    alt.Chart(embeddings_2d)
    .mark_circle(opacity=0.5)
    .encode(
        alt.X("tsne_1:Q").axis(None).title(None),
        alt.Y("tsne_2:Q").axis(None).title(None),
        alt.Color("trajectory_age:Q").title("Trajectory Age (yrs)"),
        tooltip=TOOLTIP,
    )
    .properties(width=800, height=500)
)

chart.configure_view(strokeOpacity=0)
# -

# ## Performance By Patient Age

results["trajectory_age_bin"] = np.floor(results["trajectory_age"]).astype(int)
results.head()

# +
age_binned_endpoint_metrics = []
for age_bin, group in results.groupby("trajectory_age_bin"):
    endpoint_metrics = {}
    for ep_idx in range(5):
        ep_name = cfg.data.month_endpoints[ep_idx]
        y_true_ep = group[f"y_true_{ep_idx}"]
        y_pred_ep = group[f"y_pred_{ep_idx}"]
        if y_true_ep.nunique() < 2:
            continue
        endpoint_metrics[ep_name] = compute_endpoint_metrics(y_true_ep, y_pred_ep)
    endpoint_metrics = pd.DataFrame.from_dict(endpoint_metrics, orient="index")
    endpoint_metrics["age_bin"] = age_bin
    age_binned_endpoint_metrics.append(endpoint_metrics)

age_binned_endpoint_metrics = (
    pd.concat(age_binned_endpoint_metrics).rename_axis(index="endpoint").reset_index()
)
age_binned_endpoint_metrics.head()

# +
age_bin_counts = (
    results["trajectory_age_bin"]
    .value_counts()
    .to_frame("count")
    .rename_axis(index="age_bin")
    .sort_index()
    .reset_index()
)

age_binned_counts_chart = (
    alt.Chart(age_bin_counts)
    .transform_filter(alt.datum.age_bin <= 15)
    .mark_bar(color="gray")
    .encode(
        alt.X("age_bin:O", title="Age Bin").axis(grid=False, labelAngle=0),
        alt.Y("count:Q", title="Patient Count").axis(grid=False),
    )
    .properties(width=800, height=200)
)

age_binned_auprc_chart = (
    alt.Chart(age_binned_endpoint_metrics)
    .transform_filter(alt.datum.age_bin <= 15)
    .mark_line()
    .encode(
        alt.X("age_bin:O", title="Age Bin").axis(grid=False, labelAngle=0),
        alt.Y("auPRC:Q", title="auPRC").axis(grid=False),
        alt.Color("endpoint:O"),
    )
    .properties(width=800, height=200)
)

age_binned_auroc_chart = (
    alt.Chart(age_binned_endpoint_metrics)
    .transform_filter(alt.datum.age_bin <= 15)
    .mark_line()
    .encode(
        alt.X("age_bin:O", title="Age Bin").axis(grid=False, labelAngle=0),
        alt.Y("auROC:Q", title="auROC").axis(grid=False),
        alt.Color("endpoint:O"),
    )
    .properties(width=800, height=200)
)

chart = alt.vconcat(
    age_binned_counts_chart, age_binned_auprc_chart, age_binned_auroc_chart
)
chart.configure_view(strokeOpacity=0).configure_axis(titlePadding=10)

# +
# NOTE: this is super interesting -> we can clearly see these two modes where we do well
#  occuring for very young patients (<2 yo) and for patients age 8-12
# NOTE: we also have really good performance for 0-2 yrs old
# NOTE: what if patients get put on an ASM by 2 years old? -> maybe that would explain the dropoff as seizures
#   are managed better and we see less claims -> if we add in ASM med claims, maybe we will improve perfomrance in this region
# -

# ## Performance By Trajectory Length

results["trajectory_len_bin"] = np.floor(results["trajectory_len"] / 10).astype(int)
results.head()

# +
tlen_binned_endpoint_metrics = []
for tlen_bin, group in results.groupby("trajectory_len_bin"):
    endpoint_metrics = {}
    for ep_idx in range(5):
        ep_name = cfg.data.month_endpoints[ep_idx]
        y_true_ep = group[f"y_true_{ep_idx}"]
        y_pred_ep = group[f"y_pred_{ep_idx}"]
        if y_true_ep.nunique() < 2:
            continue
        endpoint_metrics[ep_name] = compute_endpoint_metrics(y_true_ep, y_pred_ep)
    endpoint_metrics = pd.DataFrame.from_dict(endpoint_metrics, orient="index")
    endpoint_metrics["tlen_bin"] = tlen_bin
    tlen_binned_endpoint_metrics.append(endpoint_metrics)

tlen_binned_endpoint_metrics = (
    pd.concat(tlen_binned_endpoint_metrics).rename_axis(index="endpoint").reset_index()
)
tlen_binned_endpoint_metrics.head()

# +
tlen_bin_counts = (
    results["trajectory_len_bin"]
    .value_counts()
    .to_frame("count")
    .rename_axis(index="tlen_bin")
    .sort_index()
    .reset_index()
)

tlen_binned_counts_chart = (
    alt.Chart(tlen_bin_counts)
    .transform_filter(alt.datum.tlen_bin <= 15)
    .mark_bar(color="gray")
    .encode(
        alt.X("tlen_bin:O", title="Trjectory Length (units: 10)").axis(
            grid=False, labelAngle=0
        ),
        alt.Y("count:Q", title="Patient Count").axis(grid=False),
    )
    .properties(width=800, height=200)
)

tlen_binned_auprc_chart = (
    alt.Chart(tlen_binned_endpoint_metrics)
    .transform_filter(alt.datum.tlen_bin <= 15)
    .mark_line()
    .encode(
        alt.X("tlen_bin:O", title="Trjectory Length (units: 10)").axis(
            grid=False, labelAngle=0
        ),
        alt.Y("auPRC:Q", title="auPRC").axis(grid=False),
        alt.Color("endpoint:O"),
        tooltip=[
            alt.Tooltip("tlen_bin:O", title="Trjectory Length (units: 10)"),
            alt.Tooltip("auPRC:Q", title="auROC"),
        ],
    )
    .properties(width=800, height=200)
)

tlen_binned_auroc_chart = (
    alt.Chart(tlen_binned_endpoint_metrics)
    .transform_filter(alt.datum.tlen_bin <= 15)
    .mark_line()
    .encode(
        alt.X("tlen_bin:O", title="Trjectory Length (units: 10)").axis(
            grid=False, labelAngle=0
        ),
        alt.Y("auROC:Q", title="auROC").axis(grid=False),
        alt.Color("endpoint:O"),
        tooltip=[
            alt.Tooltip("tlen_bin:O", title="Trjectory Length (units: 10)"),
            alt.Tooltip("auROC:Q", title="auROC"),
        ],
    )
    .properties(width=800, height=200)
)

chart = alt.vconcat(
    tlen_binned_counts_chart, tlen_binned_auprc_chart, tlen_binned_auroc_chart
)
chart.configure_view(strokeOpacity=0).configure_axis(titlePadding=10)

# +
# plot length distribution of the embeddings

# alt.Chart(url).mark_boxplot().encode(
#     alt.X("y_true_4:O").axis(labelAngle=0).title(None),
#     alt.Y("trajectory_len:Q")
#     .axis(offset=5, grid=False, titlePadding=10)
#     .title("Trajectory Length (yrs)"),
#     alt.Color("y_true_4:O"),
#     alt.Column("age_bin:O").spacing(5).title("Age Bin (yrs)"),
# ).configure_view(strokeOpacity=0)

# +
trajectory = test_pos[0].sample_trajectories(1)[0]

code_seq = trajectory.code_seq
value_seq = np.random.normal(0, 1, len(code_seq))


# +
def interpolate_color(
    color1: t.Tuple[int], color2: t.Tuple[int], n: float
) -> t.Tuple[int]:
    """Interpolate between two colors."""
    return tuple(round(color1[i] + (color2[i] - color1[i]) * n) for i in range(3))


def create_diverging_color_scheme(
    color1: t.Tuple[int], color2: t.Tuple[int], color3: t.Tuple[int]
) -> t.List[t.Tuple[int]]:
    """Create a diverging color scheme."""
    colors_1 = [interpolate_color(color1, color2, n) for n in np.arange(0, 1, 0.1)]
    colors_2 = [interpolate_color(color3, color2, n) for n in np.arange(0, 1, 0.1)]

    return colors_1 + [color2] + list(reversed(colors_2))


def create_color_scheme(
    color1: t.Tuple[int], color2: t.Tuple[int]
) -> t.List[t.Tuple[int]]:
    """Create a color scheme."""
    return [interpolate_color(color1, color2, n) for n in np.arange(0, 1.1, 0.1)]


red_rgb = (229, 87, 86)  # red
gray_rgb = (241, 239, 238)  # gray
blue_rgb = (76, 120, 168)  # blue
purple_rgb = (178, 120, 162)  # purple

colors = create_diverging_color_scheme(blue_rgb, gray_rgb, red_rgb)
colors = ["#%02X%02X%02X" % c for c in colors]

print(colors)
# -

labels = [f"{i}" for i in range(len(colors))]
bin_seq = pd.cut(value_seq, bins=len(colors), labels=labels)
bin_to_color = dict(zip(labels, colors))

# +
SALIENCY_MAP_HTML_TEMPLATE = """
    <style>
        .container {{
            display: flex;
            flex-wrap: wrap;
            font-size: 1em;
            margin: 100px;
            max-width: 75%;
        }}
        .code {{
            color: black;
            border-radius: 2px;
            margin: 2px;
            padding: 5px;
        }}
        .tooltip-text {{
            background-color: white;
            border-radius: 5px;
            color: black;
            position: absolute;
            visibility: hidden;
            z-index: 1;
            padding-left: 10px;
            padding-right: 10px;
        }}
        .tooltip:hover .tooltip-text {{
            visibility: visible;
        }}
    </style>
    <div class='container'>
        {0}
    </div>
"""


SALIENCY_MAP_DIV_HTML_TEMPLATE = """
    <div class='code tooltip' style='background-color:{0}'>
        {1}
        <div class='tooltip-text'>
            <p style='font-size:0.9em'>Code: {2}</p>
            <p style='font-size:0.9em'>Age: {3}</p>
            <p style='font-size:0.9em'>Decription: {4}</p>
            <p style='font-size:0.9em'>Integrated Gradients: {5:.2f}</p>
        </div>
    </div>
"""


class SaliencyMapPlotter:

    def __init__(
        self,
        codes: t.Iterable[str],
        ages: t.Iterable[float],
        values: t.List[float],
        colors: t.List[t.Tuple[int]],
    ) -> None:
        self.codes = codes
        self.ages = ages
        self.values = values
        self._colors = colors
        self._init_color_scheme(colors)
        self._init_bins()

    def _init_bins(self) -> None:
        """"""
        labels = [f"{i}" for i in range(len(self._scheme))]
        self.bins = pd.cut(self.values, bins=len(self._scheme), labels=labels)
        self.bin_to_color = dict(zip(labels, self._scheme))

    def _init_color_scheme(self, colors: t.List[t.Tuple[int]]) -> t.List[str]:
        """"""
        if len(colors) == 3:
            scheme = create_diverging_color_scheme(*colors)
        elif len(colors) == 2:
            scheme = create_color_scheme(*colors)
        else:
            raise ValueError("Invalid no of colors.")

        self._scheme = ["#%02X%02X%02X" % c for c in scheme]

    def plot(self) -> HTML:
        """"""
        inner = ""
        for c, a, v, b in zip(self.codes, self.ages, self.values, self.bins):
            inner += SALIENCY_MAP_DIV_HTML_TEMPLATE.format(
                self.bin_to_color[b], c, c, a, code_to_description.get(c, "N/A"), v
            )
        return HTML(SALIENCY_MAP_HTML_TEMPLATE.format(inner))


# -

# ## Integrated Gradients

# +


def get_predicted_positive_trajectories(model, batch_gen, patient):
    _, y_pred = get_predictions_and_labels(
        model, batch_gen.flow([], [patient], shuffle=False)
    )

    is_predicted = np.max(y_pred, axis=1) > 0

    return np.any(is_predicted), y_pred[is_predicted], np.where(is_predicted)[0]


# +

predicted_pos = []

for idx, pt in enumerate(train_neg):
    is_predicted, is_predicted_logits, pos_traj_idx = (
        get_predicted_positive_trajectories(model, batch_gen, pt)
    )

    if is_predicted:
        predicted_pos.append((idx, is_predicted, is_predicted_logits, pos_traj_idx))

    if len(predicted_pos) >= 3:
        break

predicted_pos


# +
def interpolate_sequence(baseline, sequence, alphas):
    """Generate interpolated sequence embeddings."""
    input_x = sequence
    baseline_x = baseline
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]

    delta = input_x - baseline_x
    return baseline_x + alphas_x * delta


def compute_gradients(model: keras.Model, sequences, target_class_idx: int = -1):
    """"""
    with tf.GradientTape() as tape:
        tape.watch(sequences)
        logits = model(sequences)
        # probs = ops.sigmoid(logits)[:, target_class_idx]
        probs = logits[:, target_class_idx]

    return tape.gradient(probs, sequences)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def integrated_gradients(
    model: keras.Model,
    sequence,
    mask,
    target_class_idx: int = -1,
    baseline: np.ndarray | None = None,
    n_steps: int = 50,
):
    """"""
    if baseline is None:
        baseline = tf.zeros_like(sequence)

    alphas = tf.linspace(0.0, 1.0, n_steps + 1)

    interpolated_sequences = interpolate_sequence(baseline, sequence, alphas)

    path_gradients = compute_gradients(
        model, sequences=interpolated_sequences, target_class_idx=target_class_idx
    )

    avg_gradients = integral_approximation(path_gradients)
    avg_gradients *= ops.expand_dims(mask, axis=1)

    integrated_gradients = (sequence - baseline) * avg_gradients

    return integrated_gradients


# +
# FIXME: account for padding in gradient computation
# -

reverse_lookup = layers.StringLookup(
    mask_token="", invert=True, vocabulary=model.tokenizer.get_vocabulary()
)

colors = ((255, 255, 255), purple_rgb)

# +
n_steps = 50
alphas = tf.linspace(start=0.0, stop=1.0, num=n_steps + 1)

patient: Patient = train_neg[9]

valid_trajectories = patient.get_trajectories()

x, y = encode_trajectories(
    [valid_trajectories[2]], cfg.model.max_sequence_length, model.tokenizer
)

embedding_layer = model.encoder.layers[2]
ig_model = keras.models.Sequential(model.encoder.layers[3:])

x_embedding = embedding_layer(x)[0].numpy()
x_mask = tf.cast(x[0][0] != 0, dtype=tf.float32)

attributions = integrated_gradients(ig_model, x_embedding, x_mask)

# sum attributions over the embedding dimension
attributions = tf.reduce_sum(attributions, axis=-1).numpy()

# +
# TODO: add saving of StringLookup and vocabulary

# +
# process attributions for visualization

# positive polarity
attributions = np.clip(attributions, 0, 1)

# get thresholded attributions
clip_above_percentile = 99.9
clip_below_percentile = 10.0
lower_end = 0.2

flatten_attr = attributions.flatten()

total = np.sum(flatten_attr)

sorted_attributions = np.sort(np.abs(flatten_attr))[::-1]
cum_sum = 100.0 * np.cumsum(sorted_attributions) / total

indices_to_consider = np.where(cum_sum >= 100 - clip_above_percentile)[0][0]
m = sorted_attributions[indices_to_consider]

indices_to_consider = np.where(cum_sum >= 100 - clip_below_percentile)[0][0]
e = sorted_attributions[indices_to_consider]

transformed_attributions = (1 - lower_end) * (np.abs(attributions) - e) / (
    m - e
) + lower_end

transformed_attributions *= np.sign(attributions)
transformed_attributions *= transformed_attributions >= lower_end
transformed_attributions = np.clip(transformed_attributions, 0.0, 1.0)

# +
result = pd.DataFrame(
    {"code": reverse_lookup(x[0][0]), "age": x[1][0], "ig": transformed_attributions}
)
result["code"] = result["code"].str.decode("utf-8")
result["code"] = result["code"].map(lambda x: "[PAD]" if x == "" else x)

# filter padding tokens
result = result[result["code"] != "[PAD]"]
result["age"] = (result["age"] / 365).round(2)
result["idx"] = range(len(result))
# -

base = alt.Chart(result).encode(alt.X("idx:N").axis().sort(list(result["idx"])))
base.mark_circle().encode(alt.Y("ig:Q").axis(grid=False))

plotter = SaliencyMapPlotter(
    list(result["code"]),
    list((result["age"] / 365).round(2)),
    list(result["ig"]),
    colors=colors,
)
plotter.plot()

# +
# exported = plotter.plot()
# with open('Saliency_Map_Pred_Neg_3.html', 'w') as f:
#     f.write(exported.data)

# +
# NOTE: what if I display the trajectory along a vertical axis - each row is a code, age, desc, and IG value
# NOTE: alternatively, I can display it as a lolipop plot showing the IG values for each code
# -

# ## Bootstrap Validation

# +
# TODO: generate a "most recent" generatator that only gabs the most recent trajectory from the valid trajectories

batch_gen = RiskNetBatchGenerator(
    batch_size=256,
    tokenizer=model.tokenizer,
    max_codes=cfg.model.max_sequence_length,
    n_trajectories=1,
)

# +
# NOTE: what if I boostrap it with half of each number of patients?

frac = 1.0
n_pos = int(121 * frac)
n_neg = int(9597 * frac)
n_bootstrap_iters = 100

bootstrap_scores = []
for i in trange(n_bootstrap_iters):
    batch_pos = PatientCollection(*np.random.choice(test_pos, n_pos, replace=True))
    batch_neg = PatientCollection(*np.random.choice(test_neg, n_neg, replace=True))

    eval_scores = get_eval_scores(
        model,
        batch_gen.flow(batch_pos, batch_neg, shuffle=False),
        class_labels=cfg.data.month_endpoints,
    )

    endpoint_scores = (
        pd.DataFrame.from_dict(eval_scores["endpoint_metrics"], orient="index")
        .assign(iter=i)
        .rename_axis("endpoint")
        .reset_index()
    )

    bootstrap_scores.append(endpoint_scores)

bootstrap_scores = pd.concat(bootstrap_scores)
bootstrap_scores.groupby("endpoint").agg("mean")
