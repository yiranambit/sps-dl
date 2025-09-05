"""Application data loading."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import typing as t

from pathlib import Path
from omegaconf import OmegaConf

from lpm.interp import GradientVisualizer
from lpm.data.datasets.risknet import (
    Patient,
    PatientCollection,
    TrajectoryValidator,
    deserialize_patient,
    deserialize_patients_parallel,
)


def load_run_config(run_dir: Path) -> OmegaConf:
    """Loads the trajectory validator from the run directory."""
    return OmegaConf.load(run_dir / ".hydra/config.yaml")


def load_patient(run_config: OmegaConf, patient_id: str) -> Patient:
    """Loads a patient from the run directory."""
    root = Path(run_config.paths.raw)
    matches = list(root.glob(f"**/{patient_id}.json"))
    if not matches:
        raise ValueError(f"Patient {patient_id} not found.")

    return deserialize_patient(matches[0], TrajectoryValidator(run_config.data))


@st.cache_data
def load_patient_table(_run_config: OmegaConf, patient_ids: t.List[str]) -> pd.DataFrame:
    """Loads patients from the run directory."""
    root = Path(_run_config.paths.raw)
    matches = [list(root.glob(f"**/{pid}.json")) for pid in patient_ids]

    patients = deserialize_patients_parallel(
        [m[0] for m in matches if m],
        TrajectoryValidator(_run_config.data),
        num_processes=4,
    )

    table_data = [
        [
            pt.id,
            pt.age,
            pt.n_events,
            ("Dravet" if pt.future_diagnosis else "Control"),
            pt.outcome_age,
        ]
        for pt in patients
    ]
    table_data = pd.DataFrame(table_data)
    table_data.columns = ["id", "age", "events", "cohort", "age_at_outcome"]

    return table_data


@st.cache_data
def list_available_patients(run_dir: Path) -> t.List[str]:
    """Lists available patients in the run directory."""
    patient_ids = []
    for item in (run_dir / "integrated_gradients").iterdir():
        if item.is_dir():
            patient_ids.append(item.name)
    return patient_ids


@st.cache_data
def load_patient_results(
    run_dir: Path, patient_id: str
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the patient data from the run directory."""
    pt_dir = run_dir / "integrated_gradients" / patient_id
    pt_preds = pd.read_csv(pt_dir / "predictions.csv")
    pt_attrs = pd.read_csv(pt_dir / "attributions.csv")
    return pt_preds, pt_attrs


def process_attributions(
    pt_attrs: pd.DataFrame, visualizer: GradientVisualizer
) -> pd.DataFrame:
    """Processes the attributions for visualization."""
    process_attrs = lambda x: visualizer.process(x.values)
    pt_attrs["z"] = pt_attrs.groupby("trajectory_id")["attr"].transform(process_attrs)

    # sort attributions
    pt_attrs = pt_attrs.sort_values(
        ["trajectory_id", "age", "z"], ascending=[True, False, False]
    )

    # assign y coordinates
    pt_attrs["y"] = pt_attrs.groupby(["trajectory_id", "age"])["z"].transform(
        lambda x: range(len(x))
    )
    return pt_attrs
