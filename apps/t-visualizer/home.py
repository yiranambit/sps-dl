#!/usr/bin/env python
"""Trajectory visualizer for RiskNet.

FIXME: some patients have 0 trajectories - this causes an error to be thrown when we 
    sample a random patient
"""

from __future__ import annotations

import streamlit as st
import tv_components as tc

from types import SimpleNamespace
from pathlib import Path

from tv_core.loaders import (
    list_available_patients,
    load_patient_results,
    load_patient,
    load_run_config,
)
from tv_core.styles import set_default_styles


st.set_page_config(
    "Trajectory Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG = SimpleNamespace(
    run_dir=Path("/Users/yiran/sps_dl/spsnet/ambit-lpm/outputs/transformer"),
    default_num_trajectories=5,
)

# pre-loading required data
run_config = load_run_config(CONFIG.run_dir)
# FIXME: try to move all data loading to here

# FIXME: lift this into a set_session_state function
if "available_patients" not in st.session_state:
    st.session_state.available_patients = list_available_patients(CONFIG.run_dir)

if "selected_patient_ids" not in st.session_state:
    st.session_state.selected_patient_ids = st.session_state.available_patients

if "patient_id" not in st.session_state:
    st.session_state.patient_id = st.session_state.available_patients[0]

if "active_filter" not in st.session_state:
    st.session_state.active_filter = False


# loading stateful data
patient = load_patient(run_config, st.session_state.patient_id)
patient_preds, patient_attrs = load_patient_results(
    CONFIG.run_dir, st.session_state.patient_id
)

if "available_trajectory_ids" not in st.session_state:
    st.session_state.available_trajectory_ids = list(
        patient_attrs["trajectory_id"].unique()
    )

if "selected_trajectory_ids" not in st.session_state:
    default_trajectory_ids = [
        t for i, t in enumerate(st.session_state.available_trajectory_ids) if i % 2 == 0
    ]
    default_trajectory_ids = default_trajectory_ids[: CONFIG.default_num_trajectories]
    st.session_state.selected_trajectory_ids = default_trajectory_ids

with st.sidebar:
    # patient selection and filtering
    st.header("Select Patient")
    tc.PatientSelectbox()
    tc.RandomPatientButton()
    tc.PatientFilterDialogButton(run_config)
    tc.ClearPatientFiltersButton()
    st.divider()

    # trajectory selection
    st.header("Filter Trajectories")
    tc.TrajectoryMultiselect()
    st.divider()

    # visualizer settings
    st.header("Visualizer Settings")
    tc.GradientVisualizerSettings()

tc.PatientSummaryMetrics(patient)
st.divider()

tc.GradientVisualizer(patient_attrs, patient_preds, height=100)

set_default_styles()
