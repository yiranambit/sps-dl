"""Patient selectbox."""

from __future__ import annotations

import streamlit as st
import typing as t

from ..base import ComponentBase


def patient_selectbox_on_change() -> None:
    """Updates session state after patient selection."""
    del st.session_state.selected_trajectory_ids
    del st.session_state.available_trajectory_ids


class PatientSelectbox(ComponentBase):
    """Select patients by id."""

    def render(self) -> None:
        st.selectbox(
            "Patient ID",
            options=st.session_state.selected_patient_ids,
            key="patient_id",
            label_visibility="collapsed",
            on_change=patient_selectbox_on_change,
        )
