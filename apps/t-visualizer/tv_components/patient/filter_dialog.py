"""Patient browser dialog."""

from __future__ import annotations

import streamlit as st
import typing as t

from tv_core.loaders import load_patient_table

from ..base import ComponentBase

if t.TYPE_CHECKING:
    from omegaconf import DictConfig


_BROWSER_COLUMN_CONFIG = {
    "id": st.column_config.Column("Patient ID", disabled=True),
    "age": st.column_config.NumberColumn(
        "Patient Age (Yrs.)", disabled=True, format="%.1f"
    ),
    "age_at_outcome": st.column_config.NumberColumn(
        "Age at Outcome (Yrs.)", disabled=True, format="%.1f"
    ),
    "events": st.column_config.NumberColumn("Events", disabled=True),
    "cohort": st.column_config.Column("Cohort", disabled=True),
}

_BROWSER_COLUMN_ORDER = ["id", "cohort", "age", "age_at_outcome", "events"]


@st.dialog("Filter Patients", width="large")
def patient_filter_dialog(run_config: DictConfig):
    with st.container():
        with st.spinner("Loading patients..."):
            data = load_patient_table(run_config, st.session_state.available_patients)

        c1, c2 = st.columns([0.85, 0.15])

        query = c1.text_input(
            "Query",
            placeholder="Enter a query to filter patients.",
            label_visibility="collapsed",
        )
        if query:
            try:
                data = data.query(query)
            except Exception:
                st.error("Please enter a valid query.")

        if c2.button("Apply", use_container_width=True, type="primary"):
            selected_patient_ids = data["id"].to_list()
            st.session_state.selected_patient_ids = selected_patient_ids
            st.session_state.active_filter = True
            st.session_state.patient_id = selected_patient_ids[0]
            del st.session_state.selected_trajectory_ids
            del st.session_state.available_trajectory_ids
            st.rerun()

    st.data_editor(
        data,
        use_container_width=True,
        hide_index=True,
        column_order=_BROWSER_COLUMN_ORDER,
        column_config=_BROWSER_COLUMN_CONFIG,
    )


class PatientFilterDialogButton(ComponentBase):
    """Random patient button."""

    def __init__(self, run_config: DictConfig):
        self.run_config = run_config
        super().__init__()

    def render(self) -> None:
        st.button(
            "Filter Patients",
            on_click=patient_filter_dialog,
            use_container_width=True,
            type="secondary",
            args=(self.run_config,),
        )
