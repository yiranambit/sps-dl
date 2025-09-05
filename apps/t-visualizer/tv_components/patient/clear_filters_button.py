"""Clear patient filters button."""

from __future__ import annotations

import streamlit as st
import typing as t

from ..base import ComponentBase


def reset_filter_state() -> None:
    """Resets the active filter state."""
    st.session_state.selected_patient_ids = st.session_state.available_patients
    st.session_state.active_filter = False

    # reset trajectory selection state
    del st.session_state.selected_trajectory_ids
    del st.session_state.available_trajectory_ids

    # HACK: somehow this prevents the patient_id from being reset
    st.session_state.patient_id = st.session_state.patient_id


class ClearPatientFiltersButton(ComponentBase):
    """Reset patient filters."""

    def render(self) -> None:
        st.button(
            "Clear Filters",
            on_click=reset_filter_state,
            use_container_width=True,
            disabled=(not st.session_state.active_filter),
            type="secondary",
        )
