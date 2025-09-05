"""Random patient button."""

from __future__ import annotations

import streamlit as st
import numpy as np

from ..base import ComponentBase


def select_random_patient() -> None:
    """Selects a random patient."""
    selected_patient_ids = st.session_state.selected_patient_ids
    curr_idx = selected_patient_ids.index(st.session_state.patient_id)
    other_idxs = [i for i in range(len(selected_patient_ids)) if i != curr_idx]
    rand_idx = np.random.randint(len(other_idxs))

    st.session_state.patient_id = selected_patient_ids[other_idxs[rand_idx]]

    # reset trajectory selection state
    del st.session_state.selected_trajectory_ids
    del st.session_state.available_trajectory_ids


class RandomPatientButton(ComponentBase):
    """Random patient button."""

    def render(self) -> None:
        st.button(
            "Random Patient",
            on_click=select_random_patient,
            use_container_width=True,
            type="primary",
        )
