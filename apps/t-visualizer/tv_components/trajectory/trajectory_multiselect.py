"""Trajectory multiselect component."""

from __future__ import annotations

import streamlit as st
import typing as t

from ..base import ComponentBase


class TrajectoryMultiselect(ComponentBase):
    """Select patients by id."""

    def render(self) -> None:
        st.multiselect(
            "Trajectory IDs",
            key="selected_trajectory_ids",
            options=st.session_state.available_trajectory_ids,
            label_visibility="collapsed",
        )
