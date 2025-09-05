"""Gradient visualizer settings."""

from __future__ import annotations

import streamlit as st

from types import SimpleNamespace

from ..base import ComponentBase


class GradientVisualizerSettings(ComponentBase):
    """Gradient visualizer settings."""

    _defaults = {
        "clip_above_percentile": 95,
        "clip_below_percentile": 30,
        "lower_end": 0.2,
    }

    def __init__(self) -> None:
        for key, value in self._defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        self.render()

    def render(self) -> SimpleNamespace:
        """Renders the visualizer settings."""
        st.number_input(
            "Clip Above Percentile",
            min_value=0,
            max_value=100,
            step=5,
            key="clip_above_percentile",
        )
        st.number_input(
            "Clip Below Percentile",
            min_value=0,
            max_value=100,
            step=5,
            key="clip_below_percentile",
        )
        st.number_input(
            "Lower End",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="lower_end",
        )
        st.button(
            "Restore Defaults",
            use_container_width=True,
            on_click=self.restore_defaults,
            type="secondary",
        )

    def restore_defaults(self) -> None:
        """Restores the default visualizer settings."""
        for key, value in self._defaults.items():
            st.session_state[key] = value
