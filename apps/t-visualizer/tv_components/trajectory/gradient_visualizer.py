"""Gradient visualizer component."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from types import SimpleNamespace

from lpm.interp import GradientVisualizer as _GradientVisualizer

from tv_core.plotters import IntegratedGradientsPlotter
from tv_core.loaders import process_attributions
from ..base import ComponentBase


class GradientVisualizer(ComponentBase):
    """Gradient visualizer settings."""

    def __init__(
        self, attr_data: pd.DataFrame, pred_data: pd.DataFrame, height: int = 80
    ) -> None:
        self.attr_data = attr_data
        self.pred_data = pred_data
        self.height = height
        super().__init__()

    def render(self) -> None:
        """Renders the visualizer settings."""
        gviz = _GradientVisualizer(
            clip_above_percentile=st.session_state.clip_above_percentile,
            clip_below_percentile=st.session_state.clip_below_percentile,
            lower_end=st.session_state.lower_end,
        )

        sel_trajectory_ids = st.session_state.selected_trajectory_ids

        if not sel_trajectory_ids:
            st.error("Please select a trajectory.")

        else:
            attr_data_sel = self.attr_data.query("trajectory_id in @sel_trajectory_ids")

            # HACK: gets rid of setting with copy warning
            attr_data_sel = attr_data_sel.copy()
            attr_data_sel = process_attributions(attr_data_sel, gviz)

            pred_data_sel = self.pred_data.query("trajectory_id in @sel_trajectory_ids")
            pred_data_sel = pred_data_sel.assign(x=0, y=0)

            plotter = IntegratedGradientsPlotter(height=self.height)
            attr_chart, pred_chart = plotter.plot(attr_data_sel, pred_data_sel)

            c1, c2 = st.columns([0.9, 0.1])
            c1.altair_chart(attr_chart, use_container_width=True, theme=None)
            c2.altair_chart(pred_chart, use_container_width=True, theme=None)
