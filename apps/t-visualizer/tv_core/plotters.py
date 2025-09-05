"""Plotters for trajectory visualization."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import typing as t

from abc import ABC, abstractmethod


def configure_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(
            domainColor="gray",
            labelColor="#31333F",
            tickColor="gray",
            titleColor="#31333F",
            titleFont="Source Sans Pro",
            titleFontSize=14,
            titleFontWeight="normal",
            titlePadding=10,
        )
        .configure_view(strokeOpacity=0, strokeWidth=0)
        .configure_legend(
            titleColor="#31333F",
            titleFont="Source Sans Pro",
            titleFontSize=14,
            titleFontWeight="normal",
            labelColor="#31333F",
        )
    )


class PlotterBase(ABC):
    """Base class for trajectory plotters."""

    def __init__(self):
        pass

    @abstractmethod
    def plot(self, data: t.Any) -> alt.Chart | t.Tuple[alt.Chart, ...]:
        """"""
        ...


class IntegratedGradientsPlotter(PlotterBase):
    """Plots the integrated gradients."""

    def __init__(self, height: int = 80):
        super().__init__()
        self.height = height

    def plot(
        self, attr_data: pd.DataFrame, pred_data: pd.DataFrame
    ) -> t.Tuple[alt.Chart, alt.Chart]:
        """Plots the integrated gradients."""
        trajectory_panel = self._make_trajectory_panel(attr_data)
        prediction_panel = self._make_prediction_panel(pred_data)

        return (configure_chart(trajectory_panel), configure_chart(prediction_panel))

    def _make_trajectory_panel(self, data: pd.DataFrame) -> alt.Chart:
        """Creates a panel for a single trajectory."""
        step = 0.5
        inv = 1 / step

        x_min = np.floor(data["age"].min() * inv) / inv
        x_max = np.ceil(data["age"].max() * inv) / inv
        y_min = -1
        y_max = max(data["y"].max() + 1, 10)

        return (
            alt.Chart(data)
            .mark_circle(size=80)
            .encode(
                alt.X("age:Q")
                .axis(
                    grid=False, offset=10, values=list(np.arange(x_min, x_max + 0.5, 0.5))
                )
                .scale(domain=(x_min, x_max))
                .title("Patient Age (Yrs.)"),
                alt.Y("y:Q")
                .axis(
                    grid=True,
                    gridWidth=1,
                    gridColor="lightgray",
                    values=[-1],
                    domain=False,
                    labels=False,
                    ticks=False,
                )
                .scale(domain=(y_min, y_max))
                .title(None),
                alt.Row("trajectory_id:O")
                .sort("descending")
                .spacing(10)
                .header(None)
                .title(None),
                alt.Color("z:Q")
                .scale(scheme="lighttealblue")
                .title("Attribution")
                .legend(orient="top", tickCount=3),
                tooltip=[
                    alt.Tooltip("age:Q", format=".2f", title="Age"),
                    alt.Tooltip("code:N", title="Code"),
                    alt.Tooltip("desc:N", title="Description"),
                    alt.Tooltip("z:Q", format=".2f", title="Attribution"),
                ],
            )
            .properties(height=self.height, width="container")
            .interactive(bind_y=False)
            .resolve_axis(x="shared")
        )

    def _make_prediction_panel(self, data: pd.DataFrame) -> alt.Chart:
        """"""
        base = alt.Chart(data).encode(
            alt.X("x:O").axis(labels=False, ticks=False, domainOpacity=0).title(None),
            alt.Y("y:O").axis(labels=False, ticks=False, domainOpacity=0).title(None),
            tooltip=[
                alt.Tooltip("y_pred_4:Q", format=".2f", title="Predicted Risk"),
                alt.Tooltip("trajectory_id:O", title="Trajectory ID"),
            ],
        )

        rect = base.mark_rect(opacity=1.0).encode(
            alt.Color("y_pred_4:Q")
            .scale(scheme="redblue", reverse=True, domain=(0, 1))
            .legend(orient="top")
            .title("Predicted Risk")
        )

        text = base.mark_text(size=14).encode(
            alt.Text("y_pred_4:Q", format=".2f"),
            alt.condition(
                (alt.datum.y_pred_4 <= 0.75) & (alt.datum.y_pred_4 >= 0.25),
                alt.ColorValue("black"),
                alt.ColorValue("white"),
            ),
        )

        return (
            (rect + text)
            .properties(height=self.height, width="container")
            .resolve_axis(x="shared")
            .facet(
                row=alt.Row("trajectory_id:O")
                .sort("descending")
                .header(None)
                .title(None),
                spacing=10,
            )
        )
