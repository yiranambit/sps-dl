"""Patient summary metrics."""

from __future__ import annotations

import streamlit as st
import typing as t

from ..base import ComponentBase

if t.TYPE_CHECKING:
    from lpm.data.datasets.risknet import Patient


class PatientSummaryMetrics(ComponentBase):
    """Patient summary metrics."""

    def __init__(self, pt: Patient):
        self.pt = pt
        super().__init__()

    def render(self) -> None:
        with st.container():
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Cohort", "Dravet" if self.pt.future_diagnosis else "Control")
            c2.metric("Age (Yrs.)", f"{self.pt.age:.2f}")
            c3.metric(self._age_at_dx_label(self.pt), f"{self.pt.outcome_age:.2f}")
            c4.metric("No. Events", self.pt.n_events)
            c5.metric("Active Interal (Yrs.)", f"{self._active_interval(self.pt):.2f}")

    @staticmethod
    def _age_at_dx_label(pt: Patient) -> str:
        """Returns the age at diagnosis label."""
        return (
            "Age at Diagnosis (Yrs.)"
            if pt.future_diagnosis
            else "Age at Last Event (Yrs.)"
        )

    @staticmethod
    def _active_interval(pt: Patient) -> float:
        """Returns the active interval."""
        return (pt.outcome_date - pt.dob).days / 365.25
