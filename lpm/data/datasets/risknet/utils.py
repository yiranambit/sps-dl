"""RiskNet data utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import typing as t

from collections import namedtuple

from .config import RiskNetDataConfig

if t.TYPE_CHECKING:
    from .dataset import Patient


Event = t.Tuple[pd.Timestamp, str]
EventWithModifier = t.Tuple[pd.Timestamp, str, str]


Trajectory = namedtuple(
    "Trajectory",
    ["patient_id", "code_seq", "age_seq", "y", "y_seq", "y_mask"],
)


TrajectoryWithModifiers = namedtuple(
    "TrajectoryWithModifiers",
    ["patient_id", "code_seq", "age_seq", "modifier_seq", "y", "y_seq", "y_mask"],
)


def get_trajectory_labels(
    events: t.List[Event | EventWithModifier],
    outcome_date: pd.Timestamp,
    future_diagnosis: bool,
    month_endpoints: t.List[int],
) -> t.List[int]:
    """"""
    last_event_date = events[-1][0]
    days_until_outcome = (outcome_date - last_event_date).days

    num_time_steps = len(month_endpoints)
    max_time_step = max(month_endpoints)

    y = future_diagnosis and (days_until_outcome < max_time_step * 30)
    y_seq = np.zeros(num_time_steps, dtype=np.float64)
    y_mask = np.ones(num_time_steps, dtype=np.float64)

    if days_until_outcome < (max_time_step * 30):
        is_after_endpoint = lambda mo: days_until_outcome < (mo * 30)
        time_at_outcome = min(
            [i for i, mo in enumerate(month_endpoints) if is_after_endpoint(mo)]
        )
    else:
        time_at_outcome = num_time_steps - 1

    if y:
        y_seq[time_at_outcome:] = 1

    y_mask[time_at_outcome + 1 :] = 0

    assert time_at_outcome >= 0 and len(y_seq) == len(y_mask)

    return int(y), y_seq, y_mask, time_at_outcome, days_until_outcome


def get_time_sequence(
    dates: t.List[pd.Timestamp], reference_date: pd.Timestamp
) -> t.List[int]:
    """"""
    return [abs((reference_date - date).days) for date in dates]


class TrajectoryValidator:
    """"""

    def __init__(self, config: RiskNetDataConfig | None = None) -> None:
        if config is None:
            config = RiskNetDataConfig()
        self.config = config

    def is_valid_trajectory(
        self,
        events: t.List[Event | EventWithModifier],
        outcome_date: pd.Timestamp,
        future_diagnosis: bool,
    ) -> bool:
        """Check if a trajectory is valid."""
        if len(events) < self.config.min_events_per_sequence:
            return False

        last_event_date = events[-1][0]
        days_until_outcome = (outcome_date - last_event_date).days

        if future_diagnosis:
            # FIXME: add a param that specifies pre outcome interval here
            is_pre_outcome = last_event_date < outcome_date
            is_in_time_horizon = days_until_outcome < (
                max(self.config.month_endpoints) * 30
            )
            return is_pre_outcome and is_in_time_horizon

        return days_until_outcome // 365 > self.config.min_followup_years_if_negative

    def get_first_valid_trajactory_indice(self, patient: Patient) -> int:
        """Returns the index of the first valid trajectory."""
        for idx, (date, *_) in enumerate(patient.events):
            if patient.future_diagnosis:
                days_until_outcome = (patient.outcome_date - date).days
                if days_until_outcome < self.config.min_pre_outcome_days_if_positive:
                    continue
            if self.is_valid_trajectory(
                patient.events[: idx + 1],
                patient.outcome_date,
                patient.future_diagnosis,
            ):
                return idx
        return -1

    def get_valid_trajectory_indices(self, patient: Patient) -> t.List[int]:
        """Returns the indices of valid trajectories."""
        # valid_indices
        valid_indices = []
        for idx, (date, *_) in enumerate(patient.events):
            if patient.future_diagnosis:
                days_until_outcome = (patient.outcome_date - date).days
                if days_until_outcome < self.config.min_pre_outcome_days_if_positive:
                    continue

            if self.is_valid_trajectory(
                patient.events[: idx + 1],
                patient.outcome_date,
                patient.future_diagnosis,
            ):
                valid_indices.append(idx)
                # days_to_censor = (patient.outcome_date - date).days

                # any_trajectory_contains_diagnosis = (
                #     any_trajectory_contains_diagnosis
                #     or days_to_censor < (max(self.cfg.month_endpoint) * 30)
                # )

        return valid_indices
