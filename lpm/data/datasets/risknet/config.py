"""RiskNet dataset configuration."""

from __future__ import annotations

import typing as t

from dataclasses import dataclass, field

from lpm.constants import (
    RISKNET_MIN_TIME_EMBED_PERIOD_IN_DAYS,
    RISKNET_MAX_TIME_EMBED_PERIOD_IN_DAYS,
    RISKNET_MIN_FOLLOWUP_YEARS_IF_NEGATIVE,
    RISKNET_MIN_PRE_OUTCOME_DAYS_IF_POSITIVE,
    RISKNET_MIN_EVENTS_PER_SEQUENCE,
    RISKNET_NUM_TRAJECTORIES_PER_PATIENT,
    RISKNET_MONTH_ENDPOINTS,
)


@dataclass
class RiskNetDataConfig:
    """Data processing configuration for Riskformer."""

    min_time_embed_period_in_days: int = RISKNET_MIN_TIME_EMBED_PERIOD_IN_DAYS
    max_time_embed_period_in_days: int = RISKNET_MAX_TIME_EMBED_PERIOD_IN_DAYS
    min_followup_years_if_negative: int = RISKNET_MIN_FOLLOWUP_YEARS_IF_NEGATIVE
    min_pre_outcome_days_if_positive: int = RISKNET_MIN_PRE_OUTCOME_DAYS_IF_POSITIVE
    min_events_per_sequence: int = RISKNET_MIN_EVENTS_PER_SEQUENCE
    num_trajectories_per_patient: int = RISKNET_NUM_TRAJECTORIES_PER_PATIENT
    month_endpoints: t.List[int] = field(default_factory=lambda: RISKNET_MONTH_ENDPOINTS)


@dataclass
class RiskNetModelConfig:
    """Model configuration for RiskNet."""

    max_sequence_length: int
    token_embed_dim: int
    age_embed_dim: int
    num_heads: int
    num_blocks: int
    batch_size: int
    epochs: int
    n_trajectories: int
    learning_rate: float

    eval_batch_size: int
    eval_n_trajectories: int


@dataclass
class RiskNetPathConfig:
    """"""

    raw: str


@dataclass
class RiskNetPreprocessConfig:
    """"""

    num_negative_samples: int
    num_processes: int
    seed: int


@dataclass
class RiskNetConfig:
    """"""

    paths: RiskNetPathConfig
    preprocess: RiskNetPreprocessConfig
    data: RiskNetDataConfig
    model: RiskNetModelConfig
