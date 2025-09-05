""""""

from __future__ import annotations

import numpy as np
import typing as t

from keras import layers

from ..base.generator import Generator
from .sequence import (
    RiskNetSequence,
    RiskNetv2Sequence,
    BalancedRiskNetSequence,
    BalancedRiskNetv2Sequence,
)

if t.TYPE_CHECKING:
    from .dataset import Patient


class RiskNetBatchGenerator(Generator):
    """"""

    def __init__(
        self,
        batch_size: int,
        max_codes: int,
        tokenizer: layers.StringLookup,
        n_trajectories: int = 1,
    ) -> None:
        self.batch_size = batch_size
        self.max_codes = max_codes
        self.tokenizer = tokenizer
        self.n_trajectories = n_trajectories

    def num_batch_dims(self) -> int:
        return 1

    def flow(
        self,
        pos_patients: t.Iterable[Patient],
        neg_patients: t.Iterable[Patient],
        shuffle: bool = False,
        seed: t.Any = None,
    ) -> RiskNetSequence:
        """"""
        return RiskNetSequence(
            pos_patients,
            neg_patients,
            batch_size=self.batch_size,
            max_codes=self.max_codes,
            tokenizer=self.tokenizer,
            n_trajectories=self.n_trajectories,
            shuffle=shuffle,
            seed=seed,
        )


class BalancedRiskNetBatchGenerator(RiskNetBatchGenerator):

    def flow(
        self,
        pos_patients: t.Iterable[Patient],
        neg_patients: t.Iterable[Patient],
        shuffle: bool = False,
        seed: t.Any = None,
    ) -> BalancedRiskNetSequence:
        """"""
        return BalancedRiskNetSequence(
            pos_patients,
            neg_patients,
            batch_size=self.batch_size,
            max_codes=self.max_codes,
            tokenizer=self.tokenizer,
            n_trajectories=self.n_trajectories,
            shuffle=shuffle,
            seed=seed,
        )


class RiskNetv2BatchGenerator(Generator):
    """"""

    def __init__(
        self,
        batch_size: int,
        max_codes: int,
        tokenizer: layers.StringLookup,
        modifier_tokenizer: layers.StringLookup,
        n_trajectories: int = 1,
    ) -> None:
        self.batch_size = batch_size
        self.max_codes = max_codes
        self.tokenizer = tokenizer
        self.modifier_tokenizer = modifier_tokenizer
        self.n_trajectories = n_trajectories

    def num_batch_dims(self) -> int:
        return 1

    def flow(
        self,
        pos_patients: t.Iterable[Patient],
        neg_patients: t.Iterable[Patient],
        shuffle: bool = False,
        seed: t.Any = None,
    ) -> RiskNetv2Sequence:
        """"""
        return RiskNetv2Sequence(
            pos_patients,
            neg_patients,
            batch_size=self.batch_size,
            max_codes=self.max_codes,
            tokenizer=self.tokenizer,
            modifier_tokenizer=self.modifier_tokenizer,
            n_trajectories=self.n_trajectories,
            shuffle=shuffle,
            seed=seed,
        )


class BalancedRiskNetv2BatchGenerator(RiskNetv2BatchGenerator):

    def flow(
        self,
        pos_patients: t.Iterable[Patient],
        neg_patients: t.Iterable[Patient],
        shuffle: bool = False,
        seed: t.Any = None,
    ) -> BalancedRiskNetv2Sequence:
        """"""
        return BalancedRiskNetv2Sequence(
            pos_patients,
            neg_patients,
            batch_size=self.batch_size,
            max_codes=self.max_codes,
            tokenizer=self.tokenizer,
            modifier_tokenizer=self.modifier_tokenizer,
            n_trajectories=self.n_trajectories,
            shuffle=shuffle,
            seed=seed,
        )
