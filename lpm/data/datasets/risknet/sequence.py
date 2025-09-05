"""RiskNet keras sequence."""

from __future__ import annotations

import numpy as np
import typing as t

from tensorflow import keras
from keras import layers

from lpm.utils.parsing import flatten

if t.TYPE_CHECKING:
    from .utils import Patient, Trajectory, TrajectoryWithModifiers


EncodedTrajectory = t.Tuple[
    t.Tuple[np.ndarray, np.ndarray], t.Tuple[np.ndarray, np.ndarray]
]
EncodedTrajectoryWithModifiers = t.Tuple[
    t.Tuple[np.ndarray, np.ndarray, np.ndarray], t.Tuple[np.ndarray, np.ndarray]
]


def encode_trajectories(
    trajectories: t.Iterable[Trajectory],
    max_codes: int,
    tokenizer: layers.StringLookup,
) -> EncodedTrajectory:
    """Encodes trajectories and labels."""
    code_seqs = [t.code_seq for t in trajectories]
    code_seqs = keras.utils.pad_sequences(
        code_seqs,
        maxlen=max_codes,
        dtype=object,
        value="",
        padding="pre",
        truncating="pre",
    )
    code_seqs = tokenizer(code_seqs).numpy()

    age_seqs = [t.age_seq for t in trajectories]
    age_seqs = keras.utils.pad_sequences(
        age_seqs,
        maxlen=max_codes,
        value=-1,
        padding="pre",
        truncating="pre",
    )

    y_seqs = [t.y_seq for t in trajectories]
    y_mask = [t.y_mask for t in trajectories]

    y_seqs = np.array(y_seqs, dtype=np.float32)
    y_mask = np.array(y_mask, dtype=np.float32)

    return (code_seqs, age_seqs), (y_seqs, y_mask)


class BaseRiskNetSequence(keras.utils.Sequence):

    def __init__(
        self,
        pos_patients: t.Iterable[Patient],
        neg_patients: t.Iterable[Patient],
        batch_size: int,
        max_codes: int,
        tokenizer: layers.StringLookup,
        n_trajectories: int = 1,
        shuffle: bool = False,
        seed: t.Any | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pos_patients = pos_patients
        self.neg_patients = neg_patients
        self.batch_size = batch_size
        self.max_codes = max_codes
        self.tokenizer = tokenizer
        self.n_trajectories = n_trajectories
        self.shuffle = shuffle

        self._rs = np.random.default_rng(seed)

        self.pos_indices = np.arange(len(self.pos_patients))
        self.neg_indices = np.arange(len(self.neg_patients))
        self.pos_data_size = len(self.pos_patients)
        self.neg_data_size = len(self.neg_patients)
        self.data_size = self.pos_data_size + self.neg_data_size

        self.on_epoch_end()

    def encode_trajectories(self, trajectories: t.Iterable[Trajectory]):
        """Encodes trajectories and labels."""
        return encode_trajectories(trajectories, self.max_codes, self.tokenizer)

    def on_epoch_end(self):
        if self.shuffle:
            self._rs.shuffle(self.pos_indices)
            self._rs.shuffle(self.neg_indices)


class RiskNetSequence(BaseRiskNetSequence):

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, idx: int):
        """"""
        pos_idx_0 = idx * self.pos_batch_size
        neg_idx_0 = idx * self.neg_batch_size
        pos_idx_1 = min(pos_idx_0 + self.pos_batch_size, self.pos_data_size)
        neg_idx_1 = min(neg_idx_0 + self.neg_batch_size, self.neg_data_size)

        pos_inds = self.pos_indices[pos_idx_0:pos_idx_1]
        neg_inds = self.neg_indices[neg_idx_0:neg_idx_1]

        pos_patients = np.take(self.pos_patients, pos_inds)
        neg_patients = np.take(self.neg_patients, neg_inds)

        pos_trajectories = [
            p.sample_trajectories(self.n_trajectories) for p in pos_patients
        ]
        neg_trajectories = [
            p.sample_trajectories(self.n_trajectories) for p in neg_patients
        ]

        batch_features, batch_labels = self.encode_trajectories(
            flatten(pos_trajectories) + flatten(neg_trajectories)
        )

        return batch_features, batch_labels

    @property
    def pos_batch_size(self) -> int:
        return int(np.ceil((self.pos_data_size / self.data_size) * self.batch_size))

    @property
    def neg_batch_size(self) -> int:
        return self.batch_size - self.pos_batch_size


class BalancedRiskNetSequence(BaseRiskNetSequence):
    """"""

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(self.pos_data_size / self.pos_batch_size))

    def __getitem__(self, idx: int):
        """"""
        pos_inds = self._rs.choice(self.pos_indices, self.pos_batch_size, replace=True)
        neg_inds = self._rs.choice(self.neg_indices, self.neg_batch_size, replace=True)

        pos_patients = np.take(self.pos_patients, pos_inds)
        neg_patients = np.take(self.neg_patients, neg_inds)

        pos_trajectories = [
            p.sample_trajectories(self.n_trajectories) for p in pos_patients
        ]
        neg_trajectories = [
            p.sample_trajectories(self.n_trajectories) for p in neg_patients
        ]

        batch_features, batch_labels = self.encode_trajectories(
            flatten(pos_trajectories) + flatten(neg_trajectories)
        )

        return batch_features, batch_labels

    @property
    def pos_batch_size(self) -> int:
        return int(self.batch_size / 2)

    @property
    def neg_batch_size(self) -> int:
        return self.batch_size - self.pos_batch_size


def encode_trajectories_v2(
    trajectories: t.Iterable[TrajectoryWithModifiers],
    max_codes: int,
    tokenizer: layers.StringLookup,
    modifier_tokenizer: layers.StringLookup,
) -> EncodedTrajectoryWithModifiers:
    """Encodes trajectories and labels."""
    code_seqs = [t.code_seq for t in trajectories]
    code_seqs = keras.utils.pad_sequences(
        code_seqs,
        maxlen=max_codes,
        dtype=object,
        value="",
        padding="pre",
        truncating="pre",
    )
    code_seqs = tokenizer(code_seqs).numpy()

    age_seqs = [t.age_seq for t in trajectories]
    age_seqs = keras.utils.pad_sequences(
        age_seqs,
        maxlen=max_codes,
        value=-1,
        padding="pre",
        truncating="pre",
    )

    mod_seqs = [t.modifier_seq for t in trajectories]
    mod_seqs = keras.utils.pad_sequences(
        code_seqs,
        maxlen=max_codes,
        dtype=object,
        value="",
        padding="pre",
        truncating="pre",
    )
    mod_seqs = modifier_tokenizer(mod_seqs).numpy()

    y_seqs = [t.y_seq for t in trajectories]
    y_mask = [t.y_mask for t in trajectories]

    y_seqs = np.array(y_seqs, dtype=np.float32)
    y_mask = np.array(y_mask, dtype=np.float32)

    return (code_seqs, age_seqs, mod_seqs), (y_seqs, y_mask)


class BaseRiskNetv2Sequence(keras.utils.Sequence):

    def __init__(
        self,
        pos_patients: t.Iterable[Patient],
        neg_patients: t.Iterable[Patient],
        batch_size: int,
        max_codes: int,
        tokenizer: layers.StringLookup,
        modifier_tokenizer: layers.StringLookup,
        n_trajectories: int = 1,
        shuffle: bool = False,
        seed: t.Any | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pos_patients = pos_patients
        self.neg_patients = neg_patients
        self.batch_size = batch_size
        self.max_codes = max_codes
        self.tokenizer = tokenizer
        self.modifier_tokenizer = modifier_tokenizer
        self.n_trajectories = n_trajectories
        self.shuffle = shuffle

        self._rs = np.random.default_rng(seed)

        self.pos_indices = np.arange(len(self.pos_patients))
        self.neg_indices = np.arange(len(self.neg_patients))
        self.pos_data_size = len(self.pos_patients)
        self.neg_data_size = len(self.neg_patients)
        self.data_size = self.pos_data_size + self.neg_data_size

        self.on_epoch_end()

    def encode_trajectories(self, trajectories: t.Iterable[Trajectory]):
        """Encodes trajectories and labels."""
        return encode_trajectories_v2(
            trajectories,
            max_codes=self.max_codes,
            tokenizer=self.tokenizer,
            modifier_tokenizer=self.modifier_tokenizer,
        )

    def on_epoch_end(self):
        if self.shuffle:
            self._rs.shuffle(self.pos_indices)
            self._rs.shuffle(self.neg_indices)


class RiskNetv2Sequence(BaseRiskNetv2Sequence):

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, idx: int):
        """"""
        pos_idx_0 = idx * self.pos_batch_size
        neg_idx_0 = idx * self.neg_batch_size
        pos_idx_1 = min(pos_idx_0 + self.pos_batch_size, self.pos_data_size)
        neg_idx_1 = min(neg_idx_0 + self.neg_batch_size, self.neg_data_size)

        pos_inds = self.pos_indices[pos_idx_0:pos_idx_1]
        neg_inds = self.neg_indices[neg_idx_0:neg_idx_1]

        pos_patients = np.take(self.pos_patients, pos_inds)
        neg_patients = np.take(self.neg_patients, neg_inds)

        pos_trajectories = [
            p.sample_trajectories(self.n_trajectories) for p in pos_patients
        ]
        neg_trajectories = [
            p.sample_trajectories(self.n_trajectories) for p in neg_patients
        ]

        batch_features, batch_labels = self.encode_trajectories(
            flatten(pos_trajectories) + flatten(neg_trajectories)
        )

        return batch_features, batch_labels

    @property
    def pos_batch_size(self) -> int:
        return int(np.ceil((self.pos_data_size / self.data_size) * self.batch_size))

    @property
    def neg_batch_size(self) -> int:
        return self.batch_size - self.pos_batch_size


class BalancedRiskNetv2Sequence(BaseRiskNetv2Sequence):
    """"""

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(self.pos_data_size / self.pos_batch_size))

    def __getitem__(self, idx: int):
        """"""
        pos_inds = self._rs.choice(self.pos_indices, self.pos_batch_size, replace=True)
        neg_inds = self._rs.choice(self.neg_indices, self.neg_batch_size, replace=True)

        pos_patients = np.take(self.pos_patients, pos_inds)
        neg_patients = np.take(self.neg_patients, neg_inds)

        pos_trajectories = [
            p.sample_trajectories(self.n_trajectories) for p in pos_patients
        ]
        neg_trajectories = [
            p.sample_trajectories(self.n_trajectories) for p in neg_patients
        ]

        batch_features, batch_labels = self.encode_trajectories(
            flatten(pos_trajectories) + flatten(neg_trajectories)
        )

        return batch_features, batch_labels

    @property
    def pos_batch_size(self) -> int:
        return int(self.batch_size / 2)

    @property
    def neg_batch_size(self) -> int:
        return self.batch_size - self.pos_batch_size
