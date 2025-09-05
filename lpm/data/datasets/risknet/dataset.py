"""RiskNet dataset."""

from __future__ import annotations

import json
import random
import functools

import pandas as pd
import numpy as np
import typing as t

from collections.abc import Sequence
from dataclasses import dataclass, field, asdict
from pathlib import Path
from joblib import delayed
from tqdm import tqdm

from lpm.utils.progress import ParallelTqdm

from . import utils as du


@dataclass(repr=False)
class Patient:
    """Container for raw patient data."""

    id: str
    dob: pd.Timestamp
    outcome_date: pd.Timestamp
    future_diagnosis: bool
    events: du.Event
    trajectory_validator: du.TrajectoryValidator = field(
        default_factory=du.TrajectoryValidator
    )

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id})"

    @property
    def age(self) -> float:
        """Patient age in years."""
        return (self.last_event_date - self.dob).days / 365.25

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def outcome_age(self) -> float:
        return (self.outcome_date - self.dob).days / 365.25

    @property
    def last_event_date(self) -> pd.Timestamp:
        return self.events[-1][0]

    def describe(self) -> None:
        print(f"Patient ID: {self.id}")
        print(f"Age: {self.age:.2f}")
        print(f"Diagnosis: {self.future_diagnosis}")
        print(f"Outcome Age: {self.outcome_age:.2f}")
        print(f"No. Events: {self.n_events}")

    @functools.cached_property
    def valid_trajectory_indices(self) -> t.List[int]:
        return self.trajectory_validator.get_valid_trajectory_indices(self)

    @functools.cached_property
    def has_valid_trajectories(self) -> bool:
        """Return valid patient trajectories."""
        # print("check patient id", self.id, "diganosis: ", self.future_diagnosis)
        result = self.trajectory_validator.get_first_valid_trajactory_indice(self) != -1
        # print("check patient id", self.id, "valid trajectory: ", result)
        return result

    def sample_trajectories(self, n: int) -> t.List[du.Trajectory]:
        """Return valid patient trajectories."""
        valid_indices = self.valid_trajectory_indices
        sampled_indices = random.sample(valid_indices, min(len(valid_indices), n))

        trajectories = []
        for idx in sampled_indices:
            events = self.events[: idx + 1]
            dates, codes = zip(*events)

            code_seq = list(codes)
            age_seq = du.get_time_sequence(dates, self.dob)

            y, y_seq, y_mask, _, _ = du.get_trajectory_labels(
                events,
                self.outcome_date,
                self.future_diagnosis,
                self.trajectory_validator.config.month_endpoints,
            )
            trajectories.append(
                du.Trajectory(self.id, code_seq, age_seq, y, y_seq, y_mask)
            )

        return trajectories

    def get_trajectories(self) -> t.List[du.Trajectory]:
        """"""
        trajectories = []
        for idx in self.valid_trajectory_indices:
            events = self.events[: idx + 1]
            dates, codes = zip(*events)

            code_seq = list(codes)
            age_seq = du.get_time_sequence(dates, self.dob)

            y, y_seq, y_mask, _, _ = du.get_trajectory_labels(
                events,
                self.outcome_date,
                self.future_diagnosis,
                self.trajectory_validator.config.month_endpoints,
            )
            trajectories.append(
                du.Trajectory(self.id, code_seq, age_seq, y, y_seq, y_mask)
            )

        return trajectories

    def to_json(self, file_path: str | Path) -> None:
        """Serialize patient data to JSON."""
        serialize_patient(self, file_path)

    @classmethod
    def from_json(
        cls, file_path: str | Path, trajectory_validator: du.TrajectoryValidator
    ) -> Patient:
        """Deserialize patient data from JSON."""
        return deserialize_patient(file_path, trajectory_validator)


def serialize_patient(patient: Patient, file_path: str | Path) -> None:
    """Serialize patient data to JSON."""
    patient_dict = asdict(patient)
    del patient_dict["trajectory_validator"]
    with open(file_path, "w") as f:
        json.dump(patient_dict, f, default=str)


def serialize_patients_parallel(
    patients: t.Iterable[Patient], out_dir: Path, num_processes: int, **kwargs
) -> None:
    """Serialize patients to JSON."""
    out_dir.mkdir(exist_ok=True, parents=True)

    _ = ParallelTqdm(total_tasks=len(patients), n_jobs=num_processes, **kwargs)(
        delayed(serialize_patient)(patient, out_dir / f"{patient.id}.json")
        for patient in patients
    )


def serialize_patients(patients: t.Iterable[Patient], out_dir: Path) -> None:
    """Serialize patients to JSON."""
    out_dir.mkdir(exist_ok=True, parents=True)

    for i, patient in enumerate(tqdm(patients)):
        serialize_patient(patient, out_dir / f"patient-{i}.json")


def deserialize_patient(
    file_path: Path, trajectory_validator: du.TrajectoryValidator
) -> Patient:
    """Load a patient from a JSON file."""
    with open(file_path, "r") as f:
        patient_dict = json.load(f)

    patient_dict["dob"] = pd.Timestamp(patient_dict["dob"])
    patient_dict["outcome_date"] = pd.Timestamp(patient_dict["outcome_date"])
    patient_dict["events"] = [(pd.Timestamp(d), c) for d, c in patient_dict["events"]]

    return Patient(**patient_dict, trajectory_validator=trajectory_validator)


def deserialize_patients_parallel(
    file_list: t.List[str | Path],
    trajectory_validator: du.TrajectoryValidator,
    num_processes: int,
) -> PatientCollection:
    """Deserialize patients from JSON."""

    parallel = ParallelTqdm(total_tasks=len(file_list), n_jobs=num_processes)

    patients = parallel(
        delayed(deserialize_patient)(f, trajectory_validator) for f in file_list
    )

    return PatientCollection(*patients)


def build_vocab_from_patients(patients: PatientCollection) -> np.ndarray:
    """Build a vocabulary from a list of patients."""
    all_codes = []
    for pt in patients:
        codes = [event[1] for event in pt.events]
        all_codes.extend(codes)
    return np.unique(all_codes)


class PatientCollection(Sequence[Patient]):
    """"""

    def __init__(self, *items) -> None:
        self._container = [*items]

    def __len__(self) -> int:
        return len(self._container)

    def __getitem__(self, index: int) -> Patient:
        return self._container[index]

    def __add__(self, other: PatientCollection) -> PatientCollection:
        return PatientCollection(*self._container, *other._container)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={len(self)})"

    @property
    def positive_patients(self) -> t.List[Patient]:
        return [pt for pt in self._container if pt.future_diagnosis]

    @property
    def negative_patients(self) -> t.List[Patient]:
        return [pt for pt in self._container if not pt.future_diagnosis]

    @property
    def n_pos(self) -> int:
        return len(self.positive_patients)

    @property
    def n_neg(self) -> int:
        return len(self.negative_patients)


@dataclass
class RiskNetDataset:
    """"""

    pos: PatientCollection
    neg: PatientCollection


def load_dataset(
    data_dir: Path, trajectory_validator: du.TrajectoryValidator, num_processes: int
) -> t.Tuple[RiskNetDataset, RiskNetDataset]:
    """Load a dataset from disk."""
    deserializer = functools.partial(
        deserialize_patients_parallel,
        trajectory_validator=trajectory_validator,
        num_processes=num_processes,
    )

    train_pos = deserializer(list(data_dir.glob("train/pos/*.json")))
    train_neg = deserializer(list(data_dir.glob("train/neg/*.json")))
    test_pos = deserializer(list(data_dir.glob("test/pos/*.json")))
    test_neg = deserializer(list(data_dir.glob("test/neg/*.json")))

    return RiskNetDataset(train_pos, train_neg), RiskNetDataset(test_pos, test_neg)


def load_train_dataset(
    data_dir: Path, trajectory_validator: du.TrajectoryValidator, num_processes: int
) -> RiskNetDataset:
    """Load a dataset from disk."""
    deserializer = functools.partial(
        deserialize_patients_parallel,
        trajectory_validator=trajectory_validator,
        num_processes=num_processes,
    )

    train_pos = deserializer(list(data_dir.glob("train/pos/*.json")))
    train_neg = deserializer(list(data_dir.glob("train/neg/*.json")))

    return RiskNetDataset(train_pos, train_neg)


def load_test_dataset(
    data_dir: Path, trajectory_validator: du.TrajectoryValidator, num_processes: int
) -> RiskNetDataset:
    """Load a dataset from disk."""
    deserializer = functools.partial(
        deserialize_patients_parallel,
        trajectory_validator=trajectory_validator,
        num_processes=num_processes,
    )

    test_pos = deserializer(list(data_dir.glob("test/pos/*.json")))
    test_neg = deserializer(list(data_dir.glob("test/neg/*.json")))

    return RiskNetDataset(test_pos, test_neg)
