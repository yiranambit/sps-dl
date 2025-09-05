"""Component base classes."""

from __future__ import annotations

import streamlit as st
import typing as t

from types import SimpleNamespace
from abc import ABC, abstractmethod

if t.TYPE_CHECKING:
    from lpm.data.datasets.risknet import Patient


class ComponentBase(ABC):

    def __init__(self):
        self.render()

    @abstractmethod
    def render(self) -> None:
        """Renders the component."""
        ...
