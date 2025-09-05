""""""

from __future__ import annotations

import typing as t
import numpy as np

from abc import ABC, abstractmethod


class Generator(ABC):
    """Generator for creating keras sequences as inputs for modeling."""

    @abstractmethod
    def num_batch_dims(self):
        """
        Returns the number of batch dimensions in returned tensors (_not_ the batch size itself).

        For instance, for full batch methods like GCN, the feature has shape ``1 x number of nodes x
        feature size``, where the 1 is a "dummy" batch dimension and ``number of nodes`` is the real
        batch size (every node in the graph).
        """
        ...

    @abstractmethod
    def flow(self, *args, **kwargs):
        """
        Creates a Keras Sequence or similar input appropriate for CDRP models.
        """
        ...
