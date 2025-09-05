""""""

from __future__ import annotations

import tensorflow as tf

from abc import ABC, abstractmethod
from tensorflow import keras


__all__ = [
    "ElementWisePreprocessingLayer",
    "RandomShuffle",
    "ZeroPadding1D",
    "DownSample",
    "WeightedDownSample",
]


class ElementWisePreprocessingLayer(keras.layers.Layer, ABC):
    """"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """Element-wise preprocessing."""
        return tf.map_fn(self.preprocess, inputs)

    @abstractmethod
    def preprocess(self, element: tf.Tensor) -> tf.Tensor:
        """Preprocesses an element."""
        pass
