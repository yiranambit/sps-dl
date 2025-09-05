"""Cumulative probability layer."""

from __future__ import annotations

import keras

import tensorflow as tf

from keras import layers, ops
from keras import Sequential


__all__ = ["CumulativeProbabilityLayer"]


@keras.saving.register_keras_serializable(package="patient2gene")
class CumulativeProbabilityLayer(layers.Layer):

    def __init__(self, output_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.mask = tf.constant(ops.triu(ops.ones((5, 5)), k=0), dtype=self.dtype)
        self.init_sublayers()

    def init_sublayers(self) -> None:
        self.hazard_fc = layers.Dense(self.output_dim, activation="sigmoid")
        self.base_hazard_fc = layers.Dense(1)

    def call(self, x):
        hazards = self.hazard_fc(x)
        expanded_hazards = ops.repeat(
            ops.expand_dims(hazards, -1), self.output_dim, axis=-1
        )
        masked_hazards = expanded_hazards * self.mask
        cum_probs = ops.sum(masked_hazards, axis=1) + self.base_hazard_fc(x)
        return cum_probs

    def get_config(self):
        base_config = super().get_config()
        config = {"output_dim": self.output_dim}
        return {**base_config, **config}
