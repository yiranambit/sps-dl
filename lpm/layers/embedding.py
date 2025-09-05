"""Embedding layers for patient2gene models."""

from __future__ import annotations

import keras

import numpy as np
import tensorflow as tf
import typing as t

from keras import layers, ops


__all__ = [
    "TimeEmbedding",
    "TokenAndPositionEmbedding",
    "TokenPositionAndModifierEmbedding",
]


@keras.saving.register_keras_serializable(package="lpm")
class TimeEmbedding(layers.Layer):
    """Generates embeddings for temporal values.

    Parameters
    ----------
    min_period : int
        The minimum period of the embedding space in days.
    max_period : int
        The maximum period of the embedding space in days.
    embed_dim : int
        The dimension of the embedding space.
    """

    def __init__(
        self,
        min_period: int,
        max_period: int,
        embed_dim: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.min_period = min_period
        self.max_period = max_period
        self.embed_dim = embed_dim
        self.multipliers = self.get_multipliers()

    def call(self, x: t.Any):
        seq_len = ops.shape(x)[-1]
        x = ops.reshape(x, (-1, seq_len, 1))
        x = ops.cos(ops.matmul(x, self.multipliers))
        return x

    def get_multipliers(self):
        """"""
        embed_linspace = ops.linspace(self.min_period, self.max_period, self.embed_dim)
        return ops.reshape(2 * np.pi / embed_linspace, (1, -1))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "min_period": self.min_period,
            "max_period": self.max_period,
            "embed_dim": self.embed_dim,
        }
        return {**base_config, **config}


@keras.saving.register_keras_serializable(package="lpm")
class TokenAndPositionEmbedding(layers.Layer):
    """Embeddings for token and position values.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    token_embed_dim : int
        Dimension for token embeddings.
    age_embed_dim : int
        Dimension for age embeddings.
    min_period : int
        Minimum period for time embeddings.
    max_period : int
        Maximum period for time embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        token_embed_dim: int,
        age_embed_dim: int,
        min_period: int,
        max_period: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.token_embed_dim = token_embed_dim
        self.age_embed_dim = age_embed_dim
        self.min_period = min_period
        self.max_period = max_period
        self.init_sublayers()

    def init_sublayers(self) -> None:
        self.token_emb = layers.Embedding(
            self.vocab_size, output_dim=self.token_embed_dim
        )
        self.age_emb = TimeEmbedding(self.min_period, self.max_period, self.age_embed_dim)
        self.age_dense1 = layers.Dense(self.token_embed_dim)
        self.age_dense2 = layers.Dense(self.token_embed_dim)

    def call(self, inputs: t.Tuple[t.Any, t.Any]):
        x_code, x_age = inputs
        x_code = self.token_emb(x_code)
        x_age = self.age_emb(x_age)
        x = self.condition_on_age_embedding((x_code, x_age))
        return x

    def condition_on_age_embedding(self, inputs):
        """"""
        x_code, x_age = inputs
        return self.age_dense1(x_age) * x_code + self.age_dense2(x_age)

    def compute_padding_mask(self, inputs):
        return ops.not_equal(inputs[0], 0)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "vocab_size": self.vocab_size,
            "token_embed_dim": self.token_embed_dim,
            "age_embed_dim": self.age_embed_dim,
            "min_period": self.min_period,
            "max_period": self.max_period,
        }
        return {**base_config, **config}


@keras.saving.register_keras_serializable(package="lpm")
class TokenPositionAndModifierEmbedding(TokenAndPositionEmbedding):

    def __init__(
        self,
        vocab_size: int,
        token_embed_dim: int,
        age_embed_dim: int,
        modifier_embed_dim: int,
        modifier_vocab_size: int,
        min_period: int,
        max_period: int,
        **kwargs,
    ) -> None:
        self.modifier_embed_dim = modifier_embed_dim
        self.modifier_vocab_size = modifier_vocab_size
        super().__init__(
            vocab_size, token_embed_dim, age_embed_dim, min_period, max_period, **kwargs
        )

    def init_sublayers(self) -> None:
        self.token_emb = layers.Embedding(
            self.vocab_size, output_dim=self.token_embed_dim
        )
        self.age_emb = TimeEmbedding(self.min_period, self.max_period, self.age_embed_dim)
        self.modifier_emb = layers.Embedding(
            self.modifier_vocab_size, output_dim=self.modifier_embed_dim
        )
        self.age_dense1 = layers.Dense(self.token_embed_dim)
        self.age_dense2 = layers.Dense(self.token_embed_dim)
        self.modifier_dense1 = layers.Dense(self.token_embed_dim)
        self.modifier_dense2 = layers.Dense(self.token_embed_dim)

    def call(self, inputs: t.Tuple[t.Any, t.Any]):
        x_code, x_age, x_mod = inputs
        x_code = self.token_emb(x_code)
        x_age = self.age_emb(x_age)
        x = self.condition_on_age_embedding((x_code, x_age))
        x = self.condition_on_modifier_embedding((x, x_mod))
        return x

    def condition_on_modifier_embedding(self, inputs):
        """"""
        x_code, x_mod = inputs
        return self.modifier_dense1(x_mod) * x_code + self.modifier_dense2(x_mod)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "modifier_vocab_size": self.modifier_vocab_size,
            "modifier_embed_dim": self.modifier_embed_dim,
        }
        return {**base_config, **config}
