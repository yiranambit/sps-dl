""""""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import typing as t

from keras import ops

from .base import ElementWisePreprocessingLayer


class DownSample(ElementWisePreprocessingLayer):
    """"""

    def __init__(self, n: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n = n

    def call(self, inputs):
        """Downsample tensor elements along the last axis."""
        return tf.map_fn(self._downsample_sequence, inputs)

    def preprocess(self, element: tf.Tensor) -> tf.Tensor:
        """Downsample tensor elements."""
        idxs = tf.range(tf.shape(element)[-1])
        rand_idxs = tf.random.shuffle(idxs)
        return tf.gather(element, rand_idxs[: self.n])


class WeightedDownSample(ElementWisePreprocessingLayer):
    """"""

    def __init__(self, n: int, token_weight_dict: t.Dict[int, float], **kwargs) -> None:
        super().__init__(**kwargs)
        self.n = tf.constant(n)
        self.W = self._init_token_weights(token_weight_dict)

    def preprocess(self, seq: tf.Tensor) -> tf.Tensor:
        """Weighted downsampling of a sequence using token weights."""

        def downsample(seq):
            weights = tf.gather(self.W, ops.cast(seq, tf.int64))
            weights_normalized = ops.divide(weights, ops.sum(weights))

            log_probs = ops.convert_to_tensor(ops.log(weights_normalized + 1e-6))
            samples = tf.random.categorical(ops.expand_dims(log_probs, 0), self.n)

            return tf.gather(seq, samples[0])

        return ops.cond(
            ops.less_equal(ops.shape(seq)[-1], self.n),
            lambda: seq,
            lambda: downsample(seq),
        )

    @staticmethod
    def _init_token_weights(weight_dict: t.Dict[int, float]) -> tf.Tensor:
        """Initializes token weights."""
        # NOTE: We add one to the length to account for padding token (0).
        weights = np.zeros((len(weight_dict) + 1,))
        for k, v in weight_dict.items():
            weights[k] = v

        return tf.constant(weights, dtype=tf.float64)
