""""""

from __future__ import annotations

import tensorflow as tf

from .base import ElementWisePreprocessingLayer


def random_shuffle_sequence(seq: tf.Tensor) -> tf.Tensor:
    """Randomly shuffles a sequence."""
    idxs = tf.range(tf.shape(seq)[-1], dtype=tf.int64)
    shuffled_idxs = tf.random.shuffle(idxs)
    return tf.gather(seq, shuffled_idxs)


class RandomShuffle(ElementWisePreprocessingLayer):
    """"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, element: tf.Tensor) -> tf.Tensor:
        return random_shuffle_sequence(element)
