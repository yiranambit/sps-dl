""""""

from __future__ import annotations

import tensorflow as tf

from keras import ops

from .base import ElementWisePreprocessingLayer


class ZeroPadding1D(ElementWisePreprocessingLayer):
    """"""

    def __init__(self, maxlen: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.maxlen = maxlen

    def preprocess(self, element):
        """Pad tensor elements."""

        def pad(element, element_len):
            padding = tf.zeros((self.maxlen - element_len,), dtype=element.dtype)
            return tf.concat([element, padding], 0)

        element_len = tf.shape(element)[-1]
        return ops.cond(
            tf.greater_equal(element_len, self.maxlen),
            lambda: element,
            lambda: pad(element, element_len),
        )
