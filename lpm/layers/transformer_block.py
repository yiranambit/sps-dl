""""""

from __future__ import annotations

import keras

import tensorflow as tf

from keras import layers, ops
from keras import Sequential


__all__ = ["TransformerBlock"]


def causal_attention_mask(batch_size: int, n_dest: int, n_src: int, dtype: tf.DType):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult)


@keras.saving.register_keras_serializable(package="lpm")
class TransformerBlock(layers.Layer):
    """"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.init_sublayers()

    def init_sublayers(self) -> None:
        """"""
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.ffn = Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.add1 = layers.Add()
        self.add2 = layers.Add()

    def call(self, inputs, training=None, mask=None):
        """"""
        attn_output = self.att(inputs, inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.add1([inputs, attn_output])
        out1 = self.layernorm1(out1)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.add2([out1, ffn_output])
        out2 = self.layernorm2(out2)

        return out2

    def get_config(self):
        base_config = super().get_config()
        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        }
        return {**base_config, **config}


@keras.saving.register_keras_serializable(package="lpm")
class CausalAttentionTransformerBlock(layers.Layer):
    """"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.init_sublayers()

    def init_sublayers(self) -> None:
        """"""
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.ffn = Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        """"""
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "bool")
        attn_output = self.att(
            inputs, inputs, attention_mask=causal_mask, training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
