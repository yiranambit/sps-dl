"""RiskNet model."""

from __future__ import annotations

import keras

import numpy as np
import tensorflow as tf
import typing as t

from keras import layers, ops, metrics

from lpm.layers import (
    CumulativeProbabilityLayer,
    TokenAndPositionEmbedding,
    TransformerBlock,
)


__all__ = ["RiskNet"]


@keras.saving.register_keras_serializable(package="lpm")
class RiskNet(keras.Model):
    """RiskNet model.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    input_dim : int
        Input sequence lenhth.
    token_embed_dim : int
        Dimension for token embeddings.
    age_embed_dim : int
        Dimension for age embeddings.
    num_heads : int
        Number of attention heads in each transformer block.
    num_blocks : int
        Number of transformer blocks.
    output_dim : int
        Output dimension.
    min_time_embed_period : int
        Minimum time embedding period.
    max_time_embed_period : int
        Maximum time embedding period.
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        token_embed_dim: int,
        age_embed_dim: int,
        num_heads: int,
        num_blocks: int,
        output_dim: int,
        min_time_embed_period: int,
        max_time_embed_period: int,
        tokenizer: layers.StringLookup,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.token_embed_dim = token_embed_dim
        self.age_embed_dim = age_embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.min_time_embed_period = min_time_embed_period
        self.max_time_embed_period = max_time_embed_period
        self.tokenizer = tokenizer
        self.init_sublayers()

    def init_sublayers(self) -> None:
        self.embedding = TokenAndPositionEmbedding(
            vocab_size=self.vocab_size,
            token_embed_dim=self.token_embed_dim,
            age_embed_dim=self.age_embed_dim,
            max_period=self.max_time_embed_period,
            min_period=self.min_time_embed_period,
        )
        self.encoder = self.get_encoder()

    def get_encoder(self) -> keras.Model:
        """"""
        token_input = layers.Input(shape=(self.input_dim,), dtype=tf.int64)
        age_input = layers.Input(shape=(self.input_dim,), dtype=tf.int64)

        x = self.embedding([token_input, age_input])

        padding_mask = self.embedding.compute_padding_mask([token_input, age_input])
        attention_mask = padding_mask[:, tf.newaxis]

        for _ in range(self.num_blocks):
            transformer_block = TransformerBlock(
                self.token_embed_dim,
                self.num_heads,
                self.token_embed_dim,
                dropout_rate=0.2,
            )
            x = transformer_block(x, mask=attention_mask)

        # FIXME: Make pooling an attribute so it can be accessed
        x = layers.GlobalAveragePooling1D()(x, mask=padding_mask)
        x = CumulativeProbabilityLayer(output_dim=self.output_dim, dtype=tf.float32)(x)

        return keras.Model(inputs=[token_input, age_input], outputs=x)

    @property
    def metrics(self):
        return [self.loss_tracker, self.auc_metric, self.prc_metric]

    def compile(self, **kwargs) -> None:
        super().compile(**kwargs)

        self.loss_tracker = metrics.Mean(name="loss")
        self.auc_metric = metrics.AUC(from_logits=True, name="auc")
        self.prc_metric = metrics.AUC(from_logits=True, curve="PR", name="prc")

    def custom_loss(self, y_true, y_pred, y_mask):
        """Focal Loss implementation for handling class imbalance."""
        # Parameters from SPSNet
        alpha = 0.25
        gamma = 2.0
        
        # Convert logits to probabilities
        y_pred_prob = ops.sigmoid(y_pred)
        
        # Clip to prevent log(0)
        epsilon = 1e-7
        y_pred_prob = ops.clip(y_pred_prob, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        # For positive examples
        pos_loss = -y_true * ops.power(1 - y_pred_prob, gamma) * ops.log(y_pred_prob)
        # For negative examples
        neg_loss = -(1 - y_true) * ops.power(y_pred_prob, gamma) * ops.log(1 - y_pred_prob)
        
        # Apply alpha weighting
        focal_loss = alpha * pos_loss + (1 - alpha) * neg_loss
        
        # Apply mask and compute mean
        return ops.sum(focal_loss * y_mask) / ops.sum(y_mask)

    def call(self, inputs, training=None):
        """"""
        return self.encoder(inputs, training=training)

    def train_step(self, data) -> None:
        """"""
        x, (y_true, y_mask) = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.custom_loss(y_true, y_pred, y_mask)

        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.auc_metric.update_state(y_true[:, -1], y_pred[:, -1])
        self.prc_metric.update_state(y_true[:, -1], y_pred[:, -1])

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, (y_true, y_mask) = data

        y_pred = self(x, training=False)
        loss = self.custom_loss(y_true, y_pred, y_mask)

        self.loss_tracker.update_state(loss)
        self.auc_metric.update_state(y_true[:, -1], y_pred[:, -1])
        self.prc_metric.update_state(y_true[:, -1], y_pred[:, -1])

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        base_config = super().get_config()
        config = {
            "input_dim": self.input_dim,
            "vocab_size": self.vocab_size,
            "token_embed_dim": self.token_embed_dim,
            "age_embed_dim": self.age_embed_dim,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "output_dim": self.output_dim,
            "min_time_embed_period": self.min_time_embed_period,
            "max_time_embed_period": self.max_time_embed_period,
            "tokenizer": keras.saving.serialize_keras_object(self.tokenizer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        tokenizer_config = config.pop("tokenizer")
        tokenizer = keras.saving.deserialize_keras_object(tokenizer_config)
        return cls(tokenizer=tokenizer, **config)

    def get_compile_config(self):
        base_compile_config = super().get_compile_config()
        compile_config = {
            "loss_tracker": self.loss_tracker,
            "auc_metric": self.auc_metric,
            "prc_metric": self.prc_metric,
        }
        return {**base_compile_config, **compile_config}

    def compile_from_config(self, config) -> None:
        loss_tracker = config.pop("loss_tracker")
        auc_metric = config.pop("auc_metric")
        prc_metric = config.pop("prc_metric")

        super().compile(**config)

        self.loss_tracker = keras.utils.deserialize_keras_object(loss_tracker)
        self.auc_metric = keras.utils.deserialize_keras_object(auc_metric)
        self.prc_metric = keras.utils.deserialize_keras_object(prc_metric)
