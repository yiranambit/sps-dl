"""Integrated gradients for model interpretation."""

from __future__ import annotations

import keras

import numpy as np
import tensorflow as tf
import typing as t

from keras import ops
from tqdm import tqdm


class IntegratedGradients:
    """Integrated gradients for interpreting model predictions.

    Parameters
    ----------
    model : keras.Model
        The model to interpret.
    steps : int
        The number of steps to use in the approximation.
    class_idx : int
        The index of the class to interpret.

    References
    ----------
    [1] https://keras.io/examples/vision/integrated_gradients/
    """

    def __init__(self, model: keras.Model, steps: int = 50, class_idx: int = -1) -> None:
        self.model = model
        self.steps = steps
        self.class_idx = class_idx

    def run(
        self,
        x: tf.Tensor,
        baseline: tf.Tensor | None = None,
        mask: tf.Tensor | None = None,
    ) -> np.ndarray:
        """Run integrated gradients."""
        if baseline is None:
            baseline = tf.zeros_like(x)

        grads = self._compute_gradients(x, baseline, self.class_idx)
        avg_grads = self._integral_approximation(grads)

        if mask is not None:
            avg_grads *= ops.expand_dims(mask, axis=1)

        attrs = (x - baseline) * avg_grads

        # sum over the embedding dimension
        agg_attrs = tf.reduce_sum(attrs, axis=-1)

        return agg_attrs.numpy()

    def run_many(
        self,
        x: t.Iterable[tf.Tensor],
        baselines: t.Iterable[tf.Tensor] | None = None,
        masks: t.Iterable[tf.Tensor] | None = None,
    ) -> t.List[np.ndarray]:
        """Run integrated gradients for multiple inputs."""
        if baselines is None:
            baselines = [tf.zeros_like(xi) for xi in x]

        if masks is None:
            masks = [None] * len(x)

        integrated_grads = []
        desc = "Running integrated gradients"
        for xi, bi, mi in tqdm(zip(x, baselines, masks), total=len(x), desc=desc):
            integrated_grads.append(self.run(xi, bi, mi))

        return integrated_grads

    def _interpolate_sequence(self, x: tf.Tensor, baseline: tf.Tensor) -> tf.Tensor:
        """"""
        delta = (x - baseline) / self.steps
        return tf.convert_to_tensor([baseline + i * delta for i in range(self.steps + 1)])

    def _integral_approximation(self, grads: tf.Tensor) -> tf.Tensor:
        """Riemann_trapezoidal approximation."""
        grads = (grads[:-1] + grads[1:]) / tf.constant(2.0)
        integrated_grads = tf.math.reduce_mean(grads, axis=0)
        return integrated_grads

    def _compute_gradients(
        self, x: tf.Tensor, baseline: tf.Tensor, class_idx: int
    ) -> tf.Tensor:
        """"""
        x_interp = self._interpolate_sequence(x, baseline)
        with tf.GradientTape() as tape:
            tape.watch(x_interp)
            logits = self.model(x_interp)[:, class_idx]
            # FIXME: decide if this should be logits or probs
            probs = ops.sigmoid(logits)
        grads = tape.gradient(probs, x_interp)
        return grads


class GradientVisualizer:
    """Transform gradients into a visual representation.

    Parameters
    ----------
    clip_above_percentile : float
        The percentile above which to clip the gradients.
    clip_below_percentile : float
        The percentile below which to clip the gradients.
    lower_end : float
        The lower end of the color scale.

    References
    ----------
    [1] https://keras.io/examples/vision/integrated_gradients/
    """

    def __init__(
        self,
        clip_above_percentile: float = 99.9,
        clip_below_percentile: float = 10.0,
        lower_end: float = 0.2,
    ) -> None:
        self.clip_above_percentile = clip_above_percentile
        self.clip_below_percentile = clip_below_percentile
        self.lower_end = lower_end

    def _apply_polarity(
        self, attributions: np.ndarray, polarity: t.Literal["pos", "neg"]
    ) -> np.ndarray:
        if polarity == "pos":
            return np.clip(attributions, 0, 1)
        elif polarity == "neg":
            return np.abs(np.clip(attributions, -1, 0))
        else:
            raise ValueError("Invalid polarity.")

    def _get_thresholded_attributions(
        self, attributions: np.ndarray, percentage: float
    ) -> np.ndarray:
        """"""
        if percentage == 100.0:
            # FIXME: check this
            return np.min(attributions)

        flat_attributions = attributions.flatten()
        total = np.sum(flat_attributions)

        sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
        cumsum = 100.0 * np.cumsum(sorted_attributions) / total
        indices = np.where(cumsum >= percentage)[0][0]

        return sorted_attributions[indices]

    def _apply_linear_transform(self, attributions: np.ndarray) -> np.ndarray:
        """"""
        m = self._get_thresholded_attributions(
            attributions, 100 - self.clip_above_percentile
        )
        e = self._get_thresholded_attributions(
            attributions, 100 - self.clip_below_percentile
        )

        attributions = (1 - self.lower_end) * (np.abs(attributions) - e) / (
            m - e
        ) + self.lower_end

        attributions *= np.sign(attributions)
        attributions *= attributions >= self.lower_end
        attributions = np.clip(attributions, 0.0, 1.0)

        return attributions

    def process(
        self,
        attributions: np.ndarray,
        polarity: t.Literal["pos", "neg"] = "pos",
    ) -> np.ndarray:
        """"""
        attributions = self._apply_polarity(attributions, polarity)
        attributions = self._apply_linear_transform(attributions)
        return attributions

    def process_many(
        self,
        attributions: t.Iterable[np.ndarray],
        polarity: t.Literal["pos", "neg"] = "pos",
    ) -> t.List[np.ndarray]:
        """"""
        desc = "Processing attributions"
        processed_grads = []
        for a in tqdm(attributions, total=len(attributions), desc=desc):
            processed_grads.append(self.process(a, polarity))

        return processed_grads
