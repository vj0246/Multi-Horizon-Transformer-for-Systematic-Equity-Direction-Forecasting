"""Multi-horizon Transformer encoder (TensorFlow/Keras).

Exact architecture from the research notebook, parameterized via config:
  Dense projection -> sinusoidal positional encoding -> N encoder blocks
  (MHA + residual/LN + FFN + residual/LN) -> GlobalAveragePooling1D -> Dense(H).

Output is H raw logits (from_logits loss), one per forecast horizon. A parallel
`attention_model` sharing the same weights exposes the 2nd block's attention
scores for interpretability.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="mht")
class AttentionPooling1D(tf.keras.layers.Layer):
    """Learned attention pooling over the time axis.

    A plain mean discards the ordering the positional encoding injects; this
    learns a softmax weighting over timesteps and returns their weighted sum.

    Implemented as a registered Layer rather than a Lambda so the model is fully
    serializable - Keras refuses to deserialize a Lambda wrapping a Python
    lambda, which blocks `model.save()` (and therefore saving optimizer state).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        # build the sublayer explicitly, otherwise it is unbuilt at save time and
        # its weights fail to restore on load
        self.score.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        w = tf.nn.softmax(self.score(x), axis=1)      # (B, T, 1) over time
        return tf.reduce_sum(x * w, axis=1)           # (B, C)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def positional_encoding(length: int, d_model: int) -> tf.Tensor:
    """Standard Vaswani et al. sinusoidal positional encoding, shape (1, length, d_model)."""
    positions = np.arange(length)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angle_rads = positions * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)


def _encoder_block(x, num_heads: int, d_model: int, ff_dim: int, dropout: float):
    """One Transformer encoder block. Returns (output, attention_scores)."""
    attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )
    attn_output, attn_scores = attn_layer(x, x, return_attention_scores=True)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ffn = tf.keras.layers.Dense(d_model)(ffn)
    x = tf.keras.layers.Add()([x, ffn])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x, attn_scores


def build_model(cfg: dict, num_features: int | None = None) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build (model, attention_model). Both share weights; attention_model emits scores.

    num_features overrides the input width; the cross-sectional track passes the
    panel feature count (base stationary features + cross-sectional features).
    """
    from Source.Pipeline.dataset import resolve_feature_cols

    m = cfg["model"]
    lookback = cfg["sequence"]["lookback"]
    if num_features is None:
        num_features = len(resolve_feature_cols(cfg))
    horizons = cfg["sequence"]["horizons"]

    inputs = tf.keras.Input(shape=(lookback, num_features))
    x = tf.keras.layers.Dense(m["d_model"])(inputs)
    x = x + positional_encoding(lookback, m["d_model"])

    attn_scores = None
    for _ in range(m["num_layers"]):
        x, attn_scores = _encoder_block(
            x, m["num_heads"], m["d_model"], m["ff_dim"], m["dropout"]
        )

    # Pooling: learned attention pooling keeps the temporal-order information the
    # positional encoding injected (a plain mean discards it); GAP kept as fallback.
    if m.get("pooling", "gap") == "attention":
        x = AttentionPooling1D(name="attention_pool")(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(horizons)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    attention_model = tf.keras.Model(inputs=inputs, outputs=attn_scores)
    return model, attention_model


def compile_model(model: tf.keras.Model, cfg: dict, objective: str = "classification") -> tf.keras.Model:
    """Compile for either classification (BCE on logits) or regression (Huber).

    The Dense(20) head is linear in both cases; only the loss/metric differ, so
    the architecture is identical across objectives.
    """
    opt = tf.keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate"])
    if objective == "regression":
        delta = cfg.get("cross_section", {}).get("huber_delta", 0.03)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.Huber(delta=delta),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    else:
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.AUC(name="auc")])
    return model
