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


def build_model(cfg: dict) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build (model, attention_model). Both share weights; attention_model emits scores."""
    from Source.Pipeline.dataset import resolve_feature_cols

    m = cfg["model"]
    lookback = cfg["sequence"]["lookback"]
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

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(horizons)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    attention_model = tf.keras.Model(inputs=inputs, outputs=attn_scores)
    return model, attention_model


def compile_model(model: tf.keras.Model, cfg: dict) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate"]),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model
