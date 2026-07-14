"""Seed-ensemble training: average several models' predictions.

GPU attention / cuDNN kernels are not fully deterministic even with op-
determinism enabled, so a single training run's metrics wobble from run to run.
Training a handful of models with different seeds and averaging their
predictions collapses that variance and generally improves generalization.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import tensorflow as tf


def train_ensemble(
    build_compile: Callable[[], tuple[tf.keras.Model, tf.keras.Model | None]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    predict_sets: dict[str, np.ndarray],
    seeds: list[int],
    epochs: int,
    batch_size: int,
    patience: int,
):
    """Train one model per seed; return averaged predictions over `predict_sets`.

    build_compile() must return (model, attn_model_or_None) freshly built and
    compiled. Returns (avg_predictions, last_model, last_attn_model, last_history).
    """
    sums: dict[str, np.ndarray | None] = {k: None for k in predict_sets}
    last_model = last_attn = last_hist = None
    for s in seeds:
        tf.keras.utils.set_random_seed(int(s))
        model, attn = build_compile()
        es = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
        hist = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0,
        )
        for k, X in predict_sets.items():
            p = model.predict(X, verbose=0, batch_size=512)
            sums[k] = p if sums[k] is None else sums[k] + p
        last_model, last_attn, last_hist = model, attn, hist.history
    n = len(seeds)
    avg = {k: (v / n if v is not None else None) for k, v in sums.items()}
    return avg, last_model, last_attn, last_hist
