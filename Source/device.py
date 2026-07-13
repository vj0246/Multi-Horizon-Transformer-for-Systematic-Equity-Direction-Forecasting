"""GPU/CPU device configuration for training.

Call configure_devices(cfg) at the start of any training entrypoint. It logs the
visible compute devices, enables memory growth on GPUs (important on small cards
like a 4 GB RTX 2050), and - when training.require_gpu is true - aborts with a
clear message instead of silently falling back to CPU.

IMPORTANT (Windows): the pip `tensorflow` wheel is CPU-only from 2.11 onward
(`tf.test.is_built_with_cuda()` is False). A CUDA-capable GPU is therefore
invisible to TensorFlow on native Windows regardless of driver. To actually
train on GPU, use one of:
  - WSL2 + `pip install "tensorflow[and-cuda]"`  (CUDA, recommended)
  - native Windows: TF 2.10 + `tensorflow-directml-plugin` (DirectML)
  - a cloud GPU runtime
This module reports the situation loudly rather than pretending.
"""
from __future__ import annotations

import tensorflow as tf


def configure_devices(cfg: dict) -> str:
    """Return "GPU" or "CPU"; honor training.require_gpu; enable GPU memory growth."""
    train = cfg.get("training", {})
    require_gpu = bool(train.get("require_gpu", False))
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        print(f"[device] Training on GPU: {[g.name for g in gpus]}")
        return "GPU"

    msg = (
        "[device] No GPU visible to TensorFlow "
        f"(built_with_cuda={tf.test.is_built_with_cuda()}). "
        "On native Windows the pip 'tensorflow' wheel is CPU-only from 2.11+. "
        "Use WSL2 + tensorflow[and-cuda], the DirectML plugin on TF 2.10, or a "
        "cloud GPU to train on the RTX 2050."
    )
    if require_gpu:
        raise SystemExit(msg + "\n[device] training.require_gpu is set -> aborting.")
    print(msg + "\n[device] Falling back to CPU (set training.require_gpu: true to forbid).")
    return "CPU"
