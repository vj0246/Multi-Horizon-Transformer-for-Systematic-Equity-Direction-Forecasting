# 2. Getting Started

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | 3.12 works in WSL |
| TensorFlow | 2.21 | GPU requires WSL2 on Windows (see below) |
| Node.js | 18+ | Frontend only |
| NVIDIA GPU | optional | `training.require_gpu: true` in config aborts CPU runs |

## Install

```bash
pip install -r requirements.txt      # 15 packages, all actually imported
cd frontend && npm install && cd ..
```

`requirements.txt` is derived from the imports in `Source/`, `scripts/` and
`tests/`. The core pipeline needs only numpy, pandas, scipy, scikit-learn,
pyyaml, joblib, tensorflow and yfinance; lightgbm, fastapi and the FinBERT stack
are optional and marked as such in the file.

## The five-minute path

```bash
# 1. Verify everything is correct (39 tests: leakage, costs, metrics, drift, gate)
python -m pytest tests/test_rigorous.py -q

# 2. Look at the results without retraining anything
cat frontend/public/data/summary.json

# 3. See the site
cd frontend && npm run dev        # http://localhost:3000
```

Every number on the site is read from `frontend/public/data/*.json`. Those files
are committed, so you can explore the full result set without a GPU.

## Regenerating artifacts

Order matters — later steps consume earlier outputs.

```bash
# Refresh market data (optional; the CSVs are committed)
python -m Source.Ingestion.Fetch_Market_Data     # ^NSEI OHLCV
python -m Source.Ingestion.fetch_macro           # VIX, USDINR, crude, S&P
python -m Source.Ingestion.fetch_universe        # 85 NSE names (gitignored)

# Index track -> summary/horizons/strategies/calibration/decile/... json
python -m Source.Backtest.run

# Cross-sectional track -> cross_section.json, stock_signals.json
python -m Source.Backtest.run_cross_section

# Freeze the paper model (only when the architecture or feature set changes)
python scripts/save_paper_model.py

# Step the paper book and rebuild current predictions
python -m Source.Paper.run --refresh
python -m Source.Insights.build

# Adaptive audit: drift detection + provenance (trains nothing)
python -m Source.Adaptive.run

# Rebuild the static site
cd frontend && npm run build      # -> frontend/out
```

### Faster iteration

`REUSE=1 python -m Source.Backtest.run` rebuilds the JSON from
`Data/Processed_Data/run_cache_v2.npz` without retraining. Use it for
presentation-only changes. It cannot produce `stock_signals.json`, which needs a
live model.

## GPU training on Windows

Native Windows TensorFlow 2.11+ is **CPU-only** — this is a hard upstream limit,
not a configuration problem. Train through WSL2:

```bash
# One-time: CUDA-enabled venv inside WSL Ubuntu
wsl -d Ubuntu bash -lc 'python3 -m venv ~/venvs/mht && \
  ~/venvs/mht/bin/pip install "tensorflow[and-cuda]==2.21.0" \
  numpy pandas scipy scikit-learn pyyaml yfinance joblib'

# Every run
scripts/train_gpu.sh Source.Backtest.run
```

**Known trap:** the pip GPU wheel does not put the `nvidia-*-cu12` libraries on
`LD_LIBRARY_PATH`. TensorFlow then reports *"Cannot dlopen some GPU libraries"*
and silently falls back to CPU with `GPUs: []`. `scripts/wsl_gpu_env.sh` fixes
this by adding `site-packages/nvidia/*/lib` to the path. Source it before any
training command.

`Source/device.py` logs the device in use and, with `training.require_gpu: true`,
aborts rather than silently training on CPU.

### Calling WSL from Git Bash

Paths get mangled unless you disable conversion and embed the command:

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu bash -c \
  "source '/mnt/c/.../scripts/wsl_gpu_env.sh'; cd '/mnt/c/...'; python -m Source.Backtest.run"
```

### GPU memory

The RTX 2050 exposes only ~1.7 GB usable. Batch sizes above 128 fail with
`Unexpected Event status: 1`. Config ships `batch_size: 32` for the index track;
the cross-sectional panel uses batch 128 with `window_stride: 3`.

## Configuration

Everything is driven by `config.yaml`. Nothing is hardcoded in the modules.

| Key | Value | Meaning |
|-----|-------|---------|
| `sequence.lookback` | 60 | Days of history per window |
| `sequence.horizons` | 20 | Forward horizons predicted |
| `model.d_model` | 64 | Embedding width |
| `model.num_heads` | 4 | Attention heads |
| `model.num_layers` | 2 | Encoder blocks |
| `model.dropout` | 0.4 | Validation-selected |
| `model.pooling` | attention | `attention` or `gap` |
| `training.learning_rate` | 0.0003 | Validation-selected |
| `training.n_seeds` | 3 | Ensemble size |
| `training.require_gpu` | true | Abort on CPU |
| `split.train_frac` / `val_frac` | 0.70 / 0.15 | Remainder is test |
| `backtest.holding_period` | 20 | Non-overlapping hold length |
| `backtest.quantile_upper` | 70 | Entry threshold percentile |
| `backtest.cost_model` | india | Full India cost stack |

## Determinism

`TF_DETERMINISTIC_OPS` and `TF_CUDNN_DETERMINISM` are set before TensorFlow is
imported in both run scripts, and seeds are fixed. Re-running
`scripts/save_paper_model.py` reproduces the same weights and the same paper
book.

Caveat: GPU `MultiHeadAttention` is not perfectly deterministic across driver
versions. The cross-sectional track is noise-dominated enough that the same
config can swing long-only from +19% to +33% between machines. Do not read
precision into cross-sectional point estimates — this is documented in
[Results](07-results.md).

Continue to [Data](03-data.md).
