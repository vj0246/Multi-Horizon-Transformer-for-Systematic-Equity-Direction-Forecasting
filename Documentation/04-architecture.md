# 4. Architecture

## System pipeline

```mermaid
flowchart TD
    A["yfinance<br/>^NSEI OHLCV + macro + 85 NSE names"] --> B["data_loader.load_ohlcv<br/>strip 3 header rows, drop dup adj-close"]
    B --> C["Features/<br/>Returns · Volatility · Macro<br/>11 price + 8 macro = 19"]
    C --> D["dataset.build_features<br/>attach 20 forward targets, dropna"]
    D --> E["make_windows<br/>60-day windows, stop 20 early"]
    E --> F["temporal_split_and_scale<br/>70/15/15 chronological<br/>StandardScaler fit on TRAIN only"]
    F --> G["Transformer x3 seeds<br/>Models/ensemble.py"]
    G --> H["20 raw logits per day"]
    H --> I["Platt calibration<br/>fit on VALIDATION only"]
    H --> J["ensemble_signal<br/>z-score by val stats, mean over 20 heads"]
    J --> K["Long/flat timing rule<br/>rolling 250d percentile"]
    K --> L["Backtest/metrics.py<br/>non-overlapping returns, India costs"]
    I --> M["Evaluation/suite.py<br/>AUC · IC · DSR · multiple testing"]
    L --> M
    M --> N["frontend/public/data/*.json"]
    N --> O["Next.js static site"]

    style G fill:#1e3a5f,color:#fff
    style M fill:#5f1e1e,color:#fff
    style N fill:#1e5f2e,color:#fff
```

## Model architecture

```mermaid
flowchart TD
    IN["Input<br/>(batch, 60, 19)"] --> PROJ["Dense projection<br/>19 → 64"]
    PROJ --> PE["+ sinusoidal positional encoding<br/>(1, 60, 64)"]

    PE --> B1

    subgraph B1["Encoder block 1"]
        direction TB
        A1["MultiHeadAttention<br/>4 heads · key_dim 16 · dropout 0.4"] --> R1["Add residual"]
        R1 --> N1["LayerNorm ε=1e-6"]
        N1 --> F1["Dense 128 ReLU → Dense 64"]
        F1 --> R2["Add residual"]
        R2 --> N2["LayerNorm ε=1e-6"]
    end

    B1 --> B2

    subgraph B2["Encoder block 2"]
        direction TB
        A2["MultiHeadAttention<br/>4 heads · key_dim 16 · dropout 0.4"] --> R3["Add residual"]
        R3 --> N3["LayerNorm ε=1e-6"]
        N3 --> F2["Dense 128 ReLU → Dense 64"]
        F2 --> R4["Add residual"]
        R4 --> N4["LayerNorm ε=1e-6"]
    end

    B2 --> POOL["AttentionPooling1D<br/>softmax over time, weighted sum<br/>(batch, 60, 64) → (batch, 64)"]
    POOL --> OUT["Dense(20), linear<br/>20 RAW LOGITS"]
    OUT --> LOSS["BinaryCrossentropy(from_logits=True)"]

    B2 -.attention scores.-> VIZ["attention_model<br/>interpretability only"]

    style POOL fill:#1e3a5f,color:#fff
    style OUT fill:#5f1e1e,color:#fff
```

### Parameters

| Component | Setting | Rationale |
|-----------|---------|-----------|
| `d_model` | 64 | Small — 3k training windows cannot support a large model |
| `num_heads` | 4 | key_dim = 64/4 = 16 per head |
| `ff_dim` | 128 | 2x d_model, standard ratio |
| `num_layers` | 2 | Deeper overfits at this sample size |
| `dropout` | 0.4 | High, validation-selected. The main overfitting control |
| `pooling` | attention | Learned weighting over timesteps |
| Output | Dense(20) linear | Raw logits |
| Loss | BCE `from_logits=True` | Numerically stable |
| Optimizer | Adam, lr 3e-4 | Validation-selected |

Roughly 120k parameters against ~3,000 training windows. The dropout of 0.4 plus
early stopping (patience 7, best weights restored) does the regularisation work.

## Three design decisions worth understanding

### 1. Attention pooling, not global average pooling

Global average pooling discards the ordering that the positional encoding just
injected. Attention pooling learns which timesteps matter:

```python
@tf.keras.utils.register_keras_serializable(package="mht")
class AttentionPooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        self.score.build(input_shape)   # else unbuilt at save time
        super().build(input_shape)

    def call(self, x):
        w = tf.nn.softmax(self.score(x), axis=1)   # (B, T, 1) over time
        return tf.reduce_sum(x * w, axis=1)        # (B, C)
```

**This must stay a registered `Layer`, never a `Lambda`.** Keras refuses to
deserialize a `Lambda` wrapping a Python lambda, which makes `model.save()` raise
and therefore makes optimizer state unsaveable. The inner `Dense` must be built
in `build()` — otherwise it is unbuilt at save time and its weights silently fail
to restore, producing a model that loads without error and predicts nonsense.

Changing this invalidates existing `.weights.h5` files. Re-run
`scripts/save_paper_model.py`.

### 2. Raw logits everywhere

The model outputs raw logits. `tf.sigmoid` is applied only at inference for
display. The logits themselves are the alpha signal, because their *magnitude*
carries confidence information that a thresholded probability throws away.

### 3. Seed ensembling

GPU `MultiHeadAttention` is not fully deterministic. Training 3 seeds and
averaging their predictions collapses run-to-run variance and improves
generalisation slightly. `training.n_seeds: 3`.

## Signal construction

```mermaid
flowchart LR
    L["20 logits"] --> Z["z-score by<br/>validation mean/std"]
    Z --> M["mean over<br/>all 20 heads"]
    M --> S["ensemble signal"]
    S --> T{"signal ≥ 250-day<br/>70th percentile?"}
    T -->|yes| LONG["LONG"]
    T -->|no| FLAT["FLAT"]
```

Z-scoring uses **validation** statistics so no test information enters signal
construction. All 20 heads are used rather than h=20 alone — averaging reduces
variance.

The threshold rule is *rolling and past-only*: the 70th percentile is computed
over the trailing 250 days, never the full sample. A fixed threshold fit on
validation was tried first and turned out degenerate on test (see
[Data, rule 6](03-data.md#6-everything-selected-on-validation)).

## Cross-sectional track

```mermaid
flowchart TD
    U["85 NSE names<br/>daily OHLCV"] --> P["Pipeline/cross_section.py<br/>panel builder"]
    P --> XS["Cross-sectional features<br/>universe-demeaned momentum<br/>per-date ranks<br/>sector-relative momentum"]
    XS --> SPLIT["Split by CALENDAR DATE<br/>not by row"]
    SPLIT --> MODEL["Same Transformer<br/>num_features override"]
    MODEL --> RANK["Rank stocks per date"]
    RANK --> Q["Quintile baskets"]
    Q --> SPREAD["Long top / short bottom<br/>charged real India costs"]
```

Targets are **relative**: does this stock beat the universe median over `h` days?
This neutralises market beta, forcing the model to find relative information.

Config `cross_section.objective` switches between `classification` (beat-median
labels) and `regression` (continuous excess log-return, Huber loss). The head
stays `Dense(20)` linear in both cases; only the loss changes. Classification is
primary — regression underperformed (IC −0.012).

Continue to [File Reference](05-file-reference.md).
