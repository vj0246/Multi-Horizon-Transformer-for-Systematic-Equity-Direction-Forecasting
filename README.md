Multi-Horizon Transformer for Systematic Equity Direction Forecasting
Summary

This project develops a Transformer-based time-series model to predict equity price direction across multiple future horizons using temporal self-attention. The model consumes a 60-day window of engineered financial features and outputs probabilities for price movement over the next 1 to 20 days.

A custom encoder-only Transformer is implemented from scratch, including positional encoding and multi-head self-attention, to capture temporal dependencies in market data. The system is trained using leakage-free preprocessing with strict chronological validation to ensure realistic performance estimation.

Model evaluation goes beyond classification metrics and focuses on economic usefulness. A fully vectorized long–short backtesting framework is used to simulate trading strategies based on model predictions.

The model achieves approximately 0.62 AUC at the 20-day horizon and a long–short Sharpe ratio of approximately 0.6 after correcting for overlapping returns, indicating statistically meaningful and economically relevant predictive signal.

This project emphasizes robustness, interpretability, and real-world applicability rather than purely optimizing predictive accuracy.

Problem Formulation

Given historical market data:

X ∈ R^(60 × 16)

predict:

P(Y_h = 1 | X)

for horizons:

h ∈ {1, 2, ..., 20}

where:

Y_h = 1 if log(P_(t+h) / P_t) > 0

Input and Output

Input
A rolling 60-day window of 16 engineered financial features:

X_t = [x_(t−60), ..., x_(t−1)]

Output
A 20-dimensional vector:

ŷ = [p_1, p_2, ..., p_20]

where each p_h represents the probability of a positive return at horizon h.

Feature Engineering

Features are derived strictly from past data to avoid leakage.

Includes:

Price and volume: close, high, low, open, volume
Returns: daily returns
Trend: rolling mean returns (5, 10, 20)
Volatility: rolling standard deviation (5, 10, 20)
Momentum: 10-day momentum
Volume transformations: log volume
Technical spreads: moving average differences

All features are aligned temporally and cleaned for invalid values before model input.

Model Architecture
Transformer Encoder

The model uses an encoder-only Transformer architecture.

Each encoder block consists of:

Multi-head self-attention
Residual connection
Layer normalization
Feed-forward neural network
Residual connection
Layer normalization

Self-attention is defined as:

Attention(Q, K, V) = softmax((QKᵀ) / √d) V

This allows the model to dynamically weight past timesteps based on relevance.

Positional Encoding

To preserve temporal order, sinusoidal positional encoding is added:

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Output Layer

The final representation is aggregated via global average pooling and passed through a dense layer:

ŷ = Dense(20)

Each output corresponds to a future horizon prediction.

Training Objective

The model is trained using binary cross-entropy across all horizons:

L = − Σ_h [ y_h log(p_h) + (1 − y_h) log(1 − p_h) ]

Training uses:

Adam optimizer
Learning rate = 1e-4
Early stopping based on validation loss

Data Pipeline

Data is split chronologically:

70% training
15% validation
15% testing

Feature scaling is applied using StandardScaler fitted only on training data.

No future data is used in feature computation or scaling.

Evaluation Metrics
AUC

Measures ranking ability:

AUC = P(model ranks positive sample higher than negative)

Observed:

Horizon 20 AUC ≈ 0.62

Sharpe Ratio

Measures risk-adjusted return:

Sharpe = E[R] / σ(R)

Backtesting Framework

A vectorized long–short strategy is used.

Signals:

Long positions: top quantile predictions
Short positions: bottom quantile predictions

Returns:

R_LS = R_long − R_short

To avoid bias, returns are evaluated using non-overlapping samples:

r_t, r_(t+20), r_(t+40), ...

Results

Horizon 20 AUC ≈ 0.62
Long-only Sharpe ≈ 0.33
Filtered Sharpe ≈ 0.39
Long–Short Sharpe ≈ 0.6 (non-overlapping)

Interpretation

The model captures medium-term temporal structure in financial data.

Attention weights show higher importance for recent timesteps while retaining some sensitivity to earlier patterns.

The economic evaluation confirms that the model’s predictions are not purely driven by market drift.

Limitations

Limited dataset size
Single asset modeling
No transaction cost modeling
Potential regime sensitivity

Future Improvements

Add volatility regime features
Introduce multi-task learning for return magnitude
Replace pooling with attention-based aggregation
Expand to multi-asset setting
Apply walk-forward validation

Tech Stack

TensorFlow
NumPy
Pandas
scikit-learn
SciPy

Conclusion

This project demonstrates that Transformer architectures can extract statistically and economically meaningful signals from financial time series when combined with proper data handling and evaluation methodology.
