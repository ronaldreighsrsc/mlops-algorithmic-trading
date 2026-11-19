---
name: timeseries-leakage-guard
description: Audits and enforces strict temporal validation in time series, demand forecasting, and quantitative trading models. Prevents lookahead bias, data leakage in rolling/expanding windows, target leakage in lag features, and improper temporal train-test splits.
---

# Time Series & Quant Leakage Guard

Enforce rigorous temporal integrity and prevent lookahead bias (data leakage) in time series forecasting, demand planning, and algorithmic trading models.

---

## When This Skill Activates

This skill activates in three scenarios:
1. **During Feature Engineering:** Whenever code computes lags (`shift`), rolling aggregations (`rolling`), exponential moving averages, or normalizations (`StandardScaler`, `MinMaxScaler`).
2. **During Model Validation & Splits:** Whenever code creates train/validation/test partitions, cross-validation folds, or backtesting loops.
3. **Upon User Request:** When the user asks: *"¿Hay data leakage?"*, *"Audita los lags / splits"*, *"Valida el backtest"*, o *"Chequea que no mire al futuro"*.

---

## The 4 Cardinal Rules of Temporal Integrity

### 1. The Preprocessing Boundary (Scaler & Encoder Leakage)
- **Violation:** Calling `fit()` or `fit_transform()` on the entire dataset before splitting, or calculating global means/medians to impute missing values.
- **Rule:** Every scaler, imputer, and encoder MUST be fitted **exclusively on the training split**:
  ```python
  # CORRECT:
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_val_scaled = scaler.transform(X_val)
  X_test_scaled = scaler.transform(X_test)
  ```

### 2. Lag & Target Leakage (The shift(1) Law)
- **Violation:** Computing a rolling mean of the target variable without shifting it first:
  ```python
  # FATAL LEAKAGE: includes target y[t] in the feature for predicting y[t]
  df['target_roll_7'] = df['sales'].rolling(7).mean()
  ```
- **Rule:** When creating features from the target variable or contemporaneous signals, shift backwards by at least 1 step (or by the forecasting horizon $H$):
  ```python
  # CORRECT:
  df['target_roll_7'] = df['sales'].shift(1).rolling(7).mean()
  ```

### 3. Chronological Splits (Never Shuffle Time)
- **Violation:** Using random splitting (e.g., `train_test_split(..., shuffle=True)` or standard `KFold`).
- **Rule:** Always use strictly chronological splits or `TimeSeriesSplit` / Walk-Forward Expansion:
  ```python
  # Verify:
  assert train_df.index.max() < val_df.index.min(), "Leakage: Training dates overlap with validation dates!"
  assert val_df.index.max() < test_df.index.min(), "Leakage: Validation dates overlap with test dates!"
  ```

### 4. Quant Trading Execution Timing (Signal vs. Fill)
- **Violation:** Calculating a trading signal using the `Close` price of candle $T$, and assuming the trade is filled at the `Open` or `Close` of candle $T$ without latency.
- **Rule:** If the decision is computed on candle $T$'s Close, the execution price is at earliest candle $T+1$'s Open (plus realistic slippage and transaction fees):
  ```python
  # CORRECT (Trading backtest):
  df['signal'] = compute_alpha(df['close'])  # Available at end of bar t
  df['trade_execution_price'] = df['open'].shift(-1)  # Filled at open of bar t+1
  ```

---

## Audit Checklist & Output Format

When auditing a pipeline or script, output:
```markdown
### ⏱️ Temporal Leakage Audit Report
- **Scaler / Imputer Isolation:** [✅ Safe / ⚠️ Leakage Detected]
- **Target & Feature Lags:** [✅ Strictly Shifted / ⚠️ Contemporaneous Leak]
- **Split Chronology:** [✅ Walk-Forward / Strict TimeSplit / ⚠️ Random Shuffle]
- **Execution / Fill Realism:** [✅ Feasible Timing / ⚠️ Lookahead Fill]

👉 **Summary & Fix:** [Concise explanation of any detected leakage and exact code fix]
```
