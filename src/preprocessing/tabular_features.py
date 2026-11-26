"""
Ingeniería de Características Tabulares Multi-Escala (v3.0 — Sin Aplanado de 480 Columnas).

Reemplaza el antiguo aplanado bruto `window[t-60:t].flatten()` (hasta 480 columnas
altamente colineales para árboles) por 15-22 features tabulares causales de alta
relación señal/ruido alineadas al evento t0 y a la dirección `primary_side`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def build_meta_feature_matrix(
    df: pd.DataFrame,
    primary_sides: pd.Series | np.ndarray,
    bars_since_cusum: Optional[pd.Series | np.ndarray] = None,
    hmm_choppy_prob: Optional[pd.Series | np.ndarray] = None,
) -> pd.DataFrame:
    """
    Construye una matriz tabular 100% causal (sin .bfill()) de features de contexto
    para el Meta-Modelo (Capa 2). Toda estadística usa únicamente datos <= t.
    """
    if "close" not in df.columns:
        raise ValueError("Falta columna 'close' para construir features tabulares v3.0.")

    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    sides = pd.Series(np.asarray(primary_sides, dtype=np.float64), index=df.index)

    feats = pd.DataFrame(index=df.index)

    # 1. Estructura y distancia a medias normalizada por ATR
    atr = df["ATR"].astype(float) if "ATR" in df.columns else (high - low).rolling(14, min_periods=5).mean()
    atr_safe = atr.replace(0.0, np.nan).ffill().fillna(close * 0.01)

    sma50 = close.rolling(50, min_periods=20).mean()
    sma200 = close.rolling(200, min_periods=50).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    feats["dist_sma50_atr"] = (close - sma50) / atr_safe
    feats["dist_sma200_atr"] = (close - sma200) / atr_safe
    feats["ema_spread_atr"] = (ema20 - ema50) / atr_safe

    # 2. Volatilidad condicional y compresión de rangos
    log_ret_1 = np.log(close / close.shift(1))
    vol_20 = log_ret_1.rolling(20, min_periods=10).std().replace(0.0, np.nan)

    if "EGARCH_Vol" in df.columns:
        feats["EGARCH_Vol"] = df["EGARCH_Vol"].astype(float)
    else:
        feats["EGARCH_Vol"] = vol_20 * 100.0

    atr_60 = atr_safe.rolling(60, min_periods=20).mean().replace(0.0, np.nan)
    feats["atr_ratio_14_60"] = atr_safe / atr_60
    feats["bar_range_atr"] = (high - low) / atr_safe

    # 3. Momentum multi-escala normalizado por volatilidad
    for win in (5, 20, 60):
        ret_w = np.log(close / close.shift(win))
        feats[f"mom_{win}d_vol"] = ret_w / (vol_20 * np.sqrt(win) + 1e-12)

    if "RSI" in df.columns:
        feats["RSI"] = df["RSI"].astype(float)
    if "MACD_Hist" in df.columns:
        feats["MACD_Hist"] = df["MACD_Hist"].astype(float)
        feats["macd_hist_slope_3d"] = feats["MACD_Hist"].diff(3)

    # 4. Memoria fraccional (FFD congelado) y exógenas macro si están disponibles
    for col in ("close_FFD", "open_FFD", "DXY_close_FFD", "Yield10Y_FFD"):
        if col in df.columns:
            feats[col] = df[col].astype(float)

    if "VIX_close" in df.columns:
        vix = df["VIX_close"].astype(float)
        feats["VIX_close"] = vix
        feats["vix_chg_5d"] = vix.pct_change(5)
        feats["vix_chg_20d"] = vix.pct_change(20)

    if "tick_volume" in df.columns:
        tv = df["tick_volume"].astype(float)
        tv_mean = tv.rolling(20, min_periods=10).mean()
        tv_std = tv.rolling(20, min_periods=10).std().replace(0.0, np.nan)
        feats["tick_vol_z20"] = (tv - tv_mean) / tv_std

    # 4b. Diferenciales Macro v3.1, Estructura VIX, Crédito, Ratio Oro/Plata y Microestructura de Ticks
    v31_macro_and_micro_cols = (
        "rate_diff_us2y_ecb",
        "delta_rate_diff_5d",
        "delta_rate_diff_20d",
        "rate_diff_fed_ecb",
        "yield_curve_10y_2y",
        "us_real_yield_10y",
        "delta_real_yield_5d",
        "delta_real_yield_20d",
        "breakeven_infl_10y",
        "delta_breakeven_20d",
        "vix3m_close",
        "vix_term_ratio",
        "vix_contango",
        "credit_hy_ief_ratio",
        "delta_credit_ratio_20d",
        "gold_silver_ratio",
        "delta_gsr_20d",
        "tick_imbalance_norm",
        "vpin_proxy",
        "spread_zscore",
        "tick_arrival_rate",
    )
    for col in v31_macro_and_micro_cols:
        if col in df.columns:
            feats[col] = df[col].astype(float)

    # Flag de calendario determinista Turn-of-the-Month (últimos 2 días o primeros 3 días hábiles del mes)
    if "time" in df.columns:
        ts = pd.to_datetime(df["time"])
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(df.index)
    else:
        ts = None

    if ts is not None:
        ts_ser = pd.Series(ts, index=df.index)
        days_left = ts_ser.dt.days_in_month - ts_ser.dt.day
        feats["is_turn_of_month"] = np.where((ts_ser.dt.day <= 4) | (days_left <= 1), 1.0, 0.0)

    # 5. Contexto direccional del Modelo Primario (alineación señal/momentum)
    feats["primary_side"] = sides
    feats["side_aligned_mom_20d"] = sides * feats["mom_20d_vol"]
    feats["side_aligned_dist_sma200"] = sides * feats["dist_sma200_atr"]

    if "delta_rate_diff_20d" in feats.columns:
        feats["side_aligned_rate_diff_20d"] = sides * (-feats["delta_rate_diff_20d"])
    if "vix_contango" in feats.columns:
        feats["side_aligned_vix_contango"] = sides * feats["vix_contango"]

    if bars_since_cusum is not None:
        feats["bars_since_cusum"] = np.asarray(bars_since_cusum, dtype=np.float64)
    if hmm_choppy_prob is not None:
        feats["hmm_prob_choppy"] = np.asarray(hmm_choppy_prob, dtype=np.float64)

    # Limpieza estrictamente causal: reemplazar infinitos por NaN y propagar SOLO hacia adelante
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill()
    return feats
