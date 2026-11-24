"""
Barras Impulsadas por Información (Information-Driven Bars — AFML Cap. 2, v3.0).

Diseñado específicamente para brokers OTC / MT5 donde `real_volume == 0` en Forex/CFDs:
utiliza flujos de ticks (`mt5.copy_ticks_range` o DataFrame con `time`, `price`/`bid`)
para construir:
1. Tick Bars (cada M ticks) — neutraliza la asimetría de liquidez intradía (ej. Asia vs. NY).
2. Tick Imbalance Bars (TIBs, AFML §2.3.2.1) — cierra barra cuando el desequilibrio
   acumulado de ticks firmados |sum(b_t)| excede E_0[T] * |2 P[b_t=1] - 1|.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis


def _extract_tick_arrays(ticks_df: pd.DataFrame, price_col: str = "price") -> tuple[np.ndarray, np.ndarray]:
    if price_col not in ticks_df.columns:
        if "bid" in ticks_df.columns:
            price_col = "bid"
        elif "last" in ticks_df.columns:
            price_col = "last"
        else:
            raise ValueError(f"Falta columna de precio ('{price_col}', 'bid' o 'last') en ticks_df.")

    prices = ticks_df[price_col].astype(float).values
    if "time" in ticks_df.columns:
        times = pd.to_datetime(ticks_df["time"]).values
    else:
        times = ticks_df.index.values
    return times, prices


def _tick_rule_signs(prices: np.ndarray) -> np.ndarray:
    """Aplica la Tick Rule (AFML §2.3.2.1): b_t = sign(Δp_t), manteniendo b_{t-1} si Δp_t == 0."""
    n = len(prices)
    signs = np.ones(n, dtype=np.int8)
    prev_b = 1
    for t in range(1, n):
        diff = prices[t] - prices[t - 1]
        if diff > 0.0:
            prev_b = 1
        elif diff < 0.0:
            prev_b = -1
        signs[t] = prev_b
    return signs


def build_tick_bars(
    ticks_df: pd.DataFrame,
    ticks_per_bar: int = 500,
    price_col: str = "price",
) -> pd.DataFrame:
    """Agrega un flujo de ticks en barras OHLC cada `ticks_per_bar` llegadas de información."""
    if ticks_per_bar < 2:
        raise ValueError("ticks_per_bar debe ser >= 2.")

    times, prices = _extract_tick_arrays(ticks_df, price_col=price_col)
    n = len(prices)
    if n < ticks_per_bar:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])

    n_bars = n // ticks_per_bar
    rows = []
    for k in range(n_bars):
        start = k * ticks_per_bar
        end = start + ticks_per_bar
        chunk_p = prices[start:end]
        rows.append({
            "time": times[end - 1],
            "open": float(chunk_p[0]),
            "high": float(np.max(chunk_p)),
            "low": float(np.min(chunk_p)),
            "close": float(chunk_p[-1]),
            "tick_volume": int(ticks_per_bar),
        })
    return pd.DataFrame(rows)


def build_tick_imbalance_bars(
    ticks_df: pd.DataFrame,
    expected_init_t: int = 200,
    ewma_span: int = 20,
    min_ticks: int = 20,
    max_ticks: int = 2000,
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Construye Tick Imbalance Bars (TIBs — Marcos López de Prado, AFML §2.3.2.1).

    Muestrea barras con mayor frecuencia cuando aparece flujo informado unidireccional
    y comprime periodos de ruido balanceado.
    """
    times, prices = _extract_tick_arrays(ticks_df, price_col=price_col)
    n = len(prices)
    if n < min_ticks:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume", "imbalance"])

    b = _tick_rule_signs(prices)
    alpha = 2.0 / (ewma_span + 1.0)

    ewma_t = float(expected_init_t)
    ewma_b_pos_prob = 0.55  # Inicialización no degenerada (2*p - 1 = 0.10)

    rows = []
    bar_start = 0
    theta = 0.0
    pos_ticks = 0

    for i in range(n):
        theta += float(b[i])
        if b[i] > 0:
            pos_ticks += 1
        bar_len = i - bar_start + 1

        exp_imbalance = max(0.05, abs(2.0 * ewma_b_pos_prob - 1.0))
        threshold = ewma_t * exp_imbalance

        should_close = (bar_len >= min_ticks and abs(theta) >= threshold) or (bar_len >= max_ticks)
        if should_close:
            chunk_p = prices[bar_start : i + 1]
            rows.append({
                "time": times[i],
                "open": float(chunk_p[0]),
                "high": float(np.max(chunk_p)),
                "low": float(np.min(chunk_p)),
                "close": float(chunk_p[-1]),
                "tick_volume": int(bar_len),
                "imbalance": float(theta),
            })

            # Actualización EWMA estrictamente causal al cerrar la barra
            ewma_t = (1.0 - alpha) * ewma_t + alpha * float(bar_len)
            bar_pos_prob = pos_ticks / float(bar_len)
            ewma_b_pos_prob = (1.0 - alpha) * ewma_b_pos_prob + alpha * bar_pos_prob

            bar_start = i + 1
            theta = 0.0
            pos_ticks = 0

    return pd.DataFrame(rows)


def compare_normality_vs_time_bars(
    time_bar_returns: np.ndarray,
    info_bar_returns: np.ndarray,
) -> Dict[str, float]:
    """
    Compara propiedades estadísticas (curtosis excesiva y estadístico Jarque-Bera)
    entre retornos de Time Bars y retornos de Information Bars (AFML Cap. 2).
    """
    r_time = np.asarray(time_bar_returns, dtype=np.float64)
    r_info = np.asarray(info_bar_returns, dtype=np.float64)
    r_time = r_time[np.isfinite(r_time)]
    r_info = r_info[np.isfinite(r_info)]

    kurt_time = float(kurtosis(r_time, fisher=True)) if len(r_time) > 5 else 0.0
    kurt_info = float(kurtosis(r_info, fisher=True)) if len(r_info) > 5 else 0.0
    jb_time = float(jarque_bera(r_time).statistic) if len(r_time) > 5 else 0.0
    jb_info = float(jarque_bera(r_info).statistic) if len(r_info) > 5 else 0.0

    return {
        "time_bar_excess_kurtosis": kurt_time,
        "info_bar_excess_kurtosis": kurt_info,
        "time_bar_jb_stat": jb_time,
        "info_bar_jb_stat": jb_info,
        "kurtosis_reduction_pct": float((kurt_time - kurt_info) / max(abs(kurt_time), 1e-9)),
    }
