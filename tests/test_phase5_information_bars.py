"""
Tests Fase 5 (v3.0): Information-Driven Bars (Tick Bars & Tick Imbalance Bars — AFML Cap. 2).

Simula un mercado intradía con una fuerte 'sonrisa de volumen' (apertura/cierre 25x más
activos que las horas muertas del mediodía, exactamente el fenómeno documentado en AFML Cap. 2)
y demuestra estadísticamente que las Tick Bars y TIBs recuperan retornos mucho más cercanos
a la normalidad (menor curtosis excesiva) que las Time Bars cronológicas.
"""

import numpy as np
import pandas as pd

from src.preprocessing.information_bars import (
    build_tick_bars,
    build_tick_imbalance_bars,
    compare_normality_vs_time_bars,
)


def _simulate_intraday_u_shape_ticks(n_minutes: int = 390, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera ticks con alta actividad en apertura/cierre y calma en el mediodía."""
    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp("2026-11-09 09:30:00")

    tick_times = []
    tick_prices = []
    time_bar_closes = []

    price = 100.0
    for m in range(n_minutes):
        # Sonrisa de volumen: primeros 30 min y últimos 30 min tienen 25x más ticks que el medio
        is_busy = (m < 30) or (m >= n_minutes - 30)
        n_ticks = 150 if is_busy else 6

        t_min = base_time + pd.Timedelta(minutes=m)
        offsets = np.linspace(0, 59, n_ticks)
        # Cada tick aporta varianza constante (subordinación de Clark 1973 -> reloj de actividad)
        deltas = rng.normal(0.0, 0.0004, size=n_ticks)
        for off, d in zip(offsets, deltas):
            price *= float(np.exp(d))
            tick_times.append(t_min + pd.Timedelta(seconds=float(off)))
            tick_prices.append(price)

        time_bar_closes.append(price)

    ticks_df = pd.DataFrame({"time": tick_times, "price": tick_prices})
    time_bars_df = pd.DataFrame({
        "time": [base_time + pd.Timedelta(minutes=m) for m in range(n_minutes)],
        "close": time_bar_closes,
    })
    return ticks_df, time_bars_df


def test_tick_bars_valid_ohlc_and_lower_kurtosis_than_time_bars():
    ticks_df, time_bars_df = _simulate_intraday_u_shape_ticks(n_minutes=390, seed=42)

    tick_bars = build_tick_bars(ticks_df, ticks_per_bar=60)
    assert len(tick_bars) > 100
    assert (tick_bars["high"] >= tick_bars[["open", "close"]].max(axis=1)).all()
    assert (tick_bars["low"] <= tick_bars[["open", "close"]].min(axis=1)).all()

    r_time = np.diff(np.log(time_bars_df["close"].values))
    r_tick = np.diff(np.log(tick_bars["close"].values))

    comp = compare_normality_vs_time_bars(r_time, r_tick)
    # Bajo subordinación de volumen U-shaped, Time Bars tienen colas fuertemente leptocúrticas
    # mientras Tick Bars recuperan normalidad cercana a curtosis excesiva ~ 0
    assert comp["time_bar_excess_kurtosis"] > 3.0
    assert comp["info_bar_excess_kurtosis"] < comp["time_bar_excess_kurtosis"] * 0.5


def test_tick_imbalance_bars_sample_more_during_informed_flow():
    """TIBs deben cerrar barras más rápido durante flujo informado unidireccional."""
    rng = np.random.default_rng(7)
    n_quiet = 600
    n_informed = 600

    # Fase 1: ruido balanceado (50/50 arriba/abajo)
    p1 = 100.0 + np.cumsum(rng.choice([-0.01, 0.01], size=n_quiet))
    # Fase 2: flujo informado alcista (85% arriba)
    p2 = p1[-1] + np.cumsum(rng.choice([-0.01, 0.01], size=n_informed, p=[0.15, 0.85]))
    prices = np.concatenate([p1, p2])
    times = pd.date_range("2026-11-09 09:30:00", periods=len(prices), freq="1s")
    ticks_df = pd.DataFrame({"time": times, "price": prices})

    tibs = build_tick_imbalance_bars(ticks_df, expected_init_t=100, ewma_span=10, min_ticks=15, max_ticks=300)
    assert len(tibs) >= 5
    assert (tibs["high"] >= tibs[["open", "close"]].max(axis=1)).all()
    assert (tibs["low"] <= tibs[["open", "close"]].min(axis=1)).all()

    cutoff_time = times[n_quiet]
    bars_quiet = tibs[tibs["time"] < cutoff_time]
    bars_informed = tibs[tibs["time"] >= cutoff_time]
    assert len(bars_informed) > len(bars_quiet), (
        f"TIBs deberían muestrear más barras en flujo informado ({len(bars_informed)}) que en ruido ({len(bars_quiet)})"
    )
