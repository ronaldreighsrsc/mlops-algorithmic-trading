"""
Tests Fase 3 (v3.0): Filtro CUSUM Simétrico, Estrategias Primarias Económicas,
Meta-Labeling Verdadero de 2 Etapas (simetría Long/Short) y Pesos de Unicidad AFML Cap. 4.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cusum_filter import SymmetricCUSUMFilter
from src.preprocessing.meta_labeling import MetaBarrierLabeler
from src.preprocessing.sample_weights import (
    compute_average_uniqueness,
    compute_concurrency,
    compute_sample_weights,
)
from src.strategies.primary_strategies import (
    DonchianBreakoutPrimary,
    TrendPullbackPrimary,
    VolatilityMeanReversionPrimary,
    get_primary_strategy,
)


def _make_ohlcv(prices: np.ndarray, vol_pct: float = 1.0) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame({
        "time": pd.bdate_range("2020-01-01", periods=n),
        "open": prices,
        "high": prices * 1.003,
        "low": prices * 0.997,
        "close": prices,
        "EGARCH_Vol": np.full(n, vol_pct, dtype=float),
    })


def test_cusum_filter_triggers_on_moves_and_ignores_flat_noise():
    # Mercado casi plano vs. mercado con saltos claros
    flat_prices = np.full(100, 100.0) + np.sin(np.linspace(0, 6, 100)) * 0.02
    df_flat = _make_ohlcv(flat_prices, vol_pct=1.0)

    moving_prices = 100.0 * np.exp(np.cumsum(np.full(100, 0.012)))
    df_move = _make_ohlcv(moving_prices, vol_pct=1.0)

    cusum = SymmetricCUSUMFilter(kappa=1.0, vol_col="EGARCH_Vol")
    res_flat = cusum.filter_events(df_flat)
    res_move = cusum.filter_events(df_move)

    assert len(res_flat.event_positions) == 0
    assert len(res_move.event_positions) >= 30
    assert np.all(res_move.event_directions == 1)


def test_cusum_filter_is_strictly_causal():
    """Alterar precios futuros t > 120 jamás debe cambiar ningún evento detectado en t <= 120."""
    rng = np.random.default_rng(42)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, 200)))
    df_a = _make_ohlcv(prices.copy(), vol_pct=1.0)

    prices_b = prices.copy()
    prices_b[121:] *= 3.0  # Choque masivo en el futuro
    df_b = _make_ohlcv(prices_b, vol_pct=1.0)

    cusum = SymmetricCUSUMFilter(kappa=1.0, vol_col="EGARCH_Vol")
    ev_a = cusum.filter_events(df_a).event_positions
    ev_b = cusum.filter_events(df_b).event_positions

    assert np.array_equal(ev_a[ev_a <= 120], ev_b[ev_b <= 120])


def test_primary_strategies_are_causal_and_respect_bilateral_flag():
    n = 300
    t = np.arange(n)
    # Tendencia alcista seguida de tendencia bajista
    prices = np.where(t < 160, 100.0 + 0.5 * t, 180.0 - 0.6 * (t - 160))
    df = _make_ohlcv(prices, vol_pct=1.0)

    # TrendPullback: bilateral=False jamás emite -1
    strat_long = TrendPullbackPrimary(sma_slow=50, ema_fast=10, ema_slow=25, bilateral=False)
    sides_long = strat_long.generate_sides(df)
    assert set(sides_long.unique()).issubset({0, 1})
    assert (sides_long == 1).sum() > 20

    # bilateral=True emite tanto +1 como -1
    strat_bi = TrendPullbackPrimary(sma_slow=50, ema_fast=10, ema_slow=25, bilateral=True)
    sides_bi = strat_bi.generate_sides(df)
    assert (sides_bi == 1).sum() > 20
    assert (sides_bi == -1).sum() > 20

    # Causalidad: alterar precios futuros tras t=100 no cambia sides[:101]
    df_shock = df.copy()
    df_shock.loc[101:, ["open", "high", "low", "close"]] *= 0.2
    sides_shock = strat_bi.generate_sides(df_shock)
    assert np.array_equal(sides_bi.iloc[:101].values, sides_shock.iloc[:101].values)

    # Factory registry funciona para las 3 estrategias
    for name in ("TrendPullbackPrimary", "DonchianBreakoutPrimary", "VolatilityMeanReversionPrimary"):
        s_inst = get_primary_strategy(name, bilateral=True)
        out = s_inst.generate_sides(df)
        assert len(out) == len(df)
        assert set(out.unique()).issubset({-1, 0, 1})


def test_meta_barrier_labeler_exact_long_and_short_symmetry():
    """
    Prueba que corrige de raíz el bug F-04 de EURUSD:
    - En un rally alcista claro, side=+1 logra meta_label=1 y side=-1 logra meta_label=0.
    - En un crash bajista claro, side=-1 logra meta_label=1 y side=+1 logra meta_label=0.
    """
    n = 80
    bull_prices = 100.0 * np.exp(np.cumsum(np.full(n, 0.01)))   # +1%/día sostenido
    bear_prices = 100.0 * np.exp(np.cumsum(np.full(n, -0.01)))  # -1%/día sostenido

    df_bull = _make_ohlcv(bull_prices, vol_pct=1.0)
    df_bear = _make_ohlcv(bear_prices, vol_pct=1.0)

    labeler = MetaBarrierLabeler(k_up=2.0, k_down=1.5, max_hold=10, round_trip_cost=0.0002, require_tp_hit=True)

    # En bull: Long gana, Short pierde
    ev_bull_long = labeler.label_events(df_bull, np.ones(n, dtype=np.int8))
    ev_bull_short = labeler.label_events(df_bull, -np.ones(n, dtype=np.int8))
    assert ev_bull_long["meta_label"].mean() > 0.90
    assert ev_bull_short["meta_label"].mean() < 0.10
    assert (ev_bull_long["t1_pos"] > ev_bull_long["t0_pos"]).all()

    # En bear: Short gana, Long pierde
    ev_bear_short = labeler.label_events(df_bear, -np.ones(n, dtype=np.int8))
    ev_bear_long = labeler.label_events(df_bear, np.ones(n, dtype=np.int8))
    assert ev_bear_short["meta_label"].mean() > 0.90
    assert ev_bear_long["meta_label"].mean() < 0.10
    assert (ev_bear_short["net_ret"] > 0).mean() > 0.90


def test_meta_barrier_labeler_pessimistic_dual_hit_and_friction_gate():
    """Doble toque TP+SL en misma barra => SL_DUAL (meta_label=0); ganancia menor a fricción => meta_label=0."""
    n = 30
    df = _make_ohlcv(np.full(n, 100.0), vol_pct=1.0)
    # Día 2 toca simultáneamente TP (102) y SL (98.5)
    df.loc[2, "high"] = 110.0
    df.loc[2, "low"] = 90.0

    labeler = MetaBarrierLabeler(k_up=2.0, k_down=1.5, max_hold=10, round_trip_cost=0.0001)
    ev = labeler.label_events(df, np.ones(n, dtype=np.int8), event_positions=np.array([0]))
    assert len(ev) == 1
    assert ev.iloc[0]["barrier"] == "SL_DUAL"
    assert ev.iloc[0]["meta_label"] == 0

    # Fricción enorme que devora el TP => meta_label debe ser 0
    df_tp = _make_ohlcv(np.full(n, 100.0), vol_pct=1.0)
    df_tp.loc[2, "high"] = 103.0
    labeler_expensive = MetaBarrierLabeler(k_up=2.0, k_down=1.5, max_hold=10, round_trip_cost=0.05)
    ev_exp = labeler_expensive.label_events(df_tp, np.ones(n, dtype=np.int8), event_positions=np.array([0]))
    assert ev_exp.iloc[0]["barrier"] == "TP"
    assert ev_exp.iloc[0]["net_ret"] < 0
    assert ev_exp.iloc[0]["meta_label"] == 0


def test_sample_weights_isolated_vs_overlapping_and_time_decay():
    """Verifica concurrencia c_t, unicidad u_i, decaimiento temporal y normalización sum(w)=N."""
    # Caso 1: 3 eventos completamente aislados -> u_i = 1.0 cada uno
    t0_iso = np.array([0, 10, 20])
    t1_iso = np.array([4, 14, 24])
    u_iso = compute_average_uniqueness(t0_iso, t1_iso, n_bars=30)
    assert np.allclose(u_iso, 1.0)

    # Caso 2: 4 eventos idénticos completamente solapados -> c_t = 4, u_i = 0.25
    t0_ov = np.array([5, 5, 5, 5])
    t1_ov = np.array([10, 10, 10, 10])
    c_ov = compute_concurrency(t0_ov, t1_ov, n_bars=20)
    assert c_ov[7] == 4.0
    u_ov = compute_average_uniqueness(t0_ov, t1_ov, n_bars=20)
    assert np.allclose(u_ov, 0.25)

    # Caso 3: Time-decay sobre eventos aislados -> el evento más reciente pesa más y sum(w) == N
    w, avg_u = compute_sample_weights(
        t0_iso, t1_iso, n_bars=30, close_prices=None, decay_c=0.5, use_return_attribution=False
    )
    assert abs(np.sum(w) - len(t0_iso)) < 1e-10
    assert w[2] > w[1] > w[0]
    assert np.allclose(avg_u, 1.0)
