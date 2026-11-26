"""
Tests Slice C (v3.1): Nuevas Estrategias Primarias Económicas y Features v3.1.

Verifica:
1. `MacroCarryMomentumPrimary` (fx_rate_diff y real_yield) emite lados económicamente
   coherentes y es 100% causal ante choques futuros oráculo.
2. `VixRegimeEquityPrimary` bloquea compras en backwardation (VIX/VIX3M > 1) o estrés
   de crédito y aprueba en contango alcista.
3. `build_meta_feature_matrix` propaga las nuevas señales macro, microestructura y
   `is_turn_of_month` sin `.bfill()` ni fuga temporal.
"""

import numpy as np
import pandas as pd

from src.preprocessing.tabular_features import build_meta_feature_matrix
from src.strategies.primary_strategies import (
    MacroCarryMomentumPrimary,
    VixRegimeEquityPrimary,
    get_primary_strategy,
)


def _make_v31_df(n: int = 200) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    t = np.arange(n)
    close = np.where(t < 100, 100.0 + 0.3 * t, 130.0 - 0.3 * (t - 100))
    return pd.DataFrame({
        "time": dates,
        "open": close,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "ATR": close * 0.01,
        "EGARCH_Vol": np.full(n, 1.0),
        "VIX_close": np.where(t < 100, 15.0, 28.0),
        "delta_rate_diff_20d": np.where(t < 100, -0.25, 0.30),
        "delta_real_yield_20d": np.where(t < 100, -0.15, 0.25),
        "vix_term_ratio": np.where(t < 100, 0.85, 1.12),
        "vix_contango": np.where(t < 100, 3.0, -3.0),
        "credit_hy_ief_ratio": np.full(n, 0.82),
        "delta_credit_ratio_20d": np.where(t < 100, 0.01, -0.05),
        "gold_silver_ratio": np.full(n, 82.0),
        "vpin_proxy": np.full(n, 0.35),
    })


def test_macro_carry_momentum_primary_signals_and_causality():
    df = _make_v31_df(200)

    fx_strat = get_primary_strategy("MacroCarryMomentumPrimary", mode="fx_rate_diff", ema_window=20, bilateral=True)
    sides_fx = fx_strat.generate_sides(df)
    assert (sides_fx.iloc[40:95] == 1).all()
    assert (sides_fx.iloc[120:] == -1).all()

    gold_strat = MacroCarryMomentumPrimary(mode="real_yield", sma_window=30, bilateral=False)
    sides_gold = gold_strat.generate_sides(df)
    assert set(sides_gold.unique()).issubset({0, 1})
    assert (sides_gold.iloc[40:95] == 1).all()
    assert (sides_gold.iloc[130:] == 0).all()

    # Oracle future shock tras t=70 no altera <= 70
    df_shock = df.copy()
    df_shock.loc[71:, ["close", "delta_rate_diff_20d"]] *= -5.0
    sides_shock = fx_strat.generate_sides(df_shock)
    assert np.array_equal(sides_fx.iloc[:71].values, sides_shock.iloc[:71].values)


def test_vix_regime_equity_primary_blocks_backwardation_and_credit_stress():
    df = _make_v31_df(200)
    eq_strat = VixRegimeEquityPrimary(vix_ratio_max=0.98, credit_delta_min=-0.03, sma_window=40, bilateral=False)
    sides = eq_strat.generate_sides(df)

    # En régimen alcista contango (t=50..95) aprueba Long (+1)
    assert (sides.iloc[50:95] == 1).all()
    # En régimen backwardation + estrés de crédito (t=105..199) se mantiene fuera (0)
    assert (sides.iloc[105:] == 0).all()


def test_tabular_features_v31_propagates_macro_micro_and_is_causal():
    df = _make_v31_df(160)
    sides = VixRegimeEquityPrimary(sma_window=40).generate_sides(df)
    feats_a = build_meta_feature_matrix(df, primary_sides=sides)

    for expected_col in (
        "delta_rate_diff_20d",
        "vix_term_ratio",
        "delta_credit_ratio_20d",
        "gold_silver_ratio",
        "vpin_proxy",
        "is_turn_of_month",
        "side_aligned_rate_diff_20d",
        "side_aligned_vix_contango",
    ):
        assert expected_col in feats_a.columns

    df_b = df.copy()
    df_b.loc[91:, ["close", "vix_term_ratio", "delta_rate_diff_20d", "vpin_proxy"]] = 999.0
    sides_b = VixRegimeEquityPrimary(sma_window=40).generate_sides(df_b)
    feats_b = build_meta_feature_matrix(df_b, primary_sides=sides_b)

    pd.testing.assert_frame_equal(feats_a.iloc[:91], feats_b.iloc[:91])
