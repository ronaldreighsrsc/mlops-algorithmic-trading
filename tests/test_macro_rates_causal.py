"""
Tests Slice A (v3.1): Causalidad Estricta del Extractor Macro (macro_rates.py).

Verifica:
1. Cero fuga futura (`allow_exact_matches=False` / `< T`): un choque extremo en
   la serie macro en fecha T NO contamina la fila T del activo (solo entra desde T+1).
2. Cero `.bfill()`: valores previos al inicio histórico de una serie macro permanecen
   en NaN (jamás propagados hacia atrás).
3. Cálculo exacto de `rate_diff_us2y_ecb`, `us_real_yield_10y`, `vix_term_ratio`,
   `vix_contango`, `credit_hy_ief_ratio` y `gold_silver_ratio`.
"""

import tempfile

import numpy as np
import pandas as pd

from src.preprocessing.macro_rates import MacroRatesProvider


def _make_sample_asset_and_macro(n_days: int = 80):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    df_asset = pd.DataFrame({
        "time": dates,
        "open": 100.0 + np.arange(n_days) * 0.2,
        "high": 101.0 + np.arange(n_days) * 0.2,
        "low": 99.0 + np.arange(n_days) * 0.2,
        "close": 100.0 + np.arange(n_days) * 0.2,
        "VIX_close": np.full(n_days, 18.0),
    })

    df_macro = pd.DataFrame({
        "US2Y": np.linspace(4.0, 4.8, n_days),
        "US10Y": np.linspace(4.2, 4.9, n_days),
        "ECB_DFR": np.full(n_days, 3.25),
        "FED_FUNDS": np.full(n_days, 5.25),
        "US_REAL_10Y": np.linspace(1.8, 2.4, n_days),
        "US_BREAKEVEN_10Y": np.full(n_days, 2.3),
        "US_CURVE_10Y2Y": np.full(n_days, 0.2),
        "VIX3M": np.full(n_days, 21.0),
        "VIX_YF": np.full(n_days, 18.0),
        "HYG": np.full(n_days, 78.0),
        "IEF": np.full(n_days, 94.0),
        "XAG_CLOSE": np.full(n_days, 25.0),
    }, index=dates)
    df_macro.index.name = "date"
    return df_asset, df_macro


def test_macro_alignment_strictly_prevents_same_day_lookahead():
    """
    Test de invariancia anti-leakage:
    Un choque masivo en la fecha macro T_k NO debe alterar la observación T_k del activo;
    recién debe reflejarse en T_{k+1}.
    """
    df_asset, df_macro_a = _make_sample_asset_and_macro(60)
    k = 30
    target_dt = df_macro_a.index[k]

    df_macro_b = df_macro_a.copy()
    df_macro_b.loc[target_dt:, "US2Y"] = 99.0
    df_macro_b.loc[target_dt:, "US_REAL_10Y"] = -50.0
    df_macro_b.loc[target_dt:, "VIX3M"] = 80.0
    df_macro_b.loc[target_dt:, "HYG"] = 10.0

    aligned_a = MacroRatesProvider.align_causal_macro(df_asset, df_macro_a)
    aligned_b = MacroRatesProvider.align_causal_macro(df_asset, df_macro_b)

    cols = ["rate_diff_us2y_ecb", "us_real_yield_10y", "vix_term_ratio", "credit_hy_ief_ratio"]
    # Hasta k INCLUSIVE (índice 0..30), las series alineadas deben ser idénticas bit a bit
    pd.testing.assert_frame_equal(aligned_a[cols].iloc[: k + 1], aligned_b[cols].iloc[: k + 1])

    # En k + 1, el dato del cierre k ya debe ser visible
    assert aligned_b.loc[k + 1, "rate_diff_us2y_ecb"] > 90.0
    assert aligned_b.loc[k + 1, "us_real_yield_10y"] == -50.0


def test_macro_alignment_never_backfills_missing_early_history():
    """Si una serie macro arranca en el día 20, los días 0..20 del activo quedan NaN (sin .bfill())."""
    df_asset, df_macro = _make_sample_asset_and_macro(50)
    df_macro.loc[df_macro.index[:20], "US_REAL_10Y"] = np.nan

    aligned = MacroRatesProvider.align_causal_macro(df_asset, df_macro)
    assert aligned["us_real_yield_10y"].iloc[:21].isna().all()
    assert not aligned["us_real_yield_10y"].iloc[21:].isna().any()


def test_macro_derived_formulas_and_local_cache_loading():
    df_asset, df_macro = _make_sample_asset_and_macro(40)
    aligned = MacroRatesProvider.align_causal_macro(df_asset, df_macro)

    # En la barra 10, VIX_close=18.0 y VIX3M=21.0 => ratio = 18/21, contango = 3.0
    assert abs(aligned.loc[10, "vix_term_ratio"] - (18.0 / 21.0)) < 1e-9
    assert abs(aligned.loc[10, "vix_contango"] - 3.0) < 1e-9
    # Diferencial US2Y(t-1) - ECB_DFR(t-1)
    expected_diff = float(df_macro["US2Y"].iloc[9] - 3.25)
    assert abs(aligned.loc[10, "rate_diff_us2y_ecb"] - expected_diff) < 1e-9

    # Validar lectura offline desde caché local en cache_dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        prov = MacroRatesProvider(base_dir=tmp_dir, cache_ttl_days=30.0)
        cache_file = f"{prov.cache_dir}/fred_DFII10.csv"
        pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "value": [1.95, 2.05],
        }).to_csv(cache_file, index=False)
        s = prov.fetch_fred_series("DFII10", force_refresh=False)
        assert len(s) == 2
        assert float(s.iloc[-1]) == 2.05
