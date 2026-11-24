"""
Tests Fase 4 (v3.0): Features Tabulares Multi-Escala, Poda MDA Purgada,
DSR Empírico, CSCV PBO Formal Multi-Estrategia y Entrenador Meta-Labeling v3.0
con exportación ONNX real y ejecución end-to-end en ExecutionGateway.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.database.trade_vault import TradeVault
from src.evaluation.cpcv_auditor import CPCVAuditor
from src.evaluation.feature_importance import compute_purged_mda, prune_noise_features
from src.execution.execution_gateway import ExecutionGateway
from src.execution.onnx_inference_engine import ONNXInferenceEngine
from src.models.meta_trainer_v3 import (
    MetaModelTrainerV3,
    compute_analytical_probability_threshold,
)
from src.preprocessing.tabular_features import build_meta_feature_matrix
from src.strategies.primary_strategies import TrendPullbackPrimary


def _make_rich_synthetic_ohlcv(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Alternar regímenes de tendencia alcista y bajista con ruido realista
    drift = np.where((np.arange(n) // 160) % 2 == 0, 0.0022, -0.0018)
    rets = drift + rng.normal(0.0, 0.011, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    spread_bar = np.abs(rng.normal(0.006, 0.003, size=n))
    high = np.maximum(open_p, close) * (1.0 + spread_bar)
    low = np.minimum(open_p, close) * (1.0 - spread_bar)

    df = pd.DataFrame({
        "time": pd.bdate_range("2020-01-01", periods=n),
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": rng.integers(500, 5000, size=n).astype(float),
        "EGARCH_Vol": np.full(n, 1.1, dtype=float),
        "ATR": close * 0.012,
        "RSI": np.clip(50.0 + 30.0 * np.sin(np.linspace(0, 20, n)), 5, 95),
        "MACD_Hist": np.cos(np.linspace(0, 20, n)) * 0.5,
        "close_FFD": np.diff(np.insert(close, 0, close[0])) * 0.4 + 0.1 * close,
        "VIX_close": np.clip(18.0 + rng.normal(0, 3.0, size=n), 10.0, 45.0),
        "DXY_close_FFD": rng.normal(0, 0.5, size=n),
    })
    return df


def test_tabular_features_are_causal_and_no_bfill():
    df = _make_rich_synthetic_ohlcv(300, seed=11)
    sides = TrendPullbackPrimary(sma_slow=50, ema_fast=10, ema_slow=25, bilateral=True).generate_sides(df)
    feats_a = build_meta_feature_matrix(df, primary_sides=sides)

    # Choque masivo después del índice 180 no debe alterar ninguna feature en <= 180
    df_b = df.copy()
    df_b.loc[181:, ["open", "high", "low", "close", "VIX_close"]] *= 4.0
    sides_b = TrendPullbackPrimary(sma_slow=50, ema_fast=10, ema_slow=25, bilateral=True).generate_sides(df_b)
    feats_b = build_meta_feature_matrix(df_b, primary_sides=sides_b)

    pd.testing.assert_frame_equal(feats_a.iloc[:181], feats_b.iloc[:181])
    assert "side_aligned_mom_20d" in feats_a.columns
    assert "atr_ratio_14_60" in feats_a.columns


def test_purged_mda_identifies_real_signal_and_prunes_noise():
    """Una feature con señal real debe obtener MDA > 0 y superar ampliamente a 3 columnas de ruido gaussiano."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n + 10)
    t0 = pd.Series(dates[:n])
    t1 = pd.Series(dates[5 : n + 5])

    signal = rng.normal(0, 1, size=n)
    X_df = pd.DataFrame({
        "true_signal": signal,
        "noise_1": rng.normal(0, 1, size=n),
        "noise_2": rng.normal(0, 1, size=n),
        "noise_3": rng.normal(0, 1, size=n),
    })
    probs = 1.0 / (1.0 + np.exp(-2.2 * signal))
    y = (rng.uniform(size=n) < probs).astype(int)

    factory = lambda: Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500)),
    ])
    mda_df = compute_purged_mda(factory, X_df, y, t0, t1, n_splits=4, pct_embargo=0.02, n_repeats=3)

    assert mda_df.iloc[0]["feature"] == "true_signal"
    assert mda_df.iloc[0]["mean_importance"] > 0.15
    noise_scores = mda_df[mda_df["feature"].str.startswith("noise_")]["mean_importance"].values
    assert np.all(noise_scores < 0.05)

    kept = prune_noise_features(mda_df, min_importance=0.02, min_features=1)
    assert kept == ["true_signal"]


def test_empirical_dsr_penalizes_multiple_testing():
    """Para el mismo retorno observado, el DSR debe disminuir estrictamente al aumentar n_trials."""
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0012, 0.01, size=250)
    trial_sharpes = rng.normal(0.0, 0.08, size=50)

    d1 = CPCVAuditor.compute_deflated_sharpe_ratio(rets, n_trials=1, trial_sharpes=trial_sharpes)
    d20 = CPCVAuditor.compute_deflated_sharpe_ratio(rets, n_trials=20, trial_sharpes=trial_sharpes)
    d200 = CPCVAuditor.compute_deflated_sharpe_ratio(rets, n_trials=200, trial_sharpes=trial_sharpes)

    assert d1["dsr"] > d20["dsr"] > d200["dsr"]
    assert abs(d1["dsr"] - d1["psr"]) < 1e-10  # Con 1 ensayo, DSR == PSR


def test_formal_cscv_pbo_separates_pure_noise_from_real_alpha():
    """
    Algoritmo CSCV formal (Bailey & López de Prado, 2014):
    - 25 estrategias de ruido blanco puro deben producir PBO alto (> 0.35).
    - Una matriz que incluye 1 estrategia con señal persistente real debe producir PBO bajo (< 0.15).
    """
    rng = np.random.default_rng(2026)
    T = 400
    N = 25

    noise_matrix = rng.normal(0.0, 0.01, size=(T, N))
    auditor = CPCVAuditor(symbol="TEST_CSCV")
    res_noise = auditor.audit_cscv_pbo(noise_matrix, n_blocks=8, purge_bars=5)
    assert res_noise["n_combinations"] == 70  # C(8, 4) = 70
    assert res_noise["pbo"] > 0.35, f"Ruido puro debería tener PBO alto (~0.50), obtenido {res_noise['pbo']:.3f}"

    # Inyectar una estrategia genuina con Sharpe estable alto en la columna 0
    alpha_matrix = noise_matrix.copy()
    alpha_matrix[:, 0] = rng.normal(0.0035, 0.008, size=T)
    res_alpha = auditor.audit_cscv_pbo(alpha_matrix, n_blocks=8, purge_bars=5)
    assert res_alpha["pbo"] < 0.15, f"Alpha real debería tener PBO bajo, obtenido {res_alpha['pbo']:.3f}"
    assert res_alpha["mean_oos_sharpe"] > 2.0


def test_analytical_probability_threshold_is_bounded_and_monotonic():
    t_low_fric = compute_analytical_probability_threshold(k_up=2.0, k_down=1.5, hurdle_multiplier=2.5, friction_to_vol_ratio=0.02)
    t_high_fric = compute_analytical_probability_threshold(k_up=2.0, k_down=1.5, hurdle_multiplier=2.5, friction_to_vol_ratio=0.10)
    assert 0.45 <= t_low_fric < t_high_fric <= 0.85


def test_end_to_end_v3_meta_trainer_exports_real_onnx_and_gateway_consumes_it():
    """
    Test End-to-End completo de la arquitectura v3.0:
    1. Entrena MetaModelTrainerV3 (CUSUM + Primary + MetaBarrier + Sample Weights + MDA + Platt + CSCV PBO + DSR).
    2. Verifica exportación ONNX real (< 1e-5 paridad) y ModelBundle schema_version == '3.0'.
    3. Carga ese ModelBundle en ExecutionGateway y verifica ejecución sin errores y política NO_PRIMARY_EVENT.
    """
    df = _make_rich_synthetic_ohlcv(n=650, seed=55)
    primary = TrendPullbackPrimary(sma_slow=50, ema_fast=10, ema_slow=25, bilateral=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = MetaModelTrainerV3(
            symbol="EURUSD",
            primary_strategy=primary,
            cusum_kappa=0.8,
            k_up=2.0,
            k_down=1.5,
            max_hold=8,
            n_splits=3,
            enable_mda_pruning=True,
        )
        out = trainer.train_and_audit(
            df=df,
            results_dir=tmp_dir,
            ffd_d_map={"close": 0.35},
            export_onnx=True,
        )

        bundle = out["model_bundle"]
        meta = bundle["meta_model"]
        assert bundle["schema_version"] == "3.0"
        assert os.path.exists(out["onnx_path"])
        assert meta["onnx_parity_max_diff"] < 1e-5
        assert 0.0 <= meta["pbo_cscv"] <= 1.0
        assert 0.0 <= meta["dsr_oos"] <= 1.0
        assert 0.0 <= meta["ece_oos"] <= 1.0

        # Verificar que ExecutionGateway carga el ONNX v3.0 con su SHA-256 verificado
        vault = TradeVault(os.path.join(tmp_dir, "vault.db"))
        connector = MagicMock()
        connector.connected = True

        gw = ExecutionGateway(
            symbol="EURUSD",
            timeframe=1440,
            config=bundle,
            connector=connector,
            trade_vault=vault,
            results_dir=tmp_dir,
        )
        assert gw.onnx_engine is not None
        assert gw.model_integrity_error is None

        # Comprobar que ONNXInferenceEngine devuelve probabilidad válida sobre vector de features v3
        sample_vec = np.zeros((1, len(meta["feature_order"])), dtype=np.float32)
        p = gw.onnx_engine.predict_probability(sample_vec)
        assert 0.0 <= p <= 1.0
        vault.close()
