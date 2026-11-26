"""
Tests Slice D (v3.1): Torneo Pre-Registrado Cerrado (N=6 = 3 Primarias x 2 Modelos),
compuertas estadísticas duras (LogLoss, ECE, DSR, CSCV PBO, Benchmark) y paridad ONNX v3.1.
"""

import os
import tempfile

import numpy as np
import pandas as pd

from src.models.meta_trainer_v3 import run_preregistered_v31_tournament
from src.strategies.primary_strategies import (
    DonchianBreakoutPrimary,
    MacroCarryMomentumPrimary,
    TrendPullbackPrimary,
)


def _make_pure_noise_ohlcv_macro(n: int = 700, seed: int = 101) -> pd.DataFrame:
    """Caminata aleatoria pura + ruido macro: debe ser RECHAZADA por gate_approved == False."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    high = np.maximum(open_p, close) * (1.0 + np.abs(rng.normal(0.004, 0.002, size=n)))
    low = np.minimum(open_p, close) * (1.0 - np.abs(rng.normal(0.004, 0.002, size=n)))

    return pd.DataFrame({
        "time": pd.bdate_range("2018-01-01", periods=n),
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "EGARCH_Vol": np.full(n, 1.0),
        "ATR": close * 0.01,
        "RSI": rng.uniform(30, 70, size=n),
        "MACD_Hist": rng.normal(0, 0.2, size=n),
        "delta_rate_diff_20d": rng.normal(0, 0.15, size=n),
        "vix_term_ratio": rng.uniform(0.85, 1.05, size=n),
    })


def test_preregistered_v31_tournament_runs_6_trials_and_rejects_pure_noise():
    df_noise = _make_pure_noise_ohlcv_macro(n=720, seed=101)
    primaries = [
        TrendPullbackPrimary(sma_slow=50, ema_fast=10, ema_slow=25, bilateral=True),
        DonchianBreakoutPrimary(lookback=15, bilateral=True),
        MacroCarryMomentumPrimary(mode="fx_rate_diff", ema_window=20, bilateral=True),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        out = run_preregistered_v31_tournament(
            symbol="EURUSD",
            df=df_noise,
            primaries=primaries,
            results_dir=tmp_dir,
            candidate_names=("LOGISTIC_L2", "RANDOM_FOREST"),
            cusum_kappa=0.8,
            enable_mda_pruning=True,
            export_onnx=True,
        )

        # Deben evaluarse hasta 3 x 2 = 6 ensayos pre-registrados y nunca sobrepasar 6
        assert 2 <= len(out["trials_table"]) <= 6
        bundle = out["model_bundle"]
        meta = bundle["meta_model"]
        assert bundle["schema_version"] == "3.1"
        assert meta["n_trials_preregistered"] == len(out["trials_table"])
        assert meta["onnx_parity_max_diff"] < 1e-5
        assert os.path.exists(os.path.join(tmp_dir, "saved_models", meta["onnx_file"]))

        # Sobre ruido aleatorio puro, al menos una compuerta estadística debe rechazar el despliegue
        assert out["gate_approved"] is False
        assert set(out["gates"].keys()) == {
            "oos_log_loss_lt_067",
            "oos_ece_lt_005",
            "dsr_ge_095",
            "cscv_pbo_lt_020",
            "sharpe_gt_benchmark",
        }
