"""
Tests Fase 1 (v3.0): Paridad Numérica Real ONNX (< 1e-5), Fail-Closed Gateway,
FFD determinista congelado y corrección de shape 3-features en AnalyticsDaemon.
"""

import os
import json
import tempfile
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from src.database.trade_vault import TradeVault
from src.execution.analytics_daemon import AnalyticsDaemon
from src.execution.execution_gateway import ExecutionGateway
from src.execution.onnx_inference_engine import ONNXInferenceEngine
from src.models.calibration import PurgedPlattCalibrator, compute_brier_and_ece
from src.models.onnx_exporter import (
    export_legacy_trainer_pkl_to_onnx,
    export_sklearn_pipeline_to_onnx,
    update_onnx_manifest,
)
from src.preprocessing.stationarity import FractionalDifferencer


@pytest.fixture
def synthetic_tabular_dataset():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    # Inyectar algunos NaNs para verificar el SimpleImputer embebido en ONNX
    X[5, 2] = np.nan
    X[17, 0] = np.nan
    logits = 1.2 * np.nan_to_num(X[:, 0]) - 0.8 * np.nan_to_num(X[:, 1]) + 0.4 * np.nan_to_num(X[:, 3])
    probs_true = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=300) < probs_true).astype(int)
    return X[:220], y[:220], X[220:], y[220:]


@pytest.mark.parametrize("model_name", ["LR", "RF", "XGB", "LGB"])
def test_real_onnx_parity_all_tabular_models_with_platt_calibration(synthetic_tabular_dataset, model_name):
    """
    Valida paridad numérica estricta (< 1e-5) entre Pipeline + PurgedPlattCalibrator
    en Python y su exportación ONNX autocontenida para LR, RF, XGBoost y LightGBM.
    """
    X_tr, y_tr, X_val, y_val = synthetic_tabular_dataset

    if model_name == "LR":
        clf = LogisticRegression(C=1.0, max_iter=500)
    elif model_name == "RF":
        clf = RandomForestClassifier(n_estimators=25, max_depth=4, random_state=42)
    elif model_name == "XGB":
        clf = xgb.XGBClassifier(n_estimators=25, max_depth=3, learning_rate=0.08, random_state=42, eval_metric="logloss")
    else:
        clf = lgb.LGBMClassifier(n_estimators=25, max_depth=3, learning_rate=0.08, random_state=42, verbose=-1)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    pipe.fit(X_tr, y_tr)

    raw_val_probs = pipe.predict_proba(X_val)[:, 1]
    calibrator = PurgedPlattCalibrator().fit(raw_val_probs, y_val)
    cal_val_probs = calibrator.transform(raw_val_probs)

    brier, ece = compute_brier_and_ece(y_val, cal_val_probs)
    assert 0.0 <= brier <= 1.0
    assert 0.0 <= ece <= 1.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, f"real_{model_name.lower()}.onnx")
        out_path, sha256, max_diff = export_sklearn_pipeline_to_onnx(
            pipeline=pipe,
            output_path=onnx_path,
            X_parity_sample=X_val,
            calibrator=calibrator,
            max_parity_tol=1e-5,
        )
        assert os.path.exists(out_path)
        assert len(sha256) == 64
        assert max_diff < 1e-5

        # Verificar también inferencia unitaria vía ONNXInferenceEngine con SHA-256
        engine = ONNXInferenceEngine(out_path, expected_sha256=sha256)
        p_single = engine.predict_probability(X_val[0])
        assert abs(p_single - float(cal_val_probs[0])) < 1e-5


def test_legacy_xgboost_pkl_exports_real_weights_not_dummy(synthetic_tabular_dataset):
    """Un .pkl legacy {'scaler', 'master_model', 'look_back'} debe exportar pesos reales con paridad < 1e-5."""
    X_tr, y_tr, X_val, _ = synthetic_tabular_dataset
    X_tr_clean = np.nan_to_num(X_tr)
    X_val_clean = np.nan_to_num(X_val)

    scaler = StandardScaler().fit(X_tr_clean)
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3, random_state=42, eval_metric="logloss")
    model.fit(scaler.transform(X_tr_clean), y_tr)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pkl_path = os.path.join(tmp_dir, "legacy_xgb.pkl")
        onnx_path = os.path.join(tmp_dir, "legacy_xgb.onnx")
        joblib.dump({"scaler": scaler, "master_model": model, "look_back": 1}, pkl_path)

        _, sha, max_diff = export_legacy_trainer_pkl_to_onnx(pkl_path, onnx_path, X_parity_sample=X_val_clean)
        assert max_diff < 1e-5
        assert os.path.getsize(onnx_path) > 1000  # Grafo real de árboles, jamás un dummy de 252 bytes

        engine = ONNXInferenceEngine(onnx_path, expected_sha256=sha)
        p_onnx = engine.predict_proba_batch(X_val_clean)
        p_native = model.predict_proba(scaler.transform(X_val_clean))[:, 1]
        assert np.max(np.abs(p_onnx - p_native)) < 1e-5


def test_gateway_fail_closed_on_tampered_onnx_checksum(synthetic_tabular_dataset):
    """
    Si el manifiesto SHA-256 no coincide con el .onnx en disco, ExecutionGateway debe:
    1. Leer onnx_manifest.json sin NameError (import json presente),
    2. Rechazar el modelo y guardar model_integrity_error,
    3. Abortar con MODEL_INTEGRITY_ERROR (fail-closed) en lugar de usar fallback 0.50.
    """
    X_tr, y_tr, X_val, _ = synthetic_tabular_dataset
    X_tr_clean = np.nan_to_num(X_tr)
    pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression())]).fit(X_tr_clean, y_tr)

    with tempfile.TemporaryDirectory() as tmp_dir:
        models_dir = os.path.join(tmp_dir, "saved_models")
        os.makedirs(models_dir, exist_ok=True)
        onnx_path = os.path.join(models_dir, "meta_eurusd.onnx")
        export_sklearn_pipeline_to_onnx(pipe, onnx_path, X_parity_sample=np.nan_to_num(X_val))

        # Escribir manifiesto con hash corrupto/manipulado
        manifest_path = os.path.join(models_dir, "onnx_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"meta_eurusd.onnx": "deadbeef" * 8}, f)

        db_path = os.path.join(tmp_dir, "vault.db")
        vault = TradeVault(db_path)
        connector = MagicMock()
        connector.connected = True

        cfg = {
            "schema_version": "3.0",
            "activo": "EURUSD",
            "meta_model": {"onnx_file": "meta_eurusd.onnx", "model_type": "LOGISTIC"},
        }
        gw = ExecutionGateway(
            symbol="EURUSD",
            timeframe=1440,
            config=cfg,
            connector=connector,
            trade_vault=vault,
            results_dir=tmp_dir,
        )

        assert gw.onnx_engine is None
        assert gw.model_integrity_error is not None
        assert "Fallo de integridad criptográfica" in gw.model_integrity_error

        with patch("MetaTrader5.copy_rates_from_pos") as mock_rates, \
             patch("MetaTrader5.account_info", return_value=None), \
             patch.object(gw.engine, "has_open_positions", return_value=False):
            mock_rates.return_value = [{"time": 1000}, {"time": 2000}]
            res = gw.process_tick_event()

        assert res["status"] == "MODEL_INTEGRITY_ERROR"
        audits = vault.get_recent_audits(symbol="EURUSD", limit=5)
        assert len(audits) >= 1
        assert audits.iloc[0]["status"] == "MODEL_INTEGRITY_ERROR"
        vault.close()


def test_ffd_transform_with_fixed_d_is_deterministic_without_refit():
    """transform_with_fixed_d debe aplicar exactamente los pesos d* congelados sin recalcular ADF."""
    rng = np.random.default_rng(7)
    prices = np.cumsum(rng.normal(0, 1, 400)) + 100.0
    df = pd.DataFrame({"close": prices, "open": prices + 0.1})

    ffd = FractionalDifferencer(threshold=1e-4)
    df_train = ffd.apply_ffd(df.copy())
    assert "close" in ffd.d_map

    # Aplicar exactamente el d_map congelado sobre una ventana viva más corta
    df_live = ffd.transform_with_fixed_d(df.copy(), ffd.d_map)
    common = df_train.index.intersection(df_live.dropna().index)
    assert len(common) > 50
    if "close_FFD" in df_train.columns:
        assert np.allclose(
            df_train.loc[common, "close_FFD"].values,
            df_live.loc[common, "close_FFD"].values,
            atol=1e-10,
        )


def test_analytics_daemon_accepts_3feature_autoencoder_telemetry():
    """
    Verifica que AnalyticsDaemon procesa correctamente la telemetría de 3 features
    [net_ret, duration, win_flag] del StrategyLSTMAutoencoder y dispara cuarentena
    sin fallar por incompatibilidad de shape (N, 1) vs (N, 3).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "vault.db")
        vault = TradeVault(db_path)

        # Crear un archivo sentinel .keras para que el daemon entre al flujo del Autoencoder
        ae_base = os.path.join(tmp_dir, "campeon_EURUSD_autoencoder")
        with open(ae_base + ".keras", "wb") as f:
            f.write(b"dummy")

        mock_ae = MagicMock()
        mock_ae.scaler = MagicMock(n_features_in_=3)
        mock_ae.is_anomalous.return_value = True
        mock_ae.check_concept_drift.return_value = True

        # Simular tanto telemetría 3D explícita como lista 1D legacy (auto-convertida a 3 features)
        metrics_3d = [[0.01, 3, 1]] * 29 + [[-0.08, 10, 0]]
        cfg = [{"activo": "EURUSD", "oos_trade_metrics": metrics_3d}]
        daemon = AnalyticsDaemon(symbols_config=cfg, trade_vault=vault, results_dir=tmp_dir)

        with patch("src.models.anomaly_detector.StrategyLSTMAutoencoder", return_value=mock_ae):
            healthy = daemon.run_shadow_journal_for_symbol(cfg[0])

        assert healthy is False
        assert vault.is_symbol_quarantined("EURUSD") is True
        passed_row = mock_ae.is_anomalous.call_args[0][0]
        assert passed_row.shape == (3,)
        passed_window = mock_ae.check_concept_drift.call_args[0][0]
        assert passed_window.shape == (30, 3)

        # Caso 1D legacy: debe auto-expandirse a (30, 3) sin crashear
        mock_ae.reset_mock()
        mock_ae.is_anomalous.return_value = False
        mock_ae.check_concept_drift.return_value = False
        cfg_1d = [{"activo": "EURUSD", "oos_returns": [0.005] * 30}]
        daemon_1d = AnalyticsDaemon(symbols_config=cfg_1d, trade_vault=vault, results_dir=tmp_dir)
        with patch("src.models.anomaly_detector.StrategyLSTMAutoencoder", return_value=mock_ae):
            healthy_1d = daemon_1d.run_shadow_journal_for_symbol(cfg_1d[0])

        assert healthy_1d is True
        assert mock_ae.check_concept_drift.call_args[0][0].shape == (30, 3)
        vault.close()
