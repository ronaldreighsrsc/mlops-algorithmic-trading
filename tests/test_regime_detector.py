"""
Tests unitarios para el Detector de Regímenes Institucionales HMM 3-Estados y Test KS.
"""

import os
import tempfile
import pytest
import numpy as np
from src.models.regime_detector import (
    InstitutionalRegimeDetector,
    REGIME_BULL,
    REGIME_BEAR,
    REGIME_CHOPPY,
)


@pytest.fixture
def synthetic_market_data():
    """Genera datos sintéticos con 3 regímenes claros: alcista, bajista y rango turbulento."""
    np.random.seed(42)
    n_per_regime = 200

    # Régimen 0: Bull (retornos positivos, vol baja)
    bull_rets = np.random.normal(0.01, 0.005, n_per_regime)
    bull_vol = np.random.uniform(0.5, 1.0, n_per_regime)

    # Régimen 1: Bear (retornos negativos, vol media)
    bear_rets = np.random.normal(-0.01, 0.008, n_per_regime)
    bear_vol = np.random.uniform(1.0, 1.5, n_per_regime)

    # Régimen 2: Choppy (retornos oscilantes alrededor de 0, vol altísima)
    chop_rets = np.random.normal(0.0, 0.03, n_per_regime)
    chop_vol = np.random.uniform(2.5, 4.0, n_per_regime)

    rets = np.concatenate([bull_rets, bear_rets, chop_rets])
    vols = np.concatenate([bull_vol, bear_vol, chop_vol])

    X = np.column_stack([rets, vols])
    return X, rets


def test_hmm_training_and_semantic_mapping(synthetic_market_data):
    """Verifica que el HMM se entrene y asigne los 3 estados semánticos correctamente."""
    X, rets = synthetic_market_data
    detector = InstitutionalRegimeDetector(n_components=3, random_state=42)
    detector.fit(X, returns_col_idx=0)

    assert detector.model is not None
    assert len(detector.state_mapping) == 3
    assert set(detector.state_mapping.values()) == {REGIME_BULL, REGIME_BEAR, REGIME_CHOPPY}


def test_ks_drift_no_drift_when_same_distribution(synthetic_market_data):
    """Retornos extraídos de la misma distribución in-sample no deben disparar alarma KS."""
    X, rets = synthetic_market_data
    detector = InstitutionalRegimeDetector(n_components=3, random_state=42)
    detector.fit(X, returns_col_idx=0)

    # Muestra reciente de la misma distribución
    np.random.seed(99)
    recent_same = np.random.choice(rets, size=50, replace=True)

    is_drift, p_val, safety = detector.evaluate_ks_drift(recent_same)
    assert is_drift is False
    assert p_val >= 0.01
    assert safety == 1.0


def test_ks_drift_triggered_on_regime_shift(synthetic_market_data):
    """Un cambio brusco en la distribución de retornos debe disparar la alarma KS (p < 0.01)."""
    X, rets = synthetic_market_data
    detector = InstitutionalRegimeDetector(n_components=3, random_state=42)
    detector.fit(X, returns_col_idx=0)

    # Distribución severamente alterada (Cisne Negro / Volatilidad extrema)
    np.random.seed(123)
    recent_drifted = np.random.normal(-0.08, 0.06, 50)

    is_drift, p_val, safety = detector.evaluate_ks_drift(recent_drifted)
    assert is_drift is True
    assert p_val < 0.01
    assert safety == 0.50


def test_diagnose_combines_regime_and_safety_factors(synthetic_market_data):
    """Verifica que el factor de Kelly efectivo combine el régimen HMM y el factor de deriva KS."""
    X, rets = synthetic_market_data
    detector = InstitutionalRegimeDetector(n_components=3, random_state=42)
    detector.fit(X, returns_col_idx=0)

    # Simular una ventana reciente con drift
    recent_drifted = np.random.normal(-0.08, 0.06, 50)
    diagnosis = detector.diagnose(X[-5:], recent_drifted)

    assert diagnosis.regime_state in [REGIME_BULL, REGIME_BEAR, REGIME_CHOPPY]
    assert diagnosis.effective_kelly_factor == diagnosis.regime_kelly_multiplier * diagnosis.ks_safety_multiplier
    assert "Kelly Efectivo:" in diagnosis.summary


def test_save_and_load_regime_detector(synthetic_market_data):
    """Verifica la persistencia y reproducibilidad de inferencia tras guardar y cargar."""
    X, rets = synthetic_market_data
    detector = InstitutionalRegimeDetector(n_components=3, random_state=42)
    detector.fit(X, returns_col_idx=0)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    try:
        detector.save(temp_path)
        loaded = InstitutionalRegimeDetector()
        loaded.load(temp_path)

        test_point = X[-1].reshape(1, -1)
        orig_pred = detector.predict_regime(test_point)
        load_pred = loaded.predict_regime(test_point)

        assert orig_pred == load_pred
        assert loaded.reference_returns is not None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
