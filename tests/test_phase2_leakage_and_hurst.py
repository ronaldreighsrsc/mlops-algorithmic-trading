"""
Tests Fase 2 (v3.0): Integridad Temporal Estricta (Zero Leakage) y Corrección de Hurst.

Valida:
1. PurgedKFold basado en intervalos exactos [t0, t1] (cero solapamiento + embargo).
2. Test de Choque Futuro Oráculo ("Oracle Future Shock") en Walk-Forward Retraining:
   corromper las últimas `max_hold` etiquetas futuras no altera ni un bit las predicciones
   hasta el instante de reentrenamiento.
3. Ausencia de 1-bar lookahead en CPCVAuditor.
4. Hurst sobre retornos estacionarios y Variance Ratio en Random Walk vs. Momentum vs. Mean Reversion.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cpcv_auditor import CPCVAuditor
from src.models.random_forest import RandomForestTrainer
from src.models.xgb_model import XGBoostTrainer
from src.preprocessing.asset_screener import AssetScreener
from src.validation.purged_cv import PurgedKFold, get_causal_walk_forward_train_mask


def test_purged_kfold_enforces_zero_overlap_and_embargo():
    """Ningún evento de entrenamiento [t0_j, t1_j] puede intersectar el bloque de validación ni su embargo."""
    n = 120
    dates = pd.bdate_range("2020-01-01", periods=n + 15)
    t0 = pd.Series(dates[:n])
    # Cada trade dura 8 días hábiles
    t1 = pd.Series(dates[8 : n + 8])

    cv = PurgedKFold(n_splits=4, pct_embargo=0.05)
    folds = list(cv.split(t0, t1))
    assert len(folds) == 4

    for train_idx, val_idx in folds:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(set(train_idx).intersection(set(val_idx))) == 0

        val_start = t0.iloc[val_idx[0]]
        val_end = t1.iloc[val_idx].max()

        # Verificar cero solapamiento de [t0, t1]
        for j in train_idx:
            assert not (t1.iloc[j] >= val_start and t0.iloc[j] <= val_end), (
                f"Leakage detectado: train[{j}]=[{t0.iloc[j]}, {t1.iloc[j]}] solapa val=[{val_start}, {val_end}]"
            )

        # Verificar embargo posterior cuando existe bloque después de validación
        if val_idx[-1] < n - 1:
            embargo_count = int(np.ceil(n * 0.05))
            embargo_indices = set(range(val_idx[-1] + 1, min(n, val_idx[-1] + 1 + embargo_count)))
            assert len(set(train_idx).intersection(embargo_indices)) == 0


def test_causal_walk_forward_mask_excludes_open_barriers():
    dates = pd.bdate_range("2022-01-01", periods=30)
    t1 = pd.Series(dates[10:25])  # 15 eventos cerrando entre el día 10 y 24
    current_time = dates[15]

    mask = get_causal_walk_forward_train_mask(t1, current_time)
    # Solo eventos con t1 < dates[15] (índices 0..4 -> dates[10..14])
    assert int(mask.sum()) == 5
    assert (t1[mask] < current_time).all()
    assert (t1[~mask] >= current_time).all()


@pytest.mark.parametrize("trainer_cls,params", [
    (XGBoostTrainer, {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0}),
    (RandomForestTrainer, {"n_estimators": 20, "max_depth": 4}),
])
def test_walk_forward_oracle_future_shock_invariance(trainer_cls, params):
    """
    Test definitivo de No-Leakage:
    Con retrain_step=20 y max_hold=10, en el paso i=19 (20º punto de test), las etiquetas
    y_test[10:20] corresponden a barreras AÚN ABIERTAS.
    Si invertimos completamente y_test[10:20], la predicción del paso 20 (justo tras el
    reentrenamiento en i=19) DEBE permanecer idéntica bit a bit.
    """
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(120, 6))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(30, 6))
    y_test_orig = (X_test[:, 0] > 0).astype(int)

    # Corromper SOLO las etiquetas de test dentro de la ventana abierta [10:20]
    y_test_shocked = y_test_orig.copy()
    y_test_shocked[10:20] = 1 - y_test_shocked[10:20]

    t1 = trainer_cls(look_back=2, retrain_step=20, max_hold=10)
    p_orig, _ = t1.walk_forward_predict(X_train, y_train, X_test, y_test_orig, params)

    t2 = trainer_cls(look_back=2, retrain_step=20, max_hold=10)
    p_shock, _ = t2.walk_forward_predict(X_train, y_train, X_test, y_test_shocked, params)

    # Hasta el paso 29 (único retrain en i=19 usa y_test[:10]), TODAS las predicciones deben ser idénticas
    assert np.allclose(p_orig, p_shock, atol=1e-12), (
        "LEAKAGE DETECTADO: modificar etiquetas aún abiertas (y_test[10:20]) alteró las predicciones!"
    )


def test_cpcv_auditor_has_no_same_bar_lookahead():
    """
    Si le pasamos una señal oráculo del día t (prob=1 cuando close[t] > close[t-1]),
    al aplicar shift(1) sobre un Random Walk NO debe generar un Sharpe estratosférico (> 5),
    porque el retorno del día t+1 es independiente del retorno del día t.
    """
    rng = np.random.default_rng(99)
    n = 400
    rets = rng.normal(0.0, 0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    df_proc = pd.DataFrame({"close": prices})

    # Señal contemporánea (conoce si la vela t cerró arriba de t-1)
    same_bar_up = (np.insert(np.diff(prices), 0, 0.0) > 0).astype(float)
    probs = np.where(same_bar_up == 1.0, 0.9, 0.1)

    auditor = CPCVAuditor(symbol="TEST_NO_LOOKAHEAD")
    res = auditor.audit_strategy(df_proc, probs, confidence_threshold=0.5, n_groups=6, k_test=2)
    # Con lookahead sin shift(1) esto daría Sharpe ~ +15; con shift(1) causal debe estar cerca de 0
    assert abs(res["sharpe_mean"]) < 3.0, f"Posible 1-bar lookahead: Sharpe={res['sharpe_mean']:.2f}"


def test_hurst_on_random_walk_is_near_050_not_085():
    """
    Verifica corrección F-09: sobre caminatas aleatorias geométricas (precios I(1)),
    Hurst debe calcularse sobre retornos estacionarios arrojando ~0.50 (y jamás ~0.85).
    """
    hurst_vals = []
    vr_vals = []
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        rets = rng.normal(0.0, 0.01, size=1500)
        prices = pd.Series(100.0 * np.exp(np.cumsum(rets)))
        h = AssetScreener.calculate_hurst_exponent(prices, max_lag=100)
        vr, _ = AssetScreener.calculate_variance_ratio(prices, q=10)
        hurst_vals.append(h)
        vr_vals.append(vr)

    mean_h = float(np.mean(hurst_vals))
    mean_vr = float(np.mean(vr_vals))
    assert 0.44 <= mean_h <= 0.59, f"Hurst en Random Walk mal calibrado: {mean_h:.4f} (valores={hurst_vals})"
    assert 0.80 <= mean_vr <= 1.20, f"Variance Ratio en Random Walk fuera de rango: {mean_vr:.4f}"


def test_variance_ratio_and_hurst_separate_momentum_from_mean_reversion():
    """Un proceso AR(1) positivo (momentum) debe tener H y VR mayores que un AR(1) negativo (reversión)."""
    rng = np.random.default_rng(77)
    n = 1500
    eps = rng.normal(0.0, 0.01, size=n)

    # Proceso persistente (AR(1) con phi = +0.45)
    r_mom = np.zeros(n)
    # Proceso anti-persistente (AR(1) con phi = -0.45)
    r_mr = np.zeros(n)
    for t in range(1, n):
        r_mom[t] = 0.45 * r_mom[t - 1] + eps[t]
        r_mr[t] = -0.45 * r_mr[t - 1] + eps[t]

    p_mom = pd.Series(100.0 * np.exp(np.cumsum(r_mom)))
    p_mr = pd.Series(100.0 * np.exp(np.cumsum(r_mr)))

    h_mom = AssetScreener.calculate_hurst_exponent(p_mom)
    h_mr = AssetScreener.calculate_hurst_exponent(p_mr)
    vr_mom, z_mom = AssetScreener.calculate_variance_ratio(p_mom, q=10)
    vr_mr, z_mr = AssetScreener.calculate_variance_ratio(p_mr, q=10)

    assert h_mom > h_mr + 0.15
    assert vr_mom > 1.3 and z_mom > 2.0
    assert vr_mr < 0.7 and z_mr < -2.0
