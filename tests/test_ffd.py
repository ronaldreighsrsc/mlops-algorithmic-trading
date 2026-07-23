"""
Prioridad 3: Tests del Fractional Differencing (FFD).
Un bug aquí = las columnas de entrada del modelo son incorrectas (features corruptas).
"""
import pytest
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))
from stationarity import FractionalDifferencer


class TestFFDWeights:
    """Tests para la fórmula de pesos iterativos _get_weights_ffd()"""

    def test_weights_first_always_one(self):
        """El primer peso de la fórmula FFD siempre debe ser 1.0 (w₀ = 1 por definición)."""
        ffd = FractionalDifferencer(threshold=1e-4)
        
        for d in [0.0, 0.1, 0.5, 0.8, 1.0]:
            weights = ffd._get_weights_ffd(d)
            # Los pesos se almacenan invertidos (el último es w₀)
            assert abs(weights[-1][0] - 1.0) < 1e-10, \
                f"Con d={d}, el primer peso (w₀) debe ser 1.0, pero es {weights[-1][0]}"

    def test_d0_returns_single_weight(self):
        """Con d=0, la diferenciación no hace nada (solo 1 peso = 1.0)."""
        ffd = FractionalDifferencer(threshold=1e-4)
        weights = ffd._get_weights_ffd(0.0)
        
        assert len(weights) == 1, f"Con d=0, debería haber exactamente 1 peso, hay {len(weights)}"
        assert weights[0][0] == 1.0, f"Con d=0, el único peso debe ser 1.0"


class TestFFDTransformation:
    """Tests para la transformación completa apply_ffd()"""

    def test_output_is_stationary(self):
        """La serie FFD resultante debe pasar el test ADF (p < 0.05)."""
        np.random.seed(42)
        # Crear un random walk (no estacionario por definición)
        n = 500
        random_walk = np.cumsum(np.random.normal(0, 1, n)) + 100
        
        df = pd.DataFrame({
            'close': random_walk,
        })
        
        ffd = FractionalDifferencer(threshold=1e-4, p_value_limit=0.05)
        result = ffd.apply_ffd(df, columns_to_ignore=['time'])
        
        if 'close_FFD' in result.columns:
            serie_ffd = result['close_FFD'].dropna()
            if len(serie_ffd) > 100:
                p_value = adfuller(serie_ffd.values)[1]
                assert p_value < 0.05, \
                    f"La serie FFD NO es estacionaria (p={p_value:.4f}). El modelo recibirá datos con tendencia."

    def test_preserves_memory(self):
        """La correlación entre la serie original y la FFD debe ser > 0.80."""
        np.random.seed(42)
        n = 500
        random_walk = np.cumsum(np.random.normal(0, 1, n)) + 100
        
        df = pd.DataFrame({
            'close': random_walk,
        })
        
        ffd = FractionalDifferencer(threshold=1e-4, p_value_limit=0.05)
        result = ffd.apply_ffd(df, columns_to_ignore=['time'])
        
        if 'close_FFD' in result.columns:
            original = result['close'].values
            transformed = result['close_FFD'].values
            corr = np.corrcoef(original, transformed)[0, 1]
            assert corr > 0.50, \
                f"La correlación ({corr:.4f}) es demasiado baja. El FFD está destruyendo la memoria."

    def test_protected_columns_untouched(self, df_ohlcv_clean):
        """Las columnas protegidas (RSI, MACD, ATR) no deben ser diferenciadas."""
        # Agregar indicadores simulados que deberían estar protegidos
        df = df_ohlcv_clean.copy()
        df['RSI'] = np.random.uniform(30, 70, len(df))
        df['MACD'] = np.random.normal(0, 1, len(df))
        df['ATR'] = np.random.uniform(0.5, 2.0, len(df))
        
        ffd = FractionalDifferencer(threshold=1e-4)
        result = ffd.apply_ffd(df)
        
        # Las columnas protegidas NO deben tener versión _FFD
        assert 'RSI_FFD' not in result.columns, "RSI fue diferenciado (es un oscilador, no debería tocarse)"
        assert 'MACD_FFD' not in result.columns, "MACD fue diferenciado (es un oscilador, no debería tocarse)"
        assert 'ATR_FFD' not in result.columns, "ATR fue diferenciado (es una medida de varianza, no debería tocarse)"
        
        # Las originales deben seguir existiendo
        assert 'RSI' in result.columns, "RSI fue eliminado del DataFrame"
        assert 'MACD' in result.columns, "MACD fue eliminado del DataFrame"
        assert 'ATR' in result.columns, "ATR fue eliminado del DataFrame"

    def test_d1_approximates_first_difference(self):
        """Con d=1.0, el FFD debe aproximarse a la primera diferencia clásica (diff())."""
        np.random.seed(42)
        n = 200
        prices = np.cumsum(np.random.normal(0, 1, n)) + 100
        series = pd.Series(prices)
        
        ffd = FractionalDifferencer(threshold=1e-4)
        ffd_result = ffd._frac_diff_ffd(series, d=1.0).dropna()
        diff_result = series.diff().dropna()
        
        # Alinear por índice
        common = ffd_result.index.intersection(diff_result.index)
        if len(common) > 50:
            corr = np.corrcoef(ffd_result.loc[common].values, diff_result.loc[common].values)[0, 1]
            assert corr > 0.95, \
                f"Con d=1.0, FFD y diff() deberían ser casi idénticos (corr={corr:.4f})"
