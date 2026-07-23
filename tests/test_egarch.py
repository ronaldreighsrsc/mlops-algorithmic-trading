"""
Prioridad 4: Tests del EGARCH (Volatilidad Condicional).
Un bug aquí = barreras TP/SL mal calibradas (demasiado anchas o estrechas).
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))
from volatility import VolatilityModeler


class TestEGARCH:
    """Tests para VolatilityModeler.compute_egarch()"""

    def test_output_always_positive(self, df_ohlcv_clean):
        """La volatilidad condicional siempre debe ser > 0 (un número positivo)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modeler = VolatilityModeler(window_size=500, cache_dir=tmpdir)
            result = modeler.compute_egarch(df_ohlcv_clean.copy())
        
            vol_col = result['EGARCH_Vol'].dropna()
            assert (vol_col > 0).all(), \
                f"Se encontraron {(vol_col <= 0).sum()} valores de volatilidad <= 0. Esto haría que el SL colapse sobre el precio."

    def test_output_no_nans_after_warmup(self, df_ohlcv_clean):
        """Después del periodo de calentamiento (window_size), no debe haber NaN."""
        window = 500
        with tempfile.TemporaryDirectory() as tmpdir:
            modeler = VolatilityModeler(window_size=window, cache_dir=tmpdir)
            result = modeler.compute_egarch(df_ohlcv_clean.copy())
        
            # Solo verificar después del periodo de calentamiento
            post_warmup = result['EGARCH_Vol'].iloc[window + 10:]
            nan_count = post_warmup.isna().sum()
            assert nan_count == 0, \
                f"Hay {nan_count} NaN después del warmup. Esto significa trades sin Stop Loss."

    def test_sanity_cap_below_5_percent(self, df_ohlcv_clean):
        """La volatilidad diaria no debería superar el 5% (filtro de cordura)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            modeler = VolatilityModeler(window_size=500, cache_dir=tmpdir)
            result = modeler.compute_egarch(df_ohlcv_clean.copy())
        
            vol_col = result['EGARCH_Vol'].dropna()
            max_vol = vol_col.max()
            assert max_vol < 5.0, \
                f"Volatilidad máxima ({max_vol:.2f}%) supera el 5% diario. El SL estaría absurdamente lejos."

    def test_fallback_produces_valid_output(self):
        """Con datos muy ruidosos, el sistema de fallback (GARCH → Vol Histórica) debe producir un resultado válido."""
        np.random.seed(42)
        n = 600
        # Crear datos extremadamente ruidosos para forzar el fallback
        prices = 100 + np.cumsum(np.random.normal(0, 5, n))
        prices = np.abs(prices) + 1  # Asegurar precios positivos
        
        df = pd.DataFrame({
            'time': pd.date_range('2018-01-01', periods=n, freq='B'),
            'open': prices * 0.99,
            'high': prices * 1.05,
            'low': prices * 0.95,
            'close': prices,
            'tick_volume': np.random.randint(1000, 5000, n),
            'real_volume': np.random.randint(1000, 5000, n),
            'spread': np.zeros(n),
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            modeler = VolatilityModeler(window_size=500, cache_dir=tmpdir)
            result = modeler.compute_egarch(df)
        
            # El resultado debe existir y ser usable (sin importar si usó EGARCH, GARCH o Vol Histórica)
            assert 'EGARCH_Vol' in result.columns, "La columna EGARCH_Vol no fue creada"
            vol_valid = result['EGARCH_Vol'].dropna()
            assert len(vol_valid) > 0, "No se produjo ningún valor de volatilidad válido"
            assert (vol_valid > 0).all(), "Algunos valores de volatilidad son <= 0 incluso con fallback"
