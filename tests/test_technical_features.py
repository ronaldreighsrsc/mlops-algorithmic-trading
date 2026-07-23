"""
Prioridad 5: Tests de Indicadores Técnicos (MACD, RSI, ATR).
Un bug aquí = features mal calculadas, pero con impacto menor 
(los modelos son robustos a ruido en features individuales).
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))
from technical_features import TechnicalFeatureEngineer


class TestTechnicalFeatures:
    """Tests para TechnicalFeatureEngineer.add_indicators()"""

    def test_rsi_bounded_0_100(self, df_ohlcv_clean):
        """RSI siempre debe estar entre 0 y 100 (es un oscilador acotado)."""
        eng = TechnicalFeatureEngineer(target_price_col='close')
        result = eng.add_indicators(df_ohlcv_clean.copy())
        
        rsi = result['RSI']
        assert rsi.min() >= 0, f"RSI mínimo ({rsi.min():.2f}) está por debajo de 0 (imposible matemáticamente)"
        assert rsi.max() <= 100, f"RSI máximo ({rsi.max():.2f}) está por encima de 100 (imposible matemáticamente)"

    def test_atr_always_positive(self, df_ohlcv_clean):
        """ATR siempre debe ser > 0 (es una medida de rango, nunca negativa)."""
        eng = TechnicalFeatureEngineer(target_price_col='close')
        result = eng.add_indicators(df_ohlcv_clean.copy())
        
        atr = result['ATR']
        assert (atr > 0).all(), \
            f"Se encontraron {(atr <= 0).sum()} valores de ATR <= 0. ATR negativo es matemáticamente imposible."

    def test_warmup_period_cleaned(self, df_ohlcv_clean):
        """Los indicadores técnicos no deben contener NaN en el output final."""
        eng = TechnicalFeatureEngineer(target_price_col='close')
        result = eng.add_indicators(df_ohlcv_clean.copy())
        
        # No debe haber NaN en los indicadores calculados
        for col in ['MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'ATR']:
            nan_count = result[col].isna().sum()
            assert nan_count == 0, f"La columna {col} tiene {nan_count} NaN después de la limpieza"

    def test_missing_columns_raises_error(self):
        """Si falta 'high' o 'low', debe lanzar un error claro (fail-fast)."""
        df_broken = pd.DataFrame({
            'close': [1.0, 2.0, 3.0],
            # Falta 'high' y 'low' intencionalmente
        })
        
        eng = TechnicalFeatureEngineer(target_price_col='close')
        
        with pytest.raises(ValueError, match="Falta la columna"):
            eng.add_indicators(df_broken)
