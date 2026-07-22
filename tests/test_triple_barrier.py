"""
Prioridad 2: Tests del Triple Barrier Labeling (Meta-Labeling).
Un bug aquí = el modelo aprende "al revés" y pierde dinero sistemáticamente.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))
from meta_labeling import TripleBarrierLabeler


class TestTripleBarrierLabeling:
    """Tests para TripleBarrierLabeler.apply_labels()"""

    def test_label_obvious_bullrun(self, df_bullrun):
        """En una subida fuerte y consistente, la mayoría de etiquetas deben ser 1 (TP alcanzado)."""
        labeler = TripleBarrierLabeler(k_up=2.0, k_down=1.5, max_hold=10)
        result = labeler.apply_labels(df_bullrun)
        
        win_rate = result['Label'].mean()
        assert win_rate > 0.5, \
            f"En un bull run obvio, el win rate ({win_rate:.2%}) debería ser > 50%"

    def test_label_obvious_crash(self, df_crash):
        """En una caída fuerte, la mayoría de etiquetas deben ser 0 (SL alcanzado)."""
        labeler = TripleBarrierLabeler(k_up=2.0, k_down=1.5, max_hold=10)
        result = labeler.apply_labels(df_crash)
        
        win_rate = result['Label'].mean()
        assert win_rate < 0.5, \
            f"En un crash obvio, el win rate ({win_rate:.2%}) debería ser < 50%"

    def test_label_max_hold_expiry(self, df_flat):
        """En un mercado plano, las posiciones deben expirar por tiempo (Label = 0)."""
        labeler = TripleBarrierLabeler(k_up=2.0, k_down=1.5, max_hold=10)
        result = labeler.apply_labels(df_flat)
        
        # En mercado plano, las barreras no se tocan → expira por tiempo → Label = 0
        win_rate = result['Label'].mean()
        assert win_rate < 0.3, \
            f"En mercado plano, el win rate ({win_rate:.2%}) debería ser < 30% (expiraciones por tiempo)"

    def test_label_both_barriers_same_day(self):
        """Si TP y SL se tocan el mismo día, la heurística pesimista debe asignar Label = 0."""
        n = 30
        # Escenario extremo: High muy alto Y Low muy bajo el mismo día (día 2)
        prices = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        
        # Día 2: rango intradiario absurdamente amplio (toca ambas barreras)
        highs[2] = 115.0  # Toca TP
        lows[2] = 85.0    # Toca SL
        
        df = pd.DataFrame({
            'time': pd.date_range('2020-01-01', periods=n, freq='B'),
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'EGARCH_Vol': np.full(n, 2.0),  # Vol alta para barreras anchas pero no tanto
        })
        
        labeler = TripleBarrierLabeler(k_up=2.0, k_down=1.5, max_hold=10)
        result = labeler.apply_labels(df)
        
        # El día 0 (señal) debería tener Label = 0 (heurística pesimista)
        assert result['Label'].iloc[0] == 0, \
            "Cuando TP y SL se tocan el mismo día, Label debe ser 0 (heurística pesimista)"

    def test_output_has_no_nans_in_label(self, df_bullrun):
        """La columna Label no debe contener NaN después del procesamiento."""
        labeler = TripleBarrierLabeler(k_up=2.0, k_down=1.5, max_hold=10)
        result = labeler.apply_labels(df_bullrun)
        
        nan_count = result['Label'].isna().sum()
        assert nan_count == 0, f"Label tiene {nan_count} NaN. Esto crasheará XGBoost silenciosamente."

    def test_output_shape_trimmed(self, df_bullrun):
        """El output debe tener max_hold+1 filas menos que el input (recorte anti look-ahead)."""
        max_hold = 10
        labeler = TripleBarrierLabeler(k_up=2.0, k_down=1.5, max_hold=max_hold)
        
        input_len = len(df_bullrun)
        result = labeler.apply_labels(df_bullrun)
        output_len = len(result)
        
        expected_len = input_len - max_hold - 1
        assert output_len == expected_len, \
            f"Output ({output_len} filas) debería ser input ({input_len}) - max_hold ({max_hold}) - 1 = {expected_len}"
