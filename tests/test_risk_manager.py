"""
Prioridad 1: Tests del Risk Manager (Position Sizing y Barreras de Precio).
Un bug aquí = pérdida directa de dinero real en MT5/Quantfury.
"""
import pytest
import math


class TestTripleBarrierLevels:
    """Tests para calculate_triple_barrier_levels()"""

    def test_tp_above_price_sl_below(self, risk_manager_default):
        """TP siempre debe estar ARRIBA del precio y SL siempre ABAJO."""
        price = 1.1000
        vol = 0.5  # 0.5% de volatilidad diaria
        
        tp, sl = risk_manager_default.calculate_triple_barrier_levels(price, vol)
        
        assert tp > price, f"Take Profit ({tp}) debería estar por encima del precio ({price})"
        assert sl < price, f"Stop Loss ({sl}) debería estar por debajo del precio ({price})"

    def test_zero_volatility_no_crash(self, risk_manager_default):
        """Si la volatilidad es 0, las barreras deben ser iguales al precio (sin crash)."""
        price = 1.1000
        vol = 0.0
        
        tp, sl = risk_manager_default.calculate_triple_barrier_levels(price, vol)
        
        # Con volatilidad 0, TP y SL colapsan al precio de entrada
        assert tp == price, "Con vol=0, TP debería ser igual al precio"
        assert sl == price, "Con vol=0, SL debería ser igual al precio"


class TestPositionSizing:
    """Tests para calculate_position_size()"""
    
    def test_lots_never_negative(self, risk_manager_default):
        """El volumen calculado nunca debe ser negativo."""
        lots = risk_manager_default.calculate_position_size(
            balance=10000.0,
            current_price=1.1000,
            stop_loss_price=1.0950,
            tick_size=0.00001,
            tick_value=1.0,
            volume_step=0.01,
            prediction_prob=0.55,
            confidence_threshold=0.50
        )
        assert lots >= 0, f"Los lotes ({lots}) nunca deben ser negativos"

    def test_lots_respects_volume_step(self, risk_manager_default):
        """Los lotes deben ser múltiplos exactos del volume_step del broker."""
        volume_step = 0.01
        lots = risk_manager_default.calculate_position_size(
            balance=10000.0,
            current_price=1.1000,
            stop_loss_price=1.0950,
            tick_size=0.00001,
            tick_value=1.0,
            volume_step=volume_step,
            prediction_prob=0.65,
            confidence_threshold=0.50
        )
        
        if lots > 0:
            # Verificar que sea múltiplo exacto del step
            remainder = round(lots % volume_step, 10)
            assert remainder == 0 or remainder == volume_step, \
                f"Lotes ({lots}) no es múltiplo de volume_step ({volume_step})"

    def test_kelly_multiplier_boundaries(self, risk_manager_default):
        """Verificar que los 3 niveles de Kelly (0.5x, 1.0x, 2.0x) se apliquen correctamente."""
        balance = 10000.0
        params = dict(
            current_price=1.1000,
            stop_loss_price=1.0950,
            tick_size=0.00001,
            tick_value=1.0,
            volume_step=0.01,
            confidence_threshold=0.50
        )
        
        # Prob apenas sobre el umbral (delta <= 0.05) → Kelly 0.5x
        lots_low = risk_manager_default.calculate_position_size(
            balance=balance, prediction_prob=0.53, **params
        )
        # Prob moderada (0.05 < delta <= 0.15) → Kelly 1.0x
        lots_mid = risk_manager_default.calculate_position_size(
            balance=balance, prediction_prob=0.60, **params
        )
        # Prob alta (delta > 0.15) → Kelly 2.0x
        lots_high = risk_manager_default.calculate_position_size(
            balance=balance, prediction_prob=0.70, **params
        )
        
        assert lots_low <= lots_mid <= lots_high, \
            f"Kelly debería escalar: 0.5x ({lots_low}) <= 1.0x ({lots_mid}) <= 2.0x ({lots_high})"

    def test_equal_prices_returns_zero(self, risk_manager_default):
        """Si precio == SL (distancia 0), debe retornar 0 lotes para evitar división por cero."""
        lots = risk_manager_default.calculate_position_size(
            balance=10000.0,
            current_price=1.1000,
            stop_loss_price=1.1000,  # Mismo precio = distancia 0
            tick_size=0.00001,
            tick_value=1.0,
            volume_step=0.01,
            prediction_prob=0.60,
            confidence_threshold=0.50
        )
        assert lots == 0.0, f"Con precio == SL, lotes debe ser 0, no {lots}"
