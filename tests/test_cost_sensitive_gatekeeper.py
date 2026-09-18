"""
Tests unitarios para el Filtro Sensible al Costo Microestructural (CostSensitiveGatekeeper).
"""

import pytest
from src.execution.cost_sensitive_gatekeeper import (
    CostSensitiveGatekeeper,
    TradeFrictionEstimate,
    GatekeeperDecision
)


@pytest.fixture
def gatekeeper():
    return CostSensitiveGatekeeper(hurdle_multiplier=2.5)


def test_approve_trade_with_strong_edge(gatekeeper):
    """Operación con convicción alta y barreras asimétricas favorables debe aprobarse."""
    decision = gatekeeper.evaluate_trade_utility(
        predicted_p_tp=0.75,
        target_tp_points=50.0,
        target_sl_points=25.0,
        current_spread_points=1.0,
        volatility_atr_points=4.0,
        expected_hold_days=1.5,
        daily_swap_points=0.2
    )

    assert decision.is_approved is True
    assert decision.status == "FILLED"
    assert decision.hurdle_ratio >= 2.5
    assert decision.net_expected_utility > 0
    assert "APROBADO" in decision.reason


def test_reject_trade_eaten_by_spread_and_slippage(gatekeeper):
    """Operación marginal con spread amplio y alta volatilidad debe rechazarse por fricción."""
    decision = gatekeeper.evaluate_trade_utility(
        predicted_p_tp=0.53,
        target_tp_points=12.0,
        target_sl_points=12.0,
        current_spread_points=3.5,
        volatility_atr_points=15.0,
        expected_hold_days=3.0,
        daily_swap_points=0.5
    )

    assert decision.is_approved is False
    assert decision.status == "REJECTED_BY_FRICTION"
    assert decision.hurdle_ratio < 2.5
    assert "RECHAZADO_POR_FRICCION" in decision.reason


def test_volume_spike_increases_estimated_slippage(gatekeeper):
    """Un volumen anormalmente alto (liquidez tensionada) debe incrementar el slippage estimado."""
    friction_normal = gatekeeper.estimate_friction(
        spread_points=2.0,
        volatility_atr_points=10.0,
        current_volume=100.0,
        average_volume=100.0
    )

    friction_spike = gatekeeper.estimate_friction(
        spread_points=2.0,
        volatility_atr_points=10.0,
        current_volume=400.0,  # 4x volumen promedio
        average_volume=100.0
    )

    assert friction_spike.estimated_slippage > friction_normal.estimated_slippage
    assert friction_spike.total_cost_points > friction_normal.total_cost_points


def test_zero_friction_handled_safely(gatekeeper):
    """Si la fricción es 0, no debe ocurrir ZeroDivisionError."""
    decision = gatekeeper.evaluate_trade_utility(
        predicted_p_tp=0.80,
        target_tp_points=20.0,
        target_sl_points=10.0,
        current_spread_points=0.0,
        volatility_atr_points=0.0,
        daily_swap_points=0.0
    )

    assert decision.is_approved is True
    assert decision.hurdle_ratio == 999.0
