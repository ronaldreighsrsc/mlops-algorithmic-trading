"""
Filtro de Decisión Microestructural Sensible al Costo (Cost-Sensitive Gatekeeper).
Evalúa si la Utilidad Neta Esperada supera holgadamente el costo total de fricción:
- Spread dinámico en tiempo real
- Deslizamiento estocástico (Slippage) dependiente de volatilidad y liquidez
- Costo de financiamiento overnight (Swap acumulado por horizonte temporal)
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("CostGatekeeper")


@dataclass
class TradeFrictionEstimate:
    spread_points: float
    estimated_slippage: float
    expected_swap: float
    total_cost_points: float


@dataclass
class GatekeeperDecision:
    is_approved: bool
    status: str
    reason: str
    net_expected_utility: float
    gross_expected_return: float
    hurdle_ratio: float
    friction: TradeFrictionEstimate


class CostSensitiveGatekeeper:
    """
    Compuerta de Riesgo Microestructural Institucional.
    Aplica una tasa de rentabilidad mínima (Hurdle Rate) estricta antes de autorizar órdenes a MT5.
    """

    def __init__(self, hurdle_multiplier: float = 2.5):
        """
        :param hurdle_multiplier: Razón mínima exigida entre retorno esperado bruto y fricción total (ej. 2.5x).
        """
        self.hurdle_multiplier = hurdle_multiplier

    def estimate_friction(
        self,
        spread_points: float,
        volatility_atr_points: float,
        current_volume: float = 1.0,
        average_volume: float = 1.0,
        daily_swap_points: float = 0.5,
        expected_hold_days: float = 2.0,
    ) -> TradeFrictionEstimate:
        """
        Calcula la descomposición estocástica del costo de fricción.
        """
        spread = max(0.0, float(spread_points))
        atr = max(0.0, float(volatility_atr_points))
        
        vol_ratio = 1.0
        if average_volume > 0 and current_volume > 0:
            vol_ratio = math.sqrt(current_volume / average_volume)

        # Slippage: componente de spread + componente de impacto ATR y volumen relativo
        estimated_slippage = (0.20 * spread) + (0.05 * atr * vol_ratio)
        expected_swap = max(0.0, float(daily_swap_points)) * max(0.0, float(expected_hold_days))
        total_cost = spread + estimated_slippage + expected_swap

        return TradeFrictionEstimate(
            spread_points=round(spread, 5),
            estimated_slippage=round(estimated_slippage, 5),
            expected_swap=round(expected_swap, 5),
            total_cost_points=round(total_cost, 5),
        )

    def evaluate_trade_utility(
        self,
        predicted_p_tp: float,
        target_tp_points: float,
        target_sl_points: float,
        current_spread_points: float,
        volatility_atr_points: float,
        expected_hold_days: float = 2.0,
        daily_swap_points: float = 0.5,
        current_volume: float = 1.0,
        average_volume: float = 1.0,
    ) -> GatekeeperDecision:
        """
        Calcula la Utilidad Neta Esperada E[U] y decide si se aprueba el envío de la orden.
        """
        p_tp = max(0.0, min(1.0, float(predicted_p_tp)))
        p_sl = 1.0 - p_tp
        r_tp = max(0.0, float(target_tp_points))
        r_sl = max(0.0, float(target_sl_points))

        # 1. Estimar fricción
        friction = self.estimate_friction(
            spread_points=current_spread_points,
            volatility_atr_points=volatility_atr_points,
            current_volume=current_volume,
            average_volume=average_volume,
            daily_swap_points=daily_swap_points,
            expected_hold_days=expected_hold_days,
        )

        # 2. Retorno Bruto Ponderado por Probabilidades
        gross_expected_return = (p_tp * r_tp) - (p_sl * r_sl)

        # 3. Utilidad Neta
        net_expected_utility = gross_expected_return - friction.total_cost_points

        # 4. Hurdle Rate Ratio
        if friction.total_cost_points > 0:
            hurdle_ratio = gross_expected_return / friction.total_cost_points
        else:
            hurdle_ratio = 999.0 if gross_expected_return > 0 else 0.0

        required_edge = self.hurdle_multiplier * friction.total_cost_points

        # 5. Criterio de Decisión
        if gross_expected_return < required_edge or hurdle_ratio < self.hurdle_multiplier:
            reason = (
                f"RECHAZADO_POR_FRICCION: Edge bruto ({gross_expected_return:.2f} pts | ratio {hurdle_ratio:.2f}x) "
                f"< Umbral exigido ({required_edge:.2f} pts | {self.hurdle_multiplier:.1f}x de fricción: {friction.total_cost_points:.2f} pts)"
            )
            return GatekeeperDecision(
                is_approved=False,
                status="REJECTED_BY_FRICTION",
                reason=reason,
                net_expected_utility=round(net_expected_utility, 4),
                gross_expected_return=round(gross_expected_return, 4),
                hurdle_ratio=round(hurdle_ratio, 2),
                friction=friction,
            )

        reason = (
            f"APROBADO: Utilidad Neta Esperada {net_expected_utility:.2f} pts "
            f"(Edge bruto: {gross_expected_return:.2f} pts, Hurdle ratio: {hurdle_ratio:.2f}x >= {self.hurdle_multiplier:.1f}x)"
        )
        return GatekeeperDecision(
            is_approved=True,
            status="FILLED",
            reason=reason,
            net_expected_utility=round(net_expected_utility, 4),
            gross_expected_return=round(gross_expected_return, 4),
            hurdle_ratio=round(hurdle_ratio, 2),
            friction=friction,
        )
