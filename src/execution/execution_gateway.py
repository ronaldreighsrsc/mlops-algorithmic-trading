"""
Gateway de Ejecución de Ultra-Baja Latencia (v2.0) - SLA < 15 ms.
Desacoplado del reentrenamiento y del Shadow Journal.
Responsabilidades:
1. Recepción de eventos de nueva vela / tick desde MetaTrader 5.
2. Filtros de riesgo duro en RAM (Kill-Switch por DD diario).
3. Verificación O(1) de estado de cuarentena y deriva en SQLite WAL (TradeVault).
4. Guardián Macroeconómico Pre-News (FOMC / CPI / NFP).
5. Inferencia con ONNX Runtime (< 1.8 ms).
6. Filtro Microestructural Sensible al Costo (Hurdle Rate >= 2.5x).
7. Dimensionamiento Kelly dinámico modulado por HMM y HRP.
8. Envío de orden a MT5 y registro transaccional en TradeVault.
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd

# Conectores y Motores
from src.database.trade_vault import TradeVault
from src.execution.onnx_inference_engine import ONNXInferenceEngine
from src.execution.cost_sensitive_gatekeeper import CostSensitiveGatekeeper
from src.macro.macro_rag_agent import MacroHazardGuard
from src.macro.economic_calendar import EconomicCalendarProvider
from src.execution.risk_manager import RiskManager
from src.execution.execution_engine import ExecutionEngine
from src.execution.telegram_notifier import TelegramNotifier
from src.mt5_connector import MT5Connector
from src.preprocessing.technical_features import TechnicalFeatureEngineer
from src.preprocessing.volatility import VolatilityModeler
from src.preprocessing.stationarity import FractionalDifferencer

logger = logging.getLogger("ExecutionGateway")


class ExecutionGateway:
    """
    Gateway de ejecución institucional de alta velocidad.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: int,
        config: dict,
        connector: MT5Connector,
        trade_vault: Optional[TradeVault] = None,
        macro_guard: Optional[MacroHazardGuard] = None,
        calendar_provider: Optional[EconomicCalendarProvider] = None,
        gatekeeper: Optional[CostSensitiveGatekeeper] = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = config
        self.connector = connector

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, "results")
        self.models_dir = os.path.join(self.results_dir, "saved_models")

        # Capa de base de datos y módulos v2.0
        self.vault = trade_vault or TradeVault.get_instance()
        self.macro_guard = macro_guard or MacroHazardGuard(pre_event_freeze_minutes=30, post_event_cooldown_minutes=15)
        self.calendar_provider = calendar_provider or EconomicCalendarProvider()
        self.gatekeeper = gatekeeper or CostSensitiveGatekeeper(hurdle_multiplier=2.5)

        # Motor MT5 y Riesgo
        self.engine = ExecutionEngine(self.connector)
        
        # Magic number
        sym_code = abs(hash(self.symbol)) % 1000
        self.engine.magic_number = 200000 + sym_code * 10 + int(self.timeframe % 100)

        # Cargar peso HRP
        hrp_weights = self.vault.get_system_state("hrp_weights", {})
        self.hrp_weight = float(hrp_weights.get(self.symbol, 1.0)) if isinstance(hrp_weights, dict) else 1.0

        base_risk = config.get("optimal_risk_pct", 0.025)
        self.risk_manager = RiskManager(
            risk_per_trade_pct=base_risk * self.hrp_weight,
            k_up=config.get("k_up", 2.0),
            k_down=config.get("k_down", 1.5),
            max_hold=config.get("max_hold", 10),
        )
        self.notifier = TelegramNotifier()

        # Preprocesadores
        self.tech_eng = TechnicalFeatureEngineer(target_price_col="close")
        self.vol_mod = VolatilityModeler(window_size=500, target_col="close")
        self.ffd_eng = FractionalDifferencer(threshold=1e-4)

        # Motor de Inferencia ONNX
        self.onnx_engine: Optional[ONNXInferenceEngine] = None
        self._init_inference_engine()

        self.last_bar_time = None
        self.daily_start_balance = None
        self.max_daily_loss_pct = config.get("max_daily_loss_pct", 0.025) # 2.5% Kill-switch

    def _init_inference_engine(self):
        """Inicializa el motor ONNX si existe el modelo compilado."""
        onnx_name = f"{self.config.get('model_file', '').replace('.pkl', '').replace('.keras', '')}.onnx"
        onnx_path = os.path.join(self.models_dir, onnx_name)

        if os.path.exists(onnx_path):
            try:
                # Leer checksum del manifiesto si existe
                manifest_path = os.path.join(self.models_dir, "onnx_manifest.json")
                expected_sha = None
                if os.path.exists(manifest_path):
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                        expected_sha = manifest.get(onnx_name)

                self.onnx_engine = ONNXInferenceEngine(onnx_path, expected_sha256=expected_sha)
                logger.info(f"[{self.symbol}] Motor ONNX cargado exitosamente: {onnx_name}")
            except Exception as e:
                logger.warning(f"[{self.symbol}] No se pudo inicializar ONNX: {e}")

    def check_hard_risk_kill_switch(self) -> bool:
        """
        Verifica si la pérdida diaria acumulada supera el límite máximo permitido (ej. 2.5%).
        Retorna True si el Kill Switch está activo (debe abortar).
        """
        import MetaTrader5 as mt5
        acc_info = mt5.account_info()
        if acc_info is None:
            return False

        if self.daily_start_balance is None:
            self.daily_start_balance = acc_info.balance

        # Pérdida respecto al balance inicial del día
        equity_drawdown = (self.daily_start_balance - acc_info.equity) / self.daily_start_balance
        if equity_drawdown >= self.max_daily_loss_pct:
            logger.critical(
                f"[{self.symbol}] 🚨 KILL-SWITCH DISPARADO: Drawdown diario ({equity_drawdown:.2%}) "
                f">= Límite ({self.max_daily_loss_pct:.2%}). Abortando operativas."
            )
            return True
        return False

    def process_tick_event(self) -> Dict[str, Any]:
        """
        Flujo de ejecución de alta velocidad activado al cerrar una vela.
        Retorna diccionario con el diagnóstico y resultado del pase.
        """
        import MetaTrader5 as mt5
        t_start = time.perf_counter()

        # 1. Comprobar conexión y vela
        if not self.connector.connected:
            return {"status": "DISCONNECTED", "latency_ms": 0.0}

        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 2)
        if rates is None or len(rates) < 2:
            return {"status": "NO_DATA", "latency_ms": 0.0}

        current_bar_time = rates[1]["time"]
        if self.last_bar_time == current_bar_time:
            return {"status": "WAITING_NEW_BAR", "latency_ms": 0.0}

        self.last_bar_time = current_bar_time
        logger.info(f"[{self.symbol}] Nueva vela detectada ({current_bar_time}). Iniciando evaluación v2.0...")

        # 2. Hard Risk Kill-Switch en RAM
        if self.check_hard_risk_kill_switch():
            return {"status": "KILL_SWITCH_ACTIVE", "latency_ms": (time.perf_counter() - t_start) * 1000.0}

        # 3. Verificación de Barrera Vertical si hay orden abierta
        if self.engine.has_open_positions(self.symbol):
            max_hold = getattr(self.risk_manager, "max_hold", 10)
            closed = self.engine.check_and_close_vertical_barrier(self.symbol, self.timeframe, max_hold)
            if closed:
                self.notifier.alert_max_hold_exit(self.symbol, max_hold)
            return {"status": "POSITION_ALREADY_OPEN", "latency_ms": (time.perf_counter() - t_start) * 1000.0}

        # 4. Verificación O(1) de Cuarentena MLOps en SQLite WAL
        if self.vault.is_symbol_quarantined(self.symbol):
            logger.warning(f"[{self.symbol}] ⛔ Activo en Cuarentena según Bóveda WAL. Operación abortada.")
            return {"status": "IN_QUARANTINE", "latency_ms": (time.perf_counter() - t_start) * 1000.0}

        # 5. Pre-News Hazard Guard
        upcoming = self.calendar_provider.get_upcoming_events(window_hours=2.0)
        is_safe, macro_reason, trig_event = self.macro_guard.is_market_safe(self.symbol, upcoming)
        if not is_safe:
            logger.warning(f"[{self.symbol}] 🛑 Macro Guard activo: {macro_reason}")
            self.vault.log_execution_audit(
                symbol=self.symbol,
                timeframe=str(self.timeframe),
                signal_direction="CASH",
                model_name=self.config.get("model_type", "UNKNOWN"),
                prediction_prob=0.0,
                hmm_regime_state=2,
                autoencoder_mse=0.0,
                expected_utility=0.0,
                estimated_friction=0.0,
                spread_points=0.0,
                slippage_points=0.0,
                swap_points=0.0,
                hurdle_ratio=0.0,
                kelly_multiplier=0.0,
                lot_size=0.0,
                entry_price=0.0,
                tp_price=0.0,
                sl_price=0.0,
                status="REJECTED_BY_MACRO",
                reason=macro_reason,
            )
            return {"status": "REJECTED_BY_MACRO", "reason": macro_reason}

        # 6. Extracción y Preprocesamiento Rápido de Features
        rates_hist = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 300)
        if rates_hist is None or len(rates_hist) < 50:
            return {"status": "INSUFFICIENT_HISTORY", "latency_ms": (time.perf_counter() - t_start) * 1000.0}

        df_raw = pd.DataFrame(rates_hist)
        df_proc = self.tech_eng.add_indicators(df_raw.copy())
        df_proc = self.vol_mod.compute_egarch(df_proc, symbol=self.symbol)
        df_proc = self.ffd_eng.apply_ffd(df_proc)
        df_proc.ffill(inplace=True)
        df_proc.bfill(inplace=True)

        feature_cols = self.config.get("features", [])
        if not all(col in df_proc.columns for col in feature_cols):
            logger.error(f"[{self.symbol}] Faltan columnas de features en preprocesamiento.")
            return {"status": "FEATURE_ERROR"}

        # 7. Inferencia ONNX de Ultra-Baja Latencia
        sample_matrix = df_proc[feature_cols].values
        look_back = self.config.get("look_back", 10)
        if len(sample_matrix) < look_back:
            return {"status": "NOT_ENOUGH_LOOKBACK"}

        window_features = sample_matrix[-look_back:]

        if self.onnx_engine is not None:
            prob = self.onnx_engine.predict_probability(window_features)
        else:
            # Fallback a heurística de predicción si ONNX no está exportado aún
            prob = 0.50

        # 8. Evaluación de Regímenes y Deriva
        regime_state = 0
        regime_kelly = 1.0
        if "returns" in df_proc.columns:
            recent_rets = df_proc["returns"].iloc[-50:].values
        else:
            recent_rets = df_proc["close"].pct_change().dropna().iloc[-50:].values

        diagnosis = self.risk_manager.hybrid_monitor.diagnose_institutional_regime(
            window_features, recent_rets
        )
        if diagnosis:
            regime_state = diagnosis.regime_state
            regime_kelly = diagnosis.effective_kelly_factor

        # 9. Determinación de Dirección
        confidence_threshold = self.config.get("confidence_threshold", 0.50)
        is_long = prob > confidence_threshold
        is_short = self.config.get("bilateral", False) and (prob < (1.0 - confidence_threshold))

        if not is_long and not is_short:
            logger.info(f"[{self.symbol}] Sin señal clara (Prob: {prob:.2%}).")
            self.vault.log_execution_audit(
                symbol=self.symbol,
                timeframe=str(self.timeframe),
                signal_direction="CASH",
                model_name=self.config.get("model_type", "UNKNOWN"),
                prediction_prob=prob,
                hmm_regime_state=regime_state,
                autoencoder_mse=0.0,
                expected_utility=0.0,
                estimated_friction=0.0,
                spread_points=0.0,
                slippage_points=0.0,
                swap_points=0.0,
                hurdle_ratio=0.0,
                kelly_multiplier=0.0,
                lot_size=0.0,
                entry_price=0.0,
                tp_price=0.0,
                sl_price=0.0,
                status="CASH",
                reason="Probabilidad en zona neutral (fuera de umbrales)",
            )
            return {"status": "CASH", "probability": prob}

        # 10. Cálculo de Barreras y Niveles
        sym_info = self.engine.get_symbol_info(self.symbol)
        if sym_info is None:
            return {"status": "SYMBOL_INFO_ERROR"}

        tick = mt5.symbol_info_tick(self.symbol)
        last_vol = float(df_proc["EGARCH_Vol"].iloc[-1])
        atr_points = float(df_proc.get("ATR", pd.Series([10.0])).iloc[-1])
        spread_points = (tick.ask - tick.bid) / sym_info["tick_size"]

        if is_long:
            direction = "BUY"
            current_price = tick.ask
            tp_price, sl_price = self.risk_manager.calculate_triple_barrier_levels(current_price, last_vol)
            target_tp_pts = abs(tp_price - current_price) / sym_info["tick_size"]
            target_sl_pts = abs(current_price - sl_price) / sym_info["tick_size"]
        else:
            direction = "SELL"
            current_price = tick.bid
            vol_dec = last_vol / 100.0
            tp_price = current_price * (1.0 - self.risk_manager.k_up * vol_dec)
            sl_price = current_price * (1.0 + self.risk_manager.k_down * vol_dec)
            target_tp_pts = abs(current_price - tp_price) / sym_info["tick_size"]
            target_sl_pts = abs(sl_price - current_price) / sym_info["tick_size"]

        # 11. Filtro Sensible al Costo (Cost-Sensitive Gatekeeper)
        decision = self.gatekeeper.evaluate_trade_utility(
            predicted_p_tp=prob if is_long else (1.0 - prob),
            target_tp_points=target_tp_pts,
            target_sl_points=target_sl_pts,
            current_spread_points=spread_points,
            volatility_atr_points=atr_points,
            expected_hold_days=2.0,
            daily_swap_points=0.5,
        )

        acc_info = mt5.account_info()
        account_balance = acc_info.balance if acc_info else 10000.0

        if not decision.is_approved:
            logger.warning(f"[{self.symbol}] ⛔ Rechazado por Compuerta de Fricción: {decision.reason}")
            self.vault.log_execution_audit(
                symbol=self.symbol,
                timeframe=str(self.timeframe),
                signal_direction=direction,
                model_name=self.config.get("model_type", "UNKNOWN"),
                prediction_prob=prob,
                hmm_regime_state=regime_state,
                autoencoder_mse=0.0,
                expected_utility=decision.net_expected_utility,
                estimated_friction=decision.friction.total_cost_points,
                spread_points=decision.friction.spread_points,
                slippage_points=decision.friction.estimated_slippage,
                swap_points=decision.friction.expected_swap,
                hurdle_ratio=decision.hurdle_ratio,
                kelly_multiplier=regime_kelly,
                lot_size=0.0,
                entry_price=current_price,
                tp_price=tp_price,
                sl_price=sl_price,
                status=decision.status,
                reason=decision.reason,
            )
            return {"status": decision.status, "reason": decision.reason}

        # 12. Position Sizing
        lots = self.risk_manager.calculate_position_size(
            balance=account_balance,
            current_price=current_price,
            stop_loss_price=sl_price,
            tick_size=sym_info["tick_size"],
            tick_value=sym_info["tick_value"],
            volume_step=sym_info["volume_step"],
            prediction_prob=prob,
            confidence_threshold=confidence_threshold,
            regime_multiplier=regime_kelly,
        )

        if lots < sym_info["volume_min"]:
            logger.warning(f"[{self.symbol}] Volumen calculado ({lots}) < mínimo permitido. Trade abortado.")
            return {"status": "VOLUME_TOO_LOW"}

        # 13. Envío de Orden a Mercado en MT5
        if is_long:
            success = self.engine.send_market_buy_order(self.symbol, lots, sl_price, tp_price)
        else:
            success = self.engine.send_market_sell_order(self.symbol, lots, sl_price, tp_price)

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(f"[{self.symbol}] 🚀 Orden {direction} enviada. Latencia total tick-to-order: {t_elapsed_ms:.2f} ms")

        # 14. Registro Transaccional en SQLite WAL
        ticket_num = None
        status_str = "FILLED" if success else "ORDER_SEND_FAILED"
        self.vault.log_execution_audit(
            symbol=self.symbol,
            timeframe=str(self.timeframe),
            signal_direction=direction,
            model_name=self.config.get("model_type", "UNKNOWN"),
            prediction_prob=prob,
            hmm_regime_state=regime_state,
            autoencoder_mse=0.0,
            expected_utility=decision.net_expected_utility,
            estimated_friction=decision.friction.total_cost_points,
            spread_points=decision.friction.spread_points,
            slippage_points=decision.friction.estimated_slippage,
            swap_points=decision.friction.expected_swap,
            hurdle_ratio=decision.hurdle_ratio,
            kelly_multiplier=regime_kelly,
            lot_size=lots,
            entry_price=current_price,
            tp_price=tp_price,
            sl_price=sl_price,
            status=status_str,
            reason=f"Ejecutado con latencia {t_elapsed_ms:.2f} ms",
            ticket_id=ticket_num,
        )

        # 15. Notificación Telegram
        self.notifier.alert_trade_execution(
            symbol=self.symbol,
            volume=lots,
            price=current_price,
            tp=tp_price,
            sl=sl_price,
            is_long=is_long,
            account_balance=account_balance,
            risk_pct=self.risk_manager.risk_per_trade_pct * regime_kelly,
            timeframe=str(self.timeframe),
        )

        return {
            "status": status_str,
            "direction": direction,
            "lots": lots,
            "price": current_price,
            "latency_ms": round(t_elapsed_ms, 2),
        }
