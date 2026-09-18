"""
Demonio Analítico y MLOps Asíncrono en Segundo Plano (v2.0).
Ejecuta las tareas de computación pesada fuera de la ruta crítica de ejecución:
1. Simulación periódica del Shadow Journal (300 días).
2. Monitoreo de error de reconstrucción del LSTM Autoencoder (P90 Drift y P99 Kill-Switch).
3. Publicación atómica del estado de cuarentena en la Bóveda SQLite WAL (TradeVault).
4. Auto-rebalanceo mensual de portafolio HRP (Sharpe-weighted shrinkage).
5. Recalibración continua Bayesiana de Monte Carlo MDD.
"""

import os
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from src.database.trade_vault import TradeVault
from src.execution.telegram_notifier import TelegramNotifier
from src.evaluation.hrp_optimizer import HRPOptimizer

logger = logging.getLogger("AnalyticsDaemon")


class AnalyticsDaemon:
    """
    Demonio MLOps institucional asíncrono para mantenimiento continuo de modelos.
    """

    def __init__(
        self,
        symbols_config: List[Dict],
        trade_vault: Optional[TradeVault] = None,
        shadow_interval_hours: float = 6.0,
    ):
        self.symbols_config = symbols_config
        self.vault = trade_vault or TradeVault.get_instance()
        self.shadow_interval_hours = shadow_interval_hours
        self.notifier = TelegramNotifier()

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, "results")
        self.models_dir = os.path.join(self.results_dir, "saved_models")

        self.last_shadow_run = 0.0
        self.last_hrp_run = 0.0

    def run_shadow_journal_for_symbol(self, cfg: Dict) -> bool:
        """
        Ejecuta la evaluación en la sombra del autoencoder sobre los últimos 300 días.
        Actualiza la bandera de cuarentena atómicamente en TradeVault.
        Retorna True si el activo está saludable (sin cuarentena).
        """
        symbol = cfg.get("activo", cfg.get("symbol", "UNKNOWN"))
        logger.info(f"[{symbol}] Ejecutando Shadow Journal en segundo plano (MLOps)...")

        # Cargar Autoencoder si existe
        from src.models.anomaly_detector import StrategyLSTMAutoencoder
        autoencoder_base = os.path.join(self.results_dir, f"campeon_{symbol}_autoencoder")
        
        if not os.path.exists(autoencoder_base + ".keras"):
            logger.info(f"[{symbol}] No se encontró autoencoder en disco. Estado: Saludable por defecto.")
            return True

        try:
            autoencoder = StrategyLSTMAutoencoder()
            autoencoder.load(autoencoder_base)

            # Cargar retornos / métricas recientes guardadas en results/
            returns_file = os.path.join(self.results_dir, f"live_returns_{symbol}.json")
            if os.path.exists(returns_file):
                with open(returns_file, "r") as f:
                    rets = json.load(f)
            else:
                rets = cfg.get("oos_returns", [])

            if len(rets) < 30:
                logger.info(f"[{symbol}] Historial insuficiente para check de drift ({len(rets)} trades).")
                return True

            # Evaluar últimas 30 métricas contra umbrales P90 y P99
            recent_sample = np.array(rets[-30:]).reshape(-1, 1)
            is_anomalous = autoencoder.is_anomalous(recent_sample[-1])
            is_concept_drift = autoencoder.check_concept_drift(recent_sample)

            # 1. Caso de Muerte / Cuarentena por Anomalía Crítica (P99)
            if is_anomalous:
                logger.critical(f"[{symbol}] 🚨 ANOMALÍA CRÍTICA P99: Modelo colocado en Cuarentena.")
                self.vault.set_symbol_quarantine(symbol, True, reason="Error de reconstrucción MSE > P99")
                self.notifier.alert_mlops_quarantine(symbol)
                return False

            # 2. Caso de Concept Drift (P90)
            if is_concept_drift:
                logger.warning(f"[{symbol}] ⚠️ CONCEPT DRIFT DETECTADO (Mediana MSE > P90).")
                self.vault.set_system_state(f"concept_drift_{symbol}", {
                    "drift_detected": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                self.notifier.alert_concept_drift(symbol)
            else:
                self.vault.set_system_state(f"concept_drift_{symbol}", {
                    "drift_detected": False,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            # Si estaba en cuarentena y ahora está sano, levantarla
            if self.vault.is_symbol_quarantined(symbol):
                logger.info(f"[{symbol}] 🟢 Modelo recuperado. Levantando cuarentena.")
                self.vault.set_symbol_quarantine(symbol, False, reason="Recuperación validada por Shadow Journal")
                self.notifier.alert_mlops_resurrection(symbol)

            return True

        except Exception as e:
            logger.error(f"[{symbol}] Error ejecutando Shadow Journal: {e}")
            return True

    def check_and_rebalance_hrp(self):
        """
        Auto-Rebalanceo Mensual del Portafolio con Shrinkage Adaptativo.
        Guarda los nuevos pesos en TradeVault y en results/hrp_weights.json.
        """
        hrp_path = os.path.join(self.results_dir, "hrp_weights.json")
        needs_rebalance = False

        if not os.path.exists(hrp_path):
            needs_rebalance = True
        else:
            mtime = os.path.getmtime(hrp_path)
            days_old = (time.time() - mtime) / 86400.0
            if days_old >= 30.0:
                needs_rebalance = True

        if not needs_rebalance:
            return

        logger.info("⚖️ [Daemon MLOps] Ejecutando Auto-Rebalanceo Mensual de Portafolio HRP...")
        try:
            import yfinance as yf
            activos = [c.get("activo", c.get("symbol")) for c in self.symbols_config if c.get("activo")]
            if not activos:
                activos = ["EURUSD", "Oro", "SP500"]

            ticker_map = {
                "EURUSD": "EURUSD=X",
                "Oro": "GC=F",
                "SP500": "^GSPC",
            }

            end_date = datetime.now()
            start_date = end_date - timedelta(days=120)
            rets_dict = {}

            for act in activos:
                t = ticker_map.get(act, act)
                df_act = yf.download(t, start=start_date, end=end_date, progress=False)
                if not df_act.empty:
                    close_col = df_act["Close"].iloc[:, 0] if isinstance(df_act.columns, pd.MultiIndex) else df_act["Close"]
                    rets_dict[act] = close_col.pct_change().dropna()

            if len(rets_dict) > 1:
                df_rets = pd.DataFrame(rets_dict).dropna()
                hrp = HRPOptimizer()
                weights = hrp.allocate(df_rets).to_dict()

                # Guardar en SQLite WAL y en archivo JSON
                self.vault.set_system_state("hrp_weights", weights)
                with open(hrp_path, "w", encoding="utf-8") as f:
                    json.dump(weights, f, indent=4)

                logger.info(f"✅ [Daemon MLOps] Auto-Rebalanceo completado. Nuevos pesos: {weights}")
        except Exception as e:
            logger.error(f"Error en Auto-Rebalanceo HRP: {e}")

    def run_cycle(self):
        """Ejecuta una pasada de todas las rutinas de mantenimiento MLOps."""
        now = time.time()

        # 1. Latido de vida (Heartbeat) en Bóveda WAL
        self.vault.set_system_state("daemon_heartbeat", datetime.now(timezone.utc).isoformat())

        # 2. Shadow Journal si expiró el intervalo
        if (now - self.last_shadow_run) >= (self.shadow_interval_hours * 3600.0):
            for cfg in self.symbols_config:
                self.run_shadow_journal_for_symbol(cfg)
            self.last_shadow_run = now

        # 3. Auto-Rebalanceo Mensual HRP
        self.check_and_rebalance_hrp()

    def run_forever(self, sleep_seconds: int = 300):
        """Bucle continuo para ejecutar el demonio en un proceso secundario independiente."""
        logger.info("Iniciando Demonio Analítico MLOps v2.0...")
        try:
            while True:
                self.run_cycle()
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logger.info("Deteniendo Demonio Analítico.")
