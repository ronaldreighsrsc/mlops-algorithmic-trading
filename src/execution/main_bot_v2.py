"""
Orquestador Principal Quant Trading Bot v2.0 (Arquitectura Institucional Desacoplada).
Soporta tres modos de ejecución:
1. 'gateway': Proceso de ultra-baja latencia (< 15 ms tick-to-order) sin TensorFlow.
2. 'daemon': Trabajador analítico asíncrono para Shadow Journal, HRP y Autoencoder.
3. 'dual': Modo unificado con ejecución desacoplada mediante hilos de alta prioridad y SQLite WAL.
"""

import os
import sys

# Agregar carpeta raíz del proyecto y src al PYTHONPATH
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(src_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import glob
import json
import time
import signal
import logging
import argparse
import threading
from typing import List

from src.mt5_connector import MT5Connector
from src.database.trade_vault import TradeVault
from src.macro.macro_rag_agent import MacroHazardGuard
from src.macro.economic_calendar import EconomicCalendarProvider
from src.execution.cost_sensitive_gatekeeper import CostSensitiveGatekeeper
from src.execution.execution_gateway import ExecutionGateway
from src.execution.analytics_daemon import AnalyticsDaemon

# Configuración de Logging Institucional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("MainBotV2")


def load_champion_configs() -> List[dict]:
    """Carga todas las configuraciones de modelos campeones desde results/campeon_*.json."""
    results_dir = os.path.join(root_dir, "results")
    campeon_files = glob.glob(os.path.join(results_dir, "campeon_*.json"))
    configs = []

    for fpath in campeon_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                configs.append(cfg)
        except Exception as e:
            logger.error(f"Error cargando {fpath}: {e}")

    return configs


def run_gateway_loop(gateways: List[ExecutionGateway], interval_seconds: float = 5.0):
    """Bucle principal de ejecución del Gateway."""
    logger.info(f"⚡ [Gateway] Monitoreando {len(gateways)} activos a < 15 ms SLA (intervalo {interval_seconds}s)...")
    while True:
        for gw in gateways:
            try:
                gw.process_tick_event()
            except Exception as e:
                logger.error(f"[{gw.symbol}] Error en pase de tick: {e}")
        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Institutional Quant Trading Bot v2.0")
    parser.add_argument(
        "--mode",
        choices=["dual", "gateway", "daemon"],
        default="dual",
        help="Modo de ejecución: 'gateway' (SLA < 15ms), 'daemon' (MLOps asíncrono) o 'dual' (unificado desacoplado).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Intervalo de sondeo en segundos para el Gateway.",
    )
    args = parser.parse_args()

    logger.info("=================================================================")
    logger.info("  🚀 INSTITUTIONAL QUANT TRADING BOT v2.0 - ALPHAEDGE SENTINEL   ")
    logger.info(f"  Modo Operativo: {args.mode.upper()} | SLA: < 15 ms Tick-to-Order")
    logger.info("=================================================================")

    # 1. Bóveda Transaccional SQLite WAL
    vault = TradeVault.get_instance()
    logger.info(f"✅ Bóveda Transaccional SQLite WAL inicializada en: {vault.db_path}")

    # 2. Cargar configuraciones de campeones
    configs = load_champion_configs()
    if not configs:
        logger.warning("No se encontraron archivos campeon_*.json en results/. Creando configuración de contingencia.")
        configs = [
            {"activo": "EURUSD", "timeframe": "D1", "model_type": "XGBOOST", "features": ["open_FFD", "high_FFD", "low_FFD"], "confidence_threshold": 0.50},
            {"activo": "Oro", "timeframe": "D1", "model_type": "BILSTM", "features": ["open_FFD", "MACD_Hist", "RSI", "ATR"], "confidence_threshold": 0.50},
            {"activo": "SP500", "timeframe": "D1", "model_type": "BILSTM", "features": ["VIX_close", "DXY_close_FFD"], "confidence_threshold": 0.55},
        ]

    # 3. Conexión compartida a MetaTrader 5
    connector = MT5Connector()
    if args.mode in ["gateway", "dual"]:
        if not connector.connect():
            logger.critical("No se pudo conectar a MetaTrader 5. Abortando Gateway.")
            sys.exit(1)

    # 4. Instanciar módulos transversales
    macro_guard = MacroHazardGuard(pre_event_freeze_minutes=30, post_event_cooldown_minutes=15)
    calendar_provider = EconomicCalendarProvider()
    gatekeeper = CostSensitiveGatekeeper(hurdle_multiplier=2.5)

    # Manejo de señales de apagado elegante
    def signal_handler(sig, frame):
        logger.info("Recibida señal de terminación. Apagando componentes...")
        connector.shutdown()
        vault.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 5. Ejecución según el modo seleccionado
    if args.mode == "daemon":
        daemon = AnalyticsDaemon(symbols_config=configs, trade_vault=vault)
        daemon.run_forever(sleep_seconds=300)

    elif args.mode == "gateway":
        import MetaTrader5 as mt5
        tf_map = {"D1": mt5.TIMEFRAME_D1, "H4": mt5.TIMEFRAME_H4, "H1": mt5.TIMEFRAME_H1}
        gateways = []
        for cfg in configs:
            sym = cfg.get("activo", cfg.get("symbol", "EURUSD"))
            tf = tf_map.get(cfg.get("timeframe", "D1"), mt5.TIMEFRAME_D1)
            gw = ExecutionGateway(
                symbol=sym,
                timeframe=tf,
                config=cfg,
                connector=connector,
                trade_vault=vault,
                macro_guard=macro_guard,
                calendar_provider=calendar_provider,
                gatekeeper=gatekeeper,
            )
            gateways.append(gw)

        run_gateway_loop(gateways, interval_seconds=args.interval)

    elif args.mode == "dual":
        # Iniciar AnalyticsDaemon en hilo en segundo plano (no bloqueante)
        daemon = AnalyticsDaemon(symbols_config=configs, trade_vault=vault)
        daemon_thread = threading.Thread(target=daemon.run_forever, args=(300,), daemon=True)
        daemon_thread.start()
        logger.info("🧵 Hilo asíncrono de AnalyticsDaemon MLOps iniciado exitosamente.")

        # Iniciar ExecutionGateway en el hilo principal
        import MetaTrader5 as mt5
        tf_map = {"D1": mt5.TIMEFRAME_D1, "H4": mt5.TIMEFRAME_H4, "H1": mt5.TIMEFRAME_H1}
        gateways = []
        for cfg in configs:
            sym = cfg.get("activo", cfg.get("symbol", "EURUSD"))
            tf = tf_map.get(cfg.get("timeframe", "D1"), mt5.TIMEFRAME_D1)
            gw = ExecutionGateway(
                symbol=sym,
                timeframe=tf,
                config=cfg,
                connector=connector,
                trade_vault=vault,
                macro_guard=macro_guard,
                calendar_provider=calendar_provider,
                gatekeeper=gatekeeper,
            )
            gateways.append(gw)

        run_gateway_loop(gateways, interval_seconds=args.interval)


if __name__ == "__main__":
    main()
