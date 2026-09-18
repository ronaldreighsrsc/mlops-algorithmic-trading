"""
Tests de Integración y Arquitectura Desacoplada v2.0 (Execution Gateway vs. Analytics Daemon).
"""

import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime, timezone, timedelta

from src.database.trade_vault import TradeVault
from src.execution.execution_gateway import ExecutionGateway
from src.execution.analytics_daemon import AnalyticsDaemon
from src.macro.macro_rag_agent import MacroHazardGuard
from src.macro.economic_calendar import EconomicCalendarProvider
from src.execution.cost_sensitive_gatekeeper import CostSensitiveGatekeeper


@pytest.fixture
def test_environment():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    vault = TradeVault(db_path)
    connector = MagicMock()
    connector.connected = True

    macro_guard = MacroHazardGuard(pre_event_freeze_minutes=30, post_event_cooldown_minutes=15)
    calendar_provider = EconomicCalendarProvider()
    gatekeeper = CostSensitiveGatekeeper(hurdle_multiplier=2.5)

    config = {
        "activo": "EURUSD",
        "timeframe": 1440,
        "model_type": "XGBOOST",
        "model_file": "test_model.pkl",
        "features": ["open_FFD", "high_FFD"],
        "confidence_threshold": 0.50,
        "k_up": 2.0,
        "k_down": 1.5,
    }

    gateway = ExecutionGateway(
        symbol="EURUSD",
        timeframe=1440,
        config=config,
        connector=connector,
        trade_vault=vault,
        macro_guard=macro_guard,
        calendar_provider=calendar_provider,
        gatekeeper=gatekeeper,
    )

    yield vault, connector, gateway, calendar_provider

    vault.close()
    for ext in ["", "-wal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def test_gateway_aborts_immediately_on_quarantine(test_environment):
    """Si el activo está en cuarentena en TradeVault, el Gateway aborta O(1) de inmediato."""
    vault, connector, gateway, _ = test_environment

    # Activar cuarentena en la bóveda
    vault.set_symbol_quarantine("EURUSD", True, reason="Anomalía crítica detectada por MLOps")

    with patch("MetaTrader5.copy_rates_from_pos") as mock_rates:
        mock_rates.return_value = [{"time": 1000}, {"time": 2000}]
        res = gateway.process_tick_event()

    assert res["status"] == "IN_QUARANTINE"
    assert res["latency_ms"] < 20.0  # Verificación rápida en RAM/WAL


def test_gateway_aborts_on_pre_news_freeze(test_environment):
    """Si hay una noticia crítica (FOMC) en 10 minutos, el Gateway rechaza la orden por macro riesgo."""
    vault, connector, gateway, calendar = test_environment

    # Agregar evento FOMC a 10 min
    t_fomc = datetime.now(timezone.utc) + timedelta(minutes=10)
    calendar.add_event("FOMC Rate Decision", "USD", t_fomc, impact="HIGH")

    with patch("MetaTrader5.copy_rates_from_pos") as mock_rates, \
         patch.object(gateway.engine, "has_open_positions", return_value=False):
        mock_rates.return_value = [{"time": 1000}, {"time": 2000}]
        res = gateway.process_tick_event()

    assert res["status"] == "REJECTED_BY_MACRO"
    assert "FREEZE_PRE_NEWS" in res["reason"]

    # Verificar que el rechazo quedó auditado en TradeVault
    audits = vault.get_recent_audits(symbol="EURUSD", limit=5)
    assert len(audits) >= 1
    assert audits.iloc[0]["status"] == "REJECTED_BY_MACRO"


def test_analytics_daemon_run_cycle_and_heartbeat(test_environment):
    """El daemon analítico debe registrar su latido de vida (heartbeat) y monitorear activos."""
    vault, _, _, _ = test_environment
    cfg = [{"activo": "EURUSD", "timeframe": "D1", "oos_returns": [0.01] * 20}]

    daemon = AnalyticsDaemon(symbols_config=cfg, trade_vault=vault)
    daemon.run_cycle()

    heartbeat = vault.get_system_state("daemon_heartbeat")
    assert heartbeat is not None
    assert len(heartbeat) > 10
