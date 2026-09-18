"""
Tests unitarios para la Bóveda Transaccional SQLite WAL (TradeVault).
Verifica:
- Inicialización en modo WAL y durabilidad ACID.
- Inserción y consulta de bitácoras de auditoría de ejecución.
- Almacenamiento atómico de variables y banderas de estado (cuarentena, HRP, etc.).
- Robustez ante escrituras concurrentes desde múltiples hilos.
"""

import os
import tempfile
import threading
import pytest
import pandas as pd
from src.database.trade_vault import TradeVault


@pytest.fixture
def temp_vault():
    """Fixture que crea una instancia de TradeVault en archivo temporal."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name

    vault = TradeVault(temp_path)
    yield vault
    vault.close()
    for ext in ["", "-wal", "-shm"]:
        p = temp_path + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def test_vault_initialization_and_wal_mode(temp_vault):
    """Verifica que la base de datos se inicialice con journal_mode=wal."""
    conn = temp_vault._get_connection()
    cursor = conn.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    assert mode.lower() == "wal", f"Se esperaba modo WAL pero se obtuvo: {mode}"


def test_log_execution_audit(temp_vault):
    """Verifica el registro completo de una decisión de auditoría de trading."""
    audit_id = temp_vault.log_execution_audit(
        symbol="EURUSD",
        timeframe="D1",
        signal_direction="BUY",
        model_name="XGBOOST_PRECIO_PURO",
        prediction_prob=0.72,
        hmm_regime_state=0,
        autoencoder_mse=0.0012,
        expected_utility=15.4,
        estimated_friction=4.2,
        spread_points=1.2,
        slippage_points=0.5,
        swap_points=0.2,
        hurdle_ratio=3.66,
        kelly_multiplier=1.0,
        lot_size=0.15,
        entry_price=1.08500,
        tp_price=1.09200,
        sl_price=1.08100,
        status="FILLED",
        reason="APROBADO: Utilidad Neta supera con creces el umbral",
        ticket_id=12345678,
        raw_metadata={"vix": 14.5, "test": True}
    )

    assert audit_id is not None
    assert audit_id > 0

    # Consultar registros recientes
    df = temp_vault.get_recent_audits(symbol="EURUSD", limit=10)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["symbol"] == "EURUSD"
    assert row["status"] == "FILLED"
    assert row["ticket_id"] == 12345678
    assert abs(row["prediction_prob"] - 0.72) < 1e-4
    assert abs(row["hurdle_ratio"] - 3.66) < 1e-4


def test_system_state_storage_and_quarantine(temp_vault):
    """Verifica el almacenamiento atómico de estados y la bandera de cuarentena."""
    # Probar estado arbitrario
    temp_vault.set_system_state("hrp_weights", {"EURUSD": 0.40, "Oro": 0.35, "SP500": 0.25})
    weights = temp_vault.get_system_state("hrp_weights")
    assert isinstance(weights, dict)
    assert weights["EURUSD"] == 0.40

    # Probar cuarentena
    assert not temp_vault.is_symbol_quarantined("EURUSD")
    temp_vault.set_symbol_quarantine("EURUSD", True, reason="Anomalía P99 detectada por Autoencoder")
    assert temp_vault.is_symbol_quarantined("EURUSD")

    # Levantar cuarentena
    temp_vault.set_symbol_quarantine("EURUSD", False, reason="Recalibración exitosa")
    assert not temp_vault.is_symbol_quarantined("EURUSD")


def test_concurrent_multithreaded_writes(temp_vault):
    """Verifica que múltiples hilos puedan escribir auditorías en SQLite WAL sin error de bloqueo."""
    num_threads = 5
    records_per_thread = 20
    errors = []

    def worker(worker_id):
        try:
            for i in range(records_per_thread):
                temp_vault.log_execution_audit(
                    symbol=f"ASSET_{worker_id}",
                    timeframe="H1",
                    signal_direction="CASH",
                    model_name="TEST_MODEL",
                    prediction_prob=0.50,
                    hmm_regime_state=2,
                    autoencoder_mse=0.005,
                    expected_utility=0.0,
                    estimated_friction=1.0,
                    spread_points=1.0,
                    slippage_points=0.0,
                    swap_points=0.0,
                    hurdle_ratio=0.0,
                    kelly_multiplier=0.0,
                    lot_size=0.0,
                    entry_price=100.0,
                    tp_price=0.0,
                    sl_price=0.0,
                    status="CASH",
                    reason=f"Worker {worker_id} pass {i}"
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Ocurrieron errores durante escritura concurrente: {errors}"
    df = temp_vault.get_recent_audits(limit=num_threads * records_per_thread + 10)
    assert len(df) == num_threads * records_per_thread
