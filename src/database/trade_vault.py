"""
Bóveda Transaccional ACID In-Process (SQLite WAL).
Reemplaza los archivos planos (.json, .csv) de la v1.0, garantizando:
- Modo Write-Ahead Logging (WAL) para lecturas concurrentes sin bloqueo.
- Auditabilidad financiera estricta de cada tick/decisión (execution_audit_log).
- Persistencia de estados operacionales y banderas de riesgo (system_state).
- Cero latencia de red (proceso embebido).
"""

import os
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class TradeVault:
    """
    Gestor de base de datos transaccional SQLite WAL in-process.
    Totalmente thread-safe y optimizado para entornos de trading institucional.
    """

    _instances: Dict[str, "TradeVault"] = {}
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            results_dir = os.path.join(base_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            db_path = os.path.join(results_dir, "trading_vault.db")

        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> "TradeVault":
        """Patrón singleton por ruta de base de datos."""
        with cls._lock:
            key = db_path or "default"
            if key not in cls._instances:
                cls._instances[key] = cls(db_path)
            return cls._instances[key]

    def _get_connection(self) -> sqlite3.Connection:
        """Obtiene o crea una conexión por hilo con timeout seguro."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Activar Write-Ahead Logging y sincro normal para alto rendimiento y durabilidad
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        """Inicializa los esquemas de tablas transaccionales."""
        conn = self._get_connection()
        with conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS execution_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id INTEGER,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                signal_direction TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prediction_prob REAL NOT NULL,
                hmm_regime_state INTEGER NOT NULL,
                autoencoder_mse REAL NOT NULL,
                expected_utility REAL NOT NULL,
                estimated_friction REAL NOT NULL,
                spread_points REAL NOT NULL,
                slippage_points REAL NOT NULL,
                swap_points REAL NOT NULL,
                hurdle_ratio REAL NOT NULL,
                kelly_multiplier REAL NOT NULL,
                lot_size REAL NOT NULL,
                entry_price REAL NOT NULL,
                tp_price REAL NOT NULL,
                sl_price REAL NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                raw_metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_audit_symbol_time 
            ON execution_audit_log(symbol, timestamp_utc);

            CREATE INDEX IF NOT EXISTS idx_audit_status 
            ON execution_audit_log(status);

            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id INTEGER UNIQUE,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                lots REAL NOT NULL,
                open_time_utc TEXT NOT NULL,
                open_price REAL NOT NULL,
                close_time_utc TEXT,
                close_price REAL,
                profit_usd REAL,
                profit_pips REAL,
                exit_reason TEXT
            );
            """)

    def log_execution_audit(
        self,
        symbol: str,
        timeframe: str,
        signal_direction: str,
        model_name: str,
        prediction_prob: float,
        hmm_regime_state: int,
        autoencoder_mse: float,
        expected_utility: float,
        estimated_friction: float,
        spread_points: float,
        slippage_points: float,
        swap_points: float,
        hurdle_ratio: float,
        kelly_multiplier: float,
        lot_size: float,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        status: str,
        reason: Optional[str] = None,
        ticket_id: Optional[int] = None,
        raw_metadata: Optional[Dict[str, Any]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> int:
        """
        Registra una decisión de trading completa en la bitácora transaccional.
        Retorna el id del registro creado.
        """
        if timestamp_utc is None:
            timestamp_utc = datetime.now(timezone.utc).isoformat()

        meta_json = json.dumps(raw_metadata or {})
        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO execution_audit_log (
                    ticket_id, symbol, timeframe, timestamp_utc, signal_direction,
                    model_name, prediction_prob, hmm_regime_state, autoencoder_mse,
                    expected_utility, estimated_friction, spread_points, slippage_points,
                    swap_points, hurdle_ratio, kelly_multiplier, lot_size,
                    entry_price, tp_price, sl_price, status, reason, raw_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id, symbol, timeframe, timestamp_utc, signal_direction,
                    model_name, float(prediction_prob), int(hmm_regime_state),
                    float(autoencoder_mse), float(expected_utility),
                    float(estimated_friction), float(spread_points),
                    float(slippage_points), float(swap_points), float(hurdle_ratio),
                    float(kelly_multiplier), float(lot_size), float(entry_price),
                    float(tp_price), float(sl_price), status, reason, meta_json
                ),
            )
            return cursor.lastrowid

    def set_system_state(self, key: str, value: Any):
        """Guarda un estado de sistema serializado en JSON de forma atómica."""
        val_str = json.dumps(value) if not isinstance(value, str) else value
        now_utc = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO system_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, val_str, now_utc),
            )

    def get_system_state(self, key: str, default: Any = None) -> Any:
        """Recupera un estado del sistema deserializando JSON si es posible."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT value FROM system_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
            return default
        raw = row["value"]
        try:
            return json.loads(raw)
        except Exception:
            return raw

    def get_recent_audits(self, symbol: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """Retorna las auditorías más recientes como DataFrame de pandas."""
        conn = self._get_connection()
        if symbol:
            query = "SELECT * FROM execution_audit_log WHERE symbol = ? ORDER BY id DESC LIMIT ?"
            params = (symbol, limit)
        else:
            query = "SELECT * FROM execution_audit_log ORDER BY id DESC LIMIT ?"
            params = (limit,)
        return pd.read_sql_query(query, conn, params=params)

    def is_symbol_quarantined(self, symbol: str) -> bool:
        """Verifica en O(1) si el símbolo se encuentra en cuarentena MLOps."""
        key = f"quarantine_{symbol}"
        state = self.get_system_state(key, False)
        if isinstance(state, dict):
            return bool(state.get("in_quarantine", False))
        return bool(state)

    def set_symbol_quarantine(self, symbol: str, in_quarantine: bool, reason: str = ""):
        """Establece el estado de cuarentena de un activo."""
        key = f"quarantine_{symbol}"
        self.set_system_state(key, {
            "in_quarantine": in_quarantine,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def close(self):
        """Cierra la conexión del hilo actual si existe."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None
