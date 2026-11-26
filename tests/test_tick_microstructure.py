"""
Tests Slice B (v3.1): Ingesta de Ticks MT5 y Features Microestructurales (VPIN, Spread Z-Score, Tick Arrival Rate).

Verifica:
1. Filtro de sesiones líquidas (`07:00–17:00 UTC`).
2. Mayor `vpin_proxy` bajo flujo informado unidireccional vs flujo balanceado.
3. Causalidad estricta en `compute_tick_microstructure_features` (Oracle Future Shock).
4. `DataExtractor.extract_ticks_chunked` guarda correctamente Parquet deduplicado.
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.data_extractor import DataExtractor
from src.preprocessing.information_bars import (
    build_tick_imbalance_bars_with_microstructure,
    compute_tick_microstructure_features,
    filter_liquid_session_ticks,
)


def test_filter_liquid_session_ticks_drops_rollover_hours():
    times = pd.date_range("2026-11-10 00:00:00", periods=48, freq="30min")
    df_ticks = pd.DataFrame({
        "time": times,
        "bid": np.full(len(times), 1.0850),
        "ask": np.full(len(times), 1.0852),
        "price": np.full(len(times), 1.0851),
    })

    filtered = filter_liquid_session_ticks(df_ticks, start_hour_utc=7, end_hour_utc=17)
    hours = pd.to_datetime(filtered["time"]).dt.hour
    assert len(filtered) == 20  # 10 horas * 2 ticks/h
    assert (hours >= 7).all() and (hours < 17).all()


def test_vpin_proxy_detects_toxic_order_flow_and_is_causal():
    rng = np.random.default_rng(123)
    n_bal = 1500
    n_tox = 1500

    p_bal = 1.10 + np.cumsum(rng.choice([-0.0002, 0.0002], size=n_bal))
    p_tox = p_bal[-1] + np.cumsum(rng.choice([-0.0002, 0.0002], size=n_tox, p=[0.1, 0.9]))
    prices = np.concatenate([p_bal, p_tox])

    spreads = np.concatenate([np.full(n_bal, 0.0001), np.full(n_tox, 0.0005)])
    df_ticks = pd.DataFrame({
        "time": pd.date_range("2026-11-10 08:00:00", periods=len(prices), freq="1s"),
        "bid": prices - 0.5 * spreads,
        "ask": prices + 0.5 * spreads,
        "price": prices,
    })

    tibs = build_tick_imbalance_bars_with_microstructure(
        df_ticks, expected_init_t=60, ewma_span=10, min_ticks=15, max_ticks=150
    )
    feats = compute_tick_microstructure_features(tibs, vpin_buckets=5, z_window=15)

    assert {"tick_imbalance_norm", "vpin_proxy", "spread_zscore", "tick_arrival_rate"}.issubset(feats.columns)

    cutoff = pd.Timestamp("2026-11-10 08:00:00") + pd.Timedelta(seconds=n_bal)
    vpin_bal = feats[feats["time"] < cutoff]["vpin_proxy"].dropna().mean()
    vpin_tox = feats[feats["time"] >= cutoff]["vpin_proxy"].dropna().mean()
    assert vpin_tox > vpin_bal

    # Causalidad estricta: alterar barras futuras después del índice 12 no afecta <= 12
    tibs_shock = tibs.copy()
    tibs_shock.loc[13:, ["imbalance", "spread_close", "duration_sec"]] *= 10.0
    feats_shock = compute_tick_microstructure_features(tibs_shock, vpin_buckets=5, z_window=15)
    cols = ["tick_imbalance_norm", "vpin_proxy", "spread_zscore", "tick_arrival_rate"]
    pd.testing.assert_frame_equal(feats[cols].iloc[:13], feats_shock[cols].iloc[:13])


def test_extract_ticks_chunked_saves_deduplicated_parquet():
    t0_ms = int(pd.Timestamp("2026-11-01 10:00:00", tz="UTC").timestamp() * 1000)
    fake_ticks = np.array([
        (t0_ms, 1.0850, 1.0852, 0.0, 0),
        (t0_ms + 500, 1.0851, 1.0853, 0.0, 0),
        (t0_ms + 500, 1.0851, 1.0853, 0.0, 0),  # duplicado a filtrar
        (t0_ms + 1000, 1.0852, 1.0854, 0.0, 0),
    ], dtype=[("time_msc", "i8"), ("bid", "f8"), ("ask", "f8"), ("last", "f8"), ("volume", "i8")])

    connector = MagicMock()
    extractor = DataExtractor(connector)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_pq = os.path.join(tmp_dir, "EURUSD_ticks.parquet")
        with patch("MetaTrader5.symbol_select", return_value=True), \
             patch("MetaTrader5.copy_ticks_range", return_value=fake_ticks):
            df_out = extractor.extract_ticks_chunked(
                "EURUSD",
                datetime(2026, 11, 1),
                datetime(2026, 11, 15),
                chunk_days=30,
                out_path=out_pq,
            )

        assert os.path.exists(out_pq)
        assert len(df_out) == 3
        assert abs(df_out.loc[0, "price"] - 1.0851) < 1e-8
        df_loaded = pd.read_parquet(out_pq)
        assert len(df_loaded) == 3
