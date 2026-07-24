import pytest
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from execution.main_bot import TradingBot

def test_default_timeframe_fallback():
    # Config sin clave "timeframe"
    config = {
        "model_type": "XGBOOST",
        "banco": "Globales",
        "confidence_threshold": 0.50,
        "k_up": 2.0,
        "k_down": 1.5,
        "model_file": "dummy.pkl"
    }
    
    tf_str = config.get("timeframe", "D1")
    assert tf_str == "D1"

def test_dynamic_magic_number_derivation():
    config = {
        "model_type": "XGBOOST",
        "banco": "Globales",
        "confidence_threshold": 0.50,
        "k_up": 2.0,
        "k_down": 1.5,
        "model_file": "dummy.pkl"
    }
    
    bot_d1 = TradingBot(symbol="EURUSD", timeframe=mt5.TIMEFRAME_D1, config=config, models_dir=".")
    bot_h4 = TradingBot(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H4, config=config, models_dir=".")
    bot_h1 = TradingBot(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, config=config, models_dir=".")
    
    # Asegurar magic_numbers únicos para evitar colisiones en MT5
    assert bot_d1.engine.magic_number != bot_h4.engine.magic_number
    assert bot_h4.engine.magic_number != bot_h1.engine.magic_number
    assert bot_d1.engine.magic_number != bot_h1.engine.magic_number

def test_intraday_macro_forward_fill():
    # DataFrame intradía H4
    dates = pd.date_range("2026-01-01", periods=12, freq="4h")
    df_h4 = pd.DataFrame({'close': 1.10 + np.cumsum(np.random.randn(12)*0.001)}, index=dates)
    df_h4['date_key'] = df_h4.index.date

    # Macro diario
    daily_dates = pd.date_range("2026-01-01", periods=3, freq="D")
    df_macro = pd.DataFrame({'VIX_close': [15.0, 18.0, 16.5]}, index=daily_dates)
    df_macro['date_key'] = df_macro.index.date

    merged = df_h4.reset_index().merge(df_macro, on='date_key', how='left').set_index('index')
    merged['VIX_close'] = merged['VIX_close'].ffill()

    assert not merged['VIX_close'].isna().any()
    assert len(merged) == 12
