"""
Fixtures reutilizables para la suite de pruebas unitarias del Quant Trading Bot.
Provee DataFrames sintéticos con precios simulados (bull run, crash, mercado plano)
y configuraciones estándar para Risk Manager y modelos de preprocesamiento.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio src al path para poder importar los módulos del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'execution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'preprocessing'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'evaluation'))


@pytest.fixture
def df_bullrun():
    """DataFrame sintético de 200 días con una subida constante (~+0.5% diario)."""
    np.random.seed(42)
    n = 200
    base_price = 1.1000
    # Tendencia alcista con ruido pequeño
    returns = np.random.normal(0.005, 0.002, n)
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n, freq='B'),
        'open': prices * (1 - np.random.uniform(0.001, 0.003, n)),
        'high': prices * (1 + np.random.uniform(0.002, 0.008, n)),
        'low': prices * (1 - np.random.uniform(0.002, 0.008, n)),
        'close': prices,
        'tick_volume': np.random.randint(1000, 5000, n),
        'real_volume': np.random.randint(1000, 5000, n),
        'spread': np.zeros(n),
        'EGARCH_Vol': np.random.uniform(0.3, 0.8, n),  # Volatilidad baja (bull market)
    })
    return df


@pytest.fixture
def df_crash():
    """DataFrame sintético de 200 días con una caída fuerte (~-0.8% diario)."""
    np.random.seed(123)
    n = 200
    base_price = 1.2000
    returns = np.random.normal(-0.008, 0.003, n)
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n, freq='B'),
        'open': prices * (1 + np.random.uniform(0.001, 0.003, n)),
        'high': prices * (1 + np.random.uniform(0.002, 0.008, n)),
        'low': prices * (1 - np.random.uniform(0.002, 0.015, n)),
        'close': prices,
        'tick_volume': np.random.randint(1000, 5000, n),
        'real_volume': np.random.randint(1000, 5000, n),
        'spread': np.zeros(n),
        'EGARCH_Vol': np.random.uniform(0.5, 1.5, n),  # Volatilidad alta (crash)
    })
    return df


@pytest.fixture
def df_flat():
    """DataFrame sintético de 200 días con mercado completamente plano (sin tendencia)."""
    np.random.seed(99)
    n = 200
    base_price = 1.1500
    # Oscilación mínima alrededor del precio base
    noise = np.random.normal(0, 0.0005, n)
    prices = base_price + np.cumsum(noise) * 0.001  # Prácticamente plano

    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n, freq='B'),
        'open': prices - 0.0001,
        'high': prices + 0.0003,  # High apenas por encima del close
        'low': prices - 0.0003,   # Low apenas por debajo del close
        'close': prices,
        'tick_volume': np.random.randint(1000, 5000, n),
        'real_volume': np.random.randint(1000, 5000, n),
        'spread': np.zeros(n),
        'EGARCH_Vol': np.random.uniform(0.1, 0.3, n),  # Volatilidad muy baja
    })
    return df


@pytest.fixture
def df_ohlcv_clean():
    """DataFrame OHLCV limpio con 600 filas para tests de indicadores técnicos y EGARCH."""
    np.random.seed(7)
    n = 600
    base_price = 100.0
    returns = np.random.normal(0.0003, 0.01, n)
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'time': pd.date_range('2018-01-01', periods=n, freq='B'),
        'open': prices * (1 - np.random.uniform(0.001, 0.005, n)),
        'high': prices * (1 + np.random.uniform(0.003, 0.015, n)),
        'low': prices * (1 - np.random.uniform(0.003, 0.015, n)),
        'close': prices,
        'tick_volume': np.random.randint(1000, 10000, n),
        'real_volume': np.random.randint(1000, 10000, n),
        'spread': np.zeros(n),
    })
    return df


@pytest.fixture
def risk_manager_default():
    """Instancia de RiskManager con parámetros por defecto (2.5% riesgo, k_up=2, k_down=1.5)."""
    from risk_manager import RiskManager
    return RiskManager(risk_per_trade_pct=0.025, k_up=2.0, k_down=1.5, max_hold=10)
