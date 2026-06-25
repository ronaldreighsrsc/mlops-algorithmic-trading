import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class TechnicalFeatureEngineer:
    """
    Calcula indicadores de momentum y volatilidad (MACD, RSI, ATR).
    Asume que el DataFrame de entrada contiene los precios del activo base (IPSA)
    en las columnas 'Price', 'High' y 'Low'.
    """
    def __init__(self, target_price_col: str = 'close'):
        self.target = target_price_col

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Calculando indicadores técnicos (MACD, RSI, ATR)...")
        df_calc = df.copy()
        
        # ---> SEGURIDAD FAIL-FAST <---
        # Verifica que existan las columnas necesarias antes de hacer matemáticas
        columnas_requeridas = [self.target, 'high', 'low']
        for col in columnas_requeridas:
            if col not in df_calc.columns:
                raise ValueError(f"❌ Error Crítico: Falta la columna '{col}' necesaria para calcular el ATR o RSI.")

        # ==========================================
        # 1. MACD (12, 26, 9)
        # ==========================================
        ema12 = df_calc[self.target].ewm(span=12, adjust=False).mean()
        ema26 = df_calc[self.target].ewm(span=26, adjust=False).mean()
        df_calc['MACD'] = ema12 - ema26
        df_calc['MACD_Signal'] = df_calc['MACD'].ewm(span=9, adjust=False).mean()
        df_calc['MACD_Hist'] = df_calc['MACD'] - df_calc['MACD_Signal']

        # ==========================================
        # 2. RSI (14 periodos - Suavizado de Wilder Original)
        # ==========================================
        delta = df_calc[self.target].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        
        # Truco pro: Reemplazar ceros por un número ínfimo (1e-10) para evitar error de "División por Cero"
        rs = avg_gain / avg_loss.replace(0, 1e-10) 
        df_calc['RSI'] = 100 - (100 / (1 + rs))

        # ==========================================
        # 3. VOLATILIDAD: ATR (14 periodos)
        # ==========================================
        high_low = df_calc['high'] - df_calc['low']
        high_close = np.abs(df_calc['high'] - df_calc[self.target].shift())
        low_close = np.abs(df_calc['low'] - df_calc[self.target].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Suavizado de Wilder para el True Range
        df_calc['ATR'] = true_range.ewm(alpha=1/14, adjust=False).mean()

        # ==========================================
        # 4. LIMPIEZA DEL PERIODO DE CALENTAMIENTO
        # ==========================================
        # El MACD necesita ~26 días para tener sentido matemático. Las primeras filas serán NaNs.
        df_calc = df_calc.dropna(subset=['MACD_Hist', 'RSI', 'ATR'])
        
        print(f"✅ Indicadores inyectados con éxito.")
        return df_calc