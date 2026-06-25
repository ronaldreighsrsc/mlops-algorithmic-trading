import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

class FractionalDifferencer:
    """
    Asegura la estacionariedad de las series de tiempo financieras sin perder 
    la memoria de largo plazo utilizando Diferenciación Fraccionaria (FFD).
    """
    def __init__(self, threshold: float = 1e-4, p_value_limit: float = 0.05, protected_columns: list = None):
        """
        :param threshold: Umbral para descartar pesos pequeños y evitar perder demasiados datos.
        :param p_value_limit: Límite del Test de Dickey-Fuller (5% por defecto).
        :param protected_columns: Variables de "Nivel" o "Régimen" que no deben diferenciarse.
        """
        self.threshold = threshold
        self.p_value_limit = p_value_limit
        
        # GRUPO A: Variables de "Nivel" o "Régimen" (NO TOCAR)
        # - Indicadores (RSI, CCI, MACD): Construidos para ser osciladores estacionarios.
        # - Volatilidad (EGARCH, ATR): Es una medida de varianza, no un precio.
        if protected_columns is None:
            self.protected_columns = ['time', 'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'ATR', 'EGARCH_Vol', 'TPM', 'EMBI', 'VIX_close', 'tick_volume', 'real_volume', 'spread']
        else:
            self.protected_columns = protected_columns

    def _get_weights_ffd(self, d: float) -> np.ndarray:
        """Calcula los pesos iterativos de la diferenciación fraccional."""
        w, k = [1.], 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < self.threshold:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def _frac_diff_ffd(self, series: pd.Series, d: float) -> pd.Series:
        """Aplica los pesos calculados a la serie de tiempo original."""
        df_series = pd.DataFrame(series).ffill().dropna()
        w = self._get_weights_ffd(d)
        width = len(w) - 1
        
        series_val = df_series.values.reshape(-1, 1)
        
        # Proteccion: Si la ventana de pesos FFD es mayor que nuestra cantidad de datos
        if width >= len(series_val):
            return pd.Series([np.nan] * len(df_series), index=df_series.index)
            
        ffd_vals = [np.nan] * width # Rellenamos con NaN el periodo consumido por la ventana
        
        for i in range(width, len(series_val)):
            val = np.dot(w.T, series_val[i-width:i+1])[0, 0]
            ffd_vals.append(val)
            
        return pd.Series(ffd_vals, index=df_series.index)

    def _find_optimal_d(self, series: pd.Series) -> tuple:
        """
        Busca el valor mínimo de d (entre 0.0 y 1.0) que hace que la serie 
        pase el test de Dickey-Fuller (p-value < 0.05) preservando máxima memoria.
        Retorna (d_optimo, correlacion_con_original).
        """
        series_clean = series.ffill().dropna()
        
        for d in np.linspace(0.0, 1.0, 40):
            df_ffd = self._frac_diff_ffd(series_clean, d).dropna()
            if len(df_ffd) < 100: # Conservador: mínimo 100 obs para un ADF fiable
                continue
            
            try:
                p_val = adfuller(df_ffd.values.flatten(), autolag='AIC')[1]
                
                if p_val <= self.p_value_limit:
                    # Calcular correlación con serie original para medir memoria preservada
                    original_aligned = series_clean.loc[df_ffd.index]
                    corr = np.corrcoef(
                        original_aligned.values.flatten(),
                        df_ffd.values.flatten()
                    )[0, 1]
                    return d, corr
            except Exception:
                continue
                
        return 1.0, 0.0 # Fallback: diferenciación entera, sin memoria preservada

    def apply_ffd(self, df: pd.DataFrame, columns_to_ignore: list = None) -> pd.DataFrame:
        """
        Método público: Evalúa columnas, aplica FFD a las rebeldes y devuelve el DF limpio.
        """
        if columns_to_ignore is None:
            columns_to_ignore = self.protected_columns
            
        df_calc = df.copy()
        cols_to_drop = []
        
        print(f"Evaluando Estacionariedad y aplicando FFD (Threshold: {self.threshold})...")
        
        for col in df_calc.columns:
            # Ignorar variables categóricas, fechas o variables que decidamos proteger
            if col in columns_to_ignore or not pd.api.types.is_numeric_dtype(df_calc[col]):
                continue
                
            serie_limpia = df_calc[col].dropna()
            # 1. Test Dickey-Fuller Aumentado (Pre-Check)
            try:
                p_value = adfuller(serie_limpia)[1]
            except Exception:
                p_value = 1.0 # Si falla, asumir no estacionaria para forzar búsqueda de d*
            
            if p_value < self.p_value_limit:
                print(f"  [SKIP] {col} YA es estacionaria (p={p_value:.4f}).")
            else:
                print(f"  [PROCESAR] {col} NO es estacionaria (p={p_value:.4f}). Buscando d*...")
                
                # 2. Encontrar d óptimo usando SOLO datos del pasado (Train Set - 80%) para evitar Look-Ahead Bias
                train_size = int(len(serie_limpia) * 0.8)
                serie_train = serie_limpia.iloc[:train_size]
                
                d_optimo, corr = self._find_optimal_d(serie_train)
                
                # 3. Aplicar FFD a toda la serie usando el d_optimo encontrado sin ver el futuro
                nueva_col = f"{col}_FFD"
                df_calc[nueva_col] = self._frac_diff_ffd(serie_limpia, d_optimo)
                
                # Reporte de pérdida de ventana y memoria preservada
                width = len(self._get_weights_ffd(d_optimo)) - 1
                print(f"  [FINAL] {nueva_col} salvada con d={d_optimo:.2f} (Ventana: {width} días, Corr: {corr:.4f}).")
                
                # Agendar la original para su destrucción (Excepto precios crudos para el backtest real)
                if col not in ['close', 'open', 'high', 'low']:
                    cols_to_drop.append(col)

        # 3. Destrucción de no-estacionarias y Limpieza de NaNs
        df_calc = df_calc.drop(columns=cols_to_drop)
        filas_antes = len(df_calc)
        df_calc = df_calc.dropna()
        filas_despues = len(df_calc)
        
        print(f"✅ Matemáticas FFD completadas. Filas resultantes: {filas_despues} (Se consumieron {filas_antes - filas_despues} días en cálculos previos).")
        return df_calc