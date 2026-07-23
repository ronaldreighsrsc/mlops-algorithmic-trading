import pandas as pd
import numpy as np
import warnings
import json
import os
from arch import arch_model

warnings.filterwarnings("ignore")

class VolatilityModeler:
    """
    Calcula la volatilidad condicional de la serie financiera usando un modelo EGARCH
    con ventana móvil (Rolling Window). Implementa un sistema de respaldo (fallback)
    hacia GARCH simple o Volatilidad Histórica en caso de inestabilidad matemática.
    
    Incluye un sistema de caché en disco que guarda los pronósticos diarios, permitiendo
    que en ejecuciones posteriores solo se calcule el delta incremental (días nuevos)
    en lugar de recalcular toda la historia desde cero (~10 min → ~5 seg).
    """
    def __init__(self, window_size: int = 500, target_col: str = 'close', cache_dir: str = None):
        """
        Inicializa el modelador.
        :param window_size: Días para la ventana móvil (ej. 500 días).
        :param target_col: Columna sobre la cual calcular los retornos.
        :param cache_dir: Directorio donde guardar el caché. Si es None, se usa 'cache/' en el proyecto.
        """
        self.window_size = window_size
        self.target_col = target_col
        
        if cache_dir is None:
            # Detectar el directorio base del proyecto (quant-trading-bot/)
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.cache_dir = os.path.join(base, "cache")
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, symbol: str = "default") -> str:
        return os.path.join(self.cache_dir, f"egarch_cache_{symbol}.json")

    def _load_cache(self, symbol: str = "default") -> dict:
        """Carga el caché de pronósticos EGARCH del disco."""
        path = self._cache_path(symbol)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self, cache_data: dict, symbol: str = "default"):
        """Guarda el caché de pronósticos EGARCH al disco."""
        path = self._cache_path(symbol)
        with open(path, 'w') as f:
            json.dump(cache_data, f)

    def _forecast_single_day(self, train_window: pd.Series) -> float:
        """Pronostica la volatilidad de un solo día usando EGARCH con fallback."""
        vol_historica = train_window.std()
        
        try:
            # Intento 1: EGARCH (Asimétrico, ideal pero matemáticamente inestable)
            model = arch_model(train_window, vol='EGarch', p=1, o=1, q=1, dist='t', rescale=False)
            res = model.fit(disp='off', show_warning=False)
            pred = res.forecast(horizon=1, reindex=False)
            vol_pred = np.sqrt(pred.variance.values[-1, 0])
            
            # FILTRO DE CORDURA: Si es NaN, infinito, prácticamente cero, o irrealmente alto (> 5% diaria)
            if np.isnan(vol_pred) or np.isinf(vol_pred) or vol_pred > 5.0 or vol_pred < 1e-6:
                
                # Intento 2: GARCH(1,1) estándar (Simétrico, mucho más robusto)
                model_fallback = arch_model(train_window, vol='Garch', p=1, q=1, rescale=False)
                res_fallback = model_fallback.fit(disp='off', show_warning=False)
                pred_fallback = res_fallback.forecast(horizon=1, reindex=False)
                vol_pred = np.sqrt(pred_fallback.variance.values[-1, 0])
                
                # Intento 3: Si todo el álgebra lineal falla
                if np.isnan(vol_pred) or np.isinf(vol_pred) or vol_pred > 5.0 or vol_pred < 1e-6:
                    vol_pred = vol_historica
                    
        except Exception:
            # Si la librería arroja cualquier error fatal de convergencia
            vol_pred = vol_historica
            
        return float(vol_pred)

    def compute_egarch(self, df: pd.DataFrame, symbol: str = "default") -> pd.DataFrame:
        """
        Calcula EGARCH con caché incremental.
        
        Primera ejecución: calcula toda la historia (~10 min) y la guarda en disco.
        Ejecuciones posteriores: lee el caché y solo calcula los días nuevos (~5 seg).
        """
        # ---> BARRERA DE SEGURIDAD (Fail-Fast) <---
        if len(df) <= self.window_size:
            raise ValueError(f"Error Crítico: Los datos tienen {len(df)} filas, pero el modelo necesita al menos {self.window_size + 1} días para iniciar la ventana móvil.")

        df_calc = df.copy()
        
        # 1. Calcular retornos logarítmicos (necesarios para modelos ARCH/GARCH)
        df_calc['log_ret'] = np.log(df_calc[self.target_col]).diff() * 100
        df_calc['log_ret'] = df_calc['log_ret'].fillna(0)
        
        # 2. Intentar cargar caché existente
        cache = self._load_cache(symbol)
        cached_forecasts = cache.get("forecasts", [])
        cached_len = cache.get("data_length", 0)
        
        # Determinar desde dónde empezar a calcular
        current_len = len(df_calc)
        
        if cached_len > 0 and cached_len <= current_len and len(cached_forecasts) > 0:
            # --- MODO INCREMENTAL (Rápido: solo calcular los días nuevos) ---
            new_days = current_len - cached_len
            
            if new_days == 0:
                # No hay días nuevos, usar caché completo
                print(f"⚡ EGARCH desde caché (0 días nuevos). Instantáneo.")
                forecasts = cached_forecasts
            else:
                print(f"⚡ EGARCH incremental: {len(cached_forecasts)} días en caché + {new_days} día(s) nuevo(s) a calcular...")
                forecasts = cached_forecasts.copy()
                
                for i in range(cached_len, current_len):
                    if i < self.window_size:
                        forecasts.append(None)  # NaN placeholder
                    else:
                        train_window = df_calc['log_ret'].iloc[i - self.window_size : i]
                        vol_pred = self._forecast_single_day(train_window)
                        forecasts.append(vol_pred)
                        
                print(f"✅ EGARCH incremental completado (+{new_days} día(s)).")
        else:
            # --- MODO COMPLETO (Primera vez: calcular toda la historia) ---
            print(f"Calculando Volatilidad Condicional EGARCH (Ventana: {self.window_size} días)...")
            print("⏳ Esto puede tomar un par de minutos debido a las iteraciones matemáticas de la ventana móvil.")
            
            forecasts = [None] * self.window_size
            
            for i in range(self.window_size, current_len):
                train_window = df_calc['log_ret'].iloc[i - self.window_size : i]
                vol_pred = self._forecast_single_day(train_window)
                forecasts.append(vol_pred)
                
                # Imprimir progreso cada 200 días para no saturar la consola
                if i % 200 == 0:
                    print(f"  > Procesando día {i}/{current_len}...")
            
            print("✅ EGARCH calculado y acoplado exitosamente.")
        
        # 3. Guardar caché actualizado al disco
        self._save_cache({
            "forecasts": forecasts,
            "data_length": current_len,
            "window_size": self.window_size
        }, symbol)
        
        # 4. Asignación y Limpieza
        df_calc['EGARCH_Vol'] = [np.nan if v is None else v for v in forecasts]
        df_calc = df_calc.drop(columns=['log_ret'])
        df_calc = df_calc.dropna(subset=['EGARCH_Vol'])
        
        return df_calc