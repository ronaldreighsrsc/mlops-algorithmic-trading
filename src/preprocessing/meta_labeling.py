import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class TripleBarrierLabeler:
    """
    Generador de etiquetas institucionales (Meta-Labeling primario).
    Asigna 1 si una posición larga alcanza el Take Profit antes que el Stop Loss o el límite de tiempo.
    Asigna 0 en caso de tocar el Stop Loss o expirar el tiempo.
    Usa la volatilidad dinámica (EGARCH) para fijar las barreras.
    """
    def __init__(self, k_up: float = 2.0, k_down: float = 1.5, max_hold: int = 10):
        self.k_up = k_up
        self.k_down = k_down
        self.max_hold = max_hold

    def apply_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Generando Etiquetas Y (Triple Barrera: TP={self.k_up}x, SL={self.k_down}x, T={self.max_hold}d)...")
        df_labeled = df.copy()
        
        # Validar existencia de columnas
        requeridas = ['open', 'high', 'low', 'close', 'EGARCH_Vol']
        for col in requeridas:
            if col not in df_labeled.columns:
                raise ValueError(f"Falta columna '{col}' para la Triple Barrera.")
                
        labels = np.zeros(len(df_labeled))
        
        opens = df_labeled['open'].values
        highs = df_labeled['high'].values
        lows = df_labeled['low'].values
        vols = df_labeled['EGARCH_Vol'].values
        
        exitos_tp = 0
        fracasos_sl = 0
        fracasos_tiempo = 0
        
        # Iteramos hasta el penúltimo día menos max_hold
        for i in range(len(df_labeled) - self.max_hold - 1):
            # La señal se emite en t=i (al cierre). La ejecución real es en t=i+1 (apertura)
            entry_price = opens[i + 1]
            vol_entry = vols[i]
            
            if pd.isna(vol_entry) or vol_entry == 0:
                continue
                
            # Barreras Dinámicas
            tp_level = entry_price * (1 + self.k_up * (vol_entry / 100))
            sl_level = entry_price * (1 - self.k_down * (vol_entry / 100))
            
            # Recorrer la ventana de tiempo (Max Hold)
            for j in range(1, self.max_hold + 1):
                curr_high = highs[i + j]
                curr_low = lows[i + j]
                
                hit_tp = curr_high >= tp_level
                hit_sl = curr_low <= sl_level
                
                if hit_tp and hit_sl:
                    labels[i] = 0 # Heurística pesimista: si toca ambos el mismo día, asumimos pérdida
                    fracasos_sl += 1
                    break
                elif hit_tp:
                    labels[i] = 1 # Éxito
                    exitos_tp += 1
                    break
                elif hit_sl:
                    labels[i] = 0 # Fracaso por pérdida
                    fracasos_sl += 1
                    break
                elif j == self.max_hold:
                    labels[i] = 0 # Fracaso por tiempo (costo de oportunidad)
                    fracasos_tiempo += 1
                    break
                    
        df_labeled['Label'] = labels
        
        total_evaluados = exitos_tp + fracasos_sl + fracasos_tiempo
        win_rate = exitos_tp / total_evaluados if total_evaluados > 0 else 0
        
        print(f"  > Hit Ratio (Clase 1): {win_rate:.2%} | TP Hits: {exitos_tp} | SL Hits: {fracasos_sl} | Time Expirations: {fracasos_tiempo}")
        
        # Eliminar las últimas filas que no pudieron ser evaluadas por el max_hold
        df_labeled = df_labeled.iloc[:-self.max_hold-1]
        return df_labeled
