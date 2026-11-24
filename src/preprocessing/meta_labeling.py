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


class MetaBarrierLabeler:
    """
    Meta-Labeling Verdadero de 2 Etapas con Triple Barrera Direccional (AFML §3.6, v3.0).

    A diferencia de TripleBarrierLabeler (unidireccional diario), esta clase:
    1. Evalúa ÚNICAMENTE los eventos filtrados (ej. CUSUM) que tienen una dirección
       propuesta `side ∈ {+1, -1}` emitida por un Modelo Primario (Capa 1).
    2. Orienta simétricamente las barreras TP y SL según `side`:
         - side = +1: TP arriba (+k_up * vol), SL abajo (-k_down * vol)
         - side = -1: TP abajo (-k_up * vol), SL arriba (+k_down * vol)
    3. Descuenta fricción estimada `round_trip_cost` y registra exactamente el
       intervalo de vida [t0, t1] de cada evento para Purging y Sample Uniqueness.
    """

    def __init__(
        self,
        k_up: float = 2.0,
        k_down: float = 1.5,
        max_hold: int = 10,
        round_trip_cost: float = 0.0001,
        require_tp_hit: bool = False,
    ):
        self.k_up = float(k_up)
        self.k_down = float(k_down)
        self.max_hold = int(max_hold)
        self.round_trip_cost = float(round_trip_cost)
        self.require_tp_hit = bool(require_tp_hit)

    def label_events(
        self,
        df: pd.DataFrame,
        sides: pd.Series | np.ndarray,
        event_positions: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Genera el DataFrame de eventos meta-etiquetados con columnas:
        [t0, t1, t0_pos, t1_pos, side, entry_price, exit_price, barrier,
         holding_bars, gross_ret, net_ret, meta_label].
        """
        requeridas = ["open", "high", "low", "close", "EGARCH_Vol"]
        for col in requeridas:
            if col not in df.columns:
                raise ValueError(f"Falta columna '{col}' para MetaBarrierLabeler.")

        n = len(df)
        sides_arr = np.asarray(sides, dtype=np.int8)
        if len(sides_arr) != n:
            raise ValueError("La longitud de 'sides' debe coincidir con len(df).")

        if event_positions is None:
            candidate_positions = np.where(sides_arr != 0)[0]
        else:
            cand = np.asarray(event_positions, dtype=np.int64)
            candidate_positions = cand[(cand >= 0) & (cand < n) & (sides_arr[cand] != 0)]

        opens = df["open"].astype(float).values
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        closes = df["close"].astype(float).values
        vols = df["EGARCH_Vol"].astype(float).values

        if "time" in df.columns:
            time_values = pd.to_datetime(df["time"]).values
        else:
            time_values = df.index.values

        rows = []
        max_valid_i = n - self.max_hold - 1

        for i in candidate_positions:
            i_int = int(i)
            if i_int >= max_valid_i:
                continue

            side = int(sides_arr[i_int])
            if side not in (-1, 1):
                continue

            vol_entry = vols[i_int]
            entry_price = opens[i_int + 1]
            if np.isnan(vol_entry) or vol_entry <= 0.0 or np.isnan(entry_price) or entry_price <= 0.0:
                continue

            vol_dec = vol_entry / 100.0
            tp_level = entry_price * (1.0 + side * self.k_up * vol_dec)
            sl_level = entry_price * (1.0 - side * self.k_down * vol_dec)

            exit_pos = i_int + self.max_hold
            exit_price = closes[exit_pos]
            barrier = "TIME"
            holding = self.max_hold

            for j in range(1, self.max_hold + 1):
                curr_idx = i_int + j
                if side == 1:
                    hit_tp = highs[curr_idx] >= tp_level
                    hit_sl = lows[curr_idx] <= sl_level
                else:
                    hit_tp = lows[curr_idx] <= tp_level
                    hit_sl = highs[curr_idx] >= sl_level

                if hit_tp and hit_sl:
                    # Heurística pesimista institucional: doble toque en la misma barra => SL
                    exit_pos = curr_idx
                    exit_price = sl_level
                    barrier = "SL_DUAL"
                    holding = j
                    break
                if hit_tp:
                    exit_pos = curr_idx
                    exit_price = tp_level
                    barrier = "TP"
                    holding = j
                    break
                if hit_sl:
                    exit_pos = curr_idx
                    exit_price = sl_level
                    barrier = "SL"
                    holding = j
                    break

            gross_ret = side * ((exit_price / entry_price) - 1.0)
            net_ret = gross_ret - self.round_trip_cost

            if self.require_tp_hit:
                meta_label = 1 if (barrier == "TP" and net_ret > 0.0) else 0
            else:
                meta_label = 1 if net_ret > 0.0 else 0

            rows.append({
                "t0": time_values[i_int],
                "t1": time_values[exit_pos],
                "t0_pos": i_int,
                "t1_pos": int(exit_pos),
                "side": side,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "barrier": barrier,
                "holding_bars": int(holding),
                "gross_ret": float(gross_ret),
                "net_ret": float(net_ret),
                "meta_label": int(meta_label),
            })

        return pd.DataFrame(rows)
