import os
import json
import logging
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AssetScreener:
    """
    Módulo Cuantitativo de Screening de Universo de Activos (Fase 0).
    Evalúa la persistencia de tendencia mediante el Exponente de Hurst (H)
    y la matriz de descorrelación cross-asset para aprobar activos ejecutables en MT5.
    """
    def __init__(self, connector=None):
        self.connector = connector
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.results_dir = os.path.join(base_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
        """
        Calcula el Exponente de Hurst (H) usando Rescaled Range (R/S Analysis).
        - H > 0.55: Tendencia persistente (alta memoria, ideal para ML)
        - H ~ 0.50: Caminata aleatoria (ruido blanco puro, descartar)
        - H < 0.45: Reversión a la media (Mean-reverting)
        """
        ts = series.values
        if len(ts) < max_lag * 2:
            return 0.50
            
        lags = range(2, max_lag)
        try:
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            # Evitar log(0) o inis
            tau = np.array(tau)
            valid_idx = tau > 0
            if not np.any(valid_idx):
                return 0.50
            
            log_lags = np.log(np.array(lags)[valid_idx])
            log_tau = np.log(tau[valid_idx])
            
            poly = np.polyfit(log_lags, log_tau, 1)
            hurst = poly[0] * 2.0
            return float(np.clip(hurst, 0.0, 1.0))
        except Exception as e:
            logging.warning(f"Error calculando Hurst: {e}")
            return 0.50

    def screen_universe(self, 
                        candidates: List[Dict], 
                        min_hurst: float = 0.53, 
                        max_corr: float = 0.85) -> Dict:
        """
        Escanea la lista de candidatos de MT5, calcula Hurst y matriz de descorrelación.
        Retorna el diccionario con activos aprobados y guarda 'results/active_assets.json'.
        """
        print(f"\n🔍 [FASE 0] INICIANDO SCREENING CUANTITATIVO DE ACTIVOS MT5")
        print(f"{'='*80}")
        
        # 1. Recopilar series históricas de cierre
        closes_dict = {}
        hurst_results = {}
        
        for item in candidates:
            name = item["nombre"]
            ticker = item["ticker"]
            tf = item.get("timeframe", mt5.TIMEFRAME_D1)
            
            rates = mt5.copy_rates_from_pos(ticker, tf, 0, 1500)
            if rates is not None and len(rates) > 200:
                df = pd.DataFrame(rates)
                closes = df['close']
                h_val = self.calculate_hurst_exponent(closes)
                hurst_results[name] = h_val
                closes_dict[name] = closes.pct_change().fillna(0.0)
                
                status = "✅ PERSISTENTE" if h_val >= min_hurst else "❌ RUIDO (Descartar)"
                print(f"  > [{name:<12}] Exponente de Hurst H = {h_val:.4f} -> {status}")
            else:
                logging.warning(f"No se pudieron descargar datos de MT5 para {name} ({ticker}).")
                
        # 2. Filtrar candidatos por Exponente de Hurst
        aprobados_hurst = [n for n, h in hurst_results.items() if h >= min_hurst]
        
        if not aprobados_hurst:
            print("⚠️ Ningún activo superó el umbral de Hurst estricto. Relajando filtro al top candidatos...")
            sorted_by_h = sorted(hurst_results.items(), key=lambda x: x[1], reverse=True)
            aprobados_hurst = [x[0] for x in sorted_by_h[:max(1, len(sorted_by_h))]]

        # 3. Matriz de Descorrelación entre Aprobados
        cesta_final = []
        if len(aprobados_hurst) > 1 and closes_dict:
            df_rets = pd.DataFrame({k: closes_dict[k] for k in aprobados_hurst if k in closes_dict}).dropna()
            corr_matrix = df_rets.corr().abs()
            
            print(f"\n📊 MATRIZ DE DESCORRELACIÓN CROSS-ASSET:")
            print(corr_matrix.round(2))
            
            # Selección codiciosa (Greedy) para evitar pares con correlación > max_corr
            for name in aprobados_hurst:
                keep = True
                for selected in cesta_final:
                    if name in corr_matrix.columns and selected in corr_matrix.index:
                        if corr_matrix.loc[name, selected] > max_corr:
                            keep = False
                            print(f"  🚫 [{name}] descartado por alta correlación ({corr_matrix.loc[name, selected]:.2f}) con {selected}")
                            break
                if keep:
                    cesta_final.append(name)
        else:
            cesta_final = aprobados_hurst

        print(f"\n🏆 CESTA FINAL DE ACTIVOS APROBADOS DE PRODUCCIÓN ({len(cesta_final)} activos):")
        for act in cesta_final:
            print(f"  👉 {act} (Hurst: {hurst_results.get(act, 0.50):.4f})")
            
        output_data = {
            "cesta_final": cesta_final,
            "hurst_scores": hurst_results,
            "min_hurst_threshold": min_hurst,
            "max_corr_threshold": max_corr,
            "candidates_evaluated": [c["nombre"] for c in candidates]
        }
        
        json_path = os.path.join(self.results_dir, "active_assets.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        print(f"✅ Configuración de activos guardada en: '{json_path}'\n")
        return output_data


if __name__ == "__main__":
    from mt5_connector import MT5Connector
    
    conn = MT5Connector()
    if conn.connect():
        candidates = [
            {"nombre": "EURUSD", "ticker": "EURUSD", "timeframe": mt5.TIMEFRAME_D1},
            {"nombre": "EURUSD_H4", "ticker": "EURUSD", "timeframe": mt5.TIMEFRAME_H4},
            {"nombre": "SP500", "ticker": "SP500", "timeframe": mt5.TIMEFRAME_D1},
            {"nombre": "SP500_H4", "ticker": "SP500", "timeframe": mt5.TIMEFRAME_H4},
            {"nombre": "Oro", "ticker": "XAUUSD", "timeframe": mt5.TIMEFRAME_D1},
            {"nombre": "Oro_H4", "ticker": "XAUUSD", "timeframe": mt5.TIMEFRAME_H4},
        ]
        screener = AssetScreener(conn)
        screener.screen_universe(candidates)
        conn.shutdown()
