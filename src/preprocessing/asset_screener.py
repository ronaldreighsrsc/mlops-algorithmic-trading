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
    Evalúa la persistencia de tendencia mediante el Exponente de Hurst (H) R/S Analysis
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
        Calcula el Exponente de Hurst (H) usando Rescaled Range (R/S Analysis Mandelbrot).
        - H > 0.55: Tendencia persistente (alta memoria, ideal para ML)
        - H ~ 0.50: Caminata aleatoria (ruido blanco puro, descartar)
        - H < 0.45: Reversión a la media (Mean-reverting)
        """
        ts = series.values
        if len(ts) < max_lag * 2:
            return 0.50
            
        try:
            rs_list = []
            lags = []
            for k in range(10, max_lag, 5):
                n_chunks = len(ts) // k
                if n_chunks < 2: 
                    break
                rs_k = []
                for i in range(n_chunks):
                    chunk = ts[i*k:(i+1)*k]
                    m = np.mean(chunk)
                    y = chunk - m
                    z = np.cumsum(y)
                    r = np.max(z) - np.min(z)
                    s = np.std(chunk)
                    if s > 0: 
                        rs_k.append(r / s)
                if rs_k:
                    rs_list.append(np.mean(rs_k))
                    lags.append(k)
                    
            if not rs_list or len(rs_list) < 2:
                return 0.50
                
            poly = np.polyfit(np.log(lags), np.log(rs_list), 1)
            hurst = poly[0]
            return float(np.clip(hurst, 0.0, 1.0))
        except Exception as e:
            logging.warning(f"Error calculando Hurst R/S: {e}")
            return 0.50

    def discover_mt5_candidates(self, max_candidates: int = 25) -> List[Dict]:
        """
        Descubre dinámicamente el universo de símbolos principales en MT5.
        Prioriza los activos líquidos institucionales en Darwinex (FX, Índices, Metals, Energy).
        """
        symbols = mt5.symbols_get()
        if not symbols:
            logging.warning("mt5.symbols_get() no retornó ningún símbolo.")
            return []
            
        priority_tickers = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", 
            "EURGBP", "EURJPY", "GBPJPY", "XAUUSD", "XAGUSD", "SP500", "US500", 
            "NAS100", "US100", "GER40", "DE40", "UK100", "WTI", "XTIUSD", "XBRUSD"
        ]
        
        candidates = []
        seen_names = set()
        available_names = {s.name for s in symbols}
        
        # 1. Agregar prioritarios disponibles en MT5
        for ticker in priority_tickers:
            if ticker in available_names and ticker not in seen_names:
                seen_names.add(ticker)
                candidates.append({
                    "nombre": ticker, 
                    "ticker": ticker, 
                    "timeframe": mt5.TIMEFRAME_D1,
                    "filename": f"{ticker}_daily.csv"
                })
                # Incluir variante H4 para los 4 principales
                if ticker in ["EURUSD", "SP500", "US500", "XAUUSD"]:
                    name_h4 = f"{ticker}_H4"
                    candidates.append({
                        "nombre": name_h4, 
                        "ticker": ticker, 
                        "timeframe": mt5.TIMEFRAME_H4,
                        "filename": f"{name_h4}_daily.csv"
                    })
                    
        # 2. Agregar otros activos seleccionados en Market Watch hasta max_candidates
        if len(candidates) < max_candidates:
            for s in symbols:
                if (s.visible or s.select) and s.name not in seen_names:
                    seen_names.add(s.name)
                    candidates.append({
                        "nombre": s.name, 
                        "ticker": s.name, 
                        "timeframe": mt5.TIMEFRAME_D1,
                        "filename": f"{s.name}_daily.csv"
                    })
                    if len(candidates) >= max_candidates:
                        break
                
        return candidates

    def screen_universe(self, 
                        candidates: List[Dict] = None, 
                        min_hurst: float = 0.55, 
                        max_corr: float = 0.80,
                        max_candidates: int = 25) -> Dict:
        """
        Escanea el universo de MT5 (dinámico o lista pasada), calcula Hurst R/S y descorrelación.
        Retorna el diccionario con activos aprobados y guarda 'results/active_assets.json'.
        """
        if not candidates:
            candidates = self.discover_mt5_candidates(max_candidates=max_candidates)
            
        total_cands = len(candidates)
        cand_names = [c["nombre"] for c in candidates]
        
        print(f"\n================================================================================", flush=True)
        print(f"[FASE 0] DISCOVERY & SCREENING CUANTITATIVO MT5 (TOTAL: {total_cands} ACTIVOS)", flush=True)
        print(f"================================================================================", flush=True)
        print(f"  > Lista de candidatos a evaluar: {', '.join(cand_names)}", flush=True)
        print(f"  > Tiempo estimado de escaneo: ~{max(2, total_cands * 1)} segundos", flush=True)
        print(f"================================================================================\n", flush=True)
        
        # 1. Recopilar series históricas de cierre
        closes_dict = {}
        hurst_results = {}
        approved_items = []
        
        for idx, item in enumerate(candidates, 1):
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
                
                status = "[OK] PERSISTENTE" if h_val >= min_hurst else "[NO] RUIDO (Descartar)"
                print(f"  [{idx:>2}/{total_cands}] [{name:<12}] Hurst R/S H = {h_val:.4f} -> {status}", flush=True)
                if h_val >= min_hurst:
                    approved_items.append(item)
            else:
                logging.warning(f"  [{idx:>2}/{total_cands}] No se pudieron descargar datos de MT5 para {name} ({ticker}).")
                
        # 2. Filtrar candidatos por Exponente de Hurst
        aprobados_hurst = [item["nombre"] for item in approved_items]
        
        if not aprobados_hurst:
            print("\nWarning: Ningun activo supero el umbral de Hurst estricto. Tomando el Top de mayor Hurst...", flush=True)
            sorted_by_h = sorted(hurst_results.items(), key=lambda x: x[1], reverse=True)
            aprobados_hurst = [x[0] for x in sorted_by_h[:max(2, len(sorted_by_h))]]
            approved_items = [c for c in candidates if c["nombre"] in aprobados_hurst]

        # 3. Matriz de Descorrelación entre Aprobados
        cesta_final = []
        final_items = []
        
        if len(aprobados_hurst) > 1 and closes_dict:
            df_rets = pd.DataFrame({k: closes_dict[k] for k in aprobados_hurst if k in closes_dict}).dropna()
            corr_matrix = df_rets.corr().abs()
            
            print(f"\nMATRIZ DE DESCORRELACION CROSS-ASSET (Aprobados Hurst >= {min_hurst}):", flush=True)
            print(corr_matrix.round(2), flush=True)
            
            # Selección codiciosa (Greedy) para evitar pares con correlación > max_corr
            for item in approved_items:
                name = item["nombre"]
                keep = True
                for selected in cesta_final:
                    if name in corr_matrix.columns and selected in corr_matrix.index:
                        if corr_matrix.loc[name, selected] > max_corr:
                            keep = False
                            print(f"  [-] [{name}] descartado por alta correlacion ({corr_matrix.loc[name, selected]:.2f}) con {selected}", flush=True)
                            break
                if keep:
                    cesta_final.append(name)
                    final_items.append(item)
        else:
            cesta_final = aprobados_hurst
            final_items = approved_items

        print(f"\n[EXITO] CESTA FINAL DE ACTIVOS APROBADOS DE PRODUCCION ({len(cesta_final)} activos):", flush=True)
        for act in cesta_final:
            print(f"  -> {act} (Hurst R/S: {hurst_results.get(act, 0.50):.4f})", flush=True)
            
        output_data = {
            "cesta_final": cesta_final,
            "items_finales": final_items,
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
        screener = AssetScreener(conn)
        screener.screen_universe()
        conn.shutdown()
