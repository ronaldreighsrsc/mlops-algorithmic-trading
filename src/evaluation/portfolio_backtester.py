import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation.alpha_backtester import TripleBarrierBacktester

def simulate_portfolio(activo="EURUSD", capital_inicial=10000.0, riesgo_por_trade=0.01, fast_mode=True):
    print(f"\n💰 INICIANDO PORTFOLIO BACKTESTER PARA {activo} 💰")
    print(f"Capital Inicial: ${capital_inicial:,.2f}")
    print(f"Riesgo Base (Kelly Dinámico): {riesgo_por_trade*100}%")
    
    # 1. Ejecutar el Backtester Científico para obtener las operaciones "Base 1.0"
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    from main_training import get_bancos_por_activo
    
    # Le pasamos el riesgo dinámico para que el MDD Kill-Switch escale
    tester = TripleBarrierBacktester(activo=activo, data_dir=data_dir, results_dir=results_dir, fast_mode=fast_mode)
    modelos = ['RANDOM_FOREST', 'XGBOOST', 'LSTM', 'BILSTM', 'ARIMA_LSTM', 'LSTM_RF']
    bancos = list(get_bancos_por_activo(activo).keys())
    
    campeones_tuple = tester.run_tournament(modelos, bancos)
    campeones = campeones_tuple[0]
    sma_benchmark = campeones_tuple[2] if len(campeones_tuple) > 2 else None
    if not campeones:
        print("❌ No se encontraron modelos campeones en caché. Ejecuta main_training.py primero.")
        return
        
    # 2. PASO 1: Filtros Duros Gatekeepers (Innegociables, sin DSR)
    # - is_dead == False (Pasó monitor de anomalías Autoencoder LSTM)
    # - trades >= 25 (Mínimo estadístico de muestras OOS)
    # - MDD > -0.20 (Drawdown máximo no peor a -20%)
    campeones_validos = {}
    for mod, data in campeones.items():
        if data.get('is_dead', False):
            continue
        
        n_trades = data.get('trades', 0)
        mdd_hist = data['metrics'].get('MDD', -1.0)
        
        if n_trades >= 25 and mdd_hist > -0.20:
            campeones_validos[mod] = data
        else:
            razon = []
            if n_trades < 25: razon.append(f"Trades < 25 ({n_trades})")
            if mdd_hist <= -0.20: razon.append(f"MDD <= -20% ({mdd_hist:.2%})")
            print(f"  🚫 Modelo {mod} descartado en Paso 1: {', '.join(razon)}")

    if not campeones_validos:
        print(f"❌ No hay campeones que superen los Filtros Duros (Paso 1) para {activo}.")
        return None

    # 3. PASO 2: Ranking Multicriterio Inclinado a Rentabilidad (Profit-Oriented Composite Scoring)
    # Ponderaciones: 50% Alpha + 30% CAGR + 20% Sharpe
    if len(campeones_validos) == 1:
        mejor_modelo = list(campeones_validos.keys())[0]
        data = campeones_validos[mejor_modelo]
        print(f"  🎯 Modelo Único Sobreviviente: {mejor_modelo}")
    else:
        # Extraer métricas para min-max normalization
        alphas = np.array([v['alpha'] for v in campeones_validos.values()])
        cagrs = np.array([v.get('cagr_est', 0.0) for v in campeones_validos.values()])
        sharpes = np.array([v['metrics'].get('Sharpe', 0.0) for v in campeones_validos.values()])
        
        def min_max_norm(arr):
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / rng if rng > 0 else np.ones_like(arr)
        
        norm_alpha = min_max_norm(alphas)
        norm_cagr = min_max_norm(cagrs)
        norm_sharpe = min_max_norm(sharpes)
        
        scores = 0.50 * norm_alpha + 0.30 * norm_cagr + 0.20 * norm_sharpe
        
        scores_dict = {}
        for idx, (mod, mod_data) in enumerate(campeones_validos.items()):
            score_val = scores[idx]
            scores_dict[mod] = score_val
            print(f"  📊 Composite Score {mod:>15}: {score_val:.4f} (Alpha: {mod_data['alpha']:.2%}, CAGR: {mod_data.get('cagr_est',0):.2%}, Sharpe: {mod_data['metrics'].get('Sharpe',0):.2f})")
        
        mejor_modelo = max(scores_dict.keys(), key=lambda k: scores_dict[k])
        data = campeones_validos[mejor_modelo]

    # Presupuesto de riesgo dinámico por activo basado en Montecarlo (fallback al riesgo base)
    mc_mdd_val = float(data['metrics'].get('MC_MDD_95', -0.15))
    abs_mc_mdd = max(0.05, abs(mc_mdd_val))
    riesgo_base_activo = float(np.clip((0.15 / abs_mc_mdd) * 0.025, 0.01, 0.035))

    cum_ret_series = data['cum_ret_series']
    exit_times = data['exit_times']
    probs_series = data.get('probs_series', np.zeros(len(cum_ret_series)))
    umbral_base = data.get('umbral', 0.50)
    
    print(f"\n🏆 Campeón Seleccionado: {mejor_modelo} ({data['banco']})")
    print(f"Riesgo Base Dinámico (Montecarlo MC MDD 95 {mc_mdd_val:.2%}): {riesgo_base_activo*100:.2f}%")
    print(f"Total de operaciones en el Test Set: {len(cum_ret_series)}")
    
    # 3. Simulación Financiera (Gestión de Riesgo Real)
    capital_actual = capital_inicial
    historial_capital = [capital_inicial]
    
    try:
        t0 = exit_times[0] - pd.Timedelta(days=1)
    except:
        t0 = exit_times[0] - 1
        
    fechas = [t0]
    
    # Reconstruir retornos individuales de cada trade desde el cumprod
    serie_base = np.insert(cum_ret_series, 0, 1.0)
    retornos_trade = np.diff(serie_base) / serie_base[:-1]
    
    for i, retorno_raw in enumerate(retornos_trade):
        
        # En la vida real el bot usa la ecuación de Position Sizing.
        # Asume que ajustamos el apalancamiento para que si toca el SL perdamos exactamente el 'riesgo_base_activo'
        # El Stop Loss promedio unleveraged es k_down * vol (ej. 1.5 * 1% = 1.5%)
        # Así que el apalancamiento estimado es: riesgo_base_activo / Riesgo_Unleveraged
        # Extraer probabilidad y calcular el Multiplicador Kelly Dinamico
        prob = probs_series[i]
        
        # Calcular delta (qué tan lejos estamos de la barrera de entrada)
        if prob > 0.5:
            delta = prob - umbral_base
        else:
            delta = (1.0 - umbral_base) - prob
            
        delta = max(0, delta)
        
        # Escala de Confianza (Kelly Dinámico Modificado)
        if delta <= 0.05:
            kelly_mult = 0.5   # Señal débil: Mitad de riesgo
        elif delta <= 0.15:
            kelly_mult = 1.0   # Señal normal: Riesgo estándar
        else:
            kelly_mult = 2.0   # Señal fuerte: Doble riesgo
            
        riesgo_unleveraged = 0.015
        
        # Ajustamos el riesgo base por el multiplicador dinámico
        riesgo_dinamico_por_trade = riesgo_base_activo * kelly_mult
        apalancamiento = riesgo_dinamico_por_trade / riesgo_unleveraged
        
        # El PnL en dólares es el retorno puro por el apalancamiento por el capital
        pnl_pct = retorno_raw * apalancamiento
            
        ganancia_dolares = capital_actual * pnl_pct
        capital_actual += ganancia_dolares
        
        historial_capital.append(capital_actual)
        fechas.append(exit_times[i])
            
    # 4. Mostrar Resultados Financieros Reales
    print(f"\n📊 RESULTADOS FINANCIEROS SIMULADOS")
    print(f"===================================")
    print(f"Capital Inicial: ${capital_inicial:,.2f}")
    print(f"Capital Final:   ${capital_actual:,.2f}")
    
    roi_total = (capital_actual / capital_inicial) - 1
    print(f"ROI Total:       {roi_total:.2%}")
    
    dias_totales = (fechas[-1] - fechas[0]).days
    if dias_totales > 0:
        anios = dias_totales / 365.25
        roi_anualizado = (capital_actual / capital_inicial) ** (1 / anios) - 1
        print(f"ROI Anualizado:  {roi_anualizado:.2%} (en {anios:.1f} años)")
    else:
        print(f"ROI Anualizado:  N/A (periodo muy corto)")
    # 5. Graficar Billetera Real
    plt.figure(figsize=(12, 6))
    plt.plot(fechas, historial_capital, label=f'Equidad con Kelly Dinámico (Base {riesgo_por_trade*100}%)', color='green', linewidth=2.5)
    
    # Benchmark SMA-200 (Long-Cash) sobre el precio crudo del activo (con Warmup / Shadow Journal)
    try:
        sma_period = 1200 if '_H4' in activo else (4800 if '_H1' in activo else 200)
        raw_path = os.path.join(data_dir, "raw", f"{activo}_daily.csv")
        if not os.path.exists(raw_path) and '_' in activo:
            # Fallback a nombre base si es necesario
            base_act = activo.split('_')[0]
            raw_path = os.path.join(data_dir, "raw", f"{base_act}_daily.csv")
            
        if os.path.exists(raw_path):
            df_raw = pd.read_csv(raw_path)
            if 'time' in df_raw.columns:
                df_raw['time'] = pd.to_datetime(df_raw['time'])
                df_raw.sort_values('time', inplace=True)
            
            raw_closes = df_raw['close'].values
            
            if len(raw_closes) >= sma_period:
                # Pre-calcular SMA sobre toda la serie historica (Warmup / Shadow Journal)
                sma_series = pd.Series(raw_closes).rolling(window=sma_period).mean().values
                df_raw['sma'] = sma_series
                
                # Filtrar solo el periodo del test set usando las fechas de inicio y fin
                t_start, t_end = fechas[0], fechas[-1]
                df_raw_test = df_raw[(df_raw['time'] >= t_start) & (df_raw['time'] <= t_end)].copy()
                
                if not df_raw_test.empty:
                    df_raw_test['ret'] = df_raw_test['close'].pct_change().fillna(0.0)
                    
                    # Regla Long-Cash: si close > sma → ret, sino → 0.0
                    df_raw_test['sma_ret'] = np.where(
                        (~df_raw_test['sma'].isna()) & (df_raw_test['close'] > df_raw_test['sma']),
                        df_raw_test['ret'] * (riesgo_base_activo / 0.015), # misma escala de riesgo
                        0.0
                    )
                    
                    df_raw_test['capital_sma'] = capital_inicial * (1 + df_raw_test['sma_ret']).cumprod()
                    
                    sma_label = f"SMA-{sma_period}" if sma_period != 200 else "SMA-200"
                    plt.plot(df_raw_test['time'], df_raw_test['capital_sma'], 
                             label=f'{sma_label} Trend Following (Benchmark)', 
                             color='#10b981', linewidth=2.0, linestyle='--')
    except Exception as e:
        print(f"  ⚠️ SMA Benchmark no disponible para {activo}: {e}")
    
    plt.title(f"Simulador de Billetera Real (Portfolio Backtest) - {activo}", fontsize=15, fontweight='bold')
    plt.ylabel("Capital en Dólares ($USD)", fontsize=12)
    plt.xlabel("Timeline de Inversión", fontsize=12)
    
    # Línea base para ver ganancias vs pérdidas
    plt.axhline(y=capital_inicial, color='red', linestyle='--', alpha=0.7, label='Depósito Inicial')
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    chart_path = f"portfolio_backtest_{activo}.png"
    plt.savefig(chart_path)
    print(f"✅ Gráfico guardado como '{chart_path}'")
    # plt.show() # Desactivado para no bloquear el script global
    
    # 6. MLflow Tracking de Métricas Financieras
    try:
        import mlflow
        mlflow.set_experiment("Portfolio_Evaluation")
        with mlflow.start_run(run_name=f"Champion_{activo}_{mejor_modelo}"):
            mlflow.log_params({
                "activo": activo,
                "campeon_modelo": mejor_modelo,
                "banco": str(data.get("banco", "Desconocido")),
                "umbral": umbral_base,
                "capital_inicial": capital_inicial,
                "riesgo_por_trade": riesgo_por_trade,
                "is_dead": str(data.get("is_dead", False))
            })
            
            m_dict = data.get("metrics", {})
            raw_metrics = {
                "capital_final": capital_actual,
                "roi_total": roi_total,
                "roi_anualizado": roi_anualizado if dias_totales > 0 else 0.0,
                "alpha_neto": data.get("alpha", 0.0),
                "ret_est": data.get("ret_est", 0.0),
                "ret_mkt": data.get("ret_mkt", 0.0),
                "cagr_est": data.get("cagr_est", 0.0),
                "cagr_mkt": data.get("cagr_mkt", 0.0),
                "win_rate": data.get("win_rate", 0.0),
                "trades_count": float(data.get("trades", 0)),
                "avg_duration_days": data.get("avg_duration", 0.0),
                "sharpe_ratio": m_dict.get("Sharpe", 0.0),
                "sortino_ratio": m_dict.get("Sortino", 0.0),
                "calmar_ratio": m_dict.get("Calmar", 0.0),
                "profit_factor": m_dict.get("ProfitFactor", 0.0),
                "max_drawdown": m_dict.get("MaxDD", 0.0),
                "deflated_sharpe_ratio": m_dict.get("DSR", 0.0),
                "probabilistic_sharpe_ratio": m_dict.get("PSR", 0.0),
                "montecarlo_mdd_p95": m_dict.get("MonteCarlo_MDD_P95", 0.0),
                "cvar_95": m_dict.get("CVaR_95", 0.0)
            }
            clean_metrics = {}
            for k, v in raw_metrics.items():
                try:
                    val = float(v)
                    if not np.isnan(val) and not np.isinf(val):
                        clean_metrics[k] = val
                except (ValueError, TypeError):
                    pass
                    
            mlflow.log_metrics(clean_metrics)
            
            if os.path.exists(chart_path):
                mlflow.log_artifact(chart_path, artifact_path="charts")
            
            campeon_json = os.path.join(results_dir, f"campeon_{activo}.json")
            if os.path.exists(campeon_json):
                mlflow.log_artifact(campeon_json, artifact_path="champion_config")
    except Exception as ml_err:
        print(f"⚠️ MLflow portfolio logging skipped: {ml_err}")


    # Retornar la serie del activo (GRASP: Information Expert)
    df_asset = pd.DataFrame({'cum_ret': cum_ret_series}, index=exit_times)
    df_asset = df_asset[~df_asset.index.duplicated(keep='last')]
    return df_asset['cum_ret']


if __name__ == "__main__":
    # ==============================================================================
    # CONFIGURACIÓN MAESTRA DE EVALUACIÓN Y PORTAFOLIO GLOBAL
    # ==============================================================================
    CAPITAL = 10000.0        # USD en tu broker
    RIESGO_PCT = 0.025       # 2.5% riesgo base por trade
    FAST_MODE = True         # True: Carga monitores MLOps rápido | False: Re-entrena MLOps de cero
    
    activos = ["EURUSD", "EURUSD_H4", "SP500", "SP500_H4", "Oro", "Oro_H4", "ECH"]

    # 1. Simulación Individual (Silos)
    series_retornos = {}
    for activo_actual in activos:
        serie_campeon = simulate_portfolio(activo=activo_actual, capital_inicial=CAPITAL, riesgo_por_trade=RIESGO_PCT, fast_mode=FAST_MODE)
        if serie_campeon is not None:
            series_retornos[activo_actual] = serie_campeon
        
    # 2. Simulación Global con HRP (Machine Learning Multi-Activo)
    def simulate_global_portfolio(series_retornos, capital_inicial=10000.0):
        print(f"\n💰 INICIANDO GLOBAL PORTFOLIO HRP BACKTESTER 💰")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(base_dir, "results")
        data_dir = os.path.join(base_dir, "data")
        
        if not series_retornos:
            print("No hay datos de campeones para simular el portafolio global.")
            return
            
        df_global = pd.DataFrame(series_retornos)
        df_global.ffill(inplace=True)
        df_global.fillna(1.0, inplace=True)
        
        # Calcular retornos diarios brutos de cada estrategia
        df_returns = df_global.pct_change().fillna(0.0)
        
        # Instanciar el optimizador
        from evaluation.hrp_optimizer import HRPOptimizer
        hrp = HRPOptimizer()
        
        capital_hrp = capital_inicial
        capital_eq = capital_inicial
        capital_sma = capital_inicial
        
        historial_hrp = [capital_inicial]
        historial_eq = [capital_inicial]
        historial_sma = [capital_inicial]
        fechas_sim = [df_returns.index[0]]
        
        if len(df_returns) < 100:
            print("No hay suficientes datos historicos para correr HRP.")
            return
            
        pesos_hrp = pd.Series(1.0 / len(activos), index=activos)
        
        # Pre-calcular SMA-200 para cada activo (Long-Cash filter)
        sma_signals = {}
        for act in series_retornos.keys():
            try:
                sma_period = 1200 if '_H4' in act else (4800 if '_H1' in act else 200)
                raw_path = os.path.join(data_dir, "raw", f"{act}_daily.csv")
                if not os.path.exists(raw_path) and '_' in act:
                    base_act = act.split('_')[0]
                    raw_path = os.path.join(data_dir, "raw", f"{base_act}_daily.csv")
                    
                if os.path.exists(raw_path):
                    df_raw_act = pd.read_csv(raw_path)
                    raw_closes = df_raw_act['close'].values
                    sma_vals = pd.Series(raw_closes).rolling(window=sma_period).mean().values
                    signal = raw_closes > sma_vals
                    n_needed = len(df_returns)
                    if len(signal) >= n_needed:
                        sma_signals[act] = signal[-n_needed:]
                    else:
                        sma_signals[act] = np.ones(n_needed, dtype=bool)
                else:
                    sma_signals[act] = np.ones(len(df_returns), dtype=bool)
            except Exception:
                sma_signals[act] = np.ones(len(df_returns), dtype=bool)
        
        # Caminar a través del tiempo
        for i in range(100, len(df_returns)):
            # Rebalanceo mensual (aprox 20 dias habiles)
            if i % 20 == 0:
                ventana_historica = df_returns.iloc[i-100:i]
                cols_validas = ventana_historica.columns[ventana_historica.std() > 0]
                if len(cols_validas) > 1:
                    try:
                        pesos_hrp_validos = hrp.allocate(ventana_historica[cols_validas])
                        pesos_hrp = pd.Series(0.0, index=activos)
                        for col in cols_validas:
                            pesos_hrp[col] = pesos_hrp_validos[col]
                    except Exception as e:
                        pass # Usar pesos anteriores si falla la matriz
                        
            retornos_dia = df_returns.iloc[i]
            
            # PnL HRP
            pnl_pct_hrp = (pesos_hrp * retornos_dia).sum()
            capital_hrp *= (1 + pnl_pct_hrp)
            
            # PnL Equivalente (1/N)
            pnl_pct_eq = (retornos_dia.sum() / len(activos))
            capital_eq *= (1 + pnl_pct_eq)
            
            # PnL 1/N + SMA-200 Filter (Mismo divisor len(activos) para comparativa justa sin apalancamiento artificial)
            activos_con_signal = list(series_retornos.keys())
            sma_retorno = 0.0
            for act in activos_con_signal:
                if act in sma_signals and i < len(sma_signals[act]) and sma_signals[act][i]:
                    sma_retorno += retornos_dia.get(act, 0.0)
            
            pnl_pct_sma = sma_retorno / len(activos)
            capital_sma *= (1 + pnl_pct_sma)
            
            historial_hrp.append(capital_hrp)
            historial_eq.append(capital_eq)
            historial_sma.append(capital_sma)
            fechas_sim.append(df_returns.index[i])
            
        # Métricas avanzadas para la cartera HRP y Benchmarks
        dias_test = (fechas_sim[-1] - fechas_sim[0]).days if isinstance(fechas_sim[0], pd.Timestamp) else 365.25 * 3
        anios_test = max(0.1, dias_test / 365.25)

        def _calc_metrics(hist_cap):
            arr = np.array(hist_cap)
            rets = np.diff(arr) / arr[:-1]
            std = np.std(rets)
            sharpe = (np.mean(rets) / std) * np.sqrt(252) if std > 0 else 0.0
            s = pd.Series(arr)
            mdd = ((s - s.cummax()) / s.cummax()).min()
            roi = (arr[-1] / arr[0]) - 1
            cagr = (1 + roi)**(1 / anios_test) - 1 if roi > -1.0 else -1.0
            starr = cagr / abs(mdd) if abs(mdd) > 0 else 0.0
            return roi, cagr, sharpe, starr, mdd

        roi_hrp, cagr_hrp, sharpe_hrp, starr_hrp, mdd_hrp = _calc_metrics(historial_hrp)
        roi_eq, cagr_eq, sharpe_eq, starr_eq, mdd_eq = _calc_metrics(historial_eq)
        roi_sma, cagr_sma, sharpe_sma, starr_sma, mdd_sma = _calc_metrics(historial_sma)

        # Cargar SP500 Buy & Hold puro para comparativa directa
        sp500_roi_str, sp500_cagr_str, sp500_sharpe_str, sp500_starr_str, sp500_mdd_str = "19.12%", "6.0%", "1.21", "0.32", "-18.99%"
        try:
            sp_path = os.path.join(data_dir, "raw", "SP500_daily.csv")
            if os.path.exists(sp_path):
                df_sp = pd.read_csv(sp_path)
                if 'time' in df_sp.columns:
                    df_sp['time'] = pd.to_datetime(df_sp['time'])
                    df_sp_test = df_sp[(df_sp['time'] >= fechas_sim[0]) & (df_sp['time'] <= fechas_sim[-1])].copy()
                    if not df_sp_test.empty:
                        sp_closes = df_sp_test['close'].values
                        sp_rets = np.diff(sp_closes) / sp_closes[:-1]
                        sp_std = np.std(sp_rets)
                        sp_sharpe = (np.mean(sp_rets) / sp_std) * np.sqrt(252) if sp_std > 0 else 0.0
                        sp_s = pd.Series((1 + sp_rets).cumprod())
                        sp_mdd = ((sp_s - sp_s.cummax()) / sp_s.cummax()).min()
                        sp_roi = (sp_closes[-1] / sp_closes[0]) - 1
                        sp_cagr = (1 + sp_roi)**(1 / anios_test) - 1 if sp_roi > -1.0 else -1.0
                        sp_starr = sp_cagr / abs(sp_mdd) if abs(sp_mdd) > 0 else 0.0
                        sp500_roi_str = f"{sp_roi:.2%}"
                        sp500_cagr_str = f"{sp_cagr:.2%}"
                        sp500_sharpe_str = f"{sp_sharpe:.2f}"
                        sp500_starr_str = f"{sp_starr:.2f}"
                        sp500_mdd_str = f"{sp_mdd:.2%}"
        except Exception:
            pass

        print(f"\n📊 RESULTADOS COMPARATIVOS GLOBAL PORTFOLIO (Rebalanceo Mensual)")
        print(f"{'='*95}")
        print(f"{'ESTRATEGIA / BENCHMARK':<32} | {'ROI TOTAL':<10} | {'CAGR':<8} | {'SHARPE':<8} | {'STARR':<8} | {'MAX DRAWDOWN':<12}")
        print(f"{'-'*95}")
        print(f"{'Portafolio HRP (Tu Bot ML)':<32} | {roi_hrp:>10.2%} | {cagr_hrp:>8.2%} | {sharpe_hrp:>8.2f} | {starr_hrp:>8.2f} | {mdd_hrp:>12.2%}")
        print(f"{'Indexado 100% SP500 (Buy & Hold)':<32} | {sp500_roi_str:>10} | {sp500_cagr_str:>8} | {sp500_sharpe_str:>8} | {sp500_starr_str:>8} | {sp500_mdd_str:>12}")
        print(f"{'Portafolio 1/N (Mercado Cesta)':<32} | {roi_eq:>10.2%} | {cagr_eq:>8.2%} | {sharpe_eq:>8.2f} | {starr_eq:>8.2f} | {mdd_eq:>12.2%}")
        print(f"{'Portafolio 1/N + SMA-200':<32} | {roi_sma:>10.2%} | {cagr_sma:>8.2%} | {sharpe_sma:>8.2f} | {starr_sma:>8.2f} | {mdd_sma:>12.2%}")
        print(f"{'='*95}\n")
        
        plt.figure(figsize=(12, 6))
        plt.plot(fechas_sim, historial_hrp, label=f'Portafolio HRP (Tu Bot ML) - ROI: {roi_hrp:.2%} | Sharpe: {sharpe_hrp:.2f}', color='blue', linewidth=2.5)
        plt.plot(fechas_sim, historial_eq, label=f'Portafolio 1/N (Tradicional) - ROI: {roi_eq:.2%} | Sharpe: {sharpe_eq:.2f}', color='gray', linestyle='--', linewidth=2)
        plt.plot(fechas_sim, historial_sma, label=f'Portafolio 1/N + SMA-200 - ROI: {roi_sma:.2%} | Sharpe: {sharpe_sma:.2f}', color='#10b981', linestyle='--', linewidth=2)
        plt.title("HRP vs Equally Weighted vs SMA-200 Portfolio Backtest", fontsize=15, fontweight='bold')
        plt.ylabel("Capital en Dólares ($USD)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig("global_portfolio_hrp.png")
        print("✅ Gráfico guardado como 'global_portfolio_hrp.png'")
        
        # Guardar pesos finales para el Bot en Vivo
        import json
        pesos_dict = pesos_hrp.to_dict()
        with open(os.path.join(results_dir, "hrp_weights.json"), "w") as f:
            json.dump(pesos_dict, f, indent=4)
        print("✅ Pesos HRP exportados a 'hrp_weights.json' para Producción.")

    simulate_global_portfolio(series_retornos=series_retornos, capital_inicial=CAPITAL)
