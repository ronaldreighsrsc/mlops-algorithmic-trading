import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation.backtester import TripleBarrierBacktester

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
    tester = TripleBarrierBacktester(activo=activo, data_dir=data_dir, results_dir=results_dir, fast_mode=fast_mode, base_risk=riesgo_por_trade)
    modelos = ['RANDOM_FOREST', 'XGBOOST', 'LSTM', 'BILSTM', 'ARIMA_LSTM', 'LSTM_RF']
    bancos = list(get_bancos_por_activo(activo).keys())
    
    campeones_tuple = tester.run_tournament(modelos, bancos)
    campeones = campeones_tuple[0]
    if not campeones:
        print("❌ No se encontraron modelos campeones en caché. Ejecuta main_training.py primero.")
        return
        
    # 2. Elegir el mejor modelo (El Campeón de Campeones por Alpha)
    mejor_modelo = max(campeones.keys(), key=lambda k: campeones[k]['alpha'])
    data = campeones[mejor_modelo]
    cum_ret_series = data['cum_ret_series']
    exit_times = data['exit_times']
    probs_series = data.get('probs_series', np.zeros(len(cum_ret_series)))
    umbral_base = data.get('umbral', 0.50)
    
    print(f"\n🏆 Campeón Seleccionado: {mejor_modelo} ({data['banco']})")
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
        # Asume que ajustamos el apalancamiento para que si toca el SL perdamos exactamente el 'riesgo_por_trade' (ej. 1%)
        # El Stop Loss promedio unleveraged es k_down * vol (ej. 1.5 * 1% = 1.5%)
        # Así que el apalancamiento estimado es: riesgo_por_trade / Riesgo_Unleveraged
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
        riesgo_dinamico_por_trade = riesgo_por_trade * kelly_mult
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
    plt.title(f"Simulador de Billetera Real (Portfolio Backtest) - {activo}", fontsize=15, fontweight='bold')
    plt.ylabel("Capital en Dólares ($USD)", fontsize=12)
    plt.xlabel("Timeline de Inversión", fontsize=12)
    
    # Línea base para ver ganancias vs pérdidas
    plt.axhline(y=capital_inicial, color='red', linestyle='--', alpha=0.7, label='Depósito Inicial')
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"portfolio_backtest_{activo}.png")
    print(f"✅ Gráfico guardado como 'portfolio_backtest_{activo}.png'")
    # plt.show() # Desactivado para no bloquear el script global
    
    # Retornar la serie del activo (GRASP: Information Expert)
    df_asset = pd.DataFrame({'cum_ret': cum_ret_series}, index=exit_times)
    df_asset = df_asset[~df_asset.index.duplicated(keep='last')]
    return df_asset['cum_ret']

if __name__ == "__main__":
    # Configura aquí tu cuenta de banco y tu riesgo!
    CAPITAL = 10000.0       # USD en tu broker
    RIESGO_PCT = 0.025       
    
    activos = ["EURUSD", "SP500", "Oro", "ECH"]
    
    # 1. Simulación Individual (Silos)
    series_retornos = {}
    for activo_actual in activos:
        serie_campeon = simulate_portfolio(activo=activo_actual, capital_inicial=CAPITAL, riesgo_por_trade=RIESGO_PCT, fast_mode=True)
        if serie_campeon is not None:
            series_retornos[activo_actual] = serie_campeon
        
    # 2. Simulación Global con HRP (Machine Learning Multi-Activo)
    def simulate_global_portfolio(series_retornos, capital_inicial=10000.0):
        print(f"\n💰 INICIANDO GLOBAL PORTFOLIO HRP BACKTESTER 💰")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(base_dir, "results")
        
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
        
        historial_hrp = [capital_inicial]
        historial_eq = [capital_inicial]
        fechas_sim = [df_returns.index[0]]
        
        if len(df_returns) < 100:
            print("No hay suficientes datos historicos para correr HRP.")
            return
            
        pesos_hrp = pd.Series(1.0 / len(activos), index=activos)
        
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
            
            historial_hrp.append(capital_hrp)
            historial_eq.append(capital_eq)
            fechas_sim.append(df_returns.index[i])
            
        print(f"\n📊 RESULTADOS FINALES GLOBAL PORTFOLIO (Rebalanceo Mensual)")
        print(f"============================================================")
        print(f"Capital Inicial: ${capital_inicial:,.2f}")
        print(f"Capital Final HRP:   ${capital_hrp:,.2f} (ROI: {(capital_hrp/capital_inicial - 1):.2%})")
        print(f"Capital Final 1/N:   ${capital_eq:,.2f} (ROI: {(capital_eq/capital_inicial - 1):.2%})")
        
        plt.figure(figsize=(12, 6))
        plt.plot(fechas_sim, historial_hrp, label='Portafolio HRP (López de Prado)', color='blue', linewidth=2.5)
        plt.plot(fechas_sim, historial_eq, label='Portafolio 1/N (Tradicional)', color='gray', linestyle='--', linewidth=2)
        plt.title("HRP vs Equally Weighted Portfolio Backtest", fontsize=15, fontweight='bold')
        plt.ylabel("Capital en Dólares ($USD)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
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
