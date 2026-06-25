import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation.backtester import TripleBarrierBacktester

def simulate_portfolio(activo="EURUSD", capital_inicial=10000.0, riesgo_por_trade=0.01):
    print(f"\n💰 INICIANDO PORTFOLIO BACKTESTER PARA {activo} 💰")
    print(f"Capital Inicial: ${capital_inicial:,.2f}")
    print(f"Riesgo Base (Kelly Dinámico): {riesgo_por_trade*100}%")
    
    # 1. Ejecutar el Backtester Científico para obtener las operaciones "Base 1.0"
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    tester = TripleBarrierBacktester(activo=activo, data_dir=data_dir, results_dir=results_dir)
    modelos = ['ARIMAX', 'RANDOM_FOREST', 'XGBOOST', 'LSTM', 'BILSTM', 'ARIMA_LSTM', 'LSTM_RF']
    bancos = ['Precio_Puro', 'Precio_Volumen', 'Tecnicos', 'Macros', 'Globales', 'Hibrido_Precio_Tec_Vol', 'Macro_Vol', 'Kitchen_Sink_Total', 'Total']
    
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
    print(f"ROI Compuesto:   {roi_total:.2%}")
    
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
    plt.show()

if __name__ == "__main__":
    # Configura aquí tu cuenta de banco y tu riesgo!
    CAPITAL = 10000.0       # USD en tu broker
    RIESGO_PCT = 0.01       # 1% por trade base (se escala a 0.5% o 2% por Kelly)
    
    activos = ["EURUSD", "SP500", "Oro", "ECH"]
    
    for activo_actual in activos:
        simulate_portfolio(activo=activo_actual, capital_inicial=CAPITAL, riesgo_por_trade=RIESGO_PCT)
