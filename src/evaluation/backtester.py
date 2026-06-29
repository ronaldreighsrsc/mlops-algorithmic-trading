import sys
import os

# Asegurar que el modulo raiz 'src' este en sys.path para importaciones
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import json
import traceback
import matplotlib.pyplot as plt
import json
from scipy.stats import norm, skew, kurtosis
from execution.risk_manager import HybridRiskMonitor
import math
import warnings

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

class TripleBarrierBacktester:
    """
    Motor de Backtesting Financiero.
    Simula inversiones reales usando las probabilidades generadas por los modelos,
    aplicando comisiones, slippage, limites de volatilidad y Monitor Híbrido LSTM.
    """
    def __init__(self, activo: str, data_dir: str, results_dir: str, fast_mode: bool = False):
        self.activo = activo
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.fast_mode = fast_mode
        
        # Friccion Institucional y Umbrales
        self.confidence_threshold = 0.60  # Solo operar si la probabilidad es > 60%
        
        # Parametros Dinámicos por Activo
        if activo == "Oro":
            self.k_up, self.k_down = 2.5, 1.5
            self.costo_movimiento = 0.00006  # ~0.012% ida y vuelta
            self.bilateral = False  # Solo largos (activo alcista por naturaleza)
        elif activo == "SP500":
            self.k_up, self.k_down = 2.0, 1.5
            self.costo_movimiento = 0.00010  # ~0.02% ida y vuelta
            self.bilateral = False  # Solo largos (activo alcista por naturaleza)
        else: # EURUSD
            self.k_up, self.k_down = 2.0, 1.5
            # Spread real Quantfury (1.16140 - 1.16128) = 1.2 pips -> ~0.01% ida y vuelta
            self.costo_movimiento = 0.00005  # ~0.01% ida y vuelta
            self.bilateral = True   # Largos Y Cortos (par lateral)
            
        self.max_hold = 10 
        self.risk_free_rate_annual = 0.05 # 5% Tasa Libre de Riesgo USA

    def calculate_advanced_metrics(self, returns, num_trades, cum_returns, confidence=0.95):
        if len(returns) == 0: 
            return {k: 0 for k in ['Sharpe', 'VaR_95', 'CVaR_95', 'STARR', 'PSR', 'MDD', 'Skew', 'Kurt', 'Std', 'N']}
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        # Sharpe anualizado (asumimos ~252 dias de trading al anio)
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0
        
        var_level = np.percentile(returns, (1 - confidence) * 100)
        cvar_level = returns[returns <= var_level].mean() if len(returns[returns <= var_level]) > 0 else 0
        starr = mean_ret / abs(cvar_level) if cvar_level != 0 else 0
        
        n = len(returns)
        if n < 2:
            return {'Sharpe': sharpe, 'VaR_95': var_level, 'CVaR_95': cvar_level, 'STARR': starr, 'PSR': 0, 'MDD': 0, 'Skew': 0, 'Kurt': 0, 'Std': std_ret, 'N': n}
            
        skew_ret, kurt_ret = skew(returns), kurtosis(returns)
        sigma_sr = np.sqrt((1 / (n - 1)) * (1 + 0.5 * sharpe**2 - skew_ret * sharpe + (kurt_ret / 4) * sharpe**2))
        psr = norm.cdf(sharpe / sigma_sr) if sigma_sr != 0 else 0
        
        cum_series = pd.Series(cum_returns)
        running_max = cum_series.cummax()
        drawdown = (cum_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {'Sharpe': sharpe, 'VaR_95': var_level, 'CVaR_95': cvar_level, 
                'STARR': starr, 'PSR': psr, 'MDD': max_drawdown,
                'Skew': skew_ret, 'Kurt': kurt_ret, 'Std': std_ret, 'N': n}

    def run_montecarlo_and_dsr(self, returns_array: np.ndarray, num_trades: int, 
                               sharpe_obs: float, skew_obs: float, kurt_obs: float, 
                               n_models_tested: int = 21, n_simulations: int = 1000):
        if len(returns_array) < 5 or sharpe_obs == 0:
            return {'DSR': 0.0, 'MC_MDD_95': 0.0}

        euler_gamma = 0.5772156649
        z_param = (1 - euler_gamma) * norm.ppf(1 - 1/n_models_tested) + euler_gamma * norm.ppf(1 - (1/n_models_tested) * np.exp(-1))
        std_sharpes = 0.5
        expected_max_sr = std_sharpes * z_param
        
        sr_var = (1 - skew_obs * sharpe_obs + (kurt_obs - 1)/4 * sharpe_obs**2) / (num_trades - 1)
        if sr_var > 0:
            dsr_z_score = (sharpe_obs - expected_max_sr) / np.sqrt(sr_var)
            dsr = norm.cdf(dsr_z_score)
        else:
            dsr = 0.0

        mdd_simulations = []
        for _ in range(n_simulations):
            sim_returns = np.random.choice(returns_array, size=len(returns_array), replace=True)
            sim_cum_returns = (1 + sim_returns).cumprod()
            sim_series = pd.Series(sim_cum_returns)
            running_max = sim_series.cummax()
            drawdowns = (sim_series - running_max) / running_max
            mdd_simulations.append(drawdowns.min())

        mc_mdd_95 = np.percentile(mdd_simulations, 5)

        return {'DSR': dsr, 'MC_MDD_95': mc_mdd_95}

    def simulate_trades(self, df, probabilities, hybrid_monitor=None, is_training_phase=False, macro_cols=None):
        """Simula las operaciones evaluando HybridRiskMonitor y Single-Position (sin superposición).
        Si self.bilateral=True (EURUSD), también abre posiciones en CORTO cuando
        la probabilidad es muy baja (el modelo predice caída)."""
        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        vols = df['EGARCH_Vol'].values
        results = []
        rolling_metrics = []
        is_dead = False
        cooldown_remaining = 0
        
        # Variables para calcular el Maximum Drawdown (MDD) y detectar Muerte Permanente
        current_equity = 1.0
        peak_equity = 1.0
        
        i = 0
        while i < len(closes) - self.max_hold - 1:
            if is_dead:
                break
                
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                i += 1
                continue
                
            if i >= len(probabilities):
                break
            
            # Determinar dirección de la señal
            prob = probabilities[i]
            is_long = prob > self.confidence_threshold
            is_short = self.bilateral and prob < (1.0 - self.confidence_threshold)
            
            if is_long or is_short:
                entry_price = opens[i + 1] 
                vol_entry = vols[i] 
                
                # --- HMM MACRO REGIME CHECK ---
                risk_multiplier = 1.0
                if hybrid_monitor is not None and hybrid_monitor.hmm_model is not None and macro_cols:
                    try:
                        X_macro = df[macro_cols].iloc[i].values
                        risk_multiplier = hybrid_monitor.check_macro_regime(X_macro)
                    except Exception:
                        pass
                
                if pd.isna(vol_entry) or vol_entry == 0:
                    i += 1
                    continue
                
                if is_long:
                    # LARGO: TP arriba, SL abajo
                    tp_level = entry_price * (1 + self.k_up * (vol_entry/100))
                    sl_level = entry_price * (1 - self.k_down * (vol_entry/100))
                else:
                    # CORTO: TP abajo, SL arriba (invertido)
                    tp_level = entry_price * (1 - self.k_up * (vol_entry/100))
                    sl_level = entry_price * (1 + self.k_down * (vol_entry/100))
                
                trade_duration = 0
                net_ret = 0
                direction = 'LONG' if is_long else 'SHORT'
                
                for j in range(1, self.max_hold + 1):
                    curr_close = closes[i + j]
                    curr_high = highs[i + j]
                    curr_low = lows[i + j]
                    
                    if is_long:
                        hit_tp = curr_high >= tp_level
                        hit_sl = curr_low <= sl_level
                    else:
                        # CORTO: TP se toca cuando el LOW baja hasta el TP
                        #         SL se toca cuando el HIGH sube hasta el SL
                        hit_tp = curr_low <= tp_level
                        hit_sl = curr_high >= sl_level
                    
                    if hit_tp and hit_sl:
                        # En caso de duda, asumimos el peor escenario (SL)
                        if is_long:
                            raw_ret = (sl_level / entry_price) - 1
                        else:
                            raw_ret = (entry_price / sl_level) - 1  # Short: ganamos si baja
                        net_ret = ((1 + raw_ret) * (1 - self.costo_movimiento)**2 - 1) * risk_multiplier
                        results.append({'idx': i+j, 'ret': net_ret, 'type': f'SL (Dual) [{direction}]', 'duration': j})
                        trade_duration = j
                        break
                    elif hit_tp:
                        if is_long:
                            raw_ret = (tp_level / entry_price) - 1
                        else:
                            raw_ret = (entry_price / tp_level) - 1
                        net_ret = ((1 + raw_ret) * (1 - self.costo_movimiento)**2 - 1) * risk_multiplier
                        results.append({'idx': i+j, 'ret': net_ret, 'type': f'TP [{direction}]', 'duration': j, 'prob': prob})
                        trade_duration = j
                        break
                    elif hit_sl:
                        if is_long:
                            raw_ret = (sl_level / entry_price) - 1
                        else:
                            raw_ret = (entry_price / sl_level) - 1
                        net_ret = ((1 + raw_ret) * (1 - self.costo_movimiento)**2 - 1) * risk_multiplier
                        results.append({'idx': i+j, 'ret': net_ret, 'type': f'SL [{direction}]', 'duration': j, 'prob': prob})
                        trade_duration = j
                        break
                    elif j == self.max_hold:
                        if is_long:
                            raw_ret = (curr_close / entry_price) - 1
                        else:
                            raw_ret = (entry_price / curr_close) - 1
                        net_ret = ((1 + raw_ret) * (1 - self.costo_movimiento)**2 - 1) * risk_multiplier
                        results.append({'idx': i+j, 'ret': net_ret, 'type': f'TIME [{direction}]', 'duration': j, 'prob': prob})
                        trade_duration = j
                        break
                        
                # Actualizar Equity y verificar Hard Kill-Switch (Muerte Permanente)
                current_equity *= (1 + net_ret)
                if current_equity > peak_equity:
                    peak_equity = current_equity
                    
                drawdown = (current_equity / peak_equity) - 1.0
                if drawdown <= -0.15:  # Caída del 15% desde su pico histórico
                    is_dead = True
                    i += trade_duration
                    continue
                        
                # Registrar métricas del trade para el Autoencoder
                trade_metrics = [net_ret, trade_duration, 1 if net_ret > 0 else 0]
                rolling_metrics.append(trade_metrics)
                
                # Si estamos en producción (Test), el Autoencoder ya está pre-entrenado
                if not is_training_phase and hybrid_monitor is not None:
                    if len(rolling_metrics) >= 10:
                        X_window = np.array(rolling_metrics[-10:])
                    else:
                        # Zero padding para los primeros trades antes de llegar a la ventana de 10
                        pad_len = 10 - len(rolling_metrics)
                        X_window = np.pad(np.array(rolling_metrics), ((pad_len, 0), (0, 0)), 'constant')
                        
                    if hybrid_monitor.check_micro_anomaly(X_window):
                        # Solo declaramos muerte si esta anomalía microestructural nos llevó a pérdida severa
                        # (Más de 3% de pérdida acumulada en los últimos 10 trades, sin apalancamiento)
                        recent_pnl = np.sum(X_window[:, 0])
                        if recent_pnl < -0.03:
                            # CI/CD MLOps: En lugar de morir para siempre, entra en Cuarentena
                            # Pausa operaciones por 60 días para que el Walk-Forward aprenda el nuevo régimen
                            cooldown_remaining = 60
                            rolling_metrics = []  # Limpiamos el historial para la resurrección
                            # is_dead = True  <-- Eliminamos la muerte permanente
                # Single-position: no abrir trades mientras este está abierto
                i += trade_duration
            else:
                i += 1
                
        return pd.DataFrame(results), is_dead, rolling_metrics

    def run_tournament(self, modelos, bancos):
        modo_str = "BILATERAL (Long + Short)" if self.bilateral else "LONG ONLY"
        print(f"\n{'='*90}")
        print(f"  TORNEO FINANCIERO NETO - {self.activo}")
        print(f"  Modo: {modo_str}")
        print(f"  Umbrales Evaluados: Optimizador Dinámico (50% - 75%)")
        print(f"  Monitor Híbrido: Desactivación por Anomalías de Autoencoder LSTM")
        print(f"{'='*90}\n")
        
        raw_path = os.path.join(self.data_dir, "raw", f"{self.activo}_daily.csv")
        processed_path = os.path.join(self.data_dir, "processed", f"{self.activo}_processed.csv")
        
        df_raw = pd.read_csv(raw_path)
        df_processed = pd.read_csv(processed_path)
        
        POSSIBLE_MACROS = ['TPM', 'EMBI', 'Copper_FFD', 'Yield10Y_FFD', 'USDCLP_FFD', 'SP500_FFD', 'VIX_close', 'FXI_FFD', 'DXY_close_FFD']
        macro_cols = [c for c in POSSIBLE_MACROS if c in df_processed.columns]
        if macro_cols:
            print(f"  📊 Variables Macro Dinámicas detectadas para HMM: {macro_cols}")
        else:
            print("  ⚠️ No se detectaron variables macro. HMM desactivado para este activo.")
            
        campeones = {}
        
        # Directorio para guardar/cargar modelos MLOps pre-entrenados
        mlops_dir = os.path.join(self.results_dir, "mlops_monitors")
        os.makedirs(mlops_dir, exist_ok=True)

        for modelo in modelos:
            mejor_alpha = -np.inf
            campeon_actual = None
            
            for banco in bancos:
                file_name = os.path.join(self.results_dir, f'probs_{modelo.lower()}_{banco.lower()}_{self.activo}.npy')
                train_probs_path = os.path.join(self.results_dir, f'train_probs_{modelo.lower()}_{banco.lower()}_{self.activo}.npy')
                
                if not os.path.exists(file_name):
                    continue
                    
                pred_probs = np.load(file_name)
                n_test = len(pred_probs)
                
                # 1. Fase de Calibración de Riesgo (Pre-entrenamiento In-Sample)
                hybrid_monitor = HybridRiskMonitor()
                
                # Rutas de los modelos MLOps para esta combinación modelo/banco
                mlops_prefix = f"{modelo.lower()}_{banco.lower()}_{self.activo}"
                hmm_save_path = os.path.join(mlops_dir, f"{mlops_prefix}_hmm.pkl")
                ae_save_path = os.path.join(mlops_dir, f"{mlops_prefix}_autoencoder")
                
                if self.fast_mode:
                    # === MODO RÁPIDO: Cargar modelos MLOps pre-entrenados ===
                    if not os.path.exists(f"{ae_save_path}.keras"):
                        # Sin Autoencoder guardado → No competir sin escudo MLOps
                        continue
                    if macro_cols and os.path.exists(hmm_save_path):
                        from models.anomaly_detector import HMMRegimeDetector
                        hybrid_monitor.hmm_model = HMMRegimeDetector(n_components=3)
                        hybrid_monitor.hmm_model.load(hmm_save_path)
                        print(f"  ⚡ HMM cargado desde disco para {modelo} ({banco})")
                    from models.anomaly_detector import StrategyLSTMAutoencoder
                    hybrid_monitor.lstm_model = StrategyLSTMAutoencoder()
                    hybrid_monitor.lstm_model.load(ae_save_path)
                    print(f"  ⚡ Autoencoder cargado desde disco para {modelo} ({banco})")
                else:
                    # === MODO FULL: Entrenar desde cero y guardar a disco ===
                    if os.path.exists(train_probs_path):
                        train_probs = np.load(train_probs_path)
                        n_train = len(train_probs)
                        df_train = df_processed.iloc[-(n_train + n_test) : -n_test].copy()
                        
                        if macro_cols:
                            from models.anomaly_detector import HMMRegimeDetector
                            hybrid_monitor.hmm_model = HMMRegimeDetector(n_components=3)
                            X_macro_train = df_train[macro_cols].values
                            hybrid_monitor.hmm_model.fit(X_macro_train)
                            hybrid_monitor.hmm_model.save(hmm_save_path)
                            
                        _, _, train_rolling_metrics = self.simulate_trades(df_train, train_probs, is_training_phase=True, macro_cols=macro_cols)
                        
                        if len(train_rolling_metrics) > 10:
                            from models.anomaly_detector import StrategyLSTMAutoencoder
                            hybrid_monitor.lstm_model = StrategyLSTMAutoencoder(epochs=50, batch_size=4)
                            X_train = np.array(train_rolling_metrics)
                            hybrid_monitor.lstm_model.fit(X_train)
                            hybrid_monitor.lstm_model.save(ae_save_path)
                    else:
                        print(f"  ⛔ Sin train_probs para {modelo} (Banco: {banco}). Saltando (no compite sin escudo MLOps).")
                        continue
                    
                # 2. Fase de Producción / Evaluación Out-Of-Sample
                df_backtest = df_processed.iloc[-n_test:].copy()
                
                for umbral in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
                    self.confidence_threshold = umbral
                    trade_results, is_dead, _ = self.simulate_trades(df_backtest, pred_probs, hybrid_monitor=hybrid_monitor, is_training_phase=False, macro_cols=macro_cols)
                    
                    if not trade_results.empty:
                        trade_results = trade_results.sort_values('idx').reset_index(drop=True)
                        trade_results['cum_ret'] = (1 + trade_results['ret']).cumprod()
                        
                        if 'time' in df_backtest.columns:
                            trade_results['exit_time'] = pd.to_datetime(df_backtest['time'].iloc[trade_results['idx']].values)
                        else:
                            trade_results['exit_time'] = trade_results['idx'] # Fallback
                            
                        retorno_estrategia = trade_results['cum_ret'].iloc[-1] - 1
                        
                        precio_inicio = df_backtest['close'].iloc[0]
                        
                        # Mercado para Alpha (hasta la muerte de la estrategia)
                        precio_final_alpha = df_backtest['close'].iloc[trade_results['idx'].iloc[-1]] if is_dead else df_backtest['close'].iloc[-1]
                        retorno_mercado_alpha = (precio_final_alpha / precio_inicio) - 1
                        alpha = retorno_estrategia - retorno_mercado_alpha
                        
                        # Mercado para Benchmark (periodo completo)
                        precio_final_total = df_backtest['close'].iloc[-1]
                        retorno_mercado_total = (precio_final_total / precio_inicio) - 1
                        
                        n_trades = len(trade_results)
                        win_rate = len(trade_results[trade_results['ret'] > 0]) / n_trades if n_trades > 0 else 0
                        
                        # Calcular CAGR (Rentabilidad Anualizada)
                        if 'time' in df_backtest.columns:
                            dias_totales = (pd.to_datetime(df_backtest['time'].iloc[-1]) - pd.to_datetime(df_backtest['time'].iloc[0])).days
                            anios = dias_totales / 365.25
                            if anios > 0:
                                cagr_est = (1 + retorno_estrategia)**(1/anios) - 1
                                cagr_mkt = (1 + retorno_mercado_total)**(1/anios) - 1
                            else:
                                cagr_est, cagr_mkt = 0, 0
                        else:
                            cagr_est, cagr_mkt = 0, 0
                        
                        metrics = self.calculate_advanced_metrics(
                            trade_results['ret'].values, n_trades, trade_results['cum_ret'].values
                        )
                        
                        mc_dsr_metrics = self.run_montecarlo_and_dsr(
                            trade_results['ret'].values, n_trades, 
                            metrics['Sharpe'], metrics['Skew'], metrics['Kurt'],
                            n_models_tested=len(modelos)*len(bancos)*6
                        )
                        metrics.update(mc_dsr_metrics)
                        
                        if alpha > mejor_alpha:
                            mejor_alpha = alpha
                            campeon_actual = {
                                'banco': f"{banco} (>{umbral:.0%})", 'alpha': alpha, 
                                'umbral': umbral,
                                'ret_est': retorno_estrategia, 'ret_mkt': retorno_mercado_total,
                                'cagr_est': cagr_est, 'cagr_mkt': cagr_mkt,
                                'trades': n_trades, 'win_rate': win_rate,
                                'avg_duration': trade_results['duration'].mean() if 'duration' in trade_results else 0.0,
                                'metrics': metrics,
                                'cum_ret_series': trade_results['cum_ret'].values,
                                'probs_series': trade_results['prob'].values if 'prob' in trade_results.columns else np.zeros(len(trade_results)),
                                'exit_times': trade_results['exit_time'].values,
                                'is_dead': is_dead,
                                'hybrid_monitor': hybrid_monitor
                            }
                        
            if campeon_actual:
                campeones[modelo] = campeon_actual
                status_str = "💀 MUERTO (Anomalía)" if campeon_actual['is_dead'] else "✅ ACTIVO"
                print(f"  Campeon {modelo.upper():>15}: {campeon_actual['banco']:<15} | Estado: {status_str} | Alpha: {campeon_actual['alpha']:>8.2%} | Win: {campeon_actual['win_rate']:>6.1%} | Trades: {campeon_actual['trades']}")

        return campeones, df_backtest if 'df_backtest' in locals() else None

    def generate_html_report(self, campeones):
        if not campeones:
            print("  No hay campeones para generar reporte.")
            return

        report_path = os.path.join(self.results_dir, f"backtest_report_{self.activo}.html")
        modo_str = "BILATERAL (Long + Short)" if self.bilateral else "LONG ONLY"
        primer = list(campeones.values())[0]

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Torneo Financiero - {self.activo}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0f172a; --panel: rgba(30, 41, 59, 0.7);
            --text: #f8fafc; --muted: #94a3b8; --border: rgba(255, 255, 255, 0.1);
            --accent: #38bdf8; --success: #10b981; --danger: #ef4444;
        }}
        body {{
            font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text);
            margin: 0; padding: 40px 20px; display: flex; justify-content: center;
        }}
        .container {{
            max-width: 1200px; width: 100%; background: var(--panel);
            backdrop-filter: blur(12px); border: 1px solid var(--border);
            border-radius: 16px; padding: 30px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        }}
        h1 {{ text-align: center; color: var(--accent); margin-top: 0; }}
        .header {{ display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 30px; }}
        .header p {{ margin: 5px 0; color: var(--muted); }}
        .header span {{ color: var(--text); font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 15px; text-align: right; border-bottom: 1px solid var(--border); }}
        th {{ color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 13px; letter-spacing: 1px; }}
        th:nth-child(1), td:nth-child(1), th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
        tr:hover td {{ background: rgba(255,255,255,0.03); }}
        .bench {{ background: rgba(56,189,248,0.05); color: var(--muted); }}
        .bench td:first-child {{ color: var(--accent); font-weight: 600; }}
        .positive {{ color: var(--success); }}
        .negative {{ color: var(--danger); }}
        .dead {{ color: var(--danger); font-weight: bold; }}
        .active {{ color: var(--success); font-weight: bold; }}
        .model-name {{ color: var(--accent); font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Torneo Financiero Quant - {self.activo}</h1>
        <div class="header">
            <div>
                <p>Activo: <span>{self.activo}</span></p>
                <p>Modo de Operación: <span>{modo_str}</span></p>
            </div>
            <div style="text-align: right;">
                <p>Umbrales Evaluados: <span>Optimizador Dinámico (50% - 75%)</span></p>
                <p>Position Sizing: <span>Kelly Dinámico Modificado (0.5x - 2.0x)</span></p>
            </div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Modelo</th><th>Banco Features</th><th>Estado</th>
                    <th>Alpha</th><th>CAGR</th><th>Trades</th><th>Win %</th><th>Duración (días)</th>
                    <th>Sharpe</th><th>PSR</th><th>DSR</th><th>MC MDD 95</th><th>CVaR 95</th>
                </tr>
            </thead>
            <tbody>
                <tr class="bench">
                    <td>BENCHMARK</td><td>Risk-Free (US)</td><td>-</td><td>-</td>
                    <td class="positive">5.00%</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
                </tr>
                <tr class="bench">
                    <td>BENCHMARK</td><td>Buy & Hold</td><td>-</td><td>0.00%</td>
                    <td class="{'positive' if primer['cagr_mkt'] >= 0 else 'negative'}">{primer['cagr_mkt']:.2%}</td>
                    <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
                </tr>
"""
        for mod, data in campeones.items():
            m = data['metrics']
            is_dead = data.get('is_dead', False)
            status_cls = "dead" if is_dead else "active"
            status_txt = "💀 MUERTO" if is_dead else "✅ ACTIVO"
            alpha_cls = "positive" if data['alpha'] >= 0 else "negative"
            cagr_cls = "positive" if data['cagr_est'] >= 0 else "negative"
            
            html += f"""
                <tr>
                    <td class="model-name">{mod.upper()}</td><td>{data['banco']}</td>
                    <td class="{status_cls}">{status_txt}</td>
                    <td class="{alpha_cls}">{data['alpha']:.2%}</td>
                    <td class="{cagr_cls}">{data['cagr_est']:.2%}</td>
                    <td>{data['trades']}</td><td>{data['win_rate']:.1%}</td><td>{data['avg_duration']:.1f}</td>
                    <td>{m['Sharpe']:.2f}</td><td>{m['PSR']:.1%}</td><td>{m['DSR']:.1%}</td>
                    <td>{m['MC_MDD_95']:.2%}</td><td>{m['CVaR_95']:.2%}</td>
                </tr>
"""
        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  📊 Reporte Profesional HTML generado: {report_path}")
        
        # Abrir automáticamente en el navegador del usuario
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(report_path)}")

    def plot_equity_curves(self, campeones, df_backtest):
        """Genera y guarda un gráfico comparativo de las curvas de Equity."""
        if not campeones:
            return
            
        plt.figure(figsize=(12, 6))
        plt.title(f"Curvas de Equity (Out-of-Sample) - {self.activo}", fontsize=14, fontweight='bold')
        
        if df_backtest is not None and not df_backtest.empty:
            if 'time' in df_backtest.columns:
                mkt_times = pd.to_datetime(df_backtest['time'])
            else:
                mkt_times = df_backtest.index
            mkt_equity = df_backtest['close'] / df_backtest['close'].iloc[0]
            
            # Obtener el CAGR del mercado calculado previamente
            primer = list(campeones.values())[0] if campeones else None
            mkt_cagr_str = f" - CAGR: {primer['cagr_mkt']:.2%}" if primer else ""
            
            plt.plot(mkt_times, mkt_equity, label=f"Buy & Hold (Benchmark){mkt_cagr_str}", color='black', linewidth=2.5, linestyle='-', zorder=1)
        
        for mod, data in campeones.items():
            cum_ret_series = data['cum_ret_series']
            exit_times = data['exit_times']
            
            if len(exit_times) > 0:
                try:
                    t0 = exit_times[0] - pd.Timedelta(days=1)
                except:
                    t0 = exit_times[0] - 1
                    
                times = [t0] + list(exit_times)
                equity_curve = [1.0] + list(cum_ret_series)
                
                status_label = "(💀 Anomalía)" if data.get('is_dead', False) else ""
                line, = plt.plot(times, equity_curve, label=f"{mod} {status_label} - Alpha: {data['alpha']:.2%} | CAGR: {data['cagr_est']:.2%}")
                
                if data.get('is_dead', False):
                    # Dibujar una 'X' roja gigante en el ultimo punto
                    plt.plot(times[-1], equity_curve[-1], marker='X', color='red', markersize=10, markeredgecolor='black')
            
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel("Fecha de Cierre del Trade", fontsize=12)
        plt.ylabel("Crecimiento del Capital (Equity)", fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plot_path = os.path.join(self.results_dir, f"equity_curve_{self.activo}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  📈 Gráfico de Equity guardado en: {plot_path}")

    def export_champion_config(self, campeones):
        """Busca el mejor modelo con Alpha > 0 y vivo, y lo exporta a JSON."""
        if not campeones:
            return
            
        # Filtrar campeones vivos y con alpha > 0
        campeones_validos = {mod: data for mod, data in campeones.items() if not data.get('is_dead', False) and data['alpha'] > 0}
        
        if not campeones_validos:
            print(f"  ⚠️ No hay campeones viables (Alpha > 0 y Vivos) para {self.activo}. No se exportará configuración para producción.")
            return
            
        mejor_modelo = max(campeones_validos.keys(), key=lambda k: campeones_validos[k]['alpha'])
        data = campeones_validos[mejor_modelo]
        
        banco_clean = data['banco'].split(" (")[0]
        model_file = f"{mejor_modelo.lower()}_{banco_clean.lower()}_{self.activo}.pkl"
        
        # Importar dinámicamente get_bancos_por_activo para obtener los features
        import sys
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        try:
            from main_training import get_bancos_por_activo
            bancos_dict = get_bancos_por_activo(self.activo)
            features = bancos_dict.get(banco_clean, [])
        except ImportError:
            features = []
            
        config = {
            "model_type": mejor_modelo,
            "model_file": model_file,
            "banco": banco_clean,
            "features": features,
            "confidence_threshold": data['umbral'],
            "k_up": self.k_up,
            "k_down": self.k_down,
            "bilateral": self.bilateral
        }
        
        json_path = os.path.join(self.results_dir, f"campeon_{self.activo}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
            
        # Exportar los Modelos del MLOps de Anomalías (Autoencoder y HMM)
        hybrid_monitor = data.get('hybrid_monitor')
        if hybrid_monitor is not None:
            if hybrid_monitor.lstm_model is not None:
                hybrid_monitor.lstm_model.save(os.path.join(self.results_dir, f"campeon_{self.activo}_autoencoder"))
            if hasattr(hybrid_monitor, 'hmm_model') and hybrid_monitor.hmm_model is not None:
                hybrid_monitor.hmm_model.save(os.path.join(self.results_dir, f"campeon_{self.activo}_hmm.pkl"))
            
        print(f"  💾 Configuración de Producción exportada: {json_path} (Alpha: {data['alpha']:.2%})")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    
    activos = ["EURUSD", "SP500", "Oro", "ECH"]
    modelos = ['RANDOM_FOREST', 'XGBOOST', 'LSTM', 'BILSTM', 'ARIMA_LSTM', 'LSTM_RF']
    
    from main_training import get_bancos_por_activo
    
    for activo in activos:
        bancos = list(get_bancos_por_activo(activo).keys())
        backtester = TripleBarrierBacktester(activo, data_dir, results_dir, fast_mode=True)
        campeones, df_backtest = backtester.run_tournament(modelos, bancos)
        if campeones:
            backtester.generate_html_report(campeones)
            backtester.plot_equity_curves(campeones, df_backtest)
            backtester.export_champion_config(campeones)
        else:
            print(f"No hay predicciones guardadas para {activo} aún.")


if __name__ == "__main__":
    main()
