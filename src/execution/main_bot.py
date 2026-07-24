import os
import sys

# Agregar la carpeta src al PYTHONPATH para que encuentre 'execution', 'preprocessing', etc.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import time
import logging
import warnings
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import tensorflow as tf

from execution.risk_manager import RiskManager
from execution.execution_engine import ExecutionEngine
from execution.telegram_notifier import TelegramNotifier
from mt5_connector import MT5Connector

from preprocessing.technical_features import TechnicalFeatureEngineer
from preprocessing.volatility import VolatilityModeler
from preprocessing.stationarity import FractionalDifferencer

from models.random_forest import RandomForestTrainer
from models.xgb_model import XGBoostTrainer
from models.lstm_model import LSTMTrainer
from models.bilstm_model import BiLSTMTrainer
from models.arima_lstm import HybridARIMALSTMTrainer
from models.lstm_rf import HybridLSTMRFTrainer

# Forzar UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TradingBot:
    def __init__(self, symbol: str, timeframe: int, config: dict, models_dir: str):
        self.symbol = symbol
        self.timeframe = timeframe # Ej. mt5.TIMEFRAME_D1
        self.config = config
        self.model_path = os.path.join(models_dir, config["model_file"])
        
        # Conectores y Motores
        self.connector = MT5Connector()
        self.engine = ExecutionEngine(self.connector)
        
        # Mapeo dinámico de Magic Number por símbolo y timeframe en MT5
        tf_code = {mt5.TIMEFRAME_D1: 1, mt5.TIMEFRAME_H4: 4, mt5.TIMEFRAME_H1: 10}.get(self.timeframe, 1)
        sym_code = abs(hash(self.symbol)) % 1000
        self.engine.magic_number = 100000 + sym_code * 10 + tf_code

        self.risk_manager = RiskManager(
            risk_per_trade_pct=0.025,
            k_up=config["k_up"],
            k_down=config["k_down"]
        )
        self.notifier = TelegramNotifier()

        
        # Preprocesadores
        self.tech_eng = TechnicalFeatureEngineer(target_price_col='close')
        self.vol_mod = VolatilityModeler(window_size=500, target_col='close')
        self.ffd_eng = FractionalDifferencer(threshold=1e-4)
        
        # Estado
        self.model = None
        self.last_bar_time = None

    def load_model(self):
        logging.info(f"Cargando modelo campeon ({self.config['model_type']}) desde {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontro el modelo en {self.model_path}")
        
        if self.config["model_type"] == "RANDOM_FOREST":
            self.model = RandomForestTrainer.load(self.model_path)
        elif self.config["model_type"] == "XGBOOST":
            self.model = XGBoostTrainer.load(self.model_path)
        elif self.config["model_type"] == "LSTM":
            self.model = LSTMTrainer.load(self.model_path)
        elif self.config["model_type"] == "BILSTM":
            self.model = BiLSTMTrainer.load(self.model_path)
        elif self.config["model_type"] == "ARIMA_LSTM":
            self.model = HybridARIMALSTMTrainer.load(self.model_path)
        elif self.config["model_type"] == "LSTM_RF":
            self.model = HybridLSTMRFTrainer.load(self.model_path)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.config['model_type']}")
            
        # Cargar Filtros MLOps Institucionales
        from models.anomaly_detector import StrategyLSTMAutoencoder, HMMRegimeDetector
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(base_dir, "results")
        
        autoencoder_path = os.path.join(results_dir, f"campeon_{self.symbol}_autoencoder")
        hmm_path = os.path.join(results_dir, f"campeon_{self.symbol}_hmm.pkl")
        
        if os.path.exists(autoencoder_path + ".keras"):
            self.risk_manager.hybrid_monitor.lstm_model = StrategyLSTMAutoencoder()
            self.risk_manager.hybrid_monitor.lstm_model.load(autoencoder_path)
            logging.info(f"Filtro Micro-Estructural (LSTM Autoencoder) cargado exitosamente.")
            
        if os.path.exists(hmm_path):
            self.risk_manager.hybrid_monitor.hmm_model = HMMRegimeDetector()
            self.risk_manager.hybrid_monitor.hmm_model.load(hmm_path)
            logging.info(f"Filtro Macro (HMM) cargado exitosamente.")
        
        logging.info(f"Modelo Predictivo {self.config['model_type']} cargado exitosamente (Banco: {self.config['banco']}, Umbral: >{self.config['confidence_threshold']:.0%}).")

    def fetch_live_data(self) -> pd.DataFrame:
        """Descarga las ultimas velas necesarias para generar las variables (ej. FFD, EGARCH)"""
        import yfinance as yf
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3000)
        
        if self.symbol == "ECH":
            df = yf.download(self.symbol, start=start_date, end=end_date, progress=False)
            if df.empty:
                logging.error(f"Fallo al descargar datos live para {self.symbol} vía yfinance.")
                return pd.DataFrame()
                
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            df.rename(columns={'volume': 'real_volume', 'adj close': 'adj_close'}, inplace=True)
            df['tick_volume'] = df['real_volume']
            df['spread'] = 0.0
            
            df.index = df.index.tz_localize(None).normalize()
            df.index.name = 'time'
            
        else:
            # Flujo normal para MT5
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 2000)
            if rates is None or len(rates) == 0:
                logging.error("No se pudieron obtener datos historicos de MT5.")
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.index = df.index.normalize()

        # --- Descargar Macro Global ---
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date, progress=False)
        tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False)
        
        macro_df = pd.DataFrame()
        if not vix.empty:
            macro_df['VIX_close'] = vix['Close'].iloc[:, 0] if isinstance(vix.columns, pd.MultiIndex) else vix['Close']
        if not dxy.empty:
            macro_df['DXY_close'] = dxy['Close'].iloc[:, 0] if isinstance(dxy.columns, pd.MultiIndex) else dxy['Close']
        if not tnx.empty:
            macro_df['Yield10Y'] = tnx['Close'].iloc[:, 0] if isinstance(tnx.columns, pd.MultiIndex) else tnx['Close']
            
        macro_df.index = macro_df.index.tz_localize(None).normalize()
        
        df = df.join(macro_df, how='left')
        if 'VIX_close' in df.columns: df['VIX_close'] = df['VIX_close'].ffill().bfill()
        if 'DXY_close' in df.columns: df['DXY_close'] = df['DXY_close'].ffill().bfill()
        if 'Yield10Y' in df.columns: df['Yield10Y'] = df['Yield10Y'].ffill().bfill()

        # --- Descargar Macro Chile (Solo si es ECH) ---
        if self.symbol == "ECH":
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
            from preprocessing.chilean_macro import ChileanMacroExtractor
            macro_chile = ChileanMacroExtractor()
            df_chile = macro_chile.get_chilean_macro_data(start_date, end_date)
            if not df_chile.empty:
                df = df.join(df_chile, how='left')
                cols_chile = df_chile.columns
                df[cols_chile] = df[cols_chile].ffill().bfill()
                
        df.reset_index(inplace=True)
        return df



    def _predict_bilstm(self, df_proc: pd.DataFrame) -> float:
        """Genera predicción usando el modelo BiLSTM."""
        feature_cols = self.config["features"]
        look_back = self.model.look_back
        dataset = df_proc[feature_cols].values
        
        if len(dataset) < look_back:
            logging.warning("No hay suficientes datos para armar la secuencia LSTM.")
            return -1.0
            
        # Tomar la ultima secuencia (ventana temporal) y escalar
        window = dataset[-look_back:]
        n_feat = len(feature_cols)
        
        # Aplicamos el scaler entrenado
        window_scaled = np.clip(self.model.scaler.transform(window), -10, 10)
        
        # Tensor 3D para la prediccion: (1, look_back, n_features)
        X_input = window_scaled.reshape(1, look_back, n_feat)
        
        # Prediccion del Modelo
        prob = self.model.master_model.predict(X_input, verbose=0)[0][0]
        return float(prob)

    def _predict_arima_lstm(self, df_proc: pd.DataFrame, df_raw: pd.DataFrame) -> float:
        """Genera predicción usando el modelo híbrido ARIMA_LSTM."""
        look_back = self.model.look_back
        
        # El ARIMA necesita la serie de retornos fraccionarios (close_FFD) para generar residuos
        close_ffd = df_proc['close_FFD'].values
        
        # Generar residuos ARIMA sobre las últimas observaciones
        try:
            arima_fitted = self.model.model_arima_curr.fittedvalues
            # Calcular residuos: valor real - valor predicho por ARIMA
            # Usamos los últimos look_back residuos
            residuals = close_ffd[-look_back:] - arima_fitted.values[-look_back:]
        except Exception:
            # Fallback: usar los valores crudos si el ARIMA no puede generar residuos
            residuals = close_ffd[-look_back:]
        
        # Preparar las features exógenas del banco
        feature_cols = self.config["features"]
        
        # Construir la ventana: residuos + exógenas (si las hay)
        resid_window = residuals.reshape(-1, 1)
        if feature_cols:
            exog_window = df_proc[feature_cols].values[-look_back:]
            current_window = np.hstack([resid_window, exog_window])
        else:
            current_window = resid_window
        
        n_feat = current_window.shape[1]
        
        # Escalar e inferir
        curr_win_scaled = np.clip(
            self.model.scaler.transform(current_window.reshape(-1, n_feat)), -10, 10
        ).reshape(1, look_back, n_feat)
        
        prob = self.model.master_lstm.predict(curr_win_scaled, verbose=0)[0][0]
        return float(prob)

    def _predict_xgboost(self, df_proc: pd.DataFrame) -> float:
        """Genera predicción usando el modelo XGBOOST (o Random Forest)."""
        feature_cols = self.config["features"]
        look_back = self.model.look_back
        dataset = df_proc[feature_cols].values
        
        if len(dataset) < look_back:
            logging.warning("No hay suficientes datos para armar la secuencia tabular.")
            return -1.0
            
        window = dataset[-look_back:]
        window_flat = window.flatten().reshape(1, -1)
        window_scaled = np.clip(self.model.scaler.transform(window_flat), -10, 10)
        
        prob = self.model.master_model.predict_proba(window_scaled)[0][1]
        return float(prob)

    def check_for_signals(self):
        """Metodo principal ejecutado en cada nueva vela."""
        if not self.connector.connected:
            return

        # 1. Comprobar si hay nueva vela cerrada
        last_tick = mt5.symbol_info_tick(self.symbol)
        if last_tick is None:
            return
            
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 2)
        if rates is None or len(rates) < 2:
            return
            
        # El indice 0 es la vela cerrada anterior, el 1 es la vela actual en formacion
        current_bar_time = rates[1]['time']
        
        # Si todavia estamos en la misma vela que ya procesamos, ignorar
        if self.last_bar_time == current_bar_time:
            return

        logging.info(f"NUEVA VELA DETECTADA. Procesando señal para {self.symbol}...")
        self.last_bar_time = current_bar_time

        # 2. (Legacy CUSUM eliminado — reemplazado por Shadow Journal + HybridRiskMonitor)

        # 3. Verificar si ya tenemos un trade corriendo y gestionar la Barrera Vertical (Max Hold)
        if self.engine.has_open_positions(self.symbol):
            max_hold = getattr(self.risk_manager, 'max_hold', 10)
            closed_by_max_hold = self.engine.check_and_close_vertical_barrier(self.symbol, self.timeframe, max_hold)
            if closed_by_max_hold:
                logging.info(f"[{self.symbol}] ⏳ Barrera Vertical alcanzada ({max_hold} velas). Posición cerrada exitosamente.")
                self.notifier.alert_max_hold_exit(self.symbol, max_hold)
            else:
                logging.info("Ya existe un trade abierto. Ignorando nueva señal (estrategia single-position).")
            return


        # 4. Descargar y preprocesar datos (necesitamos la matriz X 3D para el modelo)
        df_raw = self.fetch_live_data()
        if df_raw.empty:
            return
            
        df_proc = df_raw.copy()
        df_proc = self.tech_eng.add_indicators(df_proc)
        df_proc = self.vol_mod.compute_egarch(df_proc, symbol=self.symbol)
        df_proc = self.ffd_eng.apply_ffd(df_proc)
        df_proc.ffill(inplace=True)
        df_proc.bfill(inplace=True)
        
        # 4.5 REENTRENAMIENTO RÁPIDO DIARIO
        logging.info(f"[{self.symbol}] Ejecutando Fast-Retrain (Actualización de pesos matemáticos con la data reciente)...")
        try:
            self.model.fast_retrain(df_proc, self.config["features"])
        except Exception as e:
            logging.error(f"[{self.symbol}] Error durante el reentrenamiento rápido: {e}")
            
        # 4.8 SHADOW JOURNAL (Sin Estado)
        # Evaluamos si el modelo está en Cuarentena usando su historial simulado reciente
        if not self._run_shadow_journal(df_proc):
            return # Permiso denegado por MLOps
        
        # 5. Prediccion del Modelo Campeon
        if self.config["model_type"] == "BILSTM" or self.config["model_type"] == "LSTM" or self.config["model_type"] == "LSTM_RF":
            prob = self._predict_bilstm(df_proc)
        elif self.config["model_type"] == "ARIMA_LSTM":
            prob = self._predict_arima_lstm(df_proc, df_raw)
        elif self.config["model_type"] == "XGBOOST" or self.config["model_type"] == "RANDOM_FOREST":
            prob = self._predict_xgboost(df_proc)
        else:
            logging.error(f"Tipo de modelo no soportado: {self.config['model_type']}")
            return
            
        if prob < 0:
            return # Error en la prediccion
            
        logging.info(f"Probabilidad de tocar Take Profit (Triple Barrera): {prob:.2%} (Umbral: >{self.config['confidence_threshold']:.0%})")
        
        # Determinar dirección de la señal
        is_long = prob > self.config["confidence_threshold"]
        is_short = self.config.get("bilateral", False) and prob < (1.0 - self.config["confidence_threshold"])
        
        if is_long or is_short:
            direction = "COMPRA (Long)" if is_long else "VENTA (Short)"
            logging.info(f">>> SEÑAL DE {direction} DISPARADA <<<")
            
            # Calculos de Riesgo
            last_vol = df_proc['EGARCH_Vol'].iloc[-1]
            
            if self.symbol == "ECH":
                # ECH usa Yahoo Finance, no tenemos MT5 tick info
                current_price = df_proc['close'].iloc[-1]
                lots = 0.0 # No podemos ejecutar lotes en MT5
                account_balance = 500.0 # Balance asumido para la alerta de telegram
                
                if is_long:
                    tp_price, sl_price = self.risk_manager.calculate_triple_barrier_levels(current_price, last_vol)
                else:
                    vol_decimal = last_vol / 100.0
                    tp_price = current_price * (1 - self.risk_manager.k_up * vol_decimal)
                    sl_price = current_price * (1 + self.risk_manager.k_down * vol_decimal)
            else:
                sym_info = self.engine.get_symbol_info(self.symbol)
                if sym_info is None:
                    return
                    
                account_info = mt5.account_info()
                account_balance = account_info.balance
                
                if is_long:
                    current_price = mt5.symbol_info_tick(self.symbol).ask
                    tp_price, sl_price = self.risk_manager.calculate_triple_barrier_levels(current_price, last_vol)
                else:
                    # SHORT: TP abajo, SL arriba (invertido)
                    current_price = mt5.symbol_info_tick(self.symbol).bid
                    vol_decimal = last_vol / 100.0
                    tp_price = current_price * (1 - self.risk_manager.k_up * vol_decimal)
                    sl_price = current_price * (1 + self.risk_manager.k_down * vol_decimal)
                
                lots = self.risk_manager.calculate_position_size(
                    balance=account_balance,
                    current_price=current_price,
                    stop_loss_price=sl_price,
                    tick_size=sym_info['tick_size'],
                    tick_value=sym_info['tick_value'],
                    volume_step=sym_info['volume_step'],
                    prediction_prob=prob,
                    confidence_threshold=self.config["confidence_threshold"]
                )
            
            # Validacion de volumenes minimos del broker (Solo si no es ECH)
            if self.symbol != "ECH" and lots < sym_info['volume_min']:
                logging.warning(f"Volumen calculado ({lots}) es menor al minimo permitido ({sym_info['volume_min']}). Abortando trade.")
                return
            
            # Calcular riesgo dinámico para la alerta de Telegram
            kelly_mult = 1.0
            if prob > 0.5:
                delta = prob - self.config["confidence_threshold"]
            else:
                delta = (1.0 - self.config["confidence_threshold"]) - prob
            delta = max(0, delta)
            if delta <= 0.05: kelly_mult = 0.5
            elif delta <= 0.15: kelly_mult = 1.0
            else: kelly_mult = 2.0
            
            dynamic_risk_pct = self.risk_manager.risk_per_trade_pct * kelly_mult
            
            # Enviar notificación Telegram (Trade) con timeframe explícito
            tf_str_map = {mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_H4: "H4", mt5.TIMEFRAME_H1: "H1"}
            tf_label = tf_str_map.get(self.timeframe, self.config.get("timeframe", "D1"))

            self.notifier.alert_trade_execution(
                symbol=self.symbol, volume=lots, price=current_price, tp=tp_price, sl=sl_price, is_long=is_long, account_balance=account_balance, risk_pct=dynamic_risk_pct, timeframe=tf_label
            )

            
            # 6. Enviar orden a MT5
            if self.symbol != "ECH":
                if is_long:
                    self.engine.send_market_buy_order(self.symbol, lots, sl_price, tp_price)
                else:
                    self.engine.send_market_sell_order(self.symbol, lots, sl_price, tp_price)
            else:
                logging.info(f"Símbolo ECH detectado. Alerta enviada a Telegram. Ejecución MT5 saltada ya que se opera en Quantfury.")
        else:
            # Enviar notificación de status sin señal
            last_vol = df_proc['EGARCH_Vol'].iloc[-1]
            self.notifier.alert_daily_check(self.symbol, last_vol, has_signal=False)

    def _run_shadow_journal(self, df_proc: pd.DataFrame) -> bool:
        """
        Ejecuta la Simulación en la Sombra sobre los ultimos 300 días.
        Retorna True si el bot tiene PERMISO para operar (No esta en Cuarentena).
        """
        logging.info(f"[{self.symbol}] Ejecutando Shadow Journal (Evaluacion de MLOps en la sombra)...")
        
        import sys
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
            
        from src.evaluation.backtester import TripleBarrierBacktester
        
        eval_window = 300
        if len(df_proc) < eval_window:
            eval_window = len(df_proc)
            
        df_eval = df_proc.iloc[-eval_window:].copy()
        df_eval.reset_index(drop=True, inplace=True)
        
        try:
            if self.config["model_type"] == "BILSTM" or self.config["model_type"] == "LSTM":
                look_back = self.model.look_back
                dataset = df_eval[self.config["features"]].values
                X_batch = []
                for i in range(len(dataset) - look_back):
                    window = dataset[i:i+look_back]
                    X_batch.append(self.model.scaler.transform(window))
                if X_batch:
                    X_batch = np.array(X_batch)
                    probs_batch = self.model.master_model.predict(X_batch, verbose=0).flatten()
                    pred_probs = np.pad(probs_batch, (look_back, 0), 'constant')
                else:
                    return True 
            elif self.config["model_type"] in ["XGBOOST", "RANDOM_FOREST"]:
                look_back = self.model.look_back
                dataset = df_eval[self.config["features"]].values
                X_batch = []
                for i in range(look_back, len(dataset)):
                    window = dataset[i-look_back:i]
                    X_batch.append(window.flatten())
                if X_batch:
                    X_batch = np.array(X_batch)
                    X_batch_scaled = np.clip(self.model.scaler.transform(X_batch), -10, 10)
                    probs_batch = self.model.master_model.predict_proba(X_batch_scaled)[:, 1]
                    pred_probs = np.pad(probs_batch, (look_back, 0), 'constant')
                else:
                    return True
            else:
                logging.info(f"[{self.symbol}] Shadow Journal no soportado en batch para {self.config['model_type']}. Saltando check de Cuarentena.")
                return True
        except Exception as e:
            logging.error(f"[{self.symbol}] Error generando probabilidades en Shadow Journal: {e}")
            return True
            
        tester = TripleBarrierBacktester(self.symbol, "", "")
        tester.confidence_threshold = self.config["confidence_threshold"]
        
        trade_results, is_dead, rolling_metrics = tester.simulate_trades(
            df_eval, pred_probs, hybrid_monitor=self.risk_manager.hybrid_monitor, is_training_phase=False
        )
        
        concept_drift = False
        if self.risk_manager.hybrid_monitor and self.risk_manager.hybrid_monitor.lstm_model:
            if len(rolling_metrics) >= 30:
                X_window = np.array(rolling_metrics[-30:])
                concept_drift = self.risk_manager.hybrid_monitor.lstm_model.check_concept_drift(X_window)
                
        lock_file = os.path.join(base_dir, "cache", f"quarantine_{self.symbol}.lock")
        
        if is_dead:
            logging.warning(f"[{self.symbol}] 🚨 SHADOW JOURNAL RECHAZA OPERACIÓN: El modelo está en Cuarentena por Anomalías o Muerte por MDD.")
            
            # Verificar si ya notificamos (para evitar spam diario)
            if not os.path.exists(lock_file):
                self.notifier.alert_mlops_quarantine(self.symbol)
                os.makedirs(os.path.dirname(lock_file), exist_ok=True)
                with open(lock_file, "w") as f:
                    f.write("quarantined")
            
            return False
            
        # Si NO está en cuarentena, revisamos si estaba castigado ayer (Lock file) y lo levantamos
        if os.path.exists(lock_file):
            self.notifier.alert_mlops_resurrection(self.symbol)
            os.remove(lock_file)
            
        concept_drift_lock = os.path.join(base_dir, "cache", f"concept_drift_{self.symbol}.lock")
        
        if concept_drift:
            logging.warning(f"[{self.symbol}] ⚠️ CONCEPT DRIFT: Régimen de mercado alterado (Vejez del filtro). Se recomienda Retuning masivo.")
            if not os.path.exists(concept_drift_lock):
                self.notifier.alert_concept_drift(self.symbol)
                os.makedirs(os.path.dirname(concept_drift_lock), exist_ok=True)
                with open(concept_drift_lock, "w") as f:
                    f.write("drifted")
        else:
            if os.path.exists(concept_drift_lock):
                os.remove(concept_drift_lock)
            
        logging.info(f"[{self.symbol}] ✅ Shadow Journal: Permiso concedido para operar.")
        return True

    def process_asset(self):
        """Metodo de un solo pase (una iteracion) para ser llamado por el gestor."""
        if not self.connector.connected:
            return
        self.check_for_signals()

class MultiAssetBotManager:
    """Gestor maestro que controla múltiples bots (uno por cada activo) usando una sola conexión a MT5."""
    def __init__(self, bots: list):
        self.bots = bots
        self.connector = MT5Connector() # Conexión global compartida
        
    def run_forever(self):
        logging.info("Iniciando Gestor Multi-Activo Quant 24/7...")
        if not self.connector.connect():
            sys.exit(1)
            
        try:
            # Preparar todos los bots
            for bot in self.bots:
                # Inyectar el conector compartido
                bot.connector = self.connector
                bot.engine.connector = self.connector
                bot.load_model()
            logging.info("✅ Todos los modelos cargados.")
            
            # 1.5 Cargar Pesos HRP Dinámicos
            import json
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            hrp_path = os.path.join(base_dir, "results", "hrp_weights.json")
            if os.path.exists(hrp_path):
                with open(hrp_path, 'r') as f:
                    hrp_weights = json.load(f)
                logging.info(f"⚖️ Pesos de Portafolio HRP detectados: {hrp_weights}")
                active_hrp = {bot.symbol: hrp_weights.get(bot.symbol, 0.0) for bot in self.bots}
                total_active_weight = sum(active_hrp.values())
                
                if total_active_weight > 0:
                    norm_hrp = {sym: w / total_active_weight for sym, w in active_hrp.items()}
                else:
                    norm_hrp = {bot.symbol: 1.0 / len(self.bots) for bot in self.bots}

                for bot in self.bots:
                    peso_norm = norm_hrp.get(bot.symbol, 0.0)
                    riesgo_original = bot.risk_manager.risk_per_trade_pct
                    nuevo_riesgo = riesgo_original * peso_norm
                    bot.risk_manager.risk_per_trade_pct = nuevo_riesgo
                    logging.info(f"  > [{bot.symbol}] Riesgo Base ajustado por HRP Re-normalizado ({peso_norm:.2%}): {riesgo_original:.2%} -> {nuevo_riesgo:.2%}")

            else:
                logging.warning("⚠️ No se encontró 'hrp_weights.json'. Operando con pesos distribuidos uniformemente (1/N).")

            logging.info("Esperando cierre de velas... (Presiona Ctrl+C para detener)")
            if self.bots:
                self.bots[0].notifier.alert_startup() # Enviar una sola notificación global de encendido
            
            # Ciclo infinito
            while True:
                for bot in self.bots:
                    try:
                        bot.process_asset()
                    except Exception as e:
                        logging.error(f"Error procesando el activo {bot.symbol}: {e}")
                
                # Dormir 10 segundos antes del próximo ciclo de monitoreo
                time.sleep(10)
                
        except KeyboardInterrupt:
            logging.info("Apagando Gestor (Interrupción de usuario)...")
        finally:
            self.connector.shutdown()

if __name__ == "__main__":
    import json
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(results_dir, "saved_models")
    
    # Buscar todos los JSON de campeones en results/ (soporta multi-timeframe y retrocompatibilidad D1)
    import glob
    campeon_files = glob.glob(os.path.join(results_dir, "campeon_*.json"))
    bots_activos = []
    
    tf_map = {
        "D1": mt5.TIMEFRAME_D1,
        "H4": mt5.TIMEFRAME_H4,
        "H1": mt5.TIMEFRAME_H1
    }

    if not campeon_files:
        logging.warning("No se encontraron archivos de campeones en results/.")
    else:
        for json_path in campeon_files:
            filename = os.path.basename(json_path)
            symbol_raw = filename.replace("campeon_", "").replace(".json", "")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Obtener el timeframe de la configuración o fallback a D1
            tf_str = config.get("timeframe", "D1")
            timeframe = tf_map.get(tf_str, mt5.TIMEFRAME_D1)
            symbol = config.get("activo", symbol_raw.split("_")[0])

            model_path = os.path.join(models_dir, config["model_file"])
            if os.path.exists(model_path):
                logging.info(f"Registrando bot para {symbol} [{tf_str}] -> Campeón: {config['model_type']} ({config['banco']}) Umbral: >{config['confidence_threshold']:.0%}")
                bot = TradingBot(symbol=symbol, timeframe=timeframe, config=config, models_dir=models_dir)
                bots_activos.append(bot)
            else:
                logging.error(f"Falta el archivo binario del modelo {model_path} para {symbol} [{tf_str}].")

            
    if not bots_activos:
        logging.error("No hay bots configurados o no se encontraron los modelos. Abortando.")
        sys.exit(1)
        
    manager = MultiAssetBotManager(bots_activos)
    manager.run_forever()

