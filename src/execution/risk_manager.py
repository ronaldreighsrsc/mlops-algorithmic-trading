import math

import numpy as np
import os
import joblib
from models.anomaly_detector import StrategyLSTMAutoencoder, HMMRegimeDetector

class HybridRiskMonitor:
    """
    Monitor de Riesgo Híbrido Institucional.
    Sustituye al antiguo CUSUM. Utiliza:
    1. HMM para contexto Macro (Reducir riesgo en Crisis).
    2. LSTM Autoencoder para contexto Micro (Apagar estrategia si hay anomalía intrínseca).
    """
    def __init__(self, data_dir: str = None):
        self.hmm_model = None
        self.lstm_model = None
        self.is_dead = False
        self.current_risk_multiplier = 1.0
        
        # Intentar cargar modelos pre-entrenados si existe la ruta
        if data_dir:
            hmm_path = os.path.join(data_dir, "hmm_model.pkl")
            lstm_path = os.path.join(data_dir, "lstm_autoencoder_state.pkl")
            
            if os.path.exists(hmm_path):
                self.hmm_model = joblib.load(hmm_path)
            if os.path.exists(lstm_path):
                # Placeholder para la carga segura si decidimos usar el método .load del modelo
                pass

    def check_macro_regime(self, X_macro_recent: np.ndarray) -> float:
        """
        Verifica el régimen macro.
        Retorna el multiplicador de riesgo (1.0 normal, 0.5 crisis).
        """
        if self.hmm_model is None:
            return 1.0
            
        is_crisis = self.hmm_model.is_crisis(X_macro_recent)
        if is_crisis:
            return 0.5
        return 1.0

    def check_micro_anomaly(self, X_window: np.ndarray) -> bool:
        """
        Verifica si la ventana reciente de la estrategia es anómala.
        Si es True, la estrategia debe morir.
        """
        if self.lstm_model is None:
            return False
            
        anomalous = self.lstm_model.is_anomalous(X_window)
        if anomalous:
            self.is_dead = True
        return anomalous

class RiskManager:
    """
    Gestor de Riesgo y Position Sizing para el Bot Cuantitativo.
    """
    def __init__(self, risk_per_trade_pct: float = 0.01, k_up: float = 2.0, k_down: float = 1.5, max_hold: int = 10):
        """
        :param risk_per_trade_pct: Porcentaje del balance a arriesgar (ej. 0.01 = 1%)
        :param k_up: Multiplicador de volatilidad para el Take Profit
        :param k_down: Multiplicador de volatilidad para el Stop Loss
        :param max_hold: Maximo de velas a mantener el trade abierto
        """
        self.risk_per_trade_pct = risk_per_trade_pct
        self.k_up = k_up
        self.k_down = k_down
        self.max_hold = max_hold
        self.hybrid_monitor = HybridRiskMonitor()

    def calculate_triple_barrier_levels(self, current_price: float, egarch_vol: float):
        """
        Calcula los precios exactos de Take Profit y Stop Loss basados en la volatilidad actual.
        Nota: Asumimos operaciones de COMPRA (Largo) por ahora, ya que el Label 1 del modelo
        indica que el precio tocará la barrera superior antes que la inferior.
        """
        # La volatilidad EGARCH asume retornos porcentuales, por ej. 0.5 = 0.5%
        vol_decimal = egarch_vol / 100.0 
        
        take_profit_price = current_price * (1 + self.k_up * vol_decimal)
        stop_loss_price = current_price * (1 - self.k_down * vol_decimal)
        
        return take_profit_price, stop_loss_price

    def calculate_position_size(self, balance: float, current_price: float, stop_loss_price: float, 
                                tick_size: float, tick_value: float, volume_step: float,
                                prediction_prob: float = None, confidence_threshold: float = 0.5):
        """
        Calcula el volumen exacto (en lotes) respetando el riesgo máximo de la cuenta.
        Soporta tanto Long (SL abajo del precio) como Short (SL arriba del precio).
        """
        if current_price == stop_loss_price:
            return 0.0
            
        kelly_mult = 1.0
        if prediction_prob is not None:
            if prediction_prob > 0.5:
                delta = prediction_prob - confidence_threshold
            else:
                delta = (1.0 - confidence_threshold) - prediction_prob
                
            delta = max(0, delta)
            if delta <= 0.05:
                kelly_mult = 0.5
            elif delta <= 0.15:
                kelly_mult = 1.0
            else:
                kelly_mult = 2.0
                
        dynamic_risk_pct = self.risk_per_trade_pct * kelly_mult
        risk_amount = balance * dynamic_risk_pct
        
        # Distancia al Stop Loss en precio (valor absoluto para soportar Long y Short)
        sl_distance_price = abs(current_price - stop_loss_price)
        
        # Distancia en ticks
        sl_distance_ticks = sl_distance_price / tick_size
        
        # Valor monetario en riesgo por lote estandar
        risk_per_lot = sl_distance_ticks * tick_value
        
        if risk_per_lot <= 0:
            return 0.0
            
        # Lotes crudos
        raw_lots = risk_amount / risk_per_lot
        
        # Redondear hacia abajo al multiplo mas cercano del volume_step (ej. 0.01)
        precision = abs(int(math.log10(volume_step)))
        rounded_lots = math.floor(raw_lots / volume_step) * volume_step
        rounded_lots = round(rounded_lots, precision)
        
        return rounded_lots
