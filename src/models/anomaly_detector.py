import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
from hmmlearn import hmm

warnings.filterwarnings("ignore")

class StrategyLSTMAutoencoder:
    """
    LSTM Autoencoder adaptado para detectar anomalías en la estrategia de trading.
    Basado en el modelo Predictive Maintenance.
    Entrenado con ventanas de operaciones "Normales" (In-Sample).
    """
    def __init__(self, encoding_dim: int = 16, epochs: int = 100,
                 batch_size: int = 32, learning_rate: float = 1e-3,
                 random_state: int = 42):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self._threshold_mse = None
        self._threshold_p90 = None

    def _build_model(self, input_dim: int) -> keras.Model:
        tf.random.set_seed(self.random_state)
        
        inputs = keras.Input(shape=(1, input_dim))
        noisy_inputs = keras.layers.GaussianNoise(0.05)(inputs)
        
        # Encoder
        encoded = keras.layers.LSTM(32, activation='relu', return_sequences=False)(noisy_inputs)
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu', name='latent_space')(encoded)

        # Decoder
        decoded = keras.layers.RepeatVector(1)(encoded)
        decoded = keras.layers.LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = keras.layers.TimeDistributed(keras.layers.Dense(input_dim, activation='linear'))(decoded)

        autoencoder = keras.Model(inputs, decoded, name='strategy_lstm_autoencoder')
        autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return autoencoder

    def fit(self, X_train: np.ndarray):
        """Entrena el LSTM Autoencoder con las métricas móviles in-sample."""
        print("  🧠 Entrenando LSTM Autoencoder (Operación Normal In-Sample)...")
        keras.backend.clear_session()  # Limpia el grafo viejo de TF de la RAM
        
        X_scaled = self.scaler.fit_transform(X_train)
        X_3d = np.expand_dims(X_scaled, axis=1)

        self.model = self._build_model(X_scaled.shape[1])
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.history = self.model.fit(
            X_3d, X_3d, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=0.15, callbacks=[early_stop], verbose=0
        )

        reconstructed = self.model.predict(X_3d, verbose=0)
        mse = np.mean(np.power(X_3d - reconstructed, 2), axis=(1, 2))
        
        # El threshold será el percentil 99 (Todo por encima de esto se considerará anómalo/muerte)
        self._threshold_mse = np.percentile(mse, 99)
        self._threshold_p90 = np.percentile(mse, 90)
        print(f"  ✅ LSTM Entrenado. Threshold Anómalo (MSE P99): {self._threshold_mse:.6f} | Concept Drift Base (P90): {self._threshold_p90:.6f}")

    def is_anomalous(self, X_window: np.ndarray) -> bool:
        """Determina si la ventana actual de trades es anómala (Kill-Switch)."""
        if len(X_window.shape) == 1:
            X_window = X_window.reshape(1, -1)
            
        X_scaled = self.scaler.transform(X_window)
        X_3d = np.expand_dims(X_scaled, axis=1)
        
        reconstructed = self.model.predict(X_3d, verbose=0)
        mse = np.mean(np.power(X_3d - reconstructed, 2), axis=(1, 2))
        
        # Si el error es mayor al umbral P99 de entrenamiento, es anómalo
        return bool(mse[0] > self._threshold_mse)

    def check_concept_drift(self, recent_X_window: np.ndarray) -> bool:
        """
        Calcula si la Mediana del MSE de la ventana reciente (ej. últimos 30 trades)
        supera la frontera P90 de la distribución original In-Sample.
        Indica si el régimen base cambió estructuralmente y el modelo debe re-entrenarse.
        """
        if self._threshold_p90 is None:
            return False
            
        if len(recent_X_window.shape) == 1:
            recent_X_window = recent_X_window.reshape(1, -1)
            
        X_scaled = self.scaler.transform(recent_X_window)
        # Expandir cada fila independientemente para el autoencoder (N, 1, features)
        X_3d = np.expand_dims(X_scaled, axis=1)
        
        reconstructed = self.model.predict(X_3d, verbose=0)
        mse_array = np.mean(np.power(X_3d - reconstructed, 2), axis=(1, 2))
        
        median_mse = np.median(mse_array)
        return bool(median_mse > self._threshold_p90)

    def save(self, filepath_base: str):
        """Guarda el modelo, el scaler y el umbral para Producción."""
        if self.model is not None:
            self.model.save(f"{filepath_base}.keras")
            joblib.dump({
                'scaler': self.scaler, 
                'threshold': self._threshold_mse,
                'threshold_p90': self._threshold_p90
            }, f"{filepath_base}_meta.pkl")
            
    def load(self, filepath_base: str):
        """Carga el modelo pre-entrenado para Producción."""
        self.model = keras.models.load_model(f"{filepath_base}.keras")
        meta = joblib.load(f"{filepath_base}_meta.pkl")
        self.scaler = meta['scaler']
        self._threshold_mse = meta['threshold']
        self._threshold_p90 = meta.get('threshold_p90', self._threshold_mse * 0.5) # Fallback


class HMMRegimeDetector:
    """
    Hidden Markov Model para detectar el régimen Macro del mercado.
    Detecta si estamos en Mercado Normal (0), Volátil (1) o Crisis (2).
    """
    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="full", random_state=self.random_state, n_iter=100)
        self.scaler = StandardScaler()
        self.crisis_state = None

    def fit(self, X_macro: np.ndarray):
        print("  🕵️ Entrenando Hidden Markov Model para Regímenes Macro...")
        X_scaled = self.scaler.fit_transform(X_macro)
        self.model.fit(X_scaled)
        
        # Inferir cuál es el estado de "Crisis" asumiendo que es el estado con mayor volatilidad (Varianza)
        variances = []
        for i in range(self.n_components):
            state_cov = self.model.covars_[i]
            # Suma de la diagonal (varianza total) de todas las features en ese estado
            variances.append(np.trace(state_cov))
            
        self.crisis_state = np.argmax(variances)
        print(f"  ✅ HMM Entrenado. Estado clasificado como Crisis/Alta Volatilidad: Régimen {self.crisis_state}")

    def predict_regime(self, X_macro_recent: np.ndarray) -> int:
        """Devuelve el régimen actual. Si es igual a self.crisis_state, estamos en Crisis."""
        if len(X_macro_recent.shape) == 1:
            X_macro_recent = X_macro_recent.reshape(1, -1)
            
        X_scaled = self.scaler.transform(X_macro_recent)
        state = self.model.predict(X_scaled)[-1]
        return state

    def is_crisis(self, X_macro_recent: np.ndarray) -> bool:
        return self.predict_regime(X_macro_recent) == self.crisis_state

    def save(self, filepath: str):
        """Guarda el modelo HMM para Producción."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'crisis_state': self.crisis_state
        }, filepath)
        
    def load(self, filepath: str):
        """Carga el modelo HMM pre-entrenado para Producción."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.crisis_state = data['crisis_state']
