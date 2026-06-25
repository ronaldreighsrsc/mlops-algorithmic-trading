import os
import warnings

# Silenciar logs de C++ de TensorFlow ANTES de importarlo (Vital en Windows)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

class LSTMTrainer:
    """
    Motor de Deep Learning usando Long Short-Term Memory (LSTM).
    Maneja internamente la conversión de 2D a tensores 3D, escalamiento local,
    Purged CV y Walk-Forward con limpieza de memoria RAM para evitar fugas.
    """
    def __init__(self, look_back: int = 60, retrain_step: int = 50, 
                 n_splits: int = 3, purge_size: int = 60, embargo_size: int = 10):
        self.look_back = look_back
        self.retrain_step = retrain_step
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        self.scaler = StandardScaler()

    def prepare_sequences(self, df: pd.DataFrame, target_col: str, feature_cols: list) -> tuple:
        """Transforma el DataFrame bidimensional en Tensores 3D para la red neuronal."""
        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        y_array = df_clean[target_col].values
        features = [c for c in feature_cols if c != target_col]
        dataset = df_clean[features].values
        
        X_raw, y_raw = [], []
        for i in range(self.look_back, len(dataset)):
            X_raw.append(dataset[i-self.look_back:i, :])
            y_raw.append(y_array[i])
            
        return np.array(X_raw), np.array(y_raw)

    def _get_purged_embargoed_folds(self, num_samples: int) -> list:
        fold_size = num_samples // self.n_splits
        folds = []
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.n_splits - 1 else num_samples
            train_indices = []
            for j in range(num_samples):
                if j < val_start - self.purge_size:          
                    train_indices.append(j)
                elif j >= val_end + self.embargo_size:       
                    train_indices.append(j)
            val_indices = list(range(val_start, val_end))
            folds.append((np.array(train_indices), np.array(val_indices)))
        return folds

    def _build_model(self, input_shape: tuple, units: int, dropout: float) -> Sequential:
        """Arquitectura base de la red neuronal."""
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units // 2, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def find_best_params(self, X_train: np.ndarray, y_train: np.ndarray, param_distributions: dict, n_iter: int = 3) -> dict:
        """Búsqueda Aleatoria 3D con K-Fold Purged (Lightweight)."""
        print(f"  🔍 Buscando hiperparámetros (Purged & Embargoed CV para LSTM)...")
        tf.keras.backend.clear_session()
        
        # Escalamiento 3D Local estricto para el Grid Search
        s_tr, t_steps, n_feat = X_train.shape
        temp_scaler = StandardScaler()
        X_train_scaled = np.clip(
            temp_scaler.fit_transform(X_train.reshape(-1, n_feat)), -10, 10
        ).reshape(s_tr, t_steps, n_feat)

        folds = self._get_purged_embargoed_folds(s_tr)
        best_acc = -1
        
        from sklearn.model_selection import ParameterSampler
        param_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))
        best_params = param_list[0]

        for params in param_list:
            fold_accs = []
            for train_idx, val_idx in folds:
                if len(train_idx) == 0: continue
                
                model_cv = self._build_model((t_steps, n_feat), units=params.get('units', 64), dropout=params.get('dropout', 0.2))
                es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
                
                model_cv.fit(X_train_scaled[train_idx], y_train[train_idx], epochs=15, batch_size=32, 
                             validation_data=(X_train_scaled[val_idx], y_train[val_idx]), 
                             callbacks=[es], verbose=0, shuffle=True)
                
                preds = (model_cv.predict(X_train_scaled[val_idx], verbose=0) > 0.5).astype(int)
                fold_accs.append(accuracy_score(y_train[val_idx], preds))
                
            avg_acc = np.mean(fold_accs) if fold_accs else 0
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = params

        print(f"  ✅ Ganador: {best_params} (Acc Interno: {best_acc:.2%})")
        return best_params

    def walk_forward_predict(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_test: np.ndarray, y_test: np.ndarray, best_params: dict) -> tuple:
        """Entrena y predice paso a paso reajustando los pesos neuronales."""
        print(f"  🚀 Iniciando Walk-Forward (Reentrenamiento cada {self.retrain_step} días)...")
        tf.keras.backend.clear_session()

        s_tr, t_steps, n_feat = X_train.shape
        s_te = X_test.shape[0]

        # Escalamiento 3D Maestro sin fuga de datos
        X_train_scaled = np.clip(self.scaler.fit_transform(X_train.reshape(-1, n_feat)), -10, 10).reshape(s_tr, t_steps, n_feat)
        X_test_scaled = np.clip(self.scaler.transform(X_test.reshape(-1, n_feat)), -10, 10).reshape(s_te, t_steps, n_feat)

        # Entrenamiento Base
        master_model = self._build_model((t_steps, n_feat), units=best_params['units'], dropout=best_params['dropout'])
        master_model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0, shuffle=True)

        pred_probs = []
        
        for i in range(len(X_test_scaled)):
            x_input = X_test_scaled[i].reshape(1, t_steps, n_feat)
            prob = master_model.predict(x_input, verbose=0)[0][0]
            pred_probs.append(prob)
            
            # Re-entrenamiento periódico (Ligero, 2 epochs para actualizar pesos recientes)
            if (i + 1) % self.retrain_step == 0:
                if (i+1) % (self.retrain_step*2) == 0:
                    print(f"    > Re-calibrando pesos en paso {i+1}/{len(X_test_scaled)}...")
                curr_X = np.concatenate((X_train_scaled, X_test_scaled[:i+1]))
                curr_y = np.concatenate((y_train, y_test[:i+1]))
                master_model.fit(curr_X, curr_y, epochs=2, batch_size=32, verbose=0, shuffle=True)

        # Guardar el modelo final entrenado en la instancia
        self.master_model = master_model
        self.best_params = best_params
        # Las redes neuronales no devuelven Feature Importance clásico
        return np.array(pred_probs), None

    def save(self, filepath: str) -> None:
        """Saves the final trained model, scaler and params."""
        if not hasattr(self, 'master_model'):
            raise ValueError("No model trained yet. Run walk_forward_predict first.")
            
        keras_path = filepath.replace(".pkl", ".keras")
        self.master_model.save(keras_path)
        
        state = {
            'scaler': self.scaler,
            'look_back': self.look_back,
            'best_params': getattr(self, 'best_params', None)
        }
        joblib.dump(state, filepath)
        print(f"  💾 LSTM model saved to {filepath} and {keras_path}")

    @classmethod
    def load(cls, filepath: str):
        """Loads a previously saved model."""
        state = joblib.load(filepath)
        keras_path = filepath.replace(".pkl", ".keras")
        
        instance = cls(look_back=state['look_back'])
        instance.scaler = state['scaler']
        instance.best_params = state.get('best_params', None)
        instance.master_model = tf.keras.models.load_model(keras_path)
        return instance
        
    def fast_retrain(self, df: pd.DataFrame, feature_cols: list, target_col: str = 'Label'):
        """Reajuste ligero de pesos neuronales usando la memoria del master_model existente."""
        if getattr(self, 'master_model', None) is None:
            print("  ⚠️ No master_model found. Cannot fast retrain.")
            return
            
        df_valid = df.dropna(subset=[target_col])
        if len(df_valid) == 0:
            return
            
        X, y = self.prepare_sequences(df_valid, target_col, feature_cols)
        
        X_train = X[-1000:]
        y_train = y[-1000:]
        
        s_tr, t_steps, n_feat = X_train.shape
        X_train_scaled = np.clip(self.scaler.transform(X_train.reshape(-1, n_feat)), -10, 10).reshape(s_tr, t_steps, n_feat)
        
        self.master_model.fit(X_train_scaled, y_train, epochs=2, batch_size=32, verbose=0, shuffle=True)
        print("  ✅ Pesos de LSTM reentrenados exitosamente con datos recientes (2 epochs).")